"""第 0.5 步：日语零样本参数扫描 + 声纹打分。

背景：3 条日语里只有 1 条能用，怀疑是零样本的种子不稳定。
本脚本按「参考音频 × 种子 × temperature」扫一遍，再用已有的 Kei 声纹质心
（assets/centroid_<角色>.npy，由该角色全部语音算出）给每条打分，
把"像不像"从主观感觉变成可排序的数字。

指标：
  kei   与 Kei 质心的余弦相似度（越高越像 Kei）
  f0    中位基频 / 帧间标准差（标准差大 = 音高抖）
  hf    6kHz 以上能量占比（偏高 = 金属感/嘶声重）
  clip  削波采样点数（必须为 0）

用法：
    python sweep_ja.py
"""
from __future__ import annotations

import glob
import json
import os

import httpx
import librosa
import numpy as np
import soundfile as sf

API = "http://127.0.0.1:9880/tts"
from paths import CHARACTER
from paths import ROOT
OUT = os.path.join(ROOT, "sweep")
os.makedirs(OUT, exist_ok=True)

TEXT = "何をしているんですか"

# 参考音频 + 精确日文台词（wiki 的 Japanese 列；去掉「」（）这类转录标记）
REFS = {
    "mono4": ("Cafe_monolog_4.wav", "自分の身体を持つのは……。不便なことでもあるんですね……。"),
    "mono3": ("Cafe_monolog_3.wav", "なるほど……これが、まったりするということ……。"),
    "mono1": ("Cafe_monolog_1.wav", "私だって生徒ですから……カフェくらい……！"),
    "mono5": ("Cafe_monolog_5.wav", "ふふ……こうして立っているだけでも、なんだか楽しいです……。"),
}

SEEDS = [42, 1024, 20261026, 7, 31415]
TEMPS = [1.0, 0.8]

# 声纹质心（口径统一在 voiceprint.py）
import voiceprint  # noqa: E402
from paths import ASSETS as W  # noqa: E402

MU, SD, CK = voiceprint.load(W, CHARACTER)


def kei_sim(path: str) -> float:
    return voiceprint.score(path, MU, SD, CK)


def stats(path: str) -> dict:
    y, sr = librosa.load(path, sr=32000, mono=True)
    f0, _, _ = librosa.pyin(y, fmin=70, fmax=700, sr=sr, frame_length=1024, hop_length=256)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    n = min(len(f0), len(rms))
    f0, rms = f0[:n], rms[:n]
    voiced = f0[(rms > np.percentile(rms, 40)) & np.isfinite(f0)]
    # 只统计连续有声段的帧间波动（跨静音的巨大跳变要排除）
    d = np.abs(np.diff(voiced)) if len(voiced) > 2 else np.array([0.0])
    spec = np.abs(librosa.stft(y, n_fft=2048)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hf = float(spec[freqs > 6000].sum() / (spec.sum() + 1e-9))
    a, _ = sf.read(path, dtype="float32")
    return {
        "dur": round(len(y) / sr, 2),
        "f0": round(float(np.median(voiced)), 0) if len(voiced) else 0,
        "f0_jitter": round(float(np.median(d)), 1) if len(voiced) else 0,
        "hf": round(hf, 4),
        "clip": int((np.abs(a) >= 0.999).sum()),
        "kei": round(kei_sim(path), 3),
    }


def synth(ref_path, ref_text, seed, temp, dst) -> bool:
    payload = {
        "text": TEXT, "text_lang": "ja",
        "ref_audio_path": ref_path, "prompt_text": ref_text, "prompt_lang": "ja",
        "top_k": 15, "top_p": 1.0, "temperature": temp,
        "text_split_method": "cut5", "batch_size": 1,
        "speed_factor": 1.0, "seed": seed, "media_type": "wav",
        "repetition_penalty": 1.35, "sample_steps": 32, "super_sampling": False,
    }
    try:
        r = httpx.post(API, json=payload, timeout=300.0)
    except Exception as e:
        print(f"    请求失败 {e}")
        return False
    if r.status_code != 200:
        print(f"    失败 {r.status_code} {r.text[:120]}")
        return False
    open(dst, "wb").write(r.content)
    return os.path.getsize(dst) > 2000


def main():
    jobs = []
    for rk, (rf, rt) in REFS.items():
        for s in SEEDS:
            jobs.append((rk, rf, rt, s, 1.0))
    for s in SEEDS[:3]:
        jobs.append(("mono4", *REFS["mono4"][:1], REFS["mono4"][1], s, 0.8))

    print(f"共 {len(jobs)} 条待合成\n", flush=True)
    for i, (rk, rf, rt, s, t) in enumerate(jobs, 1):
        dst = os.path.join(OUT, f"ja_{rk}_t{t}_s{s}.wav")
        print(f"[{i}/{len(jobs)}] {rk} temp={t} seed={s}", flush=True)
        synth(os.path.join(ROOT, "ref", rf), rt, s, t, dst)

    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "ja_*.wav"))):
        d = stats(p)
        d["file"] = os.path.basename(p)
        rows.append(d)

    rows.sort(key=lambda r: -r["kei"])
    print(f"\n{'文件':26s}{'kei':>7s}{'时长':>7s}{'F0':>6s}{'抖动':>7s}{'hf':>8s}{'削波':>5s}")
    print("-" * 68)
    for r in rows:
        flag = "  <<" if r["clip"] else ""
        print(f"{r['file']:26s}{r['kei']:7.3f}{r['dur']:6.2f}s{r['f0']:6.0f}"
              f"{r['f0_jitter']:7.1f}{r['hf']:8.4f}{r['clip']:5d}{flag}")

    json.dump(rows, open(os.path.join(OUT, "sweep_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n明细 {os.path.join(OUT, 'sweep_result.json')}")


if __name__ == "__main__":
    main()
