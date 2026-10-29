# -*- coding: utf-8 -*-
"""给微调结果 + 零样本基线 + 真·Kei 原声一起打分。

指标沿用 sweep_ja.py：kei=与 Kei 声纹质心的余弦、f0、hf(6k+ 能量占比)、clip。
把"像不像"变成可排序数字，避免只靠印象。

    python score_ft.py
"""
import glob
import json
import os

import librosa
import numpy as np
import soundfile as sf

import voiceprint
from paths import CHARACTER
from paths import ASSETS as W

# 声纹特征与打分口径统一在 voiceprint.py，别在这里另写一份
MU, SD, CK = voiceprint.load(W, CHARACTER)


def kei_sim(path):
    return voiceprint.score(path, MU, SD, CK)


def stats(path):
    y, sr = librosa.load(path, sr=32000, mono=True)
    f0, _, _ = librosa.pyin(y, fmin=70, fmax=700, sr=sr, frame_length=1024, hop_length=256)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    n = min(len(f0), len(rms))
    f0, rms = f0[:n], rms[:n]
    voiced = f0[(rms > np.percentile(rms, 40)) & np.isfinite(f0)] if n else np.array([])
    spec = np.abs(librosa.stft(y, n_fft=2048)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hf = float(spec[freqs > 6000].sum() / (spec.sum() + 1e-9))
    a, _ = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    return {
        "dur": round(len(y) / sr, 2),
        "f0": int(np.median(voiced)) if len(voiced) else 0,
        "hf": round(hf, 4),
        "clip": int((np.abs(a) >= 0.999).sum()),
    }


def scan(pattern, tag):
    out = []
    for p in sorted(glob.glob(pattern)):
        d = stats(p)
        d.update(name=os.path.basename(p), group=tag, kei=round(kei_sim(p), 3))
        out.append(d)
    return out


def main():
    rows = []
    rows += scan(os.path.join(ROOT, "ft_out", "*.wav"), "微调")
    rows += scan(os.path.join(ROOT, "out", "*.wav"), "零样本")
    # 真 Kei 原声当上限参考
    rows += scan(os.path.join(ROOT, "ref_raw", "*.wav"), "原声")
    if not any(r["group"] == "原声" for r in rows):
        rows += scan(os.path.join(ROOT, "ref", "*.wav"), "原声")

    rows.sort(key=lambda r: -r["kei"])
    json.dump(rows, open(os.path.join(ROOT, "ft_score.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"{'组':<6}{'kei':>7}{'时长':>7}{'F0':>6}{'6k+':>7}{'削波':>5}  文件")
    for r in rows:
        print(f"{r['group']:<6}{r['kei']:>7.3f}{r['dur']:>7.1f}{r['f0']:>6}"
              f"{r['hf']:>7.3f}{r['clip']:>5}  {r['name']}")


if __name__ == "__main__":
    main()
