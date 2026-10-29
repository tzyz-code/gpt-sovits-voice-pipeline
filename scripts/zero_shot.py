"""第 0 步：GPT-SoVITS 零样本合成 —— 用 Kei 的参考音频念指定台词。

不训练，直接用 v2 底模 + 参考音频。用途是判断"值不值得微调"。

用法：
    python zero_shot.py
    （需先启动 GPT-SoVITS 的 api_v2.py）
"""
from __future__ import annotations

import json
import os
import sys

import httpx

API = "http://127.0.0.1:9880/tts"
from paths import ROOT
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

# 参考音频 -> 其准确的日文台词（来自 wiki 的 Japanese 列，不是英译）
REFS = {
    "neutral": {
        "path": os.path.join(ROOT, "ref", "Cafe_monolog_4.wav"),
        "text": "自分の身体を持つのは……。不便なことでもあるんですね……。",
        "note": "陈述句、无笑声、RMS 最高（0.163）——最适合当主参考",
    },
    "smile": {
        "path": os.path.join(ROOT, "ref", "Cafe_monolog_5.wav"),
        "text": "ふふ……こうして立っているだけでも、なんだか楽しいです……。",
        "note": "带轻微笑意，用来对比语气是否会被参考带跑",
    },
}

LINES = {
    "zh": "你这是在干什么？",
    "ja": "何をしているんですか？",
}

SEEDS = [42, 1024, 20261026]


def synth(text: str, lang: str, ref: dict, seed: int, dst: str) -> bool:
    payload = {
        "text": text,
        "text_lang": lang,
        "ref_audio_path": ref["path"],
        "prompt_text": ref["text"],
        "prompt_lang": "ja",
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut5",
        "batch_size": 1,
        "speed_factor": 1.0,
        "seed": seed,
        "media_type": "wav",
        "repetition_penalty": 1.35,
        "sample_steps": 32,
        "super_sampling": False,
    }
    try:
        r = httpx.post(API, json=payload, timeout=300.0)
    except Exception as e:
        print(f"  请求失败: {e}")
        return False
    if r.status_code != 200:
        print(f"  失败 {r.status_code}: {r.text[:200]}")
        return False
    with open(dst, "wb") as f:
        f.write(r.content)
    return os.path.getsize(dst) > 2000


def main():
    ok, fail = 0, 0
    for lang, text in LINES.items():
        for ref_name, ref in REFS.items():
            seeds = SEEDS if ref_name == "neutral" else SEEDS[:1]
            for seed in seeds:
                dst = os.path.join(OUT, f"kei_{lang}_{ref_name}_s{seed}.wav")
                print(f"[{lang}] 参考={ref_name} seed={seed} …", flush=True)
                if synth(text, lang, ref, seed, dst):
                    ok += 1
                    print(f"  -> {os.path.basename(dst)}")
                else:
                    fail += 1
    print(f"\n完成：成功 {ok}，失败 {fail}，输出目录 {OUT}")


if __name__ == "__main__":
    main()
