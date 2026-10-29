# -*- coding: utf-8 -*-
"""专治中文句首多出来的「哼」：试温度 / 种子 / repetition_penalty。

    python test_zh_filler.py
"""
import json
import os
import sys
import time

import soundfile as sf

from paths import CHARACTER
from paths import REF_DIR
from paths import GSV
HERE = os.path.dirname(os.path.abspath(__file__))
from paths import OUT_DIR
OUT = os.path.join(OUT_DIR, "zh_fix")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.join(GSV, "GPT_SoVITS"))
sys.path.insert(0, GSV)
os.chdir(GSV)

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

PM = os.path.join(GSV, "GPT_SoVITS", "pretrained_models")
REF = ("Cafe_monolog_4.wav", "自分の身体を持つのは……。不便なことでもあるんですね……。")

# 这两句最容易在开头冒出「哼」
TEXTS = ["我会一直在这里的。", "这种事情，还是让我来吧。"]
# (温度, 种子, repetition_penalty)
VARIANTS = [
    (1.0, 42, 1.35),   # 现状基线
    (0.8, 42, 1.35),
    (0.8, 1024, 1.35),
    (0.9, 7, 1.35),
    (0.8, 1024, 1.5),
    (0.7, 1024, 1.35),
]


def main():
    cfg = {"custom": {
        "device": "cuda", "is_half": True, "version": "v2",
        "cnhuhbert_base_path": os.path.join(PM, "chinese-hubert-base"),
        "bert_base_path": os.path.join(PM, "chinese-roberta-wwm-ext-large"),
        "vits_weights_path": os.path.join(GSV, "SoVITS_weights_v2", f"{CHARACTER}_e8_s512.pth"),
        "t2s_weights_path": os.path.join(GSV, "GPT_weights_v2", f"{CHARACTER}-e9.ckpt"),
    }}
    tts = TTS(TTS_Config(cfg))
    print("e8 + g9 已加载", flush=True)
    meta = []
    for ti, text in enumerate(TEXTS, 1):
        for temp, seed, rp in VARIANTS:
            name = f"zh_t{ti}_temp{temp}_s{seed}_rp{rp}.wav"
            inputs = {
                "text": text, "text_lang": "zh",
                "ref_audio_path": os.path.join(REF_DIR, REF[0]),
                "prompt_text": REF[1], "prompt_lang": "ja",
                "top_k": 15, "top_p": 1.0, "temperature": temp,
                "text_split_method": "cut5", "batch_size": 1,
                "speed_factor": 1.2, "seed": seed,
                "repetition_penalty": rp, "sample_steps": 32,
                "super_sampling": False, "return_fragment": False,
                "fragment_interval": 0.3, "parallel_infer": True, "split_bucket": True,
            }
            t0 = time.time()
            for sr, audio in tts.run(inputs):
                sf.write(os.path.join(OUT, name), audio, sr)
            meta.append({"file": name, "text": text, "temp": temp,
                         "seed": seed, "rp": rp})
            print(f"  {name}  {time.time()-t0:.1f}s", flush=True)
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n共 {len(meta)} 条 -> {OUT}")


if __name__ == "__main__":
    main()
