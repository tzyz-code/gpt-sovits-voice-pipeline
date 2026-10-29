# -*- coding: utf-8 -*-
"""验证「语速偏慢」能否用 speed_factor 直接补回来。

微调后中文六条全是有声 3.2s，而零样本是 2.5s —— 模型整体语速慢了下来。
这里用最优档（e8+g15 / e8+g9）加 speed_factor 生成对照。

    python test_speed.py
"""
import os
import sys
import time

import soundfile as sf

from paths import CHARACTER
from paths import GSV
HERE = os.path.dirname(os.path.abspath(__file__))
from paths import FT_OUT as OUT
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.join(GSV, "GPT_SoVITS"))
sys.path.insert(0, GSV)
os.chdir(GSV)

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

PM = os.path.join(GSV, "GPT_SoVITS", "pretrained_models")
from paths import REF_DIR
REF = os.path.join(REF_DIR, "Cafe_monolog_4.wav")
REF_TEXT = "自分の身体を持つのは……。不便なことでもあるんですね……。"

JOBS = [
    ("e8_g15", "ja", "何をしているんですか？", 42, 1.2),
    ("e8_g15", "ja", "何をしているんですか？", 1024, 1.2),
    ("e8_g15", "zh", "你这是在干什么？", 42, 1.2),
    ("e8_g9", "zh", "你这是在干什么？", 42, 1.2),
]

WEIGHTS = {
    "e8_g15": (f"SoVITS_weights_v2/{CHARACTER}_e8_s512.pth", f"GPT_weights_v2/{CHARACTER}-e15.ckpt"),
    "e8_g9": (f"SoVITS_weights_v2/{CHARACTER}_e8_s512.pth", f"GPT_weights_v2/{CHARACTER}-e9.ckpt"),
}


def main():
    cache = {}
    for tag, lang, text, seed, spd in JOBS:
        if tag not in cache:
            s, g = WEIGHTS[tag]
            cfg = {"custom": {
                "device": "cuda", "is_half": True, "version": "v2",
                "cnhuhbert_base_path": os.path.join(PM, "chinese-hubert-base"),
                "bert_base_path": os.path.join(PM, "chinese-roberta-wwm-ext-large"),
                "vits_weights_path": os.path.join(GSV, s),
                "t2s_weights_path": os.path.join(GSV, g),
            }}
            cache[tag] = TTS(TTS_Config(cfg))
            print(f"[{tag}] 已加载", flush=True)
        tts = cache[tag]
        inputs = {
            "text": text, "text_lang": lang,
            "ref_audio_path": REF, "prompt_text": REF_TEXT, "prompt_lang": "ja",
            "top_k": 15, "top_p": 1.0, "temperature": 1.0,
            "text_split_method": "cut5", "batch_size": 1,
            "speed_factor": spd, "seed": seed,
            "repetition_penalty": 1.35, "sample_steps": 32,
            "super_sampling": False, "return_fragment": False,
            "fragment_interval": 0.3, "parallel_infer": True, "split_bucket": True,
        }
        name = f"{lang}_{tag}_s{seed}_spd{spd}.wav"
        t0 = time.time()
        for sr, audio in tts.run(inputs):
            sf.write(os.path.join(OUT, name), audio, sr)
        print(f"  {name}  {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
