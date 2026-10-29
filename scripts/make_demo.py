# -*- coding: utf-8 -*-
"""用训练集之外的新台词验证泛化，并固定「日语 g15 / 中文 g9」的搭配。

    python make_demo.py
"""
import json
import os
import sys
import time

import soundfile as sf

from paths import CHARACTER
from paths import GSV
HERE = os.path.dirname(os.path.abspath(__file__))
from paths import DEMO_DIR as OUT
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.join(GSV, "GPT_SoVITS"))
sys.path.insert(0, GSV)
os.chdir(GSV)

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

PM = os.path.join(GSV, "GPT_SoVITS", "pretrained_models")
REF = ("Cafe_monolog_4.wav", "自分の身体を持つのは……。不便なことでもあるんですね……。")

# 用户听下来的结论：日语 GPT 15 轮、中文 GPT 9 轮，SoVITS 都是 e8
BEST = {"ja": "15", "zh": "9"}
SEED = {"ja": 1024, "zh": 1024}
TEMP = {"ja": 0.8, "zh": 0.8}

# 全部是训练集里没有的句子
LINES = {
    "ja": [
        "先生、そんなに急いでどうしたんですか？",
        "少し休んだほうがいいですよ。",
        "私は、先生を信じています。",
        "これは……なんというか、不思議な感じですね。",
    ],
    "zh": [
        "老师，你今天看起来很累。",
        "我会一直在这里的。",
        "这种事情，还是让我来吧。",
        "你真的打算这么做吗？",
    ],
}
# 同一条台词换种子，看稳定性
SEED_TEST = ("ja", "少し休んだほうがいいですよ。", [42, 1024, 7])


def build(lang, gpt_ver):
    sov = os.path.join(GSV, "SoVITS_weights_v2", f"{CHARACTER}_e8_s512.pth")
    gpt = os.path.join(GSV, "GPT_weights_v2", f"{CHARACTER}-e{gpt_ver}.ckpt")
    cfg = {"custom": {
        "device": "cuda", "is_half": True, "version": "v2",
        "cnhuhbert_base_path": os.path.join(PM, "chinese-hubert-base"),
        "bert_base_path": os.path.join(PM, "chinese-roberta-wwm-ext-large"),
        "vits_weights_path": sov, "t2s_weights_path": gpt,
    }}
    return TTS(TTS_Config(cfg))


def gen(tts, text, lang, seed, name, speed=1.2, temp=1.0):
    inputs = {
        "text": text, "text_lang": lang,
        "ref_audio_path": os.path.join(REF_DIR, REF[0]),
        "prompt_text": REF[1], "prompt_lang": "ja",
        "top_k": 15, "top_p": 1.0, "temperature": temp,
        "text_split_method": "cut5", "batch_size": 1,
        "speed_factor": speed, "seed": seed,
        "repetition_penalty": 1.35, "sample_steps": 32,
        "super_sampling": False, "return_fragment": False,
        "fragment_interval": 0.3, "parallel_infer": True, "split_bucket": True,
    }
    t0 = time.time()
    for sr, audio in tts.run(inputs):
        sf.write(os.path.join(OUT, name), audio, sr)
    print(f"  {name}  {time.time()-t0:.1f}s", flush=True)


def main():
    meta = []
    for lang in ("ja", "zh"):
        tts = build(lang, BEST[lang])
        print(f"[{lang}] SoVITS e8 + GPT {BEST[lang]} 已加载", flush=True)
        for i, text in enumerate(LINES[lang], 1):
            name = f"{lang}_{i:02d}_g{BEST[lang]}_s{SEED[lang]}.wav"
            gen(tts, text, lang, SEED[lang], name, temp=TEMP[lang])
            meta.append({"file": name, "lang": lang, "text": text,
                         "gpt": BEST[lang], "seed": SEED[lang], "temp": TEMP[lang]})
        # 稳定性：换三个种子
        if lang == SEED_TEST[0]:
            _, text, seeds = SEED_TEST
            for s in seeds:
                name = f"{lang}_seed_s{s}.wav"
                gen(tts, text, lang, s, name, temp=TEMP[lang])
                meta.append({"file": name, "lang": lang, "text": text,
                             "gpt": BEST[lang], "seed": s, "temp": TEMP[lang]})
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n共 {len(meta)} 条 -> {OUT}")


if __name__ == "__main__":
    main()
