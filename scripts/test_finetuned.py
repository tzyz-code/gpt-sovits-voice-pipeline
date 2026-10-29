# -*- coding: utf-8 -*-
"""用刚训好的 kei 微调权重做推理，和零样本结果对比。

    python test_finetuned.py
"""
import json
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
# sv.py 里有 sys.path.append(f"{os.getcwd()}/GPT_SoVITS/eres2net") 和相对路径的 sv 底模，
# 所以必须把工作目录切到仓库根
os.chdir(GSV)

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

PM = os.path.join(GSV, "GPT_SoVITS", "pretrained_models")
SOVITS = {
    "e4": f"SoVITS_weights_v2/{CHARACTER}_e4_s256.pth",
    "e8": f"SoVITS_weights_v2/{CHARACTER}_e8_s512.pth",
    "e10": f"SoVITS_weights_v2/{CHARACTER}_e10_s640.pth",
}
GPTS = {
    "g9": f"GPT_weights_v2/{CHARACTER}-e9.ckpt",
    "g15": f"GPT_weights_v2/{CHARACTER}-e15.ckpt",
}
# 微调后模型本身就是 Kei，参考音频只要干净、语气中性即可
from paths import REF_DIR
REF = os.path.join(REF_DIR, "Cafe_monolog_4.wav")
REF_TEXT = "自分の身体を持つのは……。不便なことでもあるんですね……。"

LINES = [
    ("ja", "何をしているんですか？", [42, 1024]),
    ("zh", "你这是在干什么？", [42]),
]

cfg_common = {
    "device": "cuda",
    "is_half": True,
    "version": "v2",
    "cnhuhbert_base_path": os.path.join(PM, "chinese-hubert-base"),
    "bert_base_path": os.path.join(PM, "chinese-roberta-wwm-ext-large"),
}


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    made = []
    for sk, srel in SOVITS.items():
        for gk, grel in GPTS.items():
            if only and only not in (sk, gk):
                continue
            # 必须包一层 custom：TTS_Config 内部是 configs_.get("custom", configs_["v2"])，
            # 直接传扁平字典会被忽略、一直用 v2 底模（表现为各档输出字节完全相同）
            cfg = {"custom": dict(cfg_common)}
            cfg["custom"]["vits_weights_path"] = os.path.join(GSV, srel)
            cfg["custom"]["t2s_weights_path"] = os.path.join(GSV, grel)
            t0 = time.time()
            tts = TTS(TTS_Config(cfg))
            print(f"[{sk}+{gk}] 加载 {time.time()-t0:.1f}s", flush=True)
            for lang, text, seeds in LINES:
                for seed in seeds:
                    inputs = {
                        "text": text,
                        "text_lang": lang,
                        "ref_audio_path": REF,
                        "prompt_text": REF_TEXT,
                        "prompt_lang": "ja",
                        "top_k": 15, "top_p": 1.0, "temperature": 1.0,
                        "text_split_method": "cut5", "batch_size": 1,
                        "speed_factor": 1.0, "seed": seed,
                        "repetition_penalty": 1.35, "sample_steps": 32,
                        "super_sampling": False,
                        "return_fragment": False, "fragment_interval": 0.3,
                        "parallel_infer": True, "split_bucket": True,
                    }
                    name = f"{lang}_{sk}_{gk}_s{seed}.wav"
                    path = os.path.join(OUT, name)
                    t1 = time.time()
                    for sr, audio in tts.run(inputs):
                        sf.write(path, audio, sr)
                    print(f"  {name}  {time.time()-t1:.1f}s", flush=True)
                    made.append(name)
    print(f"\n共 {len(made)} 条 -> {OUT}")
    json.dump(made, open(os.path.join(OUT, "index.json"), "w", encoding="utf-8"))


if __name__ == "__main__":
    main()
