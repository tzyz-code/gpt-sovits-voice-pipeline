# -*- coding: utf-8 -*-
"""情绪模型 A/B 对比：把「模型」和「参考音频」两个变量拆开。

四组对照（每组用同一批台词、同一套采样参数）：

    A  通用模型 + 中性参考    ← 现状基线
    B  通用模型 + 慌张参考    ← 只换参考音频，模型没变
    C  慌张模型 + 慌张参考    ← 完整情绪方案
    D  慌张模型 + 中性参考    ← 只换模型，参考音频没变

只有这样拆才能回答「情绪是模型学到的，还是参考音频带进来的」。
只做 C vs A 是分不清的：C 好也可能纯粹因为参考音频是慌张语气（原理三：参考音频的语气
会感染全部输出）。

    python test_emotion.py
    python test_emotion.py --sovits 16 --gpt 30
"""
import argparse
import json
import os
import sys

import soundfile as sf

from paths import WAV_DIR
from paths import GSV
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(GSV, "GPT_SoVITS"))
sys.path.insert(0, GSV)
os.chdir(GSV)

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

PM = os.path.join(GSV, "GPT_SoVITS", "pretrained_models")
from paths import EMO_OUT as OUT

# 参考音频：(wav, 台词)。都是用这套素材自己的原声，台词是确定准确的。
REF_N = (os.path.join(WAV_DIR, "Kei_Cafe_monolog_4.wav"),
         "自分の身体を持つのは……。不便なことでもあるんですね……。")
REF_F = (os.path.join(WAV_DIR, "Kei_EventShop_Buy_3_3_Radio.wav"),
         "本当に……先生がいないと、ダメなんですよ、私たち。")

# 训练集里没有的台词：三句慌张（1~3）+ 一句平静对照（4）。
# 第 4 句是关键——如果它也被说得慌张，说明模型把整个人格都改偏了，而不是学会了情绪切换。
LINES = [
    ("f1", "えっ、ええっ！？ちょ、ちょっと待ってください、そんな急に言われても困ります……！"),
    ("f2", "もう！せっかく片付けたのに、また散らかしたんですか！？"),
    ("f3", "ふ、ふざけないでください！私は真面目に言ってるんですよ！？"),
    ("n1", "今日は、静かな一日でしたね。"),
]

SOLANG = {"ja": "ja"}


def sov_key(tag, epoch, step):
    return os.path.join(GSV, "SoVITS_weights_v2", f"{tag}_e{epoch}_s{step}.pth")


def gpt_key(tag, epoch):
    return os.path.join(GSV, "GPT_weights_v2", f"{tag}-e{epoch}.ckpt")


def load(sov, gpt):
    cfg = {"custom": {
        "device": "cuda", "is_half": True, "version": "v2",
        "cnhuhbert_base_path": os.path.join(PM, "chinese-hubert-base"),
        "bert_base_path": os.path.join(PM, "chinese-roberta-wwm-ext-large"),
        "vits_weights_path": sov, "t2s_weights_path": gpt,
    }}
    return TTS(TTS_Config(cfg))


def say(tts, text, ref, lang, speed, temp, seed):
    inp = {
        "text": text, "text_lang": lang,
        "ref_audio_path": ref[0], "prompt_text": ref[1], "prompt_lang": "ja",
        "top_k": 15, "top_p": 1.0, "temperature": temp,
        "text_split_method": "cut5", "batch_size": 1,
        "speed_factor": speed, "seed": seed,
        "repetition_penalty": 1.35, "sample_steps": 32,
        "super_sampling": False, "return_fragment": False,
        "fragment_interval": 0.3, "parallel_infer": True, "split_bucket": True,
    }
    for sr, audio in tts.run(inp):
        return sr, audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sovits", default="8", help="慌张模型 SoVITS 轮次：4/8/12/16")
    ap.add_argument("--gpt", default="15", help="慌张模型 GPT 轮次：15/30")
    ap.add_argument("--speed", type=float, default=1.2)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1024)
    a = ap.parse_args()

    S2 = {"4": 192, "8": 384, "12": 576, "16": 768}
    # 通用模型固定用定档的 e8 + GPT15（第十四节结论）
    combos = [
        ("A", "通用模型+中性参考", sov_key("kei", 8, 512), gpt_key("kei", 15), REF_N),
        ("B", "通用模型+慌张参考", sov_key("kei", 8, 512), gpt_key("kei", 15), REF_F),
        ("C", f"慌张模型+慌张参考(e{a.sovits}+{a.gpt})",
         sov_key("fluster", a.sovits, S2[a.sovits]), gpt_key("fluster", a.gpt), REF_F),
        ("D", f"慌张模型+中性参考(e{a.sovits}+{a.gpt})",
         sov_key("fluster", a.sovits, S2[a.sovits]), gpt_key("fluster", a.gpt), REF_N),
    ]

    missing = [(k[0], p) for k in combos for p in k[2:4] if not os.path.isfile(p)]
    if missing:
        sys.exit("权重缺失：\n" + "\n".join(f"  {n} {p}" for n, p in missing))

    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for tag, label, sov, gpt, ref in combos:
        print(f"\n[{tag}] {label}\n    SoVITS {os.path.basename(sov)} | GPT {os.path.basename(gpt)}")
        tts = load(sov, gpt)
        for lid, text in LINES:
            name = f"{tag}_{lid}.wav"
            sr, audio = say(tts, text, ref, "ja", a.speed, a.temp, a.seed)
            p = os.path.join(OUT, name)
            sf.write(p, audio, sr)
            manifest.append({"tag": tag, "label": label, "id": lid,
                             "text": text, "wav": p,
                             "dur": round(len(audio) / sr, 2)})
            print(f"    {lid} {round(len(audio)/sr,2):>5}s  {text[:22]}…")
        del tts

    mp = os.path.join(OUT, "manifest.json")
    json.dump(manifest, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n{len(manifest)} 条 -> {OUT}\n清单 -> {mp}")


if __name__ == "__main__":
    main()
