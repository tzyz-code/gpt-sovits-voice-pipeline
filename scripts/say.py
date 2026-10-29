# -*- coding: utf-8 -*-
"""让角色念任意台词（命令行，不开 WebUI）。

    python say.py "何をしているんですか" ja
    python say.py "你这是在干什么？" zh --speed 1.2 --seed 1024
    python say.py "台詞" ja --ref-wav ref/my_ref.wav --ref-text "参考音频里念的字"

参数：
    台词        必填
    语种        ja / zh / en，默认取 config.yaml 的 language
    --speed     语速倍率，默认 1.2（微调后基准语速偏慢约 1.3 倍，1.2 接近原声）
    --temp      采样温度，默认 0.8（1.0 时中文容易在句首冒出多余语气词）
    --seed      随机种子，默认 1024（种子会影响语速和句首语气，换种子等于换一版演绎）
    --sovits    SoVITS 档位，默认 e8（见 docs/04-evaluation.md 的定档结论）
    --gpt       GPT 档位，默认按语种：日语 15（更自然）、中文 9（更准）
    --ref       参考音频预设名（见 ref/ 目录），默认用 ref/ 里第一个 wav
    --ref-wav   直接指定参考音频路径，优先级高于 --ref
    --ref-text  参考音频念的台词。GPT-SoVITS 要求参考音频必须有准确台词
    --out       输出路径，默认 output/say/<时间戳>_<语种>.wav

关于参考音频：仓库不附带任何音频（版权原因），第一次用之前先跑
    python build_refs.py
从自己的数据集里挑几条，或者手动把 3~10 秒、台词确定的片段放进 ref/。
"""
import argparse
import glob
import os
import sys
import time

import soundfile as sf

from paths import CHARACTER, GSV, LANGUAGE, OUT_DIR, REF_DIR

sys.path.insert(0, os.path.join(GSV, "GPT_SoVITS"))
sys.path.insert(0, GSV)
os.chdir(GSV)  # sv.py 和底模路径依赖 cwd

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

PM = os.path.join(GSV, "GPT_SoVITS", "pretrained_models")

# SoVITS 档位 -> 训练步数。文件名形如 <角色>_e8_s512.pth
SOV_STEPS = {"e4": 256, "e6": 384, "e8": 512, "e10": 640}


def find_sovits(tier):
    pat = os.path.join(GSV, "SoVITS_weights_v2", f"{CHARACTER}_{tier}_s*.pth")
    hit = sorted(glob.glob(pat))
    if not hit:
        have = [os.path.basename(p) for p in
                glob.glob(os.path.join(GSV, "SoVITS_weights_v2", f"{CHARACTER}_*.pth"))]
        sys.exit(f"找不到 SoVITS 权重：{pat}\n"
                 f"  当前已有：{have or '（空）'}\n"
                 f"  先按 docs/03-training.md 训一遍。")
    return hit[-1]


def find_gpt(ver):
    p = os.path.join(GSV, "GPT_weights_v2", f"{CHARACTER}-e{ver}.ckpt")
    if not os.path.isfile(p):
        have = [os.path.basename(x) for x in
                glob.glob(os.path.join(GSV, "GPT_weights_v2", f"{CHARACTER}-e*.ckpt"))]
        sys.exit(f"找不到 GPT 权重：{p}\n  当前已有：{have or '（空）'}")
    return p


def pick_ref(name, ref_wav, ref_text):
    """解析参考音频。没给就自动在 ref/ 里挑一个。"""
    if ref_wav:
        wav = ref_wav
    elif name:
        cand = glob.glob(os.path.join(REF_DIR, f"{name}.*"))
        cand = [c for c in cand if c.lower().endswith((".wav", ".flac"))]
        if not cand:
            sys.exit(f"ref/ 里没有名为 {name} 的音频。当前有："
                     f"{[os.path.basename(x) for x in glob.glob(os.path.join(REF_DIR, '*.wav'))]}")
        wav = cand[0]
    else:
        cand = sorted(glob.glob(os.path.join(REF_DIR, "*.wav")))
        if not cand:
            sys.exit("ref/ 是空的。仓库不附带音频，先跑 build_refs.py 从数据集里挑，\n"
                     "或者手动放一条 3~10 秒、台词确定的音频进去。")
        wav = cand[0]

    if not os.path.isfile(wav):
        sys.exit(f"参考音频不存在：{wav}")

    # 台词：显式给 > 同名 .txt > 报错。参考台词错了，整句韵律都会跟着错。
    text = ref_text
    if not text:
        side = os.path.splitext(wav)[0] + ".txt"
        if os.path.isfile(side):
            text = open(side, encoding="utf-8").read().strip()
    if not text:
        sys.exit(f"参考音频缺少台词：{wav}\n"
                 f"  在旁边放一个同名 .txt，或用 --ref-text 传进去。\n"
                 f"  参考音频的台词必须准确，否则输出的韵律会被带偏。")
    return wav, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("lang", nargs="?", default=LANGUAGE, choices=["ja", "zh", "en"])
    ap.add_argument("--speed", type=float, default=1.2)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--sovits", default="e8")
    ap.add_argument("--gpt", default=None)
    ap.add_argument("--ref", default=None, help="ref/ 下的预设名，默认自动挑第一个")
    ap.add_argument("--ref-wav", default=None)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # 语种差异来自实测：日语 GPT 15 轮更自然，中文 9 轮更少念错字
    if a.gpt is None:
        a.gpt = "15" if a.lang == "ja" else "9"

    sov, gpt = find_sovits(a.sovits), find_gpt(a.gpt)
    ref_wav, ref_text = pick_ref(a.ref, a.ref_wav, a.ref_text)

    cfg = {"custom": {
        "device": "cuda", "is_half": True, "version": "v2",
        "cnhuhbert_base_path": os.path.join(PM, "chinese-hubert-base"),
        "bert_base_path": os.path.join(PM, "chinese-roberta-wwm-ext-large"),
        "vits_weights_path": sov, "t2s_weights_path": gpt,
    }}
    # 注意：配置必须包一层 "custom"。TTS_Config 里是 configs_.get("custom", configs_["v2"])，
    # 传扁平字典会被静默忽略、一直用底模——而且不报错，最容易误判成「微调没用」。
    tts = TTS(TTS_Config(cfg))

    inputs = {
        "text": a.text, "text_lang": a.lang,
        "ref_audio_path": ref_wav, "prompt_text": ref_text,
        "prompt_lang": "ja" if os.path.basename(ref_wav) else a.lang,
        "top_k": 15, "top_p": 1.0, "temperature": a.temp,
        "text_split_method": "cut5", "batch_size": 1,
        "speed_factor": a.speed, "seed": a.seed,
        "repetition_penalty": 1.35, "sample_steps": 32,
        "super_sampling": False, "return_fragment": False,
        "fragment_interval": 0.3, "parallel_infer": True, "split_bucket": True,
    }
    out = a.out or os.path.join(OUT_DIR, "say",
                                f"{time.strftime('%Y%m%d_%H%M%S')}_{a.lang}.wav")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    t0 = time.time()
    for sr, audio in tts.run(inputs):
        sf.write(out, audio, sr)

    print(f"{a.text}\n-> {out}")
    print(f"   {os.path.basename(sov)} + {os.path.basename(gpt)}"
          f"，语速×{a.speed}，温度{a.temp}，种子{a.seed}，耗时 {time.time()-t0:.1f}s")
    print(f"   参考：{os.path.basename(ref_wav)}")


if __name__ == "__main__":
    main()
