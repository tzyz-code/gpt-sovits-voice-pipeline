# -*- coding: utf-8 -*-
"""把训好的 Kei 装成 ComfyUI 里的角色，供 Anomalous_TTS 节点直接调用。

设计依据是情绪 A/B 的实测结论（见 情绪模型对比.html）：
  参考音频是情绪的第一杠杆（换参考 = +119Hz），模型只是把起伏幅度再放大 45%。
所以主角色做成「一个通用模型 + 一堆情绪参考音频」——剧本里写 {慌张} 就换参考，
这才是 GPT-SoVITS 的设计用法，也是唯一能做到一句平一句慌的办法。
另配一个「Kei慌张」角色，用专门训的 fluster 模型，需要最夸张的语气时用。

为什么用硬链接而不是复制：权重单个 85MB / 155MB，两份角色就是 480MB 冗余。
硬链接在同一分区上不占额外空间，改一个不影响另一个。

    python setup_comfy_character.py
"""
import json
import os
import shutil

from paths import CHARACTER
from paths import GSV
from paths import COMFY
KEI_WAV = os.path.join(GSV, "dataset", "kei")
ROOT = os.path.join(COMFY, "models", "gpt_sovits")
USER_CFG = os.path.join(COMFY, "user", "anomalous_tts.json")

# (文件名, 台词)。全部取自本套原声，台词是确认准确的原文。
REFS = {
    "平静": ("Kei_Cafe_monolog_4.wav", "自分の身体を持つのは……。不便なことでもあるんですね……。"),
    "慌张": ("Kei_EventShop_1_2_Radio.wav", "もう！誰も言うことを聞かないんですから！"),
    "害羞": ("Kei_Lobby_1_2.wav", "い、嫌なのかって……？と、というより……！"),
    "温柔": ("Kei_MemorialLobby_4_1.wav", "だから大丈夫……絶対、大丈夫です。"),
    "得意": ("Kei_Battle_Victory_1.wav", "当然の結果です。見ていましたよね、先生？"),
    "抱怨": ("Kei_Lobby_4_1.wav", "ああ、もう！だからそんな風に子ども扱いしないでください！"),
    "无奈": ("Kei_EventShop_Buy_3_3_Radio.wav", "本当に……先生がいないと、ダメなんですよ、私たち。"),
    "嘟囔": ("Kei_Lobby_3_5.wav", "……本当にもう、手が焼けるんですから。"),
}

# 采样参数沿用第十四节定档：语速 1.2、温度 0.8、top_k 15、重复惩罚 1.35。
# 不写这些节点会用默认值，出来的语气和试听页里验证过的不是一回事。
DEFAULTS = {"speed": 1.2, "temperature": 0.8, "top_k": 15,
            "top_p": 1.0, "repetition_penalty": 1.35}


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prep_audio(dst_dir, names):
    """把参考音频放进去。文件名一律 ASCII，台词写在设置文件里——
    这样不怕中文路径在某些环节出问题，也不用手写一堆同名 txt。"""
    out = {}
    for emo in names:
        fn, text = REFS[emo]
        src = os.path.join(KEI_WAV, fn)
        dst = os.path.join(dst_dir, fn)
        link(src, dst)
        out[emo] = {"audio": f"参考音频/{fn}", "text": text, "language": "ja"}
    return out


def build(alias, sov, gpt, main_emo, emotions):
    d = os.path.join(ROOT, alias)
    ad = os.path.join(d, "参考音频")
    os.makedirs(ad, exist_ok=True)
    link(os.path.join(GSV, "SoVITS_weights_v2", sov),
         os.path.join(d, "SoVITS_weights_v2", sov))
    link(os.path.join(GSV, "GPT_weights_v2", gpt),
         os.path.join(d, "GPT_weights_v2", gpt))

    refs = prep_audio(ad, [main_emo] + emotions)
    cfg = {
        "format": 1,
        "aliases": [alias.replace("Kei", "kei")] if alias != "Kei" else ["kei", "ケイ"],
        "language": "ja",
        "gpt": f"GPT_weights_v2/{gpt}",
        "sovits": f"SoVITS_weights_v2/{sov}",
        "reference": refs[main_emo],
        "emotions": {k: refs[k] for k in emotions},
        "defaults": dict(DEFAULTS),
    }
    p = os.path.join(d, "anomalous_tts.json")
    json.dump(cfg, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  {alias:<10} {sov} + {gpt}")
    print(f"             主参考 {main_emo} / 情绪 {', '.join(emotions)}")
    return p


def main():
    os.makedirs(ROOT, exist_ok=True)
    os.makedirs(os.path.dirname(USER_CFG), exist_ok=True)

    # 复用本机已有的 GPT-SoVITS 底模，省掉首次生成时最多 1.6GB 的下载。
    # 路径指向 GPT_SoVITS 目录，节点会自己往下找 pretrained_models。
    cfg = {"format": 1, "pretrained": [os.path.join(GSV, "GPT_SoVITS").replace("\\", "/")]}
    json.dump(cfg, open(USER_CFG, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"底模复用 -> {USER_CFG}")

    print("\n角色：")
    build("Kei", f"{CHARACTER}_e8_s512.pth", f"{CHARACTER}-e15.ckpt", "平静",
          ["慌张", "害羞", "温柔", "得意", "抱怨", "嘟囔"])
    # 慌张专用模型：素材只有 31 条、3 分钟，专门为「急起来」训的
    build("Kei慌张", "fluster_e16_s768.pth", "fluster-e30.ckpt", "无奈",
          ["慌张", "害羞", "抱怨"])

    print(f"\n角色根目录 -> {ROOT}")
    for d in sorted(os.listdir(ROOT)):
        p = os.path.join(ROOT, d)
        if not os.path.isdir(p):
            continue
        n = len(os.listdir(os.path.join(p, "参考音频"))) if os.path.isdir(
            os.path.join(p, "参考音频")) else 0
        print(f"  {d:<10} 参考音频 {n} 条")


if __name__ == "__main__":
    main()
