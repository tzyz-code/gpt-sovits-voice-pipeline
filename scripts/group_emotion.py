# -*- coding: utf-8 -*-
"""按游戏音频槽位给 123 条语音分情绪桶，并为每个桶挑参考音频。

为什么不用 ASR 情绪识别：SenseVoice 在日语的这套素材上 115/123 判成 EMO_UNKNOWN，
它只在中英情绪语料上训过，对日语动漫语料基本无反应（实测记录见 emotion.json）。
反过来，游戏把每句语音放在哪个槽位（Relationship_Up / Battle_Victory / Café 独白）
本身就是设计者标注好的情绪，比模型猜得准。

产出：
  dataset/emotion_groups.json  每组包含哪些条目、总时长、推荐参考音频
  dataset/<group>.list          每组的 GPT-SoVITS 训练清单

    python emotion_group.py
"""
import json
import os
import re
from collections import defaultdict

import librosa

from paths import LANGUAGE
from paths import CHARACTER, GSV
from paths import LIST_PATH as LIST
from paths import DATASET as DS
GROUPS_OUT = os.path.join(DS, "emotion_groups.json")

# 槽位 -> 情绪桶。同名台词不同槽位（如 MemorialLobby 是温柔承诺、ExSkill 是战斗宣言）要分开。
PRE = CHARACTER + "_"

RULES = [
    # 温柔 / 关切：对老师的承诺、奖励时的鼓励、节日问候
    (rf"^{PRE}MemorialLobby", "warm"),
    (rf"^{PRE}EventMission_Get", "warm"),
    (rf"^{PRE}Season_Birthday", "warm"),
    # 慌张 / 吐槽 / 抱怨：商店手忙脚乱、被戳到痛处、拌嘴
    (rf"^{PRE}EventShop", "fluster"),
    (rf"^{PRE}EventMission_(1|2)_", "fluster"),
    (rf"^{PRE}EventMission_Login_1_2", "fluster"),
    (rf"^{PRE}Relationship_Up", "fluster"),
    (rf"^{PRE}Lobby_1_[23]|^Kei_Lobby_2", "fluster"),
    # 自信 / 从容：战斗宣言、胜利、战术指挥
    (rf"^{PRE}ExSkill", "confident"),
    (rf"^{PRE}Battle", "confident"),
    (rf"^{PRE}Tactic_In|^Kei_CommonSkill", "confident"),
    (rf"^{PRE}Formation", "confident"),
    (rf"^{PRE}EventShop_Login", "confident"),
    (rf"^{PRE}EventShop_Buy_3_[12]|^Kei_EventShop_Buy_[12]", "confident"),
    # 感慨 / 动情：成长、信任、脆弱时刻
    (rf"^{PRE}Growup", "moved"),
    (rf"^{PRE}Relationship_Up_4", "moved"),
    (rf"^{PRE}Tactic_Defeat", "moved"),
    # 平静 / 陈述：独白、自我介绍、日常
    (rf"^{PRE}Cafe_monolog", "calm"),
    (rf"^{PRE}Title|^Kei_Gachaget|^Kei_ExWeapon_Get", "calm"),
    (rf"^{PRE}Lobby", "calm"),
    (rf"^{PRE}EventMission_Login", "calm"),
    (rf"^{PRE}Season_", "calm"),
]

GROUP_DESC = {
    "calm": "平静陈述（独白、自我介绍、日常对话）",
    "warm": "温柔关切（对老师的承诺、鼓励、节日问候）",
    "fluster": "慌张吐槽（手忙脚乱、被逗到、拌嘴）",
    "confident": "自信从容（战斗宣言、胜利、战术指挥）",
    "moved": "感慨动情（成长、信任、受挫后的低语）",
}
# 训练可行性门槛：GPT 少于 25 条会严重过拟合，SoVITS 15 条以下基本学不出稳定音色
GPT_MIN, SOVITS_MIN = 25, 15


def bucket_of(name):
    # 大小写不敏感：清单里的说话人名通常小写，而音频文件名常带首字母大写，
    # 两边不一致会让所有规则静默失效、全部落进 other。
    for pat, g in RULES:
        if re.search(pat, name, re.I):
            return g
    return "other"


def main():
    os.makedirs(DS, exist_ok=True)
    rows = []
    for ln in open(LIST, encoding="utf-8"):
        p, _s, _l, text = ln.rstrip("\n").split("|", 3)
        name = os.path.splitext(os.path.basename(p))[0]
        rows.append({"name": name, "path": p, "text": text,
                     "dur": round(librosa.get_duration(path=p), 2),
                     "group": bucket_of(name)})

    g = defaultdict(list)
    for r in rows:
        g[r["group"]].append(r)

    print(f"总共 {len(rows)} 条 / {sum(r['dur'] for r in rows)/60:.2f} 分钟\n")
    print(f"{'情绪桶':<10}{'条数':>5}{'时长':>9}{'均长':>7}  GPT可训  SoVITS可训  说明")
    out = {}
    for k in ("calm", "warm", "fluster", "confident", "moved", "other"):
        if k not in g:
            continue
        items = sorted(g[k], key=lambda r: -r["dur"])
        sec = sum(r["dur"] for r in items)
        # 参考音频：优先 3~8 秒（GPT-SoVITS 官方建议区间），太长会拖慢且带进无关语调
        cand = [r for r in items if 3.0 <= r["dur"] <= 8.0] or items
        ref = max(cand, key=lambda r: r["dur"])
        print(f"{k:<10}{len(items):>5}{sec:>8.0f}s{sec/len(items):>6.1f}s"
              f"{'✓' if len(items) >= GPT_MIN else '✗':>9}"
              f"{'✓' if len(items) >= SOVITS_MIN else '✗':>11}  {GROUP_DESC.get(k,'未归类')}")
        out[k] = {
            "desc": GROUP_DESC.get(k, "未归类"),
            "n": len(items), "sec": round(sec, 1),
            "gpt_ok": len(items) >= GPT_MIN, "sovits_ok": len(items) >= SOVITS_MIN,
            "ref": {"wav": ref["path"], "text": ref["text"], "dur": ref["dur"]},
            "items": items,
        }

    json.dump(out, open(GROUPS_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 每组一份 list，方便直接训
    for k, v in out.items():
        p = os.path.join(DS, f"{k}.list")
        with open(p, "w", encoding="utf-8") as f:
            for r in v["items"]:
                f.write(f"{r['path']}|{CHARACTER}|{LANGUAGE}|{r['text']}\n")
        v["list"] = p

    print("\n=== 各组推荐的参考音频（3~8 秒、语气代表性最强的那条）===")
    for k, v in out.items():
        print(f"  {k:<10}{v['ref']['dur']:>4.1f}s  {os.path.basename(v['ref']['wav'])}")
        print(f"            {v['ref']['text']}")
    print(f"\n清单已写出 -> {DS}\\<group>.list")
    print(f"明细 -> {GROUPS_OUT}")


if __name__ == "__main__":
    main()
