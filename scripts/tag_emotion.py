# -*- coding: utf-8 -*-
"""给 123 条训练音频打情绪标签，并统计每种情绪的数据量。

双路交叉：
  1) SenseVoice 的情感识别（data-driven，看音频实际怎么念）
  2) 文件名语义（domain prior，ExSkill/Growup/Relationship_Up 各自含义明确）

输出 dataset/emotion.json 和一张分布表，用来决定「哪些情绪有资格单独训一个模型」。

    python emotion_tag.py
"""
import json
import os
import re
from collections import Counter, defaultdict

import librosa

from paths import ASCII_CACHE as _ASCII
if os.path.isdir(_ASCII):
    os.environ.setdefault("MODELSCOPE_CACHE", _ASCII)

from funasr import AutoModel  # noqa: E402

from paths import GSV
from paths import LIST_PATH as LIST
from paths import DATASET as _DS
OUT = os.path.join(_DS, "emotion.json")

# 文件名 -> 先验情绪分组（看名字就知道演的是哪种状态）
PRIOR = [
    (r"Cafe_monolog", "calm"),        # 自言自语、平静观察
    (r"MemorialLobby", "warm"),       # 对老师的温和承诺
    (r"ExSkill|Battle_Victory|Tactic_Victory|Battle_Move|Battle_Defense", "confident"),
    (r"Tactic_In|Battle_In|Formation", "reluctant"),   # 不情愿 / 无奈
    (r"Growup", "moved"),             # 感慨、成长
    (r"Relationship_Up", "shy"),      # 害羞、被调戏
    (r"EventShop", "flustered"),      # 慌张、抱怨
    (r"EventMission_Get", "gentle"),  # 鼓励 / 温柔
    (r"EventMission_Login", "diligent"),
    (r"EventMission", "tsukkomi"),    # 吐槽、拌嘴
    (r"Gachaget|Title", "graceful"),  # 登场的从容
    (r"Lobby", "casual"),
    (r"Season_Birthday|Season_Xmas|Season_Newyear|Season_", "casual"),
    (r"ExWeapon_Get", "graceful"),
]


def prior_of(name):
    for pat, tag in PRIOR:
        if re.search(pat, name):
            return tag
    return "other"


def main():
    rows = []
    for ln in open(LIST, encoding="utf-8"):
        p, _spk, _lang, text = ln.rstrip("\n").split("|", 3)
        name = os.path.splitext(os.path.basename(p))[0]
        dur = round(librosa.get_duration(path=p), 2)
        rows.append({"name": name, "path": p, "text": text,
                     "dur": dur, "prior": prior_of(name)})
    total_dur = sum(r["dur"] for r in rows)
    print(f"共 {len(rows)} 条 / {total_dur/60:.2f} 分钟，开始情绪识别…\n", flush=True)

    m = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad",
                  device="cpu", disable_update=True)
    EMO = ("HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED",
           "SURPRISED", "EMO_UNKNOWN")
    for i, r in enumerate(rows, 1):
        res = m.generate(input=r["path"], language="ja", use_itn=False, batch_size_s=60)
        raw = res[0]["text"] if res else ""
        found = next((e for e in EMO if f"<|{e}|>" in raw), "EMO_UNKNOWN")
        r["ser"] = found
        r["asr"] = re.sub(r"<\|[^|]*\|>", "", raw).strip()
        print(f"\r  {i}/{len(rows)}", end="", flush=True)
    print("\n")

    json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- 按先验分组统计 ----
    g = defaultdict(lambda: {"n": 0, "sec": 0.0, "ser": Counter()})
    for r in rows:
        d = g[r["prior"]]
        d["n"] += 1
        d["sec"] += r["dur"]
        d["ser"][r["ser"]] += 1
    print(f"{'先验分组':<12}{'条数':>5}{'时长':>8}   SenseVoice 情绪分布")
    for k, v in sorted(g.items(), key=lambda kv: -kv[1]["sec"]):
        ser = " ".join(f"{e}:{c}" for e, c in v["ser"].most_common())
        print(f"{k:<12}{v['n']:>5}{v['sec']:>7.0f}s  {ser}")

    print("\n=== SenseVoice 全局分布 ===")
    for e, c in Counter(r["ser"] for r in rows).most_common():
        sec = sum(r["dur"] for r in rows if r["ser"] == e)
        print(f"  {e:<14}{c:>4} 条  {sec:>5.0f}s")

    print("\n=== 先验分组 × SER 交叉（行=先验，列=SER）===")
    keys = [k for k, _ in sorted(g.items(), key=lambda kv: -kv[1]["sec"])]
    hdr = "".join(f"{e[:6]:>8}" for e, _ in Counter(r["ser"] for r in rows).most_common())
    print(f"{'':<12}{hdr}")
    for k in keys:
        line = "".join(f"{g[k]['ser'].get(e, 0):>8}"
                       for e, _ in Counter(r["ser"] for r in rows).most_common())
        print(f"{k:<12}{line}")


if __name__ == "__main__":
    main()
