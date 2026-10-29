# -*- coding: utf-8 -*-
"""从数据集里挑几条参考音频放进 ref/。

为什么需要这一步：仓库不附带任何音频（版权原因），而 GPT-SoVITS 推理**必须**有参考音频。
参考音频是 3~10 秒、台词完全确定的一段原声，它的语气会感染之后生成的每一句话——
所以挑得好不好，直接决定效果上限，比换权重重要得多（实测：换参考 +119Hz，换模型 +14Hz）。

挑法：
  - 3~10 秒（太短学不出音色，太长拖慢推理且带进无关语调）
  - 台词必须是陈述句，句尾不要上扬（问句会让每句结尾都上翘）
  - 语气平稳，不要喊、不要笑、不要有背景音
  - 参考台词写进同名 .txt，say.py 会自动读

    python build_refs.py                 # 每个情绪桶挑一条，最多 8 条
    python build_refs.py --top 5         # 只要 5 条
    python build_refs.py --min 4 --max 8 # 改时长区间
"""
import argparse
import glob
import json
import os
import shutil

import librosa

from paths import CHARACTER, DATASET, LIST_PATH, LANGUAGE, REF_DIR


def load_list():
    """读训练清单：路径|说话人|语言|台词"""
    out = []
    if not os.path.isfile(LIST_PATH):
        return out
    for ln in open(LIST_PATH, encoding="utf-8"):
        parts = ln.rstrip("\n").split("|", 3)
        if len(parts) == 4:
            out.append({"path": parts[0], "text": parts[3]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--min", type=float, default=3.0)
    ap.add_argument("--max", type=float, default=8.0)
    ap.add_argument("--copy", action="store_true", help="复制而不是建硬链接")
    a = ap.parse_args()

    rows = load_list()
    if not rows:
        raise SystemExit(f"训练清单不存在或为空：{LIST_PATH}\n先跑完 make_list.py。")

    cand = []
    for r in rows:
        if not os.path.isfile(r["path"]):
            continue
        dur = librosa.get_duration(path=r["path"])
        if a.min <= dur <= a.max and r["text"].strip():
            cand.append((dur, r))

    # 排除问句：句尾问号会让之后每句话都往上翘
    no_q = [c for c in cand if not c[1]["text"].rstrip().endswith(("？", "?"))]
    cand = no_q or cand
    cand.sort(key=lambda c: -c[0])

    os.makedirs(REF_DIR, exist_ok=True)
    # 情绪分组存在就用它分配，一组挑一条，保证覆盖不同语气
    groups = os.path.join(DATASET, "emotion_groups.json")
    picked, seen_group = [], set()
    if os.path.isfile(groups):
        g = json.load(open(groups, encoding="utf-8"))
        for name, info in g.items():
            ref = info.get("ref") or {}
            w = ref.get("wav")
            if w and os.path.isfile(w) and ref.get("text"):
                picked.append((name, w, ref["text"]))
                seen_group.add(w)

    for dur, r in cand:
        if len(picked) >= a.top:
            break
        if r["path"] in seen_group:
            continue
        picked.append((os.path.splitext(os.path.basename(r["path"]))[0], r["path"], r["text"]))
        seen_group.add(r["path"])

    if not picked:
        raise SystemExit(f"没有落在 {a.min}~{a.max} 秒区间的样本，放宽 --min/--max 试试。")

    print(f"{'名称':<32}{'时长':>6}  台词")
    for name, wav, text in picked[:a.top]:
        dst = os.path.join(REF_DIR, os.path.basename(wav))
        if not os.path.exists(dst):
            if a.copy:
                shutil.copy2(wav, dst)
            else:
                try:
                    os.link(wav, dst)
                except OSError:
                    shutil.copy2(wav, dst)
        # 台词写成同名 txt，say.py 会读
        open(os.path.splitext(dst)[0] + ".txt", "w", encoding="utf-8").write(text)
        print(f"{os.path.basename(dst):<32}{librosa.get_duration(path=dst):>6.1f}  {text[:40]}")

    print(f"\n{len(picked[:a.top])} 条 -> {REF_DIR}")


if __name__ == "__main__":
    main()
