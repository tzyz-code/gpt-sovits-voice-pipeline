# -*- coding: utf-8 -*-
"""情绪 A/B 结果的反向转写校验：确认四组都还在正常念字。

情绪实验里最怕的是「情绪变强了，但字念糊了」——音高、起伏这些指标全测不出来。
这个脚本把每条合成结果转回文字，和原台词逐字比对。

    python asr_emotion.py
"""
import json
import os
import re

from paths import ASCII_CACHE as _ASCII
if os.path.isdir(_ASCII):
    os.environ.setdefault("MODELSCOPE_CACHE", _ASCII)

from funasr import AutoModel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
from paths import EMO_OUT as D


def clean(s):
    s = re.sub(r"<\|[^|]*\|>", "", s)
    s = re.sub(r"[\s、。，．…！？!?（）()「」『』ー〜~・,.\-—]", "", s)
    return s


def main():
    rows = json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))
    m = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad",
                  device="cpu", disable_update=True)
    for r in rows:
        res = m.generate(input=r["wav"], language="ja", use_itn=False, batch_size_s=60)
        raw = res[0]["text"] if res else ""
        r["asr"] = clean(raw)
        r["asr_raw"] = raw.strip()
        r["match"] = "OK" if r["asr"] == clean(r["text"]) else "!!"
    json.dump(rows, open(os.path.join(D, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    ids = ["f1", "f2", "f3", "n1"]
    tags = sorted({r["tag"] for r in rows})
    for i in ids:
        print(f"\n--- {i} ---")
        tgt = next(r["text"] for r in rows if r["id"] == i)
        print(f"  应念：{tgt}")
        for t in tags:
            r = next(x for x in rows if x["tag"] == t and x["id"] == i)
            miss = "" if r["match"] == "OK" else "  <-- 不一致"
            print(f"  {t} 转写：{r['asr']}{miss}")
    ok = sum(1 for r in rows if r["match"] == "OK")
    print(f"\n完全一致 {ok}/{len(rows)}")


if __name__ == "__main__":
    main()
