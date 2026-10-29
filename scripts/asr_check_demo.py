# -*- coding: utf-8 -*-
"""对 demo 目录做反向转写 + 语速检查（训练集外的新句子能不能念对）。

    python asr_demo.py
"""
import json
import os
import re

from paths import ASCII_CACHE as _ASCII
if os.path.isdir(_ASCII):
    os.environ.setdefault("MODELSCOPE_CACHE", _ASCII)

import librosa  # noqa: E402
from funasr import AutoModel  # noqa: E402

from paths import DEMO_DIR as ROOT
META = json.load(open(os.path.join(ROOT, "meta.json"), encoding="utf-8"))


def clean(s):
    s = re.sub(r"<\|[^|]*\|>", "", s)
    s = re.sub(r"[\s、。，．…！？!?（）()「」『』ー〜~・,.\-—]", "", s)
    # 「〜ほうが」被 ASR 写成汉字「方」属同音异写，不算错
    return re.sub(r"方(?=が)", "ほう", s)


def main():
    m = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad",
                  device="cpu", disable_update=True)
    rows, ok = [], 0
    for it in META:
        p = os.path.join(ROOT, it["file"])
        res = m.generate(input=p, language=it["lang"], use_itn=False, batch_size_s=60)
        got = clean(res[0]["text"] if res else "")
        y, sr = librosa.load(p, sr=32000, mono=True)
        idx = librosa.effects.trim(y, top_db=35)[1]
        voiced = (idx[1] - idx[0]) / sr
        nchar = len(clean(it["text"]))
        match = got == clean(it["text"])
        ok += match
        rows.append({**it, "asr": got, "match": match,
                     "voiced": round(voiced, 2), "char_per_s": round(nchar / voiced, 2)})
        print(f"{'OK' if match else '!!'} {it['file']:<22} 有声{voiced:4.1f}s "
              f"{nchar / voiced:4.1f}字/秒  {got}")
    json.dump(rows, open(os.path.join(ROOT, "asr.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n完全一致 {ok}/{len(rows)}")


if __name__ == "__main__":
    main()
