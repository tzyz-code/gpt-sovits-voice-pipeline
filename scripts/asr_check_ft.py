# -*- coding: utf-8 -*-
"""反向转写校验：把合成音频转回文字，和目标台词比对。

能发现重复字、漏字、假名读错——这些是 MFCC 声纹分完全测不到的问题。

    python asr_ft.py
"""
import glob
import json
import os
import re

# 必须在 import funasr 之前设，否则 Windows 下打不开中文用户目录里的缓存
from paths import ASCII_CACHE as _ASCII
if os.path.isdir(_ASCII):
    os.environ.setdefault("MODELSCOPE_CACHE", _ASCII)

from funasr import AutoModel  # noqa: E402

from paths import FT_OUT as ROOT
TARGET = {
    "ja": "何をしているんですか",
    "zh": "你这是在干什么",
}


def clean(s):
    s = re.sub(r"<\|[^|]*\|>", "", s)
    s = re.sub(r"[\s、。，．…！？!?（）()「」『』ー〜~・,.\-—]", "", s)
    return s


def main():
    m = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad",
                  device="cpu", disable_update=True)
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "*.wav"))):
        lang = os.path.basename(p)[:2]
        tgt = TARGET.get(lang, "")
        res = m.generate(input=p, language=lang if lang in ("ja", "zh") else "auto",
                         use_itn=False, batch_size_s=60)
        raw = res[0]["text"] if res else ""
        got = clean(raw)
        tag = "OK" if got == clean(tgt) else "!!"
        rows.append({"file": os.path.basename(p), "target": tgt,
                     "asr": got, "raw": raw.strip()[:60], "match": tag})
        print(f"{tag} {os.path.basename(p):<24} 转写={got:<20} 应为={clean(tgt)}")
    json.dump(rows, open(os.path.join(ROOT, "asr_check.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    ok = sum(1 for r in rows if r["match"] == "OK")
    print(f"\n完全一致 {ok}/{len(rows)}")


if __name__ == "__main__":
    main()
