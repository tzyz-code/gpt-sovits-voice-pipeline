"""用 SenseVoice 反向转写合成结果，检查有没有重复字、漏字、跳句。

这是不用耳朵也能做的客观校验：TTS 的典型毛病（GPT 重复/漏字、口音读错）
在反向转写里会直接暴露成"文本对不上"。

用法：
    python verify_asr.py
"""
from __future__ import annotations

import glob
import os
import re
import sys

# 必须在 import funasr 之前设置，否则 Windows 下 sentencepiece 打不开中文路径
from paths import ASCII_CACHE as _ASCII_CACHE
if os.path.isdir(_ASCII_CACHE):
    os.environ.setdefault("MODELSCOPE_CACHE", _ASCII_CACHE)
os.environ.setdefault("ASR_DEVICE", "cpu")

TARGET = {
    "zh": "你这是在干什么",
    "ja": "何をしているんですか",
}

from paths import OUT_DIR as OUT
def clean(s: str) -> str:
    s = re.sub(r"<\|[^|]*\|>", "", s)          # SenseVoice 的情感/事件标签
    s = re.sub(r"[^\w\u3040-\u30ff\u4e00-\u9fff]", "", s)
    return s.lower()


def main():
    from funasr import AutoModel

    print("加载 SenseVoice（CPU）…", flush=True)
    m = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad",
                  device="cpu", disable_update=True)
    print("已加载\n", flush=True)

    files = sorted(glob.glob(os.path.join(OUT, "*.wav")))
    if not files:
        raise SystemExit("没有找到合成结果")

    rows = []
    for p in files:
        lang = "ja" if "_ja_" in os.path.basename(p) else "zh"
        try:
            r = m.generate(input=p, language=lang, use_itn=True)
            raw = r[0]["text"] if r else ""
        except Exception as e:
            rows.append((os.path.basename(p), f"转写失败: {e}", "—", "❌"))
            continue
        got = clean(raw)
        want = clean(TARGET[lang])
        emo = re.findall(r"<\|([A-Z_]+)\|>", raw)
        emo = next((e for e in emo if e not in
                    ("ja", "zh", "en", "Speech", "withitn", "woitn")), "-")

        if got == want:
            verdict = "✅ 完全一致"
        elif want in got or got in want:
            verdict = "⚠ 有增删字符"
        else:
            overlap = len(set(got) & set(want)) / max(1, len(set(want)))
            verdict = f"{'⚠' if overlap > 0.8 else '❌'} 覆盖 {overlap*100:.0f}%"

        rows.append((os.path.basename(p), got, emo, verdict))

    print(f"{'文件':30s}{'反向转写':22s}{'情绪':12s}判定")
    print("-" * 74)
    for name, got, emo, verdict in rows:
        print(f"{name:30s}{got:22s}{emo:12s}{verdict}")

    import json
    with open(os.path.join(OUT, "asr_result.json"), "w", encoding="utf-8") as f:
        json.dump([{"file": a, "text": b, "emotion": c, "verdict": d} for a, b, c, d in rows],
                  f, ensure_ascii=False, indent=2)
    print(f"\n明细已存 {os.path.join(OUT, 'asr_result.json')}")


if __name__ == "__main__":
    main()
