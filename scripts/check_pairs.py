# -*- coding: utf-8 -*-
"""抽查 123 条音文配对：找「音频和台词对不上」的样本。

错配的样本会直接教坏模型——它会学着用 A 句的语气念 B 句的字。人工听 123 条不现实，
所以先用两个互相独立的信号把可疑的挑出来，只让人复核那一小撮。

信号一 语速异常（chars per second）
    日语音节速度是稳定的，正常落在 5.5~11 字/秒。如果一条 40 字的台词配了 3 秒音频，
    或者 8 字的台词配了 12 秒音频，几乎一定是配错或音频被截断。这个信号不依赖任何模型，最可靠。

信号二 反向转写相似度
    把音频转回文字，和台词算编辑距离。SenseVoice 对动漫日语本来就一般，所以低分不等于错，
    但它能抓到「整句完全不是同一句话」这种硬错。两个信号都报警的排最前面。

    python check_pairs.py
    python check_pairs.py --no-asr    # 只跑语速，秒出
"""
import argparse
import json
import os
import re

import librosa

from paths import ASCII_CACHE as _ASCII
if os.path.isdir(_ASCII):
    os.environ.setdefault("MODELSCOPE_CACHE", _ASCII)

HERE = os.path.dirname(os.path.abspath(__file__))
from paths import GSV
from paths import LIST_PATH as LIST
from paths import DATASET as _DS
OUT = os.path.join(_DS, "pair_check.json")
# 语速阈值不能套教科书的 7~8 音节/秒：Kei 的台词里塞满「……」停顿，实测中位数
# 只有 4.2 字/秒，套固定阈值会把 80 条正常样本全标成可疑。改成用这批素材自己的
# 中位数 + 稳健标准差（MAD）判离群——只抓「和这批素材格格不入」的那几条。
MAD_K = 3.5


def texts(s):
    """去掉标点后的「字」数。日语里假名+汉字都算 1，长音和促音算 0.5（它们不占独立时长）。"""
    s = re.sub(r"[、。，．…！？!?「」『』（）()\s・,.\-—〜~]", "", s)
    return len(s) - 0.5 * (s.count("っ") + s.count("ー") + s.count("ッ"))


def norm(s):
    s = re.sub(r"<\|[^|]*\|>", "", s or "")
    s = re.sub(r"[っッー〜~、。，．…！？!?「」『』（）()・,.\-—\s]", "", s)
    return s


def sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p = d[:]
        d[0] = i
        for j, cb in enumerate(b, 1):
            d[j] = min(p[j] + 1, d[j - 1] + 1, p[j - 1] + (ca != cb))
    return 1 - d[len(b)] / len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-asr", action="store_true", help="跳过转写，只用语速判")
    a = ap.parse_args()

    rows = []
    for ln in open(LIST, encoding="utf-8"):
        p, _spk, _lang, text = ln.rstrip("\n").split("|", 3)
        dur = librosa.get_duration(path=p)
        n = texts(text)
        rows.append({"name": os.path.splitext(os.path.basename(p))[0],
                     "path": p, "text": text, "dur": round(dur, 2),
                     "chars": round(n, 1), "cps": round(n / dur, 2) if dur else 0})

    if not a.no_asr:
        from funasr import AutoModel
        m = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad",
                      device="cpu", disable_update=True)
        for r in rows:
            res = m.generate(input=r["path"], language="ja", use_itn=False, batch_size_s=60)
            r["asr"] = (res[0]["text"] if res else "").strip()
            r["sim"] = round(sim(r["asr"], r["text"]), 3)

    # 打分：语速离群 + 转写越低 = 越可疑
    import statistics
    cps_all = [r["cps"] for r in rows]
    med = statistics.median(cps_all)
    mad = statistics.median([abs(c - med) for c in cps_all]) or 0.5
    sim_all = [r["sim"] for r in rows] if "sim" in rows[0] else []
    smed = statistics.median(sim_all) if sim_all else 0
    smad = statistics.median([abs(s - smed) for s in sim_all]) if sim_all else 1

    for r in rows:
        s = 0.0
        zc = abs(r["cps"] - med) / mad
        if zc > MAD_K:
            s += (zc - MAD_K) * 3
        if "sim" in r:
            zs = (smed - r["sim"]) / (smad or 1)
            if zs > MAD_K:
                s += (zs - MAD_K) * 4
        r["zc"] = round(zc, 2)
        r["score"] = round(s, 2)

    rows.sort(key=lambda r: -r["score"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"共 {len(rows)} 条，语速中位数 {med:.1f} 字/秒（MAD {mad:.2f}），"
          f"区间 {min(cps_all):.1f} ~ {max(cps_all):.1f}")
    if sim_all:
        print(f"转写相似度中位数 {smed*100:.0f}%（MAD {smad:.2f}），"
              f"最低 {min(sim_all)*100:.0f}%")

    sus = [r for r in rows if r["score"] > 0]
    has_sim = "sim" in rows[0]
    print(f"\n可疑 {len(sus)} 条（按可疑度排序）")
    if sus:
        print(f"{'样本':<36}{'时长':>6}{'字数':>6}{'字/秒':>7}{'偏离':>7}"
              + (f"{'转写相似':>9}" if has_sim else ""))
        for r in sus[:12]:
            line = (f"{r['name']:<36}{r['dur']:>6.1f}{r['chars']:>6.0f}"
                    f"{r['cps']:>7.1f}{r['zc']:>6.1f}x")
            if has_sim:
                line += f"{r['sim']*100:>8.0f}%"
            print(line)
            print(f"    {r['text'][:60]}")
            if has_sim:
                print(f"    ASR: {norm(r['asr'])[:60]}")
    print(f"\n明细 -> {OUT}")


if __name__ == "__main__":
    main()
