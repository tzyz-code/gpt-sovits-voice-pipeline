# -*- coding: utf-8 -*-
"""把 ogg 原始素材转成 GPT-SoVITS 可用的 32k 单声道 wav，并生成训练清单 + 人工抽查表。

必须用 GPT-SoVITS 的虚拟环境跑：
    python make_list.py
"""
import json
import os
import re
import shutil
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
from paths import ROOT
from paths import DATASET as DS
OGG = os.path.join(DS, "ogg")

from paths import GSV
from paths import WAV_DIR as OUT_WAV
from paths import LIST_PATH
SR = 32000
from paths import CHARACTER as SPK

# wiki 里这两个名字是小写的，CDN 上不存在；正文其实挂在首字母大写的同名文件上
ALIAS = {
    "Kei_Carrier_select_5.ogg": "Kei_Carrier_Select_5.ogg",
    "Kei_Carrier_select_6.ogg": "Kei_Carrier_Select_6.ogg",
}


def load_mono32k(path):
    x, sr = sf.read(path, always_2d=True, dtype="float32")
    x = x.mean(axis=1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x


def trim(x, top_db=35):
    import librosa
    idx = librosa.effects.trim(x, top_db=top_db)[1]
    a, b = int(idx[0]), int(idx[1])
    # 前后各留 60ms，避免把首尾辅音削掉
    pad = int(0.06 * SR)
    a = max(0, a - pad)
    b = min(len(x), b + pad)
    return x[a:b]


def main():
    lines = json.load(open(os.path.join(DS, "lines.json"), encoding="utf-8"))
    for src, dst in ALIAS.items():
        if src in lines and dst not in lines:
            lines[dst] = lines.pop(src)

    if os.path.isdir(OUT_WAV):
        shutil.rmtree(OUT_WAV)
    os.makedirs(OUT_WAV, exist_ok=True)

    rows, failed = [], []
    for fname in sorted(lines):
        src = os.path.join(OGG, fname)
        if not os.path.isfile(src):
            failed.append((fname, "ogg 缺失"))
            continue
        try:
            x = trim(load_mono32k(src))
        except Exception as e:
            failed.append((fname, f"解码失败 {e}"))
            continue
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        if peak <= 0:
            failed.append((fname, "静音"))
            continue
        x = x * (0.9 / peak)          # 统一峰值
        dur = len(x) / SR
        if dur < 0.8:
            failed.append((fname, f"过短 {dur:.2f}s"))
            continue
        out = os.path.join(OUT_WAV, os.path.splitext(fname)[0] + ".wav")
        sf.write(out, x, SR, subtype="PCM_16")
        rows.append({"wav": out, "name": os.path.splitext(fname)[0],
                     "dur": round(dur, 2), "text": lines[fname]})

    # 纯标点的退化样本（如「……。」）对训练有害，直接扔
    def degenerate(t):
        core = re.sub(r"[\s、。，．…！？!?（）()「」『』ー〜~・,.]", "", t)
        return len(core) < 4
    keep = [r for r in rows if not degenerate(r["text"])]
    for r in rows:
        if r not in keep:
            failed.append((r["name"], f"退化文本 {r['text']!r}"))
    rows = keep

    # ---- 训练清单 ----
    with open(LIST_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r['wav']}|{SPK}|ja|{r['text']}\n")

    # ---- 人工抽查表 ----
    total = sum(r["dur"] for r in rows)
    md = [f"# Kei 音文配对清单（共 {len(rows)} 条 / {total/60:.1f} 分钟）\n",
          "听完打勾，对不上或读错的直接告诉我编号，我把那条踢掉再训。\n",
          "| # | 时长 | 文件名 | 日文台词 |", "|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        flag = " ⚠️短" if r["dur"] < 2.0 else (" ⚠️长" if r["dur"] > 10 else "")
        md.append(f"| {i} | {r['dur']:.1f}s{flag} | `{r['name']}` | {r['text']} |")
    open(os.path.join(ROOT, "配对清单.md"), "w", encoding="utf-8").write("\n".join(md))

    # ---- 可试听校对页（放 dataset 目录下，相对路径引 wav）----
    html_rows = []
    for i, r in enumerate(rows, 1):
        flag = "short" if r["dur"] < 2.0 else ("long" if r["dur"] > 10 else "")
        html_rows.append(
            f'<tr class="{flag}" data-i="{i}" data-f="kei/{r["name"]}.wav">'
            f'<td class="i">{i}</td><td class="d">{r["dur"]:.1f}s</td>'
            f'<td class="n">{r["name"]}</td><td class="t">{r["text"]}</td>'
            f'<td class="ok"><input type="checkbox" class="ck"></td></tr>')
    html = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Kei 音文配对校对</title>
<style>
:root{--bg:#fbfbfd;--fg:#1b1d22;--dim:#6b7280;--line:#e6e8ee;--accent:#2f6fed;--warn:#b45309;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
header{position:sticky;top:0;background:rgba(251,251,253,.96);backdrop-filter:blur(8px);
 border-bottom:1px solid var(--line);padding:12px 20px;z-index:9;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:600}
.sub{color:var(--dim);font-size:12px}
.ctrl{display:flex;gap:8px;margin-left:auto;align-items:center}
button{border:1px solid var(--line);background:#fff;border-radius:7px;padding:5px 11px;cursor:pointer;font-size:12.5px;color:var(--fg)}
button:hover{border-color:#c9cfdb}
button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
#np{font-size:12.5px;color:var(--dim)}
table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:7px 12px;border-bottom:1px solid var(--line);vertical-align:middle}
th{font-size:12px;color:var(--dim);font-weight:600;position:sticky;top:57px;background:var(--bg);z-index:8}
tr{cursor:pointer}
tr:hover{background:#f1f4fb}
tr.playing{background:#e8efff}
tr.playing td.n{color:var(--accent);font-weight:600}
td.i{width:44px;color:var(--dim);font-variant-numeric:tabular-nums}
td.d{width:64px;color:var(--dim);font-variant-numeric:tabular-nums;white-space:nowrap}
td.n{width:250px;font-family:ui-monospace,Consolas,monospace;font-size:12px;color:#3b4252}
td.t{font-size:13.5px}
td.ok{width:44px}
tr.short td.d{color:var(--warn)}
tr.long td.d{color:#0f766e}
tr.hide{display:none}
</style></head><body>
<header>
  <div><h1>Kei 音文配对校对</h1>
  <div class="sub">点任意一行试听，对照台词听有没有错读／漏句。确认没问题就勾上。__SUB__</div></div>
  <div class="ctrl">
    <button id="f-all" class="on">全部</button>
    <button id="f-short">短(&lt;2s)</button>
    <button id="f-long">长(&gt;10s)</button>
    <span id="np">未播放</span>
  </div>
</header>
<audio id="au" preload="none"></audio>
<table><thead><tr><th>#</th><th>时长</th><th>文件</th><th>台词</th><th>OK</th></tr></thead>
<tbody>__ROWS__</tbody></table>
<script>
const au=document.getElementById('au'),np=document.getElementById('np');
let cur=null;
document.querySelectorAll('tbody tr').forEach(tr=>{
  tr.onclick=e=>{
    if(e.target.tagName==='INPUT')return;
    if(cur)cur.classList.remove('playing');
    cur=tr;tr.classList.add('playing');
    au.src=tr.dataset.f;au.play();
    np.textContent='▶ '+tr.dataset.i+' / '+tr.querySelector('td.n').textContent;
  };
  tr.querySelector('.ck').onchange=e=>{tr.style.opacity=e.target.checked?.45:1;};
});
au.onended=()=>{if(cur)cur.classList.remove('playing');cur=null;};
function filter(mode){
  document.querySelectorAll('tbody tr').forEach(tr=>{
    tr.classList.toggle('hide', mode==='all'?false:!tr.classList.contains(mode));
  });
  ['all','short','long'].forEach(m=>document.getElementById('f-'+m).classList.toggle('on',m===mode));
}
document.getElementById('f-all').onclick=()=>filter('all');
document.getElementById('f-short').onclick=()=>filter('short');
document.getElementById('f-long').onclick=()=>filter('long');
</script></body></html>"""
    html = html.replace("__ROWS__", "\n".join(html_rows)).replace(
        "__SUB__", f"共 {len(rows)} 条 / {total/60:.1f} 分钟。橙色=偏短，绿色=偏长。")
    open(os.path.join(GSV, "dataset", "配对校对.html"), "w", encoding="utf-8").write(html)

    print(f"写出 wav {len(rows)} 个 -> {OUT_WAV}")
    print(f"总时长 {total:.1f}s = {total/60:.2f} 分钟，平均 {total/max(1,len(rows)):.2f}s")
    print(f"清单 -> {LIST_PATH}")
    print(f"校对页 -> {os.path.join(GSV, 'dataset', '配对校对.html')}")
    if failed:
        print(f"\n丢弃 {len(failed)} 条：")
        for f, why in failed:
            print(f"  {f}: {why}")


if __name__ == "__main__":
    main()
