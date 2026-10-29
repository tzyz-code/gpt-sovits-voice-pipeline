# -*- coding: utf-8 -*-
"""生成「训练集外新句子」的试听页，附带反向转写结果。

    python make_demo_page.py
"""
import json
import os

from paths import ROOT
DEMO = os.path.join(ROOT, "demo")
ROWS = json.load(open(os.path.join(DEMO, "asr.json"), encoding="utf-8"))

SECTION = {
    "ja": ("日语 · SoVITS e8 + GPT 15 (kei-e15.ckpt)",
           "下面四句都不在训练集里。最后三条是同一句换种子，用来看稳定性。"),
    "zh": ("中文 · SoVITS e8 + GPT 9 (kei-e9.ckpt)",
           "同样全是训练集外的句子。种子 42 在中文上会念错字，这里统一用 1024。"),
}


def render(lang):
    title, note = SECTION[lang]
    trs = []
    for r in ROWS:
        if r["lang"] != lang:
            continue
        asr = r["asr"]
        bad = not r["match"]
        mark = '<span class="bad">转写不符</span>' if bad else '<span class="ok">转写一致</span>'
        trs.append(
            f'<tr data-src="demo/{r["file"]}">'
            f'<td class="pl">▶</td>'
            f'<td class="tx">{r["text"]}</td>'
            f'<td class="num">{r["voiced"]:.1f}s</td>'
            f'<td class="num">{r["char_per_s"]:.1f}字/秒</td>'
            f'<td class="asr">{asr}{mark}</td></tr>')
    return f"""
<h2>{title}</h2><div class="note">{note}</div>
<table><thead><tr><th></th><th>台词</th><th>有声时长</th><th>语速</th><th>反向转写</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table>"""


html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Kei 新句子验证</title>
<style>
:root{{--bg:#12141a;--panel:#181b23;--line:#272b36;--fg:#e6e8ee;--dim:#8b93a7;
      --accent:#6f9cff;--good:#4ade80;--bad:#f87171;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);
 font:14px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;padding:0 0 64px}}
header{{padding:22px 28px 16px;border-bottom:1px solid var(--line)}}
h1{{font-size:17px;margin:0 0 6px;font-weight:600}}
.sub{{color:var(--dim);font-size:12.5px;max-width:920px}}
h2{{font-size:14px;margin:28px 28px 6px;color:var(--accent);font-weight:600}}
.note{{margin:0 28px 10px;color:var(--dim);font-size:12px}}
table{{border-collapse:collapse;width:100%}}
th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{font-size:11.5px;color:var(--dim);font-weight:600}}
tr{{cursor:pointer}} tr:hover{{background:#1c2030}} tr.on{{background:#1e2740}}
tr.on td.tx{{color:var(--accent)}}
td.pl{{width:34px;color:var(--dim)}}
td.num{{width:92px;color:var(--dim);font-variant-numeric:tabular-nums;white-space:nowrap}}
td.asr{{color:#a8b0c2;font-size:12.5px}}
.ok{{color:var(--good);font-size:11px;margin-left:6px}}
.bad{{color:var(--bad);font-size:11px;margin-left:6px}}
#np{{position:fixed;bottom:0;left:0;right:0;padding:9px 28px;background:var(--panel);
 border-top:1px solid var(--line);font-size:12.5px;color:var(--dim)}}
</style></head><body>
<header><h1>Kei 新句子验证</h1>
<div class="sub">全部是训练集里没有的句子——这一关才说明模型是真会念，而不是把那 123 条背下来了。
参数统一为 语速×1.2 / 温度 0.8 / 种子 1024，GPT 档位按语种分（日语 15 轮、中文 9 轮）。</div></header>
{render("ja")}
{render("zh")}
<div id="np">未播放</div>
<audio id="au" preload="none"></audio>
<script>
const au=document.getElementById('au'),np=document.getElementById('np');let cur=null;
document.querySelectorAll('tbody tr').forEach(tr=>{{
  tr.onclick=()=>{{
    if(cur)cur.classList.remove('on');
    cur=tr;tr.classList.add('on');
    au.src=tr.dataset.src;au.play();
    np.textContent='▶ '+tr.querySelector('td.tx').textContent;
  }};
}});
au.onended=()=>{{if(cur)cur.classList.remove('on');cur=null;}};
au.onerror=()=>{{np.textContent='⚠ 播放失败：'+au.src;}};
</script></body></html>"""

p = os.path.join(ROOT, "新句子验证.html")
open(p, "w", encoding="utf-8").write(html)
print("已生成", p, "条目", sum(1 for r in ROWS))
