# -*- coding: utf-8 -*-
"""生成试听对比页：零样本 vs 各档微调 vs 提速版。

    python make_compare_page.py
"""
import json
import os

from paths import ROOT
SCORE = json.load(open(os.path.join(ROOT, "ft_score.json"), encoding="utf-8"))
S = {os.path.splitext(r["name"])[0]: r for r in SCORE}

RECOMMEND = {"ja_e8_g15_s1024", "zh_e8_g9_s42_spd1.2"}


def label(n):
    if n.startswith("kei_"):
        lang = "日语" if "_ja_" in n else "中文"
        seed = n.rsplit("_s", 1)[-1]
        return f"零样本 · {lang} · 种子{seed}"
    parts = n.split("_")
    if len(parts) >= 4:
        ek, gk, s = parts[1], parts[2], parts[3]
        tag = f"SoVITS-{ek} + GPT-{gk[1:]}轮 · 种子{s[1:]}"
        if "_spd" in n:
            tag = f"提速{n.split('_spd')[-1]}× · SoVITS-{ek} + GPT-{gk[1:]}轮 · 种子{s[1:]}"
        return tag
    return n


def rows_for(lang):
    out = []
    for n in sorted(S):
        if n.startswith("kei_" + lang):
            out.append((n, "out/" + n + ".wav", "zero"))
    for ek in ("e4", "e8", "e10"):
        for gk in ("g9", "g15"):
            for s in (("42", "1024") if lang == "ja" else ("42",)):
                n = f"{lang}_{ek}_{gk}_s{s}"
                if n in S:
                    out.append((n, "ft_out/" + n + ".wav", "ft"))
    for n in sorted(S):
        if n.startswith(lang + "_") and "_spd" in n:
            out.append((n, "ft_out/" + n + ".wav", "spd"))
    return out


def render(lang, title, note=""):
    trs = []
    for name, rel, kind in rows_for(lang):
        r = S[name]
        badge = '<span class="rec">推荐</span>' if name in RECOMMEND else ""
        trs.append(
            f'<tr data-src="{rel}" class="{kind}">'
            f'<td class="pl">▶</td>'
            f'<td class="lb">{label(name)}{badge}</td>'
            f'<td class="num">{r["dur"]:.1f}s</td>'
            f'<td class="num">{r["kei"]:+.3f}</td>'
            f'<td class="num">{r["f0"]}Hz</td>'
            f'<td class="num">{r["hf"]*100:.1f}%</td></tr>')
    return f"""
<h2>{title}</h2>{note}
<table><thead><tr><th></th><th>版本</th><th>时长</th><th>声纹像不像</th><th>中位音高</th><th>6k+ 毛刺</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table>"""


html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Kei 语音：零样本 vs 微调</title>
<style>
:root{{--bg:#12141a;--panel:#181b23;--line:#272b36;--fg:#e6e8ee;--dim:#8b93a7;
      --accent:#6f9cff;--good:#4ade80;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);
 font:14px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;padding:0 0 64px}}
header{{padding:22px 28px 14px;border-bottom:1px solid var(--line);position:sticky;top:0;
 background:rgba(18,20,26,.94);backdrop-filter:blur(10px);z-index:9}}
h1{{font-size:17px;margin:0 0 6px;font-weight:600}}
.sub{{color:var(--dim);font-size:12.5px;max-width:920px}}
#np{{position:fixed;bottom:0;left:0;right:0;padding:9px 28px;background:var(--panel);
 border-top:1px solid var(--line);font-size:12.5px;color:var(--dim);z-index:9}}
h2{{font-size:14px;margin:28px 28px 6px;color:var(--accent);font-weight:600}}
.note{{margin:0 28px 10px;color:var(--dim);font-size:12px}}
table{{border-collapse:collapse;width:100%}}
th,td{{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line)}}
th{{font-size:11.5px;color:var(--dim);font-weight:600;letter-spacing:.02em}}
tr{{cursor:pointer}} tr:hover{{background:#1c2030}} tr.on{{background:#1e2740}}
tr.on td.lb{{color:var(--accent);font-weight:600}}
td.pl{{width:34px;color:var(--dim)}}
td.num{{width:92px;color:var(--dim);font-variant-numeric:tabular-nums}}
tr.zero td.lb{{color:var(--dim)}}
tr.spd td.lb{{color:#c084fc}}
.rec{{display:inline-block;margin-left:8px;padding:1px 7px;border-radius:99px;font-size:11px;
 background:rgba(74,222,128,.15);color:var(--good);border:1px solid rgba(74,222,128,.3)}}
</style></head><body>
<header><h1>Kei 语音：零样本 vs 微调</h1>
<div class="sub">台词一律是「何をしているんですか / 你这是在干什么」。点任意一行播放，底部显示当前文件。
「声纹像不像」= 与 Kei 声纹质心的余弦，真 Kei 原声 Cafe_monolog_4 是 <b>+0.159</b>，可当基准线；
「6k+ 毛刺」越低越不刺耳。注意：这个分数只衡量音色，测不到语调和自然度，最终还是要靠耳朵。</div></header>
{render("ja", "日语", '<div class="note">微调后 g15 配种子 42 会拖到 4.4 秒，配 1024 是正常的 2.4 秒 —— 种子仍然是变量。</div>')}
{render("zh", "中文", '<div class="note">微调后语速整体慢了约 1.3 倍（3.2s vs 零样本 2.5s），最后四条是 speed_factor=1.2 的对照，音色分几乎没掉。</div>')}
<div id="np">未播放</div>
<audio id="au" preload="none"></audio>
<script>
const au=document.getElementById('au'),np=document.getElementById('np');let cur=null;
document.querySelectorAll('tbody tr').forEach(tr=>{{
  tr.onclick=()=>{{
    if(cur)cur.classList.remove('on');
    cur=tr;tr.classList.add('on');
    au.src=tr.dataset.src;au.play();
    np.textContent='▶ '+tr.querySelector('td.lb').textContent+'   ('+tr.dataset.src+')';
  }};
}});
au.onended=()=>{{if(cur)cur.classList.remove('on');cur=null;}};
au.onerror=()=>{{np.textContent='⚠ 播放失败：'+au.src;}};
</script></body></html>"""

p = os.path.join(ROOT, "试听对比.html")
open(p, "w", encoding="utf-8").write(html)
print("已生成", p)
