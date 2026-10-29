# -*- coding: utf-8 -*-
"""生成情绪模型 A/B 试听页（四组对照 + 客观测量）。

    python make_emotion_page.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
from paths import ROOT
from paths import EMO_OUT as D
ROWS = json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))

ORDER = ["A", "B", "C", "D"]
LABEL = {
    "A": "通用模型<br>+ 中性参考",
    "B": "通用模型<br>+ 慌张参考",
    "C": "慌张模型<br>+ 慌张参考",
    "D": "慌张模型<br>+ 中性参考",
}
TAGNOTE = {
    "A": "现状基线",
    "B": "只换参考音频",
    "C": "完整情绪方案",
    "D": "只换模型",
}
SENTS = [
    ("f1", "慌张 · 被打断", "えっ、ええっ！？ちょ、ちょっと待ってください、そんな急に言われても困ります……！"),
    ("f2", "慌张 · 抱怨", "もう！せっかく片付けたのに、また散らかしたんですか！？"),
    ("f3", "慌张 · 强调", "ふ、ふざけないでください！私は真面目に言ってるんですよ！？"),
    ("n1", "平静 · 对照句", "今日は、静かな一日でしたね。"),
]


def get(tag, sid):
    return next(r for r in ROWS if r["tag"] == tag and r["id"] == sid)


def cells(sid):
    out = []
    for t in ORDER:
        r = get(t, sid)
        out.append(
            f'<td class="cell c{t}" data-src="emotion/{os.path.basename(r["wav"])}"'
            f' data-label="{t}组 {SENTS_NAME[sid]}">'
            f'<div class="play">▶</div>'
            f'<div class="m"><b>{r["f0_mean"]:.0f}</b>Hz'
            f' <span class="dim">±{r["f0_std"]:.0f}</span></div>'
            f'<div class="d">{r["dur"]:.2f}s</div></td>')
    return "".join(out)


SENTS_NAME = {s[0]: s[1] for s in SENTS}

# 转写准确度：去掉句首语气词后算编辑距离。句首语气词（「えっ」「もう」「ふ、」）
# 恰好是情绪最影响、ASR 也最难还原的部分，算进去会把「情绪变强」误判成「念错字」。
import re as _re  # noqa: E402

_INTERJ = _re.compile(r"^(え、?ええっ?|えっ?|もう|ああ|あ|まあ|もあ|じゃ|ふ、|ふ)+")


def _norm(s):
    s = _re.sub(r"<\|[^|]*\|>", "", s or "")
    s = _re.sub(r"[っッー〜~、。，．…！？!?「」『』（）()・,.\-—\s]", "", s)
    return _INTERJ.sub("", s)


def _acc(asr, tgt):
    a, b = _norm(asr), _norm(tgt)
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p = d[:]
        d[0] = i
        for j, cb in enumerate(b, 1):
            d[j] = min(p[j] + 1, d[j - 1] + 1, p[j - 1] + (ca != cb))
    return 1 - d[len(b)] / max(1, len(b))


for _r in ROWS:
    _r["acc"] = _acc(_r.get("asr"), _r["text"])


def summary():
    rows = []
    for t in ORDER:
        angry = [r for r in ROWS if r["tag"] == t and r["id"].startswith("f")]
        calm = get(t, "n1")
        f0 = sum(r["f0_mean"] for r in angry) / len(angry)
        sd = sum(r["f0_std"] for r in angry) / len(angry)
        du = sum(r["dur"] for r in angry) / len(angry)
        acc = sum(r["acc"] for r in ROWS if r["tag"] == t) / 4
        rows.append((t, f0, sd, du, calm["f0_mean"], acc))
    base = next(r for r in rows if r[0] == "A")
    body = []
    for t, f0, sd, du, cf0, acc in rows:
        df0 = f0 - base[1]
        dsd = sd - base[2]
        cls = "up" if df0 > 30 else ""
        body.append(
            f'<tr><td><b>{t}</b> <span class="dim">{TAGNOTE[t]}</span></td>'
            f'<td class="num">{f0:.0f}</td>'
            f'<td class="num {cls}">{"+" if df0>0 else ""}{df0:.0f}</td>'
            f'<td class="num">{sd:.0f}</td>'
            f'<td class="num {cls}">{"+" if dsd>0 else ""}{dsd:.0f}</td>'
            f'<td class="num">{du:.2f}s</td>'
            f'<td class="num">{cf0:.0f}</td>'
            f'<td class="num">{acc*100:.1f}%</td></tr>')
    return "".join(body)


html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Kei 情绪模型 A/B</title>
<style>
:root{{--bg:#12141a;--panel:#181b23;--line:#272b36;--fg:#e6e8ee;--dim:#8b93a7;
      --accent:#6f9cff;--good:#4ade80;--warn:#fbbf24;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);
 font:14px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;padding:0 0 72px}}
header{{padding:22px 28px 16px;border-bottom:1px solid var(--line)}}
h1{{font-size:17px;margin:0 0 6px;font-weight:600}}
.sub{{color:var(--dim);font-size:12.5px;max-width:940px}}
h2{{font-size:13px;margin:26px 28px 8px;color:var(--accent);font-weight:600;
    text-transform:uppercase;letter-spacing:.06em}}
.wrap{{margin:0 28px}}
table{{border-collapse:collapse;width:100%}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}}
th{{font-size:11.5px;color:var(--dim);font-weight:600;white-space:nowrap}}
td.num{{font-variant-numeric:tabular-nums;color:#c8cfdd}}
.up{{color:var(--warn)}}
.dim{{color:var(--dim);font-size:11.5px}}
.mx th{{text-align:center}} .mx td.cell{{text-align:center;cursor:pointer;width:22%}}
.mx td.cell:hover{{background:#1c2030}} .mx td.cell.on{{background:#1e2740;outline:1px solid var(--accent)}}
.mx td.lead{{width:28%;vertical-align:middle}}
.play{{font-size:15px;color:var(--dim)}}
.cell.on .play{{color:var(--accent)}}
.m{{font-size:13px}} .m b{{font-size:15px;font-variant-numeric:tabular-nums}}
.d{{font-size:11.5px;color:var(--dim)}}
.jp{{font-size:13px;color:#cfd5e2;margin-top:3px}}
.tag{{display:inline-block;font-size:10.5px;padding:1px 6px;border-radius:4px;
      background:#22283a;color:var(--dim);margin-left:6px;vertical-align:1px}}
ul.find{{margin:0 28px;padding-left:20px;font-size:13px;color:#cfd5e2}}
ul.find li{{margin:7px 0}}
ul.find b{{color:var(--fg)}}
#np{{position:fixed;bottom:0;left:0;right:0;padding:9px 28px;background:var(--panel);
 border-top:1px solid var(--line);font-size:12.5px;color:var(--dim)}}
</style></head><body>
<header><h1>Kei 情绪模型 A/B 对照</h1>
<div class="sub">同一个「慌张」情绪，把<b>模型</b>和<b>参考音频</b>两个变量拆开单独看。
每格数字是音高均值（Hz）和音高标准差，点一下播放。四组用同一批台词、同一套采样参数
（语速×1.2 / 温度 0.8 / 种子 1024），唯一变量就是模型权重和参考音频。</div></header>

<h2>盲听对照</h2>
<div class="wrap">
<table class="mx">
<thead><tr><th></th>{''.join(f'<th>{t}<div class="dim">{LABEL[t]}</div></th>' for t in ORDER)}</tr></thead>
<tbody>
{''.join(f'<tr><td class="lead"><span class="tag">{kind}</span><div class="jp">{jp}</div></td>{cells(sid)}</tr>' for sid, kind, jp in SENTS)}
</tbody></table>
</div>

<h2>客观测量（慌张三句均值）</h2>
<div class="wrap">
<table>
<thead><tr><th>组</th><th>F0 均值</th><th>相对 A</th><th>F0 波动</th><th>相对 A</th>
<th>时长</th><th>平静句 F0</th><th>转写准确度</th></tr></thead>
<tbody>{summary()}</tbody></table>
<div class="dim" style="margin-top:6px">F0 越高越显激动，波动越大语调起伏越猛。
「平静句 F0」是关键对照：理想情况下它应该留在 250Hz 附近不被带跑。</div>
</div>

<h2>读出来的结论</h2>
<ul class="find">
<li><b>参考音频是情绪的第一杠杆，不是模型。</b>A→B 只换了参考音频，F0 直接从 258 抬到 377（+119Hz）。
这就是原理三：参考音频的语气会感染全部输出，一个模型换个参考就能换情绪。</li>
<li><b>单独训情绪模型确实有额外收益，但落在「起伏」上。</b>B→C 音高几乎没动（377→374），
F0 波动却从 ±59 涨到 ±85（+45%）。听感上就是同一句话的抑扬更夸张、更「急」，而不是单纯变尖。</li>
<li><b>情绪模型离不开匹配的参考音频。</b>D 把慌张模型配回中性参考，F0 只有 272，几乎退回基线。
说明模型学到的是这套子集的韵律分布，需要同语气的参考把它「激活」，它不会自己切。</li>
<li><b>没有一组能做到按句切换情绪。</b>四组的平静对照句 F0 都跟着抬到了 335~344。
想一句平静一句慌张，必须换参考音频（或换模型），不能靠文本自己判断。</li>
<li><b>情绪模型不损清晰度。</b>C 组反向转写准确度 98.0%，是四组最高的。</li>
</ul>

<div id="np">未播放</div>
<audio id="au" preload="none"></audio>
<script>
const au=document.getElementById('au'),np=document.getElementById('np');let cur=null;
document.querySelectorAll('td.cell').forEach(td=>{{
  td.onclick=()=>{{
    if(cur)cur.classList.remove('on');
    cur=td;td.classList.add('on');
    au.src=td.dataset.src;au.play();
    np.textContent='▶ '+td.dataset.label;
  }};
}});
au.onended=()=>{{if(cur)cur.classList.remove('on');cur=null;}};
au.onerror=()=>{{np.textContent='⚠ 播放失败：'+au.src;}};
</script></body></html>"""

p = os.path.join(ROOT, "情绪模型对比.html")
open(p, "w", encoding="utf-8").write(html)
print("已生成", p, len(ROWS), "条")
