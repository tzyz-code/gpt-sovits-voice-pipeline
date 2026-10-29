# -*- coding: utf-8 -*-
"""把 ComfyUI 里用 {情绪} 标签生成的 5 条转成 wav、量音高，做一张试听页。

这是「情绪该怎么做」的最终形态：同一个模型、同一句台词，只在剧本里换一个 {标签}，
节点就去换参考音频——不需要五套权重。

    python make_comfy_page.py
"""
import glob
import json
import os

import librosa
import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
from paths import COMFY_OUTPUT as SRC
from paths import ROOT
from paths import OUT_DIR
OUT = os.path.join(OUT_DIR, "comfy")
SENT = "先生、これはどういうことですか。"

# 标签 -> 实际用的参考音频（对应 setup_comfy_character.py 里的 REFS）
EMO = [
    ("main", "主参考 · 平静", "Kei_Cafe_monolog_4", "自分の身体を持つのは……。不便なことでもあるんですね……。", "无起伏的陈述句，句尾下沉"),
    ("慌张", "慌张", "Kei_EventShop_1_2_Radio", "もう！誰も言うことを聞かないんですから！", "语速快、句尾上挑、气声多"),
    ("害羞", "害羞", "Kei_Lobby_1_2", "い、嫌なのかって……？と、というより……！", "句首结巴、音量偏小"),
    ("温柔", "温柔", "Kei_MemorialLobby_4_1", "だから大丈夫……絶対、大丈夫です。", "放慢、压低、尾音收得很软"),
    ("得意", "得意", "Kei_Battle_Victory_1", "当然の結果です。見ていましたよね、先生？", "音高上抬、带一点炫耀的上扬"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for i, (tag, name, ref, reftext, feel) in enumerate(EMO):
        cand = glob.glob(os.path.join(SRC, f"kei_emo_{i}_{tag}_*.flac"))
        if not cand:
            print("缺文件", tag)
            continue
        y, sr = librosa.load(cand[0], sr=32000, mono=True)
        wav = os.path.join(OUT, f"{i}_{tag}.wav")
        sf.write(wav, y, sr)
        f0, _, _ = librosa.pyin(y, fmin=70, fmax=700, sr=sr,
                                frame_length=2048, hop_length=320)
        v = f0[~np.isnan(f0)]
        rows.append({
            "i": i, "tag": tag, "name": name, "wav": os.path.basename(wav),
            "ref": ref, "reftext": reftext, "feel": feel,
            "dur": round(len(y) / sr, 2),
            "f0": round(float(np.mean(v)), 1) if v.size else 0,
            "f0sd": round(float(np.std(v)), 1) if v.size else 0,
        })
        print(f"{name:<12}{rows[-1]['dur']:>5.2f}s  F0 {rows[-1]['f0']:>6.1f}Hz  ±{rows[-1]['f0sd']:>5.1f}")

    json.dump(rows, open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    base = rows[0]["f0"]
    trs = []
    for r in rows:
        d = r["f0"] - base
        cls = "up" if d > 20 else ("down" if d < -20 else "")
        trs.append(
            f'<tr data-src="out_comfy/{r["wav"]}">'
            f'<td class="pl">▶</td>'
            f'<td><b>{r["name"]}</b><div class="dim">{r["feel"]}</div></td>'
            f'<td class="num">{r["f0"]:.0f} Hz</td>'
            f'<td class="num {cls}">{"+" if d>0 else ""}{d:.0f}</td>'
            f'<td class="num">±{r["f0sd"]:.0f}</td>'
            f'<td class="num">{r["dur"]:.2f}s</td>'
            f'<td class="ref"><code>{r["ref"]}</code><div class="dim">{r["reftext"]}</div></td>'
            f'</tr>')

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>ComfyUI 情绪标签验证</title>
<style>
:root{{--bg:#12141a;--panel:#181b23;--line:#272b36;--fg:#e6e8ee;--dim:#8b93a7;
      --accent:#6f9cff;--up:#fbbf24;--down:#7dd3fc;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);
 font:14px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;padding:0 0 72px}}
header{{padding:22px 28px 16px;border-bottom:1px solid var(--line)}}
h1{{font-size:17px;margin:0 0 6px;font-weight:600}}
.sub{{color:var(--dim);font-size:12.5px;max-width:940px}}
h2{{font-size:13px;margin:26px 28px 8px;color:var(--accent);font-weight:600;
    text-transform:uppercase;letter-spacing:.06em}}
table{{border-collapse:collapse;width:100%;margin:0 28px;width:calc(100% - 56px)}}
th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{font-size:11.5px;color:var(--dim);font-weight:600;white-space:nowrap}}
tr{{cursor:pointer}} tr:hover{{background:#1c2030}} tr.on{{background:#1e2740}}
tr.on td:first-child{{color:var(--accent)}}
td.pl{{width:34px;color:var(--dim)}}
td.num{{width:88px;font-variant-numeric:tabular-nums;color:#c8cfdd;white-space:nowrap}}
.up{{color:var(--up)}} .down{{color:var(--down)}}
.dim{{color:var(--dim);font-size:11.5px}}
.ref code{{font-size:11.5px;background:#20263a;padding:1px 5px;border-radius:4px;color:#a9b4cc}}
pre{{margin:0 28px;background:var(--panel);border:1px solid var(--line);border-radius:8px;
 padding:14px 16px;font-size:12.5px;line-height:1.9;overflow:auto;color:#cfd5e2}}
pre b{{color:var(--accent)}}
ul{{margin:0 28px;padding-left:20px;font-size:13px;color:#cfd5e2}}
ul li{{margin:7px 0}}
#np{{position:fixed;bottom:0;left:0;right:0;padding:9px 28px;background:var(--panel);
 border-top:1px solid var(--line);font-size:12.5px;color:var(--dim)}}
</style></head><body>
<header><h1>ComfyUI 里的情绪切换</h1>
<div class="sub">同一句台词「{SENT}」、同一个模型、同一套采样参数（种子 1024 / 语速 1.2 / 温度 0.8），
唯一变化是剧本开头的 <code>{{标签}}</code>。点格子播放。</div></header>

<h2>五条对照</h2>
<table><thead><tr><th></th><th>情绪</th><th>F0 均值</th><th>相对平静</th><th>波动</th>
<th>时长</th><th>换成的参考音频</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table>

<h2>剧本怎么写</h2>
<pre>{{main}}先生、これはどういうことですか。
<b>{{慌张}}</b>えっ、もうそんな時間ですか！？
<b>{{温柔}}</b>……無理はしないでくださいね。<b>[pause:0.4]</b>
<b>{{害羞}}</b>べ、別に心配しているわけではありませんから。</pre>
<ul>
<li><b>{{情绪}}</b> 之后的句子改用这条参考音频，<b>{{main}}</b> 切回主参考。</li>
<li><b>[pause:0.4]</b> 停顿，也可以写 <b>[pause:400ms]</b>。</li>
<li><b>[speed:1.2]</b> 只管紧跟的这一段（到换行或下一个标签）。</li>
<li>多角色对戏写 <b>[Kei]</b> / <b>[Kei慌张]</b>，同一段台词里可以来回切。</li>
</ul>

<h2>为什么是换参考音频，而不是训多个模型</h2>
<ul>
<li>实测：只换参考音频，F0 抬升 <b>+119Hz</b>；只换模型（参考不变）只抬 <b>+14Hz</b>。</li>
<li>专门训的慌张模型，音高上和通用模型 + 慌张参考几乎一样，多出来的是<b>起伏幅度</b>
（F0 标准差 ±59 → ±85，+45%），听感上是抑扬更夸张。</li>
<li>所以 ComfyUI 里这套角色用<b>一个通用模型 + 8 条情绪参考</b>，
比维护五套权重省事，也是唯一能做到「一句平一句慌」的写法。</li>
</ul>

<div id="np">未播放</div>
<audio id="au" preload="none"></audio>
<script>
const au=document.getElementById('au'),np=document.getElementById('np');let cur=null;
document.querySelectorAll('tbody tr').forEach(tr=>{{
  tr.onclick=()=>{{
    if(cur)cur.classList.remove('on');
    cur=tr;tr.classList.add('on');
    au.src=tr.dataset.src;au.play();
    np.textContent='▶ '+tr.querySelector('b').textContent;
  }};
}});
au.onended=()=>{{if(cur)cur.classList.remove('on');cur=null;}};
au.onerror=()=>{{np.textContent='⚠ 播放失败：'+au.src;}};
</script></body></html>"""

    p = os.path.join(ROOT, "ComfyUI情绪演示.html")
    open(p, "w", encoding="utf-8").write(html)
    print("\n->", p)


if __name__ == "__main__":
    main()
