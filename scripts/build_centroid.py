# -*- coding: utf-8 -*-
"""算一批音频的声纹统计量，用于「像不像这个角色」的客观打分。

产出三个 npy，放在 assets/：
    mu.npy                所有训练样本声纹向量的均值
    sd.npy                标准差（做标准化用，避免高能量维主导距离）
    centroid_<角色>.npy   标准化后的声纹质心

!! 读之前必须知道的局限 !!

1. **绝对值没有跨项目意义。** 质心取决于构建它的那批音频和特征口径，重建之后整体会平移。
   实测重建过一个 124 条的质心，结果「真原声 -0.002 / 合成片段 +0.126」——
   排序方向都反了，因为合成音频的高频滚降更一致，反而更靠近均值。
2. **只有同一质心下的横向排序可用**，而且每次换数据集都要先验方向：
   拿几条你确信「就是本人」和几条确信「绝对不是」的音频当标尺，先确认分数是往对的方向走。
3. **它只测音色，测不到韵律和清晰度。** 必须配反向转写一起看，
   否则很容易得出「很像但念错了字」的错误结论。

所以这个脚本的定位是「筛掉明显跑偏的」，不是「证明微调有效」。
判断微调到底成没成，看 docs/04-evaluation.md 里那套组合判据。

    python build_centroid.py
    python build_centroid.py --dir dataset/my_char
"""
import argparse
import glob
import os

import numpy as np

import voiceprint
from paths import ASSETS, CHARACTER, WAV_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=WAV_DIR, help="训练用 wav 目录")
    ap.add_argument("--out", default=ASSETS)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.wav")))
    if not files:
        raise SystemExit(f"{a.dir} 里没有 wav。先跑 make_list.py。")

    ok = []
    for p in files:
        try:
            voiceprint.feature(p)
            ok.append(p)
        except Exception as e:
            print(f"  跳过 {os.path.basename(p)}: {e}")

    mu, sd, centroid = voiceprint.build(ok)
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "mu.npy"), mu)
    np.save(os.path.join(a.out, "sd.npy"), sd)
    cp = os.path.join(a.out, f"centroid_{CHARACTER}.npy")
    np.save(cp, centroid)

    # 自检：拿质心给训练集自己打分。
    # 注意绝对值的量级：质心是 N 条标准化向量的均值，在 40 维空间里它的模长很小，
    # 所以「自己与自己」的余弦只接近 1/N 量级（N 越大越接近 0），不是接近 1。
    # 因此**只能横向比**：同一批候选之间谁高谁低，或者同一质心下不同音频谁更像。
    # 换一批数据重建质心后，绝对值会和之前不一样，这是正常的。
    sims = [voiceprint.score(p, mu, sd, centroid) for p in ok]
    print(f"{len(ok)} 条样本，声纹 {len(mu)} 维")
    print(f"自检（同一质心下的分布）："
          f"均值 {np.mean(sims):+.3f} / 中位 {np.median(sims):+.3f} / "
          f"区间 {min(sims):+.3f} ~ {max(sims):+.3f}")
    print(f"-> {a.out}/mu.npy  sd.npy  centroid_{CHARACTER}.npy")


if __name__ == "__main__":
    main()
