# -*- coding: utf-8 -*-
"""给情绪 A/B 结果做客观测量：音高、音高波动、能量、语速。

情绪在声学上有比较稳的对应关系：
  慌张/激动 -> 基频(F0)抬高、F0 标准差变大（语调起伏猛）、语速加快
  平静/温柔 -> F0 低且稳、语速慢
所以这里量四个量，只要 C 组相对于 A 组在「F0 更高、波动更大、更快」上成立，
就说明情绪真的被模型学进去了，而不是心理暗示。

音高用 librosa.pyin（比 yin 抗噪、八度错误少），只统计有声帧（voiced）。
"""
import glob
import json
import os

import librosa
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from paths import EMO_OUT as D


def measure(p):
    y, sr = librosa.load(p, sr=32000, mono=True)
    f0, vflag, _ = librosa.pyin(y, fmin=70, fmax=600, sr=sr,
                                frame_length=2048, hop_length=320)
    v = f0[~np.isnan(f0)]
    rms = librosa.feature.rms(y=y)[0]
    # 有声段时长占比 -> 说话密度；配合总时长看语速
    return {
        "dur": len(y) / sr,
        "f0_mean": float(np.mean(v)) if v.size else 0.0,
        "f0_std": float(np.std(v)) if v.size else 0.0,
        "f0_p90": float(np.percentile(v, 90)) if v.size else 0.0,
        "f0_range": float(np.percentile(v, 95) - np.percentile(v, 5)) if v.size else 0.0,
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "voiced_ratio": float(len(v) / max(1, len(f0))),
    }


def main():
    mf = os.path.join(D, "manifest.json")
    rows = json.load(open(mf, encoding="utf-8"))
    for r in rows:
        r.update(measure(r["wav"]))
    json.dump(rows, open(mf, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    tags = sorted({r["tag"] for r in rows})
    label = {r["tag"]: r["label"] for r in rows}
    ids = [x[0] for x in [("f1",), ("f2",), ("f3",), ("n1",)]] if False else ["f1", "f2", "f3", "n1"]

    print("=== 音高均值 F0 (Hz)，越高越激动 ===")
    print("句\\组  " + "".join(f"{t:>13}" for t in tags))
    for i in ids:
        line = f"{i:<6}"
        for t in tags:
            r = next(x for x in rows if x["tag"] == t and x["id"] == i)
            line += f"{r['f0_mean']:>13.1f}"
        print(line)

    print("\n=== 音高波动 F0 标准差 (Hz)，越大越起伏 ===")
    print("句\\组  " + "".join(f"{t:>13}" for t in tags))
    for i in ids:
        line = f"{i:<6}"
        for t in tags:
            r = next(x for x in rows if x["tag"] == t and x["id"] == i)
            line += f"{r['f0_std']:>13.1f}"
        print(line)

    print("\n=== 时长 (s)，越短越急 ===")
    print("句\\组  " + "".join(f"{t:>13}" for t in tags))
    for i in ids:
        line = f"{i:<6}"
        for t in tags:
            r = next(x for x in rows if x["tag"] == t and x["id"] == i)
            line += f"{r['dur']:>13.2f}"
        print(line)

    print("\n=== 慌张句(f1~f3)均值 / 平静句(n1) ===")
    print(f"{'组':<22}{'F0均值':>9}{'F0波动':>9}{'时长':>8}")
    for t in tags:
        angry = [x for x in rows if x["tag"] == t and x["id"].startswith("f")]
        calm = next(x for x in rows if x["tag"] == t and x["id"] == "n1")
        print(f"{t + ' ' + label[t][:16]:<22}"
              f"{np.mean([x['f0_mean'] for x in angry]):>9.1f}"
              f"{np.mean([x['f0_std'] for x in angry]):>9.1f}"
              f"{np.mean([x['dur'] for x in angry]):>8.2f}"
              f"   |平静 {calm['f0_mean']:.1f}Hz {calm['dur']:.2f}s")


if __name__ == "__main__":
    main()
