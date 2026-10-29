# -*- coding: utf-8 -*-
"""声纹相似度的唯一实现。build_centroid / score_ft / sweep_ja 都调这里，避免各写一份走样。

特征：40 kHz 采样，20 维 MFCC 的「均值 + 标准差」拼成 40 维向量。
      拼上标准差是因为语气平稳与否本身也是音色的一部分，只取均值分不出「同一个人、
      一句话说得激动一句话说得平淡」。

打分：(v - mu) / sd 后与质心求余弦。

**这个指标只测音色，测不到韵律和清晰度。** 必须配反向转写一起看——
单看声纹分很容易得出「很像但念错了字」的错误结论。
"""
import librosa
import numpy as np

SR = 40000
N_MFCC = 20
N_FFT = 2048
HOP = 512


def feature(path):
    """一条音频 -> 40 维声纹向量。"""
    y, _ = librosa.load(path, sr=SR, mono=True)
    m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])


def build(paths):
    """一批音频 -> (mu, sd, centroid)。"""
    V = np.vstack([feature(p) for p in paths])
    mu = V.mean(axis=0)
    sd = V.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return mu, sd, ((V - mu) / sd).mean(axis=0)


def score(path, mu, sd, centroid):
    """一条音频与质心的余弦相似度，范围大致 -1 ~ +1。"""
    v = (feature(path) - mu) / (sd + 1e-9)
    return float(np.dot(v, centroid) /
                 (np.linalg.norm(v) * np.linalg.norm(centroid) + 1e-9))


def load(assets_dir, character):
    """从 assets/ 读 mu / sd / centroid_<角色>。"""
    import os
    return (np.load(os.path.join(assets_dir, "mu.npy")),
            np.load(os.path.join(assets_dir, "sd.npy")),
            np.load(os.path.join(assets_dir, f"centroid_{character}.npy")))
