# -*- coding: utf-8 -*-
"""集中解析所有路径。改 config.yaml 一处，全部脚本生效。

优先级：环境变量 > config.yaml > 内置默认值。

为什么不做成命令行参数：这些数据集的产物（wav / list / 中间特征）本来就该固定在
一个地方，让每个脚本各自接一个 --gsv-root 只会到处传参，还容易一半用 A 一半用 B。
"""
import os

import yaml

# 仓库根目录 = scripts/ 的上一级
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config.yaml")


def _load():
    if not os.path.isfile(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_c = _load()


def _get(key, default=None, env=None):
    """环境变量优先，其次 config.yaml，最后默认值。"""
    if env and os.environ.get(env):
        return os.environ[env]
    if key in _c and _c[key] not in (None, ""):
        return _c[key]
    return default


def _need(value, key, hint):
    if not value:
        raise SystemExit(
            f"缺少配置项 {key}。请复制 config.example.yaml 为 config.yaml 并填写。\n  说明：{hint}"
        )
    return value


# ---------- 外部依赖 ----------
# GPT-SoVITS 整合包根目录（里面要有 GPT_SoVITS/ 和 api_v2.py）
GSV = _need(_get("gpt_sovits_root", env="GPT_SOVITS_ROOT"),
            "gpt_sovits_root", "GPT-SoVITS 仓库或整合包的根目录，例如 D:/GPT-SoVITS/GPT-SoVITS")

# ComfyUI 的 ComfyUI/ 子目录（里面要有 models/、custom_nodes/）
COMFY = _get("comfyui_root", env="COMFYUI_ROOT")
COMFY_OUTPUT = os.path.join(COMFY, "output") if COMFY else None

# funasr / modelscope 的缓存目录。必须是纯 ASCII 路径，
# 否则 Windows 下 sentencepiece 打不开中文用户目录。
ASCII_CACHE = _get("model_cache", env="MODELSCOPE_CACHE")

# ---------- 角色与数据 ----------
CHARACTER = _get("character", "kei")
LANGUAGE = _get("language", "ja")

# 中间产物都放这里（ogg / wav / list / 特征），不进 git
DATASET = os.path.join(ROOT, "dataset")
OGG_DIR = os.path.join(DATASET, "ogg")
WAV_DIR = os.path.join(DATASET, CHARACTER)

# GPT-SoVITS 要求清单放在它自己的工作目录下，训练脚本才能读到
LIST_PATH = os.path.join(GSV, "dataset", f"{CHARACTER}.list")
EXP_NAME = _get("experiment", CHARACTER)

# 参考音频（3~10 秒、必须带准确台词）。仓库不附带任何音频，由 build_refs.py 从数据集里挑
REF_DIR = os.path.join(ROOT, "ref")

# 各类输出
OUT_DIR = os.path.join(ROOT, "output")
FT_OUT = os.path.join(OUT_DIR, "ft")
DEMO_DIR = os.path.join(OUT_DIR, "demo")
EMO_OUT = os.path.join(OUT_DIR, "emotion")

# 声纹质心（score_ft.py / sweep_ja.py 用）。由 build_centroid.py 生成
ASSETS = os.path.join(ROOT, "assets")


def ensure_dirs():
    """按需建目录，避免各脚本各自 makedirs 漏掉。"""
    for d in (DATASET, OGG_DIR, WAV_DIR, OUT_DIR, FT_OUT, DEMO_DIR, EMO_OUT, ASSETS):
        os.makedirs(d, exist_ok=True)


def describe():
    """打印一份当前生效的路径，排查「为什么找不到文件」时先看这个。"""
    rows = [
        ("仓库根", ROOT),
        ("配置文件", CONFIG_PATH if os.path.isfile(CONFIG_PATH) else "（无，用默认值）"),
        ("GPT-SoVITS", GSV),
        ("ComfyUI", COMFY or "（未配置）"),
        ("角色 / 语言", f"{CHARACTER} / {LANGUAGE}"),
        ("数据集", DATASET),
        ("训练清单", LIST_PATH),
        ("模型缓存", ASCII_CACHE or "（未配置，用默认）"),
    ]
    w = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k.ljust(w)}  {v}")


if __name__ == "__main__":
    describe()
