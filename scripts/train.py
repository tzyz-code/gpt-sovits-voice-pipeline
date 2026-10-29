# -*- coding: utf-8 -*-
"""GPT-SoVITS v2 微调：一行命令跑完预处理 + 训练。

    python train.py prep     # 只跑预处理（文本/SSL特征/语义token）
    python train.py s2       # 只训 SoVITS
    python train.py s1       # 只训 GPT
    python train.py all      # 全流程

设计说明：直接调用官方 prepare_datasets / s*_train 脚本，参数与 webui.py 的
open1a/open1b/open1c/open1Ba/open1Bb 完全一致，只是绕开 GUI。
"""
import json
import os
import subprocess
import sys

import yaml

from paths import GSV
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
from paths import EXP_NAME as EXP
from paths import LIST_PATH as LIST
from paths import WAV_DIR
OPT = os.path.join(GSV, "logs", EXP)
TMP = os.path.join(GSV, "TEMP")
FFMPEG_BIN = os.path.join(os.path.dirname(GSV), "ffmpeg", "bin")
PM = os.path.join(GSV, "GPT_SoVITS", "pretrained_models")
V2 = os.path.join(PM, "gsv-v2final-pretrained")

S2G = os.path.join(V2, "s2G2333k.pth")
S2D = os.path.join(V2, "s2D2333k.pth")
S1 = os.path.join(V2, "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
BERT = os.path.join(PM, "chinese-roberta-wwm-ext-large")
HUBERT = os.path.join(PM, "chinese-hubert-base")

# 训练超参（8G 显存笔记本）
# 注意：步数 ≈ 样本数/batch × epoch。通用模型 123 条 × 10 轮 ≈ 640 步。
# 情绪子集只有 20~35 条，必须按比例放大 epoch，否则总步数太少等于没训。
S2_EPOCH, S2_BATCH, S2_SAVE_EVERY = 10, 2, 2
S1_EPOCH, S1_BATCH, S1_SAVE_EVERY = 15, 4, 3


def configure(exp=None, list_path=None, s2_epoch=None, s1_epoch=None,
              s2_save=None, s1_save=None):
    """切换实验（一个情绪一个实验）。必须在调用 prep/train_s2/train_s1 之前调。"""
    global EXP, LIST, OPT
    global S2_EPOCH, S1_EPOCH, S2_SAVE_EVERY, S1_SAVE_EVERY
    if exp:
        EXP = exp
        OPT = os.path.join(GSV, "logs", EXP)
    if list_path:
        LIST = list_path
    if s2_epoch:
        S2_EPOCH = s2_epoch
    if s1_epoch:
        S1_EPOCH = s1_epoch
    if s2_save:
        S2_SAVE_EVERY = s2_save
    if s1_save:
        S1_SAVE_EVERY = s1_save


def est_steps():
    """粗估总训练步数，方便判断这个情绪子集能不能训。"""
    try:
        n = sum(1 for _ in open(LIST, encoding="utf-8"))
    except OSError:
        return None
    return n // S2_BATCH * S2_EPOCH, n // S1_BATCH * S1_EPOCH


def run(cmd, env=None, log=None):
    e = os.environ.copy()
    e.update(env or {})
    print(f"\n$ {cmd[0] if isinstance(cmd, list) else cmd} ...", flush=True)
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n\n===== {' '.join(cmd)} =====\n")
            p = subprocess.run(cmd, cwd=GSV, env=e, stdout=f, stderr=subprocess.STDOUT)
    else:
        p = subprocess.run(cmd, cwd=GSV, env=e)
    if p.returncode != 0:
        raise SystemExit(f"失败，退出码 {p.returncode}（日志：{log}）")


def base_env():

    return {
        # prepare_datasets 里的脚本要 import text / feature_extractor（在 GPT_SoVITS 下）
        # 和 tools（在仓库根）。两个都塞进 PYTHONPATH。
        "PYTHONPATH": os.pathsep.join(
            [GSV, os.path.join(GSV, "GPT_SoVITS"), os.environ.get("PYTHONPATH", "")]),
        # tools/my_utils.load_audio 走 ffmpeg-python，必须能在 PATH 里找到 ffmpeg.exe
        "PATH": os.pathsep.join(
            [FFMPEG_BIN, os.environ.get("PATH", "")]),
        # 8G 显存被桌面程序占掉 ~1.3G，剩下的空间容易碎片化 -> 必须开可扩展段
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "inp_text": LIST,
        "inp_wav_dir": WAV_DIR,
        "exp_name": EXP,
        "i_part": "0",
        "all_parts": "1",
        "opt_dir": OPT,
        "is_half": "True",
        "version": "v2",
        "bert_pretrained_dir": BERT,
        "cnhubert_base_dir": HUBERT,
        "pretrained_s2G": S2G,
        "s2config_path": os.path.join(GSV, "GPT_SoVITS", "configs", "s2.json"),
        "_CUDA_VISIBLE_DEVICES": "0",
    }


def prep():
    os.makedirs(OPT, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    # 权重输出目录必须存在，否则 savee 会静默失败（webui 启动时本来就建好了）
    for d in ("SoVITS_weights_v2", "GPT_weights_v2"):
        os.makedirs(os.path.join(GSV, d), exist_ok=True)
    # 清掉上一轮跑空留下的语义文件，否则 3-get-semantic 会以为已完成而跳过
    for f in os.listdir(OPT):
        if f.startswith("6-name2semantic"):
            p = os.path.join(OPT, f)
            if os.path.getsize(p) == 0:
                os.remove(p)
    env = base_env()

    # 1a 文本 -> phones + word2ph
    run([PY, "-s", "GPT_SoVITS/prepare_datasets/1-get-text.py"], env,
        os.path.join(TMP, "log_1text.txt"))
    parts = [f for f in os.listdir(OPT) if f.startswith("2-name2text-")]
    merged = os.path.join(OPT, "2-name2text.txt")
    with open(merged, "w", encoding="utf-8") as fo:
        for f in sorted(parts):
            fo.write(open(os.path.join(OPT, f), encoding="utf-8").read())
    n = sum(1 for _ in open(merged, encoding="utf-8"))
    print(f"[1a] 文本特征 {n} 行 -> {merged}")
    if n < len(open(LIST, encoding="utf-8").read().strip().split("\n")):
        print("  ⚠️ 有行被跳过，见 log_1text.txt 里的 traceback")

    # 1b SSL 特征 + 32k wav
    run([PY, "-s", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"], env,
        os.path.join(TMP, "log_1b.txt"))
    print(f"[1b] hubert 完成")

    # 1c 语义 token
    run([PY, "-s", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"], env,
        os.path.join(TMP, "log_1c.txt"))
    sem = os.path.join(OPT, "6-name2semantic.tsv")
    parts_sem = [f for f in os.listdir(OPT) if f.startswith("6-name2semantic-")]
    opt = ["item_name\tsemantic_audio"]
    for f in sorted(parts_sem):
        opt += open(os.path.join(OPT, f), encoding="utf-8").read().strip("\n").split("\n")
        os.remove(os.path.join(OPT, f))
    if len(opt) == 1 and os.path.exists(sem):
        print("[1c] 已存在完整语义文件，跳过合并")
    else:
        open(sem, "w", encoding="utf-8").write("\n".join(opt) + "\n")
        print(f"[1c] 语义 token {len(opt)-1} 行 -> {sem}")


def train_s2():
    cfg = json.load(open(os.path.join(GSV, "GPT_SoVITS", "configs", "s2.json")))
    cfg["train"].update({
        "batch_size": S2_BATCH, "epochs": S2_EPOCH, "text_low_lr_rate": 0.4,
        "pretrained_s2G": S2G, "pretrained_s2D": S2D,
        "if_save_latest": True, "if_save_every_weights": True,
        "save_every_epoch": S2_SAVE_EVERY, "gpu_numbers": "0",
        "grad_ckpt": True, "lora_rank": 32,
    })
    cfg["model"]["version"] = "v2"
    cfg["data"]["exp_dir"] = cfg["s2_ckpt_dir"] = OPT
    cfg["save_weight_dir"] = "SoVITS_weights_v2"
    cfg["name"] = EXP
    cfg["version"] = "v2"
    os.makedirs(os.path.join(OPT, "logs_s2_v2"), exist_ok=True)
    p = os.path.join(TMP, f"tmp_s2_{EXP}.json")
    json.dump(cfg, open(p, "w"), ensure_ascii=False)
    # 走单进程包装器：官方的 mp.spawn + 单卡 DDP 在本机会触发
    # "CUDA error: resource already mapped"
    run([PY, "-u", os.path.join(HERE, "s2_single.py"), "--config", p], base_env(),
        os.path.join(TMP, "log_s2.txt"))


def train_s1():
    cfg = yaml.safe_load(open(os.path.join(GSV, "GPT_SoVITS", "configs", "s1longer-v2.yaml")))
    cfg["train"].update({
        "batch_size": S1_BATCH, "epochs": S1_EPOCH,
        "save_every_n_epoch": S1_SAVE_EVERY, "if_save_every_weights": True,
        "if_save_latest": True, "if_dpo": False,
        "half_weights_save_dir": "GPT_weights_v2",
        "exp_name": EXP,
    })
    cfg["pretrained_s1"] = S1
    cfg["train_semantic_path"] = os.path.join(OPT, "6-name2semantic.tsv")
    cfg["train_phoneme_path"] = os.path.join(OPT, "2-name2text.txt")
    cfg["output_dir"] = os.path.join(OPT, "logs_s1_v2")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    p = os.path.join(TMP, f"tmp_s1_{EXP}.yaml")
    yaml.dump(cfg, open(p, "w", encoding="utf-8"), default_flow_style=False)
    env = base_env()
    env["hz"] = "25hz"
    run([PY, "-u", "GPT_SoVITS/s1_train.py", "--config_file", p], env,
        os.path.join(TMP, "log_s1.txt"))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["prep", "s2", "s1", "all"])
    # 一个情绪一个实验：--exp fluster 会读 dataset/fluster.list、写 logs/fluster
    ap.add_argument("--exp", default=None, help="实验名，默认 kei")
    ap.add_argument("--list", dest="list_file", default=None, help="训练清单，默认 dataset/<exp>.list")
    ap.add_argument("--s2-epoch", type=int, default=None)
    ap.add_argument("--s1-epoch", type=int, default=None)
    ap.add_argument("--s2-save", type=int, default=None)
    ap.add_argument("--s1-save", type=int, default=None)
    a = ap.parse_args()

    if a.exp or a.list_file or a.s2_epoch or a.s1_epoch:
        configure(exp=a.exp,
                  list_path=a.list_file or (os.path.join(GSV, "dataset", f"{a.exp}.list") if a.exp else None),
                  s2_epoch=a.s2_epoch, s1_epoch=a.s1_epoch,
                  s2_save=a.s2_save, s1_save=a.s1_save)

    for d in ("SoVITS_weights_v2", "GPT_weights_v2"):
        os.makedirs(os.path.join(GSV, d), exist_ok=True)
    n = sum(1 for _ in open(LIST, encoding="utf-8"))
    s2s, s1s = est_steps()
    print(f"experiment={EXP}  list={LIST}  samples={n}")
    print(f"  计划：SoVITS {S2_EPOCH}e -> 约 {s2s} 步 | GPT {S1_EPOCH}e -> 约 {s1s} 步")
    if a.stage in ("prep", "all"):
        prep()
    if a.stage in ("s2", "all"):
        train_s2()
    if a.stage in ("s1", "all"):
        train_s1()
    print("\n完成。权重：")
    for d in ("SoVITS_weights_v2", "GPT_weights_v2"):
        dd = os.path.join(GSV, d)
        if os.path.isdir(dd):
            for f in sorted(os.listdir(dd)):
                print(" ", os.path.join(d, f))
