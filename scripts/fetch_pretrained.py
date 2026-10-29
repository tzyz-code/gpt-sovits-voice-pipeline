# -*- coding: utf-8 -*-
"""下载 GPT-SoVITS 底模（约 1.3 GB），放进 GPT_SoVITS/pretrained_models/。

**为什么底模不放在仓库里，而是用脚本下载**

三个硬理由，不是懒：
1. 5 个文件超过 GitHub 单文件 100 MB 硬限制（最大的 pytorch_model.bin 621 MB），
   直接 commit 会被 git 拒绝，push 根本推不上去。
2. 就算改用 LFS，免费额度 1 GB 装不下 1.3 GB，而且每次别人 clone 都要再消耗一遍带宽。
3. 底模有独立的发布渠道和许可，由官方仓库分发更合适——GPT-SoVITS 自己也是这么做的。

所以体验上等价：一条命令，断点不管、失败可重试，下完自动校验。

三个源，国内优先 ModelScope：
    ms         https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained
    hf-mirror  https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained
    hf         https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained

用法：
    python scripts/fetch_pretrained.py --check        # 只看还缺什么，不下载
    python scripts/fetch_pretrained.py                # 自动挑源下载
    python scripts/fetch_pretrained.py --source hf    # 指定源
    python scripts/fetch_pretrained.py --force        # 重新下载并覆盖
"""
import argparse
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile

from paths import GSV

ZIP_NAME = "pretrained_models.zip"
SOURCES = {
    "ms": "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/",
    "hf-mirror": "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/",
    "hf": "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/",
}

# v2 微调 + 推理实际要用的文件。体积用来判断是不是下了一半就断了。
REQUIRED = {
    "pretrained_models/gsv-v2final-pretrained/s2G2333k.pth": 101,
    "pretrained_models/gsv-v2final-pretrained/s2D2333k.pth": 89,
    "pretrained_models/gsv-v2final-pretrained/"
    "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt": 148,
    "pretrained_models/chinese-hubert-base/pytorch_model.bin": 180,
    "pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin": 621,
    # 语种自动识别才需要；我们的流程显式传语种，缺了也能跑
    "pretrained_models/fast_langdetect/lid.176.bin": 125,
}
OPTIONAL = {"pretrained_models/fast_langdetect/lid.176.bin"}


def present(path, mb_expected):
    """文件在不在、大小够不够。返回 (ok, 实际MB)。"""
    if not os.path.isfile(path):
        return False, 0.0
    got = os.path.getsize(path) / 1048576
    # 留 20% 余量：不同版本的模型体积会略有差别，别因为差几 MB 就误判
    return got >= mb_expected * 0.8, got


def check():
    """报告缺哪些、缺多少，返回缺失列表。"""
    base = os.path.join(GSV, "GPT_SoVITS")
    missing, total = [], 0.0
    print(f"{'文件':<52}{'状态':>10}{'体积':>10}")
    for rel, mb in REQUIRED.items():
        ok, got = present(os.path.join(base, rel), mb)
        mark = "✓" if ok else ("可选缺" if rel in OPTIONAL else "缺失")
        if not ok and rel not in OPTIONAL:
            missing.append(rel)
            total += mb
        print(f"{rel:<52}{mark:>10}{got:>9.1f}M")
    # 缺的按压缩包总量估，不按单文件加总（zip 里还带配置和分词器）
    if missing:
        print(f"\n还缺 {len(missing)} 个必需文件，约需下载 1.3 GB（压缩包总量）")
    else:
        print("\n底模齐全，可以训练了。")
    return missing


def download(url, dst):
    """流式下载 + 进度。用 stdlib，避免为了引导脚本再装依赖。"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, t0, last = 0, time.time(), 0.0
        while True:
            chunk = r.read(1024 * 512)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last > 1.0:                     # 每秒刷一次，别刷屏
                last = now
                mb = done / 1048576
                if total:
                    pct = done / total * 100
                    speed = mb / max(now - t0, 0.1)
                    print(f"\r  {pct:5.1f}%  {mb:7.1f} / {total/1048576:.1f} MB  "
                          f"{speed:5.1f} MB/s", end="", flush=True)
                else:
                    print(f"\r  {mb:7.1f} MB", end="", flush=True)
        print()
        return done


def extract(zip_path, dest):
    """解压到 GPT_SoVITS/，zip 内部就是 pretrained_models/ 开头。"""
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        if not any(n.startswith("pretrained_models/") for n in names):
            # 结构不对就别乱解，否则文件会散到 GPT_SoVITS/ 根目录下
            print("  ⚠ 压缩包结构不像官方的（应以 pretrained_models/ 开头），停止解压")
            print("  前几个条目：", names[:5])
            print("  请手动解压，确保得到 GPT_SoVITS/pretrained_models/ 这一层")
            return False
        z.extractall(dest)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查，不下载")
    ap.add_argument("--source", choices=list(SOURCES), default=None)
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    a = ap.parse_args()

    missing = check()
    if a.check or (not missing and not a.force):
        return

    dest = os.path.join(GSV, "GPT_SoVITS")
    if not os.path.isdir(dest):
        sys.exit(f"GPT-SoVITS 目录不存在：{dest}\n"
                 f"先改 config.yaml 里的 gpt_sovits_root。")

    order = [a.source] if a.source else ["ms", "hf-mirror", "hf"]
    tmpdir = tempfile.mkdtemp(prefix="gsv_")
    ok = False
    try:
        for name in order:
            url = SOURCES[name] + ZIP_NAME
            dst = os.path.join(tmpdir, ZIP_NAME)
            print(f"\n[{name}] {url}")
            try:
                size = download(url, dst)
            except Exception as e:
                print(f"  下载失败：{e}")
                continue
            if size < 200 * 1048576:
                print(f"  文件只有 {size/1048576:.0f} MB，不像完整压缩包，换下一个源")
                continue
            print(f"  下载完成 {size/1048576:.0f} MB，解压中…")
            if extract(dst, dest):
                ok = True
                break
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if not ok:
        sys.exit("所有源都失败了。可以手动下载 pretrained_models.zip 解压到 GPT_SoVITS/ 下。\n"
                 "地址见 docs/01-setup.md。")

    print()
    left = check()
    if left:
        sys.exit(f"解压后仍缺 {len(left)} 个文件，可能压缩包版本不对。")
    print("完成。")


if __name__ == "__main__":
    main()
