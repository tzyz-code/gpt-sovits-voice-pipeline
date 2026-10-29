"""从 bluearchive.wiki 下载 Kei 的全部语音。

页面 HTML 被 Cloudflare 挡（curl 403），但静态 CDN 不挡。
文件路径是 MediaWiki 的 md5 分桶，可以直接算出来，不用抓页面：

    md5("Kei_Gachaget.ogg") = 4f...  ->  /4/4f/Kei_Gachaget.ogg

用法：
    python fetch_audio.py            # 下载并校验
    python fetch_audio.py --report   # 只统计，不下载
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import httpx

BASE = "https://static.wikitide.net/bluearchivewiki"
LIST_JSON = "page_images.json"   # 见 docs/02-dataset.md：wiki 的图片清单
from paths import OGG_DIR as DST
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"


def url_of(fname: str) -> str:
    h = hashlib.md5(fname.encode()).hexdigest()
    return f"{BASE}/{h[0]}/{h[:2]}/{fname}"


def names() -> list[str]:
    with open(LIST_JSON, encoding="utf-8") as f:
        imgs = json.load(f)["parse"]["images"]
    return [n for n in imgs if n.lower().endswith(".ogg")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    ns = names()
    print(f"清单 {len(ns)} 个 .ogg")
    if a.report:
        return

    os.makedirs(DST, exist_ok=True)
    ok, skip, bad = 0, 0, []
    seen = {}                                   # 内容 md5 -> 文件名（去重）
    with httpx.Client(headers={"User-Agent": UA, "Referer": "https://bluearchive.wiki/"},
                      timeout=60.0, follow_redirects=True) as c:
        for i, n in enumerate(ns, 1):
            p = os.path.join(DST, n)
            if os.path.exists(p) and os.path.getsize(p) > 3000:
                skip += 1
                continue
            try:
                r = c.get(url_of(n))
            except Exception as e:
                bad.append((n, f"请求异常 {e}"))
                continue
            if r.status_code != 200:
                bad.append((n, f"HTTP {r.status_code}"))
                continue
            data = r.content
            if not data.startswith(b"OggS"):
                bad.append((n, "不是 ogg（前 4 字节不对）"))
                continue
            digest = hashlib.md5(data).hexdigest()
            if digest in seen:
                print(f"  [{i}/{len(ns)}] {n} 与 {seen[digest]} 内容相同，跳过")
                continue
            seen[digest] = n
            open(p, "wb").write(data)
            ok += 1
            print(f"  [{i}/{len(ns)}] {n} {len(data)/1024:.0f}KB")
            time.sleep(0.05)                    # 别给 CDN 压力

    total = sum(os.path.getsize(os.path.join(DST, f)) for f in os.listdir(DST))
    print(f"\n新下载 {ok}，已存在 {skip}，失败 {len(bad)}")
    print(f"目录共 {len(os.listdir(DST))} 个文件，{total/1024/1024:.1f} MB -> {DST}")
    for n, why in bad:
        print(f"  失败 {n}: {why}")


if __name__ == "__main__":
    main()
