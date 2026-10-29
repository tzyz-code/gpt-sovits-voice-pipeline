# -*- coding: utf-8 -*-
"""把 bluearchive.wiki 的 Kei/audio 页面 wikitext 解析成「ogg 文件名 -> 日文台词」。

用法：
    python parse_wiki.py

读入 dataset/wiki_s12.json（以及可选的 wiki_s3.json / wiki_s5.json），
输出 dataset/lines.json 与 dataset/list.txt（GPT-SoVITS 训练清单）。
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
from paths import DATASET as DS


def _walk_wikitext(obj, out):
    """递归找出所有 wikitext 字符串（文件可能被包成 {"s3": {"parse": ...}}）。"""
    if isinstance(obj, dict):
        if "wikitext" in obj and isinstance(obj["wikitext"], dict) and "*" in obj["wikitext"]:
            out.append(obj["wikitext"]["*"])
            return
        for v in obj.values():
            _walk_wikitext(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_wikitext(v, out)


def load_wikitext(fname):
    """返回文件里所有 wikitext 片段（s12 里有两段：Normal lines / Memorial lobby）。"""
    p = os.path.join(DS, fname)
    if not os.path.isfile(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    out = []
    _walk_wikitext(d, out)
    return out or None


def strip_tags(s):
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&")
    s = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", s)  # 内部链接
    s = re.sub(r"'{2,}", "", s)                            # 斜体/粗体
    s = re.sub(r"[ \t]+", " ", s)
    return "\n".join(x.strip() for x in s.split("\n") if x.strip())


def split_rows(wikitext):
    """按 '|-' 切表格行，每行返回原始文本（含换行）。"""
    body = wikitext
    # 去掉表头（从 {| 到第一个 |- 之间的 ! 行）
    rows = re.split(r"^\|\-\s*$", body, flags=re.M)
    return rows[1:] if len(rows) > 1 else []


def parse_row(row):
    """返回 (name, [ogg...], [ja_sentence...])。"""
    cells = []
    for line in row.split("\n"):
        if line.startswith("|") and not line.startswith("|-") and not line.startswith("|}"):
            cells.append(line[1:].strip())
    if len(cells) < 3:
        return None
    name, files_cell, ja_cell = cells[0], cells[1], cells[2]

    # 日文列可能形如: lang="ja" | <p>..</p><p>..</p>
    m = re.match(r'lang\s*=\s*"ja"\s*\|(.*)$', ja_cell, flags=re.S)
    if not m:
        return None
    ja_raw = m.group(1)

    oggs = re.findall(r"\[\[File:([^\]]+?\.ogg)\]\]", files_cell, flags=re.I)
    oggs = [o.strip().replace(" ", "_") for o in oggs]

    # 按 <p> 块拆句，句内 <br> 视为同一句的换行
    blocks = re.findall(r"<p>(.*?)</p>", ja_raw, flags=re.S)
    if not blocks:
        blocks = [ja_raw]
    sents = [strip_tags(b).replace("\n", "") for b in blocks]
    sents = [s for s in sents if s]

    return name.strip(), oggs, sents


def main():
    mapping = {}       # ogg -> 台词
    detail = []        # 审计用
    for fname in ("wiki_s12.json", "wiki_s3.json", "wiki_s5.json"):
        blocks = load_wikitext(fname)
        if not blocks:
            print(f"[跳过] {fname} 不存在或为空")
            continue
        n_rows = n_pairs = 0
        for wt in blocks:
            for row in split_rows(wt):
                r = parse_row(row)
                if not r:
                    continue
                name, oggs, sents = r
                if not oggs:
                    continue
                n_rows += 1
                if len(oggs) == len(sents):
                    for o, s in zip(oggs, sents):
                        mapping[o] = s
                        n_pairs += 1
                elif len(oggs) == 1 and sents:
                    # 一个音频对应多行字幕（同一段话分两行显示）-> 拼成一句
                    mapping[oggs[0]] = "".join(sents)
                    n_pairs += 1
                else:
                    detail.append({"src": fname, "name": name, "oggs": oggs,
                                   "sents": sents, "note": f"文件{len(oggs)}句{len(sents)}不匹配"})
        print(f"[{fname}] 行 {n_rows}，成功配对 {n_pairs}")

    # 只保留磁盘上真实存在的 ogg
    ogg_dir = os.path.join(DS, "ogg")
    have = set(os.listdir(ogg_dir)) if os.path.isdir(ogg_dir) else set()
    missing_file = sorted(k for k in mapping if k not in have)
    unused_audio = sorted(have - set(mapping))

    json.dump(mapping, open(os.path.join(DS, "lines.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"\n配对总数 {len(mapping)}；磁盘 ogg {len(have)} 个")
    print(f"有台词但缺音频 {len(missing_file)} 个：{missing_file[:10]}")
    print(f"有音频但没台词 {len(unused_audio)} 个：{unused_audio[:18]}")
    if detail:
        print(f"\n未配对明细 {len(detail)} 条：")
        for d in detail[:20]:
            print(" ", d["name"], d["note"], "|", d["oggs"])

    return mapping


if __name__ == "__main__":
    main()
