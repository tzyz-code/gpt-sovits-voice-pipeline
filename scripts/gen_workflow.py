# -*- coding: utf-8 -*-
"""生成 ComfyUI 工作流「Kei念台词」，并对着运行中的 ComfyUI 校验一遍。

**这里有个必踩的坑：`seed` 组件在前端会多占一格。**
ComfyUI 前端源码里对 `seed` / `noise_seed` 有特判，会自动追加一个
`control_after_generate` 下拉（fixed / increment / decrement / randomize），
于是 `widgets_values` 里 seed 后面必须多写一个值。少写这一格，后面**所有**组件
都往前串一位——表现出来就是 speed 收到 "Japanese"、pause_seconds 收到 15、
batch_size 收到 "adjust" 这类莫名其妙的报错。

校验方式：把工作流里的 widgets_values 按前端同样的规则映射成 API 格式的 prompt，
POST 给 /prompt，看 node_errors 是否为空。为空才敢交付。

    python gen_workflow.py
"""
import json
import os
import urllib.request

OUT = os.path.join(ROOT, "Kei念台词.json")
WF_DIR = os.path.join(COMFY, "user", "default", "workflows")
API = "http://127.0.0.1:8188"

SCRIPT = """{main}先生、おはようございます。今日の予定をまとめました。
{慌张}えっ、もうそんな時間ですか！？ちょっと待ってください！
{温柔}……無理はしないでくださいね。[pause:0.3]
{害羞}べ、別に心配しているわけではありませんから。"""

# 组件顺序 = INPUT_TYPES 声明顺序，但 seed 后面要插一格 control_after_generate。
# 用 (key, value) 形式写，key 为 None 表示这一格不是 API 参数（纯 UI 用的控制位）。
VALUES = [
    ("character", "Kei"),
    ("text", SCRIPT),
    ("seed", 1024),
    (None, "randomize"),          # ← control_after_generate，少写这格就全串位
    ("speed", 1.2),
    ("language", "Japanese"),
    ("reference_audio", ""),
    ("reference_text", ""),
    ("gpt_weights", "auto"),
    ("sovits_weights", "auto"),
    ("pause_seconds", 0.3),
    ("top_k", 15),
    ("top_p", 1.0),
    ("temperature", 0.8),
    ("repetition_penalty", 1.35),
    ("batch_size", 8),
    ("cross_lingual", "adjust"),
    ("volume", "normalize"),
    ("preset", ""),
]

workflow = {
    "id": "kei-read-lines",
    "revision": 0,
    "last_node_id": 3,
    "last_link_id": 2,
    "nodes": [
        {"id": 1, "type": "AnomalousTTS_CharacterSpeech",
         "pos": [60, 120], "size": [470, 600], "flags": {}, "order": 0, "mode": 0,
         "inputs": [],
         "outputs": [{"name": "audio", "type": "AUDIO", "links": [1, 2],
                      "slot_index": 0},
                     {"name": "info", "type": "STRING", "links": None}],
         "title": "凯伊念台词",
         "properties": {"Node name for S&R": "AnomalousTTS_CharacterSpeech"},
         "widgets_values": [v for _k, v in VALUES]},
        {"id": 2, "type": "PreviewAudio",
         "pos": [580, 120], "size": [320, 140], "flags": {}, "order": 1, "mode": 0,
         "inputs": [{"name": "audio", "type": "AUDIO", "link": 1}],
         "outputs": [],
         "properties": {"Node name for S&R": "PreviewAudio"},
         "widgets_values": []},
        {"id": 3, "type": "SaveAudio",
         "pos": [580, 320], "size": [320, 150], "flags": {}, "order": 2, "mode": 0,
         "inputs": [{"name": "audio", "type": "AUDIO", "link": 2}],
         "outputs": [],
         "properties": {"Node name for S&R": "SaveAudio"},
         "widgets_values": ["Kei台词"]},
    ],
    "links": [[1, 1, 0, 2, 0, "AUDIO"], [2, 1, 0, 3, 0, "AUDIO"]],
    "groups": [], "config": {}, "extra": {"ds": {"scale": 1, "offset": [0, 0]}},
    "version": 0.4,
}


def verify():
    """按前端规则把工作流还原成 prompt，交给 /prompt 校验。"""
    api_inputs = {k: v for k, v in VALUES if k}
    g = {"prompt": {
        "1": {"class_type": "AnomalousTTS_CharacterSpeech", "inputs": api_inputs},
        "2": {"class_type": "SaveAudio",
              "inputs": {"audio": ["1", 0], "filename_prefix": "wf_verify"}},
    }, "client_id": "verify"}
    req = urllib.request.Request(API + "/prompt", data=json.dumps(g).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print("❌ 校验失败：", body[:600])
        return False
    errs = r.get("node_errors") or {}
    if errs:
        print("❌ 节点报错：", json.dumps(errs, ensure_ascii=False)[:600])
        return False
    print("✅ 服务端校验通过，prompt_id =", r.get("prompt_id"))
    return True


def main():
    os.makedirs(WF_DIR, exist_ok=True)
    for p in (OUT, os.path.join(WF_DIR, "Kei念台词.json")):
        json.dump(workflow, open(p, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("写入", p)
    print(f"组件值 {len(VALUES)} 格（其中 1 格是 seed 的控制位）")
    verify()


if __name__ == "__main__":
    main()
