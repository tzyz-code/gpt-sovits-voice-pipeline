# 接入 ComfyUI

## 七、接入 Anomalous_TTS（用起来）

训练只在整合包里做。日常生成可以用 ComfyUI 节点，好处是支持情绪标签、多人对话、停顿。

### 安装

```bash
cd <comfyui_root>\custom_nodes
git clone https://github.com/DemonGatanjieu/Anomalous_TTS.git
```

```bash
cd F:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\Anomalous_TTS\requirements.txt
```

已核实：日语依赖 `pyopenjtalk-plus` 有 `cp313-win_amd64` 轮子，实测下载 25.6MB 成功，**不用编译**。

### 配置复用已有底模

`<comfyui_root>\user\anomalous_tts.json`：

```json
{
  "format": 1,
  "storage": "voices",
  "pretrained": ["<gpt_sovits_root>/GPT_SoVITS"]
}
```

### 角色文件夹

```
voices/Kei/
  GPT_weights_v2/kei-e15.ckpt
  SoVITS_weights_v2/kei_e8_s480.pth
  参考音频/平静.wav  （+ 同名的 平静.txt 写台词，或写进 all.txt）
  all.txt
```

情绪靠文件名：`平静.开心.wav` → 情绪名 `开心`。台词里写 `{开心}` 就切换。

---

---

# 附：部署细节

### 15.6 Anomalous_TTS 安装与踩坑

节点仓库是 `DemonGatanjieu/Anomalous_TTS`（不是 `Kitsune-77`）。它自带精简推理代码、
跑在 ComfyUI 自己的 PyTorch 上，**不需要第二个 GPT-SoVITS 环境**。

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DemonGatanjieu/Anomalous_TTS.git
cd ../..
python_embeded\python.exe -m pip install -r ComfyUI/custom_nodes/Anomalous_TTS/requirements.txt
```

- **省掉 1.6GB 底模下载**：在 `ComfyUI/user/anomalous_tts.json` 里写上本机整合包路径，
  它会自己去 `pretrained_models` 下找：
  ```json
  {"format": 1, "pretrained": ["<gpt_sovits_root>/GPT_SoVITS"]}
  ```
- **角色文件夹**放在 `ComfyUI/models/gpt_sovits/<角色名>/`，含 `GPT_weights_v2/`、
  `SoVITS_weights_v2/`、`参考音频/` 和一个 `anomalous_tts.json`。
- **权重默认取轮数最大的**，而我们的最佳档是 e8（不是 e10），所以必须在设置文件里
  显式写 `sovits`，否则会被自动选成 e10。
- **参考音频一定带台词**，写进设置文件的 `reference.text` / `emotions[].text`，
  比靠同名 `.txt` 或标注文件去找可靠得多。
- **采样参数要抄进 `defaults`**（speed 1.2 / temperature 0.8 / top_k 15 /
  repetition_penalty 1.35）。不写就是节点默认值，出来的语气和试听页验证过的不是一回事。
- 装完实测：节点 0.4 秒加载完，4 句带标签的台词 15.5 秒生成，模型缓存后每句约 1 秒。
  日志里能看到 `Kei{慌张}：生成 1 句`，说明标签确实被解析成了不同的参考。



### 15.8 ComfyUI 流程 vs 脚本调用：音质相同，流程不同

**音质不会更好，一分不差。** Anomalous_TTS 内部跑的就是 GPT-SoVITS 的精简推理代码，
和我 `say.py` 调的是同一套权重、同一套采样参数。同样的种子出来是同一段波形。
「换到 ComfyUI 音质就上去了」是错觉。

真正的差别在流程上：

| | 脚本 `say.py` | ComfyUI 节点 |
|---|---|---|
| 单句试听 | 一条命令，不用开服务 | 要开 ComfyUI、占显存 |
| 多角色对戏 | 自己写循环拼 | 剧本里 `[Kei]` `[Kei慌张]` 直接切 |
| 改一句重跑 | 全量重跑 | **句子级缓存**，只重算改过的那句 |
| 单句重 roll | 换种子重跑整段 | `{开心}[take:2]` 只换这一版 |
| 接后续链路 | 出 wav 就断了 | 音频直接进视频/对口型/字幕节点 |
| 批量 100 条 | 加载一次模型跑到底，最快 | 有 HTTP 往返，慢一些 |

实测：工作流里 6 句台词，改了 2 句重跑，日志是「新生成 2 句（2 批），缓存 4 句」，
1.66 秒完成。脚本是全量重跑。

**节点自带 `volume: normalize`**，每句自动拉到约 −20 dBFS，所以平时不需要再接
「音频调整音量」。真要用到的音频节点只有三个：

- **音频拼接**：不想用 `[pause]` 而想分段单独生成再拼起来时用；
- **修剪音频时长**：掐掉尾部多余静音；
- **音频均衡器（三段）**：给声音补一点高频会更贴耳，但这属于后期，不是合成。

⭐ **找合成节点别搜「音频」**——它叫「角色语音 (GPT-SoVITS)」，分类在 `Anomalous/TTS`，
搜「角色」「语音」「Kei」都能出来。搜「音频」出来的一整列（加载/预览/保存/拼接/修剪/
分声道/EQ）全是配音**之后**的节点，没有一个是合成器。

现成工作流：`Kei念台词.json`（角色语音 → 预览音频 + 保存音频，已放进
`ComfyUI/user/default/workflows/`，在 ComfyUI 里「工作流 → 打开」或直接拖进画布）。

#### 手写工作流 JSON 的必踩坑：`seed` 多占一格

如果自己拼 `Kei念台词.json` 这类工作流文件，`widgets_values` 数组里
**`seed` 后面必须多写一个值**（`randomize` / `fixed` / `increment` / `decrement`）。
ComfyUI 前端对 `seed` 和 `noise_seed` 有特判，会自动追加一个 `control_after_generate`
下拉，于是数组长度要比「API 参数个数」多 1。

少写这一格的后果是后面**所有**组件统一往前串一位，报错看起来完全不讲道理：

```
Failed to convert an input value to a FLOAT value: speed, Japanese
Value 15.0 bigger than max of 5.0: pause_seconds
Value 8.0 bigger than max of 2.0: repetition_penalty
Failed to convert an input value to a INT value: batch_size, adjust
Custom validation failed: cross_lingual must be one of adjust / off: 'normalize'
```

`speed` 收到 `"Japanese"`、`pause_seconds` 收到 `15`——这些值原本是隔壁组件的，
串位就会这样。**看到「某个组件收到了另一个组件的值」，先怀疑数组长度错位**，
而不是去查那个组件的取值范围。

生成脚本：`gen_workflow.py`。它会按前端规则把 `widgets_values` 映射成 API 格式
POST 给 `/prompt` 做服务端校验，`node_errors` 为空才写出文件。
这个校验能挡住「值错位」这一类问题，但挡不住纯 UI 层面的问题。
