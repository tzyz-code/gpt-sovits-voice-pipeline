# 环境准备

## 四、本机现状（已逐项核实，2026-10-26）

| 组件 | 状态 |
|---|---|
| 整合包代码 | ✅ `<gpt_sovits_root>`（含 `export_torch_script_v3v4.py`，支持 v3/v4） |
| Python 环境 | ✅ `python`（**Python 3.10.11**，不是 3.13，没问题） |
| v2 底模 | ✅ `pretrained_models\gsv-v2final-pretrained\`（s1bert / s2D2333k / s2G2333k） |
| 中文支持 | ✅ `GPT_SoVITS\text\G2PWModel` 已就位（只做日语其实用不上） |
| API 链路 | ✅ **8 月 22 日跑通过**：`POST /tts 200 OK`，`refs\pct02_tts.wav` 等产物还在 |
| 启动脚本 | `F:\GPT-SoVITS\start_api.bat.bat`（api_v2，端口 9880） |
| ASR 旁路 | `F:\GPT-SoVITS\asr_sidecar.py`（SenseVoice，端口 9881）**本项目可跳过** |
| **已训角色模型** | ❌ **一个都没有**，`SoVITS_weights` / `GPT_weights` 全空 |
| ComfyUI | ✅ `F:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable`（Py3.13, torch 2.13+cu130） |
| Anomalous_TTS | ❌ 未装（`custom_nodes` 是空的） |

**结论：管道通了九成，唯一缺的是"Kei 这个角色"本身。**

> 注：`go-webui.bat` 里写的是 `runtime\python.exe`，但本机没有 `runtime` 目录，
> 真实环境在 `F:\GPT-SoVITS\env`。所以那个 bat **不能直接双击**。手动启动：
> ```
> cd /d <gpt_sovits_root>
> ..\env\Scripts\python.exe webui.py zh_CN
> ```

---
