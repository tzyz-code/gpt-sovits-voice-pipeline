# gpt-sovits-voice-pipeline

一套把「某个角色只有零散的语音片段」变成「能念任意台词、可切换情绪、能接进 ComfyUI 出片」的完整流程。
围绕 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 做数据准备、微调、客观评测和部署，
重点是**每个环节都有可复现的量化判据**，而不是靠耳朵拍脑袋。

以游戏角色 Kei（ブルーアーカイブ，CV 田中美海）为实例跑通全程，所以文档里会大量出现她的数据——
流程本身与角色无关，把你的素材换成任何语言、任何角色都一样跑。

> **这个仓库只包含代码和文档，不包含任何语音模型、音频或数据集。**

---

## ⚠️ 先读这一段：能开源什么，不能开源什么

| | 能不能进这个仓库 | 原因 |
|---|---|---|
| 脚本、文档、工作流模板 | ✅ 可以 | 本仓库的内容 |
| 训练好的角色权重（`.pth` / `.ckpt`） | ❌ **不要** | 由第三方版权语音训练而来，属于衍生作品 |
| 从游戏 / 动画里提取的原始音频 | ❌ **不要** | 版权归原权利人，且通常违反用户协议 |
| GPT-SoVITS 底模 | ❌ 不要 | 体积大，首次使用会自动下载 |

**用自己的声音**、或有明确授权的声音训练模型，公开分享没有法律问题。
用商业作品里的角色声音训练再公开分发权重，会构成侵权——GitHub 收到 DMCA 会直接下架仓库，
严重时影响账号。

所以本仓库的做法是：**流程开源，素材自备**。你按 `docs/02-dataset.md` 准备自己的音频，
跑完训练，模型留在本地。

---

## 它解决什么问题

零样本克隆（给一段参考音频就能说话）已经很好用了，但它有三个绕不过去的限制：

1. **音色学不进去**。参考音频只有几秒，模型只记得「大概这个音高」，学不到细节。
2. **口音和韵律跟着参考走**。参考是问句，之后每句结尾都上翘。
3. **没有稳定复现**。换个种子就换个人，没法当「这个角色的声音」长期使用。

微调解决的是这三条。本仓库记录的是：**怎么用最少的素材（10 分钟以内）把微调做到可用，
以及怎么用客观指标判断它到底好没好**。

实测结果（123 条 / 9.8 分钟日语素材）：

| 指标 | 零样本 | 微调后 | 真角色原声（基准） |
|---|---|---|---|
| 声纹余弦（越高越像） | +0.092 | **+0.121 ~ +0.178** | +0.159 |
| 反向转写准确率 | 常漏字 | 11/11 全对 | — |
| 音文配对错误 | — | 0 / 123 | — |

中文微调后的声纹分（+0.162~0.178）甚至**超过了真角色原声的基准**，但韵律带日语腔——
这是数据决定的，不是参数能调出来的，详见 `docs/04-evaluation.md`。

---

## 快速开始

```bash
git clone https://github.com/<you>/gpt-sovits-voice-pipeline
cd gpt-sovits-voice-pipeline
pip install -r requirements.txt          # 装在 GPT-SoVITS 的环境里
cp config.example.yaml config.yaml       # 填 gpt_sovits_root
python scripts/paths.py                  # 检查路径是否都对了
```

之后按顺序：

```bash
# 1. 抓音视频 + 台词，转成 32k 单声道 wav 并生成训练清单
python scripts/make_list.py

# 2. 抽查音文配对（别跳过，错配会直接教坏模型）
python scripts/check_pairs.py

# 3. 训练：SoVITS 管音色，GPT 管语流
python scripts/train.py all

# 4. 挑参考音频并试听
python scripts/build_refs.py
python scripts/say.py "何をしているんですか" ja

# 5. 客观评测（声纹 + 反向转写）
python scripts/build_centroid.py
python scripts/score_ft.py
```

接进 ComfyUI：

```bash
python scripts/setup_character.py   # 建角色文件夹、写设置、挂到 ComfyUI
python scripts/gen_workflow.py      # 生成工作流 JSON
```

---

## 流程全貌

```
                    ┌─ 素材 ────────────────────────────────────┐
  游戏/动画/授权音频 ─┤  ogg/mp3 → 32k 单声道 wav（裁静音、归一化） │
  台词文本 ──────────┘  → 清单：路径|说话人|语言|台词             │
                    └──────────────┬────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ 配对抽查（语速离群 + 转写）   │  ← 错配在这里拦下来
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
     ┌─────────────────┐                       ┌─────────────────┐
     │ SoVITS 微调      │                       │ GPT 微调         │
     │ 管「谁在说」      │                       │ 管「怎么说」      │
     └────────┬────────┘                       └────────┬────────┘
              └────────────────────┬────────────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │ 评测：声纹余弦 + 反向转写      │
                    │ 定档：哪个 epoch 最好         │
                    └──────────────┬───────────────┘
                                   ▼
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
     ┌─────────────────┐                       ┌─────────────────┐
     │ 命令行 say.py    │                       │ ComfyUI 角色节点 │
     │ 单句、批量        │                       │ 多角色、情绪、出片│
     └─────────────────┘                       └─────────────────┘
```

关键设计取舍：**情绪不靠多训模型，靠换参考音频**。
实测换参考音频让基频 +119 Hz，而单独训一个情绪模型只让基频 +14 Hz（多出来的是起伏幅度 ±59→±85）。
所以最终方案是「一个通用模型 + 一组情绪参考音频」，在剧本里写 `{慌张}` 就切语气。
推导过程见 `docs/05-emotion.md`。

---

## 目录结构

```
scripts/
  paths.py              所有路径的唯一来源（读 config.yaml）
  voiceprint.py         声纹特征的唯一实现（别在别处另写一份）

  01 数据
  fetch_audio.py        按 MediaWiki md5 分桶规则直连 CDN 下载
  parse_wiki.py         wikitext → 台词表
  make_list.py          ogg → 32k wav，生成训练清单和人工校对页
  check_pairs.py        音文配对抽查（语速离群 + 反向转写）
  build_refs.py         从数据集挑参考音频

  02 训练
  train.py              预处理 + SoVITS + GPT 一条命令，支持多实验
  s2_single.py          单进程包装，绕开 Windows 上 DDP 的崩溃

  03 推理
  say.py                念任意台词
  test_finetuned.py     多档位批量推理对比
  zero_shot.py          零样本基线

  04 评测
  build_centroid.py     算声纹质心
  score_ft.py           声纹打分
  asr_check_ft.py       反向转写（微调结果）
  asr_check_demo.py     反向转写（训练集外新句子）
  asr_check_emotion.py  反向转写（情绪对比）
  measure_emotion.py    情绪实验的音高/语速测量

  05 情绪
  tag_emotion.py        用 ASR 的情绪识别打标签（实测在这类素材上不可用）
  group_emotion.py      按音频槽位分情绪桶
  test_emotion.py       四组对照实验（模型 × 参考音频）

  06 ComfyUI
  setup_character.py    建角色文件夹、写角色设置、复用已有底模
  gen_workflow.py       生成工作流 JSON 并做服务端校验

  99 实验/工具
  sweep_ja.py  test_speed.py  test_zh_filler.py  verify_asr.py
  make_*.py    生成试听页

docs/
  00-concepts.md    原理、零样本 vs 微调、时间预期、版权
  01-setup.md       环境准备、参考配置
  02-dataset.md     数据集怎么来
  03-training.md    版本选型、训练步骤
  04-evaluation.md  评测方法、定档结论、泛化验证
  05-emotion.md     情绪实验：为什么一个情绪一个模型不成立
  06-comfyui.md     接入 Anomalous_TTS
  07-pitfalls.md    坑清单
```

---

## 文档

| 文档 | 内容 |
|---|---|
| [原理与选型](docs/00-concepts.md) | 两段式架构、三条硬规则、零样本 vs 微调、时间预期、版权 |
| [环境准备](docs/01-setup.md) | GPT-SoVITS、ffmpeg、`MODELSCOPE_CACHE` 必须是 ASCII、参考配置 |
| [数据集](docs/02-dataset.md) | 抓音频、抓台词、生成清单、配对抽查 |
| [训练](docs/03-training.md) | 版本怎么选（选错白干）、参数、小数据集怎么定轮数 |
| [评测与定档](docs/04-evaluation.md) | 声纹打分、反向转写、种子效应、训练集外验证、中文天花板 |
| [情绪实验](docs/05-emotion.md) | 四组对照、四个结论、为什么放弃多模型方案 |
| [接入 ComfyUI](docs/06-comfyui.md) | Anomalous_TTS 安装、角色文件夹、剧本语法、工作流 JSON 的坑 |
| [坑清单](docs/07-pitfalls.md) | 踩过的坑与成因 |

---

## 环境

本仓库的实测环境（只作参考，不是要求）：

- Windows 11 + RTX 4060 Laptop 8 GB，16 GB 内存
- GPT-SoVITS v2 整合包
- 训练规模：123 条 / 9.8 分钟素材，SoVITS 640 步 + GPT 1380 步，约 25 分钟

8 GB 显存够用，但要开 `expandable_segments` 并把 batch 降到 2。
显存更小的话见 `docs/03-training.md` 的降级方案。

---

## 依赖说明

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — 训练与推理引擎（MIT）
- [Anomalous_TTS](https://github.com/DemonGatanjieu/Anomalous_TTS) — ComfyUI 里的角色语音节点
- [FunASR / SenseVoice](https://github.com/modelscope/FunASR) — 反向转写校验（可选）
- [librosa](https://librosa.org/) — 音高、MFCC、静音切分

## License

代码与文档 MIT，见 [LICENSE](LICENSE)。许可证**不覆盖**你用本流程训练出的模型，
也不覆盖任何第三方语音素材。
