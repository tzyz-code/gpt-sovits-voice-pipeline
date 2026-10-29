# 评测与定档

## 八、验收标准

训完必须用**训练集里没有的句子**测，否则自己骗自己：

1. 3 句短句、3 句中长句，纯日语，不夹英文
2. 换 3 个不同种子各生成一次，听**稳定性**（同一句会不会时好时坏）
3. 重点听三件事：**口音**（日语发音准不准）、**尾音**（有没有断在半截）、**重复漏字**（GPT 的典型毛病）
4. 和零样本版本 A/B——如果微调后没有明显更好，说明数据集有问题（大概率是台词对不上）

---



## 十二、第零步实测记录（2026-10-26）

目标台词：中文「你这是在干什么？」/ 日文「何をしているんですか？」

### 环境补齐

`api_v2` 报 `No module named 'pyopenjtalk'`——**即使台词是中文，只要 `prompt_lang=ja` 就需要日语前端**。
之前的成功记录只有中文参考，所以这个依赖一直没被触发。

```bash
F:/GPT-SoVITS/env/Scripts/python.exe -m pip install pyopenjtalk-plus
```

装完验证：`g2p('何をしているんですか', kana=True)` → `ナニヲシテイルンデスカ`（正确）。
**装完必须重启 api_v2**，进程内缓存的 import 不会热更新。

### 参考音频怎么选

从 wiki 下了 8 条，按三个条件筛：

| 条件 | 原因 |
|---|---|
| 3–10 秒 | 太短音色向量不稳，太长会带进多余韵律 |
| 陈述句、不带笑声 | 参考的语气会感染全部输出 |
| RMS 较高 | 底噪占比小，音色更干净 |

最终选 `Kei_Cafe_monolog_4`（5.47s，RMS 0.163，陈述句）。
预处理：转 32k 单声道 → 按 RMS 阈值裁掉首尾静音（留 30ms 余量）→ 峰值统一到 0.95。

### 客观校验：反向转写

不用耳朵也能查 TTS 的典型毛病（重复字、漏字、假名读错）：**用 SenseVoice 把合成结果转回文字，和目标台词比对**。

```bash
F:/GPT-SoVITS/env/Scripts/python.exe verify_asr.py
```

本机 `<model_cache>` 已有 SenseVoiceSmall（906MB），不用下载。
注意 `MODELSCOPE_CACHE` 必须指向**纯 ASCII 路径**，否则 Windows 下 sentencepiece 打不开中文用户目录。

结果 **8/8 完全一致**：

| 文件 | 反向转写 | SenseVoice 判定情绪 |
|---|---|---|
| kei_ja_neutral_s1024 | 何をしているんですか | SURPRISED |
| kei_ja_neutral_s20261026 | 何をしているんですか | SURPRISED |
| kei_ja_neutral_s42 | 何をしているんですか | EMO_UNKNOWN |
| kei_ja_smile_s42 | 何をしているんですか | EMO_UNKNOWN |
| kei_zh_neutral_s1024 | 你这是在干什么 | EMO_UNKNOWN |
| kei_zh_neutral_s20261026 | 你这是在干什么 | NEUTRAL |
| kei_zh_neutral_s42 | 你这是在干什么 | NEUTRAL |
| kei_zh_smile_s42 | 你这是在干什么 | EMO_UNKNOWN |

基频落在 **242–288Hz**（Kei 说话中位 271Hz），说明音色没跑偏。

### 两个观察

1. **日语被识别成 SURPRISED**。台词带问号 + 参考音频本身是"略带感慨的陈述"，
   两者叠加导致语气偏惊讶。想要平稳语气，可以：去掉问号、换更平的参考、
   或降 `temperature`。
2. **API 常驻时跑不了 ASR**。`api_v2` 占约 6GB 提交内存，ASR 初始化时
   连 1MB 都分配不出来（`DefaultCPUAllocator: not enough memory`）。
   校验前先停 API，或把 ASR 放 GPU。


---



## 十三、微调结果（2026-10-26 完成）

### 数据集

| 项 | 值 |
|---|---|
| 来源 | bluearchive.wiki 的 `Kei/audio` 页，148 个 ogg 全量下载 |
| 有日文台词的 | **123 条**（`dataset/lines.json`） |
| 总时长 | **589 秒 ≈ 9.8 分钟**，平均 4.79 秒/条 |
| 丢弃 | 25 条没日文（战斗喊声/`Extra event lines` 整块空），1 条纯标点「……。」 |

配对逻辑写在 `parse_wiki.py`：按表格行取「文件名 / `[[File:x.ogg]]` / `lang="ja"` 单元格」，
`<p>` 分句后 1 对 1 配对；`MemorialLobby` 是一组两个音频对两句台词，也能正确拆开。
`make_list.py` 负责 ogg→32k 单声道 wav（裁剪首尾静音、峰值归一到 0.9）并生成训练清单。

**人工抽查入口**：`<gpt_sovits_root>/dataset/配对校对.html`（123 条可逐条试听对照）。

### 训练配置

| 项 | 值 |
|---|---|
| 版本 | **v2**（低码率 48kbps ogg 素材，官方明确 v3/v4 不适合） |
| SoVITS | epoch 10、**batch 2**、`grad_ckpt=True`、640 步 |
| GPT | epoch 15、batch 4、31 步/epoch |
| 参考底模 | `gsv-v2final-pretrained` 的 s2G2333k / s2D2333k / s1bert25hz-5kh |
| 产出 | `SoVITS_weights_v2/kei_e{4,6,8,10}_s{256,384,512,640}.pth`、`GPT_weights_v2/kei-e{3,6,9,12,15}.ckpt` |

启动命令：

```bash
F:/GPT-SoVITS/env/Scripts/python.exe train.py prep   # 文本/SSL/语义 token
F:/GPT-SoVITS/env/Scripts/python.exe train.py s2     # SoVITS
F:/GPT-SoVITS/env/Scripts/python.exe train.py s1     # GPT
```

### 客观评分（声纹质心余弦，真 Kei 原声 = +0.159）

| 组 | 最好成绩 | 说明 |
|---|---|---|
| 微调 · 中文 | **+0.178**（e8+g9） | 全部 6 条都在 +0.162~0.178，**超过真 Kei 原声基准** |
| 微调 · 日语 | **+0.121**（e8+g15） | 零样本最好只有 +0.092 |
| 零样本 | +0.162（中文）/+0.092（日语） | — |

反向转写（SenseVoice）：**12 条日语全部与台词一致**；中文 6 条里 3 条开头多了一个「哼」
（e4/e10 档，e8 没有）。

### 结论

- **推荐组合：SoVITS `kei_e8_s512.pth` + GPT `kei-e15.ckpt`**（日语种子优先 1024，g15 配种子 42 会拖到 4.4 秒）。
- 中文用 e8 档、`speed_factor=1.2`：有声时长 3.2s→2.7s，音色分只掉 0.002。
- **微调后语速整体慢了约 1.3 倍**（中文 3.2s vs 零样本 2.5s）。这是本数据集的特征，
  用 `speed_factor` 或 Anomalous_TTS 的 `[speed:1.2]` 补回来即可，不必重训。
- 种子依然是变量：同一档位换种子，时长能从 2.4s 跳到 4.4s。要固定产出就固定种子。

### 这一轮新踩的坑

| 症状 | 原因 | 解法 |
|---|---|---|
| `WinError 1455 页面文件太小` / `shm.dll 初始化失败` | 提交内存被打满（本机 15G 物理 + 25G 页面文件），DDP 要复制一整个进程 | 关掉残留的 RVC/Applio 进程；单进程训练 |
| `CUDA error: resource already mapped` | 单卡跑官方 `mp.spawn` + DDP + gloo 在 Windows 上的老问题 | 用 `s2_single.py` 直接调 `s2_train.run(0,1,hps)`，不 spawn |
| `CUDA out of memory` 但显示还有 2.5G 空闲 | 8G 显存被桌面程序占 1.3G + 显存碎片 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + batch 2 + `grad_ckpt` |
| `saving ckpt kei_e2: FileNotFoundError` | `SoVITS_weights_v2/` 目录不存在，`savee` 不会自动建 | 训练前先建目录（`train.py` 已内置） |
| 各档权重输出**字节完全相同** | `TTS_Config` 内部是 `configs_.get("custom", configs_["v2"])`，传扁平字典会被静默忽略、一直用底模 | 配置必须包一层 `{"custom": {...}}` |
| `No module named 'text' / 'tools'` | `prepare_datasets` 脚本的 import 依赖仓库根和 `GPT_SoVITS` 同时在 `sys.path` | 设 `PYTHONPATH=仓库根;仓库根\GPT_SoVITS` |
| `load_audio` 抛 `WinError 2` | 本机没有 ffmpeg，`tools/my_utils.load_audio` 走 ffmpeg-python | `pip install imageio-ffmpeg` 后把自带二进制拷到 `F:/GPT-SoVITS/ffmpeg/bin` 并加进 PATH |
| `nvidia-smi: Failed to initialize NVML`，但 CUDA 正常 | NVML 自检失败，不影响 torch | 忽略，用 `torch.cuda.is_available()` 判断即可 |

### 复现脚本清单

| 文件 | 作用 |
|---|---|
| `fetch_audio.py` | 按 md5 分桶从 CDN 批量下 ogg |
| `parse_wiki.py` | wikitext → `dataset/lines.json`（音频名→日文台词） |
| `make_list.py` | ogg → 32k wav + 训练清单 + 校对页 |
| `train.py` | 预处理 + SoVITS + GPT 一条龙 |
| `s2_single.py` | 单进程 SoVITS 训练包装（绕开 DDP 坑） |
| `test_finetuned.py` | 按「SoVITS 档 × GPT 档 × 种子」批量推理 |
| `test_speed.py` | speed_factor 对照 |
| `score_ft.py` | 声纹打分（微调/零样本/原声同表比较） |
| `asr_ft.py` | 反向转写校验 |
| `make_compare_page.py` | 生成 `试听对比.html` |

---



## 十四、定档与泛化验证（用户听过之后的结论）

### 定下的档位

| 语种 | SoVITS | GPT | 依据 |
|---|---|---|---|
| 日语 | `kei_e8_s512.pth` | `kei-e15.ckpt` | 用户实听：e8 + 15 轮最自然 |
| 中文 | `kei_e8_s512.pth` | `kei-e9.ckpt` | 用户实听：e8 + 9 轮最准 |

两种语言最优的 GPT 档位不一样，所以 `say.py` 做成了按语种自动选。

### 另外两个必须记住的参数

1. **种子别用 42**。中文用种子 42 会念错字：`我` 念成 `宝`、`这种` 念成 `这段`，
   句首还会多出一个「哼」，时长也从 1.8s 拖到 2.8s。换 1024 或 7 全干净。
   日语同理，g15 配种子 42 会拖到 4.4 秒。**统一用 1024。**
2. **温度降到 0.8**。1.0 时中文句首容易冒出语气词，0.8 明显干净，音色不受影响。

### 新句子验证（训练集之外）

11 条全部为训练集里没有的句子，参数统一 语速×1.2 / 温度 0.8 / 种子 1024：

| 序号 | 台词 | 有声时长 | 语速 | 反向转写 |
|---|---|---|---|---|
| 日1 | 先生、そんなに急いでどうしたんですか？ | 3.2s | 5.3 字/秒 | 一致 |
| 日2 | 少し休んだほうがいいですよ。 | 3.0s | 4.3 字/秒 | 一致 |
| 日3 | 私は、先生を信じています。 | 3.9s | 2.8 字/秒 | 一致 |
| 日4 | これは……なんというか、不思議な感じですね。 | 6.2s | 2.9 字/秒 | 一致 |
| 中1 | 老师，你今天看起来很累。 | 3.4s | 2.9 字/秒 | 一致 |
| 中2 | 我会一直在这里的。 | 1.8s | 4.4 字/秒 | 一致 |
| 中3 | 这种事情，还是让我来吧。 | 4.3s | 2.3 字/秒 | 一致 |
| 中4 | 你真的打算这么做吗？ | 2.0s | 4.5 字/秒 | 一致 |

**11/11 反向转写与台词完全一致**（同音异写「ほう/方」已归一并注明）。
同一句换三个种子（42/1024/7）转写也都对 —— 说明种子影响的主要是**语速和句首语气**，
不是把字念错。

试听页：`新句子验证.html`。

### 已知的天花板：中文带日语腔

123 条训练样本**全部是日语**，模型从没听过 Kei 说中文。所以：

- 音色没问题（声纹分 0.162~0.178，超过真 Kei 原声基准）——这是 SoVITS 的功劳；
- 但中文的**韵律和句首语气词是从日语习惯迁移过来的**（「哼」就是这么来的），
  语速也整体偏慢。这是数据决定的，不是参数能调没的。

要真正解决只有加中文语料，而 Kei 官方语音只有日语。所以当前的中文效果已经是这套数据的上限，
好在「日语角色说中文带一点腔」本身也不算违和。

---

---

## 附：重建质心时的注意事项

本文上面的分数（真原声 +0.159、零样本 +0.092、微调 +0.121~0.178）是用当时那批
293 条素材算出的质心得到的，属于历史记录。**用自己的数据重建质心后，绝对值会整体平移，
不能和这些数字直接比。**

更要注意的是，重建质心可能连**排序方向**都变。实测用 124 条重建后：

| | 分数 |
|---|---|
| 真·原声 40 条 | −0.002 |
| 合成片段 4 条 | +0.126 |

合成反而更高。原因是质心本质是「标准化后向量的均值」，而合成音频的高频滚降更一致，
反而更靠近这个均值；真原声因为情绪和语速变化大，离散度更高，反而离均值更远。

所以用这个指标的正确姿势是：

1. **先验方向**。拿几条确信是本人的、几条确信不是的音频当标尺，跑一遍确认分数往哪边走。
2. **只做同一质心下的横向排序**，不要跨数据集、跨质心比较具体数值。
3. **永远配反向转写一起看**。声纹分高但念错字，是这套流程里最容易出现的假阳性。

这个指标真正的作用是「筛掉明显跑偏的候选」，不是「证明微调成功」。
判断微调到底成没成，看第八节那套组合判据：声纹分 + 反向转写 + 训练集外新句子 + 人耳盲听。
