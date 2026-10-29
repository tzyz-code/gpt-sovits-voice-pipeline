# 环境准备

## 1. 需要什么

| 组件 | 要求 | 说明 |
|---|---|---|
| GPT-SoVITS | v2 整合包或源码 | 训练和推理引擎，本仓库不改它 |
| Python | 3.9 ~ 3.11 | 用 GPT-SoVITS 自带的环境，不用另建 |
| 显卡 | 6 GB 显存起 | 8 GB 可以舒展着跑，见下面的降级方案 |
| ffmpeg | 能在 PATH 里找到 | `load_audio` 依赖它 |
| 磁盘 | 约 5 GB | 底模 1.3 GB + 数据集 + 训练中间产物 |

CPU 也能训，但 123 条素材大概要跑几个小时，不推荐。

## 2. 装 GPT-SoVITS

去 [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 按官方说明装。
Windows 用户下整合包最省事——它把 Python 环境和依赖都备好了。

装完把根目录填进 `config.yaml`：

```yaml
gpt_sovits_root: D:/GPT-SoVITS/GPT-SoVITS
```

验证：

```bash
python scripts/paths.py
```

它会打印当前生效的全部路径。**排查"为什么找不到文件"先看这个输出。**

## 3. 下载底模

底模约 1.3 GB，**不在仓库里**。原因很实在：

- 5 个文件超过 GitHub 单文件 100 MB 硬限制（最大的 `pytorch_model.bin` 621 MB），
  commit 会被 git 直接拒绝；
- 就算改用 LFS，免费额度 1 GB 也装不下，而每个 clone 的人都要再消耗一遍带宽；
- 底模有独立发布渠道，由官方仓库分发更合适。

一条命令搞定，国内优先走 ModelScope：

```bash
python scripts/fetch_pretrained.py --check    # 先看缺什么
python scripts/fetch_pretrained.py            # 自动挑源下载
python scripts/fetch_pretrained.py --source hf-mirror   # 指定源
```

三个源：`ms`（ModelScope，国内最快）、`hf-mirror`（HF 镜像）、`hf`（HuggingFace 原站）。
失败会自动换下一个。下完自动校验文件大小，缺了会报出来。

需要的是这 6 个文件：

```
GPT_SoVITS/pretrained_models/
├── gsv-v2final-pretrained/
│   ├── s2G2333k.pth                                  101 MB  声学模型生成器
│   ├── s2D2333k.pth                                   89 MB  声学模型判别器
│   └── s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt 148 MB  语言模型（GPT）
├── chinese-hubert-base/pytorch_model.bin             180 MB  语义特征提取
├── chinese-roberta-wwm-ext-large/pytorch_model.bin   621 MB  文本特征提取
└── fast_langdetect/lid.176.bin                       125 MB  语种识别（可选）
```

手动下载也可以，官方的压缩包地址：

- ModelScope：`https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/pretrained_models.zip`
- HF 镜像：`https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip`
- HuggingFace：`https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip`

解压到 `GPT_SoVITS/` 下，确保得到 `GPT_SoVITS/pretrained_models/` 这一层。

> `s1bert25hz` 那个文件名里是 `step=369668`（等号），不是 `step-369668`。
> 手抄容易错，错了训练会在加载 GPT 权重时报找不到文件。

## 4. ffmpeg

`tools/my_utils.load_audio` 走 ffmpeg-python，所以 `ffmpeg.exe` 必须在 PATH 里。
没有的话最省事的办法：

```bash
pip install imageio-ffmpeg
python -c "import imageio_ffmpeg, shutil, os; p=imageio_ffmpeg.get_ffmpeg_exe(); \
os.makedirs('D:/gsv_ffmpeg/bin', exist_ok=True); shutil.copy(p, 'D:/gsv_ffmpeg/bin/ffmpeg.exe')"
```

然后把那个 `bin` 目录加进 PATH，或者直接放进 GPT-SoVITS 整合包根目录。

缺 ffmpeg 的症状是 `WinError 2`，报错信息里不会提 ffmpeg，很容易查偏。

## 5. 模型缓存目录必须是纯 ASCII

如果要用反向转写校验（`asr_check_*.py`），必须先设 `MODELSCOPE_CACHE`，
而且**路径不能含中文**：

```yaml
# config.yaml
model_cache: D:/gsv_cache
```

原因：Windows 下中文用户名目录里的模型缓存，sentencepiece 打不开，
报错信息跟编码有关，但完全看不出是路径问题。这个坑很多人踩过。

`fetch_pretrained.py` 和训练都不需要这项；只有 ASR 相关脚本需要。

## 6. 显存不够怎么办

8 GB 显存按仓库默认参数（`batch_size=2`、`grad_ckpt=True`、`lora_rank=32`）够用。
更小的卡按这个顺序降：

1. `train.py` 里把 `S2_BATCH` 调到 1；
2. 打开 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（默认已开）；
3. 关掉浏览器和桌面程序——8 GB 卡上它们能吃掉 1.3 GB。

另外 Windows 上有个坑：**提交内存**（物理内存 + 页面文件）不够会报
`WinError 1455 页面文件太小` 或 `shm.dll 初始化失败`，这两个错误看起来跟显存无关，
实际是内存耗尽。16 GB 机器上把页面文件设到 25 GB 以上比较稳。

## 7. ComfyUI（可选）

只在要做部署时用：

```yaml
comfyui_root: D:/ComfyUI_windows_portable/ComfyUI
```

注意要填到 `ComfyUI/` 这一层（里面有 `models/` 和 `custom_nodes/`），
不是外面的便携版根目录。装节点见 [06-comfyui.md](06-comfyui.md)。
