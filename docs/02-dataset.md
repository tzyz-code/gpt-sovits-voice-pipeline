# 数据集

## 五、数据集怎么来（本项目最有价值的地方）

RVC 只需要音频，GPT-SoVITS 需要「音频 + 台词」。而 wiki 的 `Kei/audio` 页**天生就是音文配对表**：

- 150 个 `.ogg`（去重 148），共约 3.8MB、约 11 分钟
- 每一条旁边都有 **日文原文** + 英文翻译

所以这次那批素材**不是"没得补"，而是刚到好**。

### 5.1 抓音频

页面 HTML 被 Cloudflare 挡（curl 403），但两条路都通：

- 清单：`https://bluearchive.wiki/w/api.php?action=parse&page=Kei%2Faudio&prop=images&format=json`
- 直链：`https://static.wikitide.net/bluearchivewiki/<md5分桶>/<文件名>`，**无拦截，实测 200**

md5 分桶规则（已验证三个都对）：

```python
h = hashlib.md5(filename.encode()).hexdigest()
url = f"https://static.wikitide.net/bluearchivewiki/{h[0]}/{h[:2]}/{filename}"
```

### 5.2 抓台词

同一个 API 换 `prop=wikitext`（或 `action=parse&prop=text`）能拿到表格，`Name` / `Transcription` / `Japanese` 三列一一对应。
**要取的是「Japanese」列，不是 English 列。** 有些行只有英译，那种要么跳过，要么自己回译。

### 5.3 生成训练清单

格式（官方规定）：

```
音频路径|说话人|语言|台词
```

例：

```
wav/Kei_Gachaget.wav|kei|ja|先生、お会いできて光栄です。
```

要点：

- 语言码用 `ja`（不是 `japanese`）
- **文本必须和音频严格对应**。多一句少一句都会让 GPT 学歪，这是微调翻车第一大原因
- 用 wiki 的现成台词，**跳过 ASR 自动转写**——比机器转写准，且省 20 分钟
- `Extra event lines` 那 17 条是人工转录的，可能有错，建议先剔除
- 采样率不用自己转，WebUI 的切片步骤会统一

### 5.4 切片

WebUI 第 0 步「切片」：`min_length=4000, max_length=10000, min_interval=300`。
目标是把每条切成 3–10 秒、首尾干净。切完要**试听几条**，切坏的（吞字、留半句）删掉——脏数据比少数据危害大。

---
