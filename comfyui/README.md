# ComfyUI 相关文件

`setup_character.py` 和 `gen_workflow.py` 会自动生成这些内容，这里只放参考版本。

- **`workflow_template.json`** — 「角色语音 → 预览音频 + 保存音频」三节点工作流。
  拖进 ComfyUI 画布即可。里面的组件值顺序是踩过坑的：`seed` 后面多占一格
  （`control_after_generate`），少写会让后面所有参数串位。详见
  [../docs/06-comfyui.md](../docs/06-comfyui.md)。

- **`character.example.json`** — 角色设置文件 `anomalous_tts.json` 的格式示例。
  放在 `models/gpt_sovits/<角色名>/anomalous_tts.json`，节点读它来决定：
  用哪个权重、主参考音频是哪条、`{情绪}` 标签对应哪条参考、默认采样参数。

  几个容易错的点：
  - 权重**默认取轮数最大的**。如果最佳档不是最后一轮（很常见），必须显式写 `sovits` / `gpt`。
  - 参考音频的台词必须写进 `text`。留空时节点会去翻同名 `.txt` 或标注文件，找不到就直接
    退化成无参考文本模式，效果差很多。
  - `defaults` 里要抄上你验证过的采样参数（语速、温度、top_k、重复惩罚），
    否则节点用默认值，出来的语气和你试听时听到的不是一回事。
