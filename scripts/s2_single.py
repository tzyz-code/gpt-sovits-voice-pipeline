# -*- coding: utf-8 -*-
"""单进程跑 s2_train.run()，绕开 mp.spawn + 单卡 DDP 在 Windows 上的
"CUDA error: resource already mapped"。

用法（由 train.py 调用）：
    python s2_single.py --config <tmp_s2.json>
"""
import os
import random
import sys

from paths import GSV
sys.path.insert(0, os.path.join(GSV, "GPT_SoVITS"))
sys.path.insert(0, GSV)

# s2_train 在 import 阶段就解析 --config 并读取 hps，所以参数顺序要保留
import s2_train  # noqa: E402

if __name__ == "__main__":
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(random.randint(20000, 55555)))
    print(f"[single] MASTER_PORT={os.environ['MASTER_PORT']} world_size=1 rank=0", flush=True)
    s2_train.run(0, 1, s2_train.hps)
