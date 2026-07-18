# v3b_full_mke_eca_proj256

继承 v2b 的四路 ResNet-ECA 和双分支 256 维投影，在四路特征合并后的 `32×20` 特征图上增加完整 MKE-ECA。本版本不增加外部残差 `+H`，并作为后续逐维门控融合的对齐维度基础。

```bash
python train.py --version v3b_full_mke_eca_proj256 --dataset H_b --seed 42 --dry_run
python train.py --version v3b_full_mke_eca_proj256 --dataset H_b --seed 42
```
