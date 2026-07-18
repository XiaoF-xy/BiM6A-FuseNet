# v3a_full_mke_eca_native

继承 v2a 的四路 ResNet-ECA 和非对称维度融合，在四路特征合并后的 `32×20` 特征图上增加完整 MKE-ECA：SE 通道注意力，以及 kernel 3/5/7 的多尺度位置注意力。本版本不增加外部残差 `+H`。

```bash
python train.py --version v3a_full_mke_eca_native --dataset H_b --seed 42 --dry_run
python train.py --version v3a_full_mke_eca_native --dataset H_b --seed 42
```
