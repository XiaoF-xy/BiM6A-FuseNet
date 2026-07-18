# v2a_mke_res_eca_native

四种手工特征分别经过独立的 ResNet-ECA 分支，NCP 使用项目现有的 3 通道编码。融合后得到 64 维手工特征，再适配到 128 维，与原生 BiRNA 特征（当前为 1536 维）拼接。

本版本不包含融合后的完整 MKE-ECA，用于与 `v2b_mke_res_eca_proj256` 比较维度对齐的影响。

```bash
python train.py --version v2a_mke_res_eca_native --dataset H_b --seed 42 --dry_run
python train.py --version v2a_mke_res_eca_native --dataset H_b --seed 42
```
