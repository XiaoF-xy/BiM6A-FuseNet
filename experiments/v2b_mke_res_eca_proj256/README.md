# v2b_mke_res_eca_proj256

手工特征编码器与 v2a 完全相同。BiRNA 特征和四路 ResNet-ECA 的 64 维输出分别通过独立投影映射到 256 维，拼接为 512 维后分类。

本版本不包含融合后的完整 MKE-ECA，用于判断加强手工分支后统一维度是否优于非对称融合。

```bash
python train.py --version v2b_mke_res_eca_proj256 --dataset H_b --seed 42 --dry_run
python train.py --version v2b_mke_res_eca_proj256 --dataset H_b --seed 42
```
