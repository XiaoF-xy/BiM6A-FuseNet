# v0e BiRNA-BERT NUC LoRA+

本版本继承 v0a 的纯 BiRNA-BERT NUC 单分支、masked mean pooling、Wqkv LoRA 和分类头。唯一训练变量是 AdamW 参数组学习率：

- LoRA A：`5e-5`
- LoRA B：`8e-4`
- 分类头：`1e-4`

LoRA rank 8、alpha 32、dropout 0.05、batch size 32、20 epochs、weight decay 0.01 和验证 ACC 选模型均保持不变。

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v0e_birna_nuc_loraplus --dataset H_b --seed 42
CUDA_VISIBLE_DEVICES=1 python train.py --version v0e_birna_nuc_loraplus --dataset H_k --seed 42
CUDA_VISIBLE_DEVICES=2 python train.py --version v0e_birna_nuc_loraplus --dataset H_l --seed 42
```

先比较 benchmark 五折汇总指标与 v0a；独立测试集结果不用于选择实验版本。
