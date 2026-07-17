# v1b_proj256_concat

`v1b_proj256_concat` 是 `v1_baseline` 的第一个单变量改进版本。它保留 v1 的 BiRNA-BERT/FiLM 分支、手工特征 CNN、LoRA 设置和训练参数，只替换最后的双分支融合方式。

## 受控变量

```text
BiRNA 特征（1536 维） → Linear + LayerNorm + GELU + Dropout(0.2) → 256 维
手工特征（128 维） → Linear + LayerNorm + GELU + Dropout(0.2) → 256 维
                                      concat → 512 维
                                      Linear(512,256) + ReLU + Dropout(0.2)
                                      Linear(256,2)
```

BiRNA 特征维度由模型 hidden size 动态计算，1536 是当前 v1 配置的实际值。

## 评估协议

- `benchmark.csv` 上固定 seed 42 的分层五折交叉验证。
- 每折训练 20 轮，按该折验证 ACC 保留最佳 epoch。
- 五个最佳折模型对 `independent_test.csv` 输出正类概率，取平均值做 soft voting，再计算一次 ACC、MCC、AUC、AUPRC 等最终指标。
- 预测、指标和 ROC/PR 坐标验证可回读后，删除临时折权重。

## 运行

```bash
python train.py --version v1b_proj256_concat --dataset H_b --seed 42 --dry_run
python train.py --version v1b_proj256_concat --dataset H_b --seed 42
```

结果保存在 `outputs/v1b_proj256_concat/<dataset>/seed_42/`，不会覆盖 `outputs/v1_baseline/`。
