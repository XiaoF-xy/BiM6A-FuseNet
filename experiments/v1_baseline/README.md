# v1_baseline

`v1_baseline` 是从 BiRNA_m6A 的 v9a 原样迁移得到的 BiM6A-FuseNet 基线。模型结构和超参数不变；只改变数据语义命名和评估流程。

- 输入：`benchmark.csv` 和 `independent_test.csv`
- 模型选择：仅在 `benchmark.csv` 上执行 seed=42 的分层五折交叉验证，每折按验证 ACC 选择最佳 epoch
- 基准集结果：五个验证折指标的均值和样本标准差
- 独立测试结果：五个最佳折模型分别预测后，对正类概率做 soft voting，再计算一次最终指标
- 保存：逐样本概率、所有指标、ROC/PR 坐标、600 dpi PNG 和矢量 PDF
- 清理：结果完整写入并回读确认后，自动删除 `best_model.pt`，然后释放模型和 CUDA 缓存

运行：

```bash
python train.py --version v1_baseline --dataset H_b --seed 42
```
