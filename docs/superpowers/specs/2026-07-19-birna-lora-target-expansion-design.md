# BiRNA-BERT LoRA 目标层扩展实验设计

## 目标

在不改变 `v0a_birna_nuc_lora` 的数据、池化、分类头和训练超参数的前提下，验证扩大 LoRA 目标层覆盖范围能否提高纯 BiRNA-BERT 分支的性能。

## 版本

### `v0c_birna_lora_full_attention`

- 继承 v0a 的 NUC masked mean pooling 和分类头；
- 冻结 BiRNA-BERT 原始参数；
- LoRA 覆盖每层 `Wqkv` 和注意力输出投影 `attention.output.dense`；
- `r=8`、`alpha=32`、LoRA dropout `0.05`；
- AdamW、学习率 `1e-4`、weight decay `0.01`、batch size 32、20 epochs。

### `v0d_birna_lora_attention_ffn`

- 继承 v0c 的全部设置；
- LoRA 进一步覆盖每层 FFN 的 `gated_layers` 和 `wo`；
- 不改变 rank、alpha、dropout、学习率或分类头。

## 对照关系

```text
v0a: Wqkv
v0c: Wqkv + attention.output.dense
v0d: Wqkv + attention.output.dense + gated_layers + wo
```

三个版本之间唯一的实验变量是 LoRA 目标层范围。不得加入中心池化、LoRA+、rsLoRA、DoRA、学习率调整或手工特征。

## 实现边界

- 保留 `v0a`、`v0b` 和已有融合版本的行为；
- 使用显式目标模块匹配，启动时验证四类目标都实际命中；
- 继续使用 benchmark 分层五折、ACC 选最佳 epoch、独立集五模型软投票；
- 每折结果导出后删除临时权重；
- 新版本沿用现有输出目录和绘图格式。

## 验证

- 配置注册和启动命令解析；
- v0c/v0d 目标模块配置准确；
- LoRA 包装后命中的模块集合符合版本定义；
- 只有 LoRA 参数与分类头可训练；
- 现有版本配置和回归测试不变。

## 结果选择

先以 seed 42 的 benchmark 五折均值比较 ACC、MCC、AUROC 和 AUPR；独立测试集不参与版本选择。若候选版本稳定优于 v0a，再追加多个随机种子确认。
