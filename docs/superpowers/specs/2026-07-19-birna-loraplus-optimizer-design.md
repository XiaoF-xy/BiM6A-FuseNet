# BiRNA-BERT LoRA+ 优化器实验设计

## 目标

在 `v0a_birna_nuc_lora` 的模型结构和评估协议完全不变的前提下，只改变可训练参数的学习率分配，验证 LoRA+ 的非对称学习率能否改善纯 BiRNA-BERT 微调效果。

## 实验版本

版本名为 `v0e_birna_nuc_loraplus`。

- NUC 输入和 masked mean pooling 与 v0a 相同；
- LoRA 仍只覆盖 `Wqkv`；
- `r=8`、`alpha=32`、LoRA dropout `0.05`；
- 分类头、batch size 32、20 epochs、weight decay `0.01` 均与 v0a 相同；
- 不使用 center pooling、FFN LoRA、早停或 warmup；
- LoRA A 学习率为 `5e-5`；
- LoRA B 学习率为 `8e-4`；
- 分类头学习率为 `1e-4`。

LoRA B/A 学习率比为 16。独立测试集仍不参与模型选择。

## 实现方式

不升级当前服务器使用的 `peft==0.11.1`。训练代码根据可训练参数名称将参数分成三组：

1. 名称包含 `.lora_A.` 的 LoRA A 参数；
2. 名称包含 `.lora_B.` 的 LoRA B 参数；
3. 其余可训练参数，即单分支分类头。

三组参数交给现有 AdamW 创建入口。启用 LoRA+ 时，三组必须全部非空且不能遗漏任何可训练参数，否则在训练开始前报错。未启用 LoRA+ 的版本继续使用原优化器路径。

## 配置与可复现性

`TrainConfig` 增加默认关闭的 LoRA+ 开关，以及 A、B、分类头三个可选学习率。启动器仅在开关开启时传递对应命令行参数。最终的 `resolved_config.json`、启动命令和 checkpoint 参数记录能够还原三组学习率。

## 评估协议

继续使用 benchmark 分层五折交叉验证、每折验证 ACC 选择最佳 epoch，以及独立测试集五折模型正类概率 soft voting。结果文件、绘图数据和训练后删除临时模型的策略不变。

## 验证要求

- v0e 除 LoRA+ 优化器配置和版本元数据外，与 v0a 配置一致；
- 启动命令准确携带三个学习率；
- LoRA A、LoRA B 和分类头参数被完整且互斥地分组；
- 缺少任一组或存在未覆盖参数时给出明确错误；
- v0a 和其他已有版本的命令与优化器行为不变；
- 服务器完整性检查包含 v0e 配置入口。

## 结果判定

先比较 seed 42 下三个组织的 benchmark 五折 ACC、MCC、AUROC 和 AUPR。独立测试结果只用于最终泛化评估，不用于决定 v0e 是否优于 v0a。若 v0e 在多数组织上稳定改善 benchmark 指标，再追加多随机种子验证。
