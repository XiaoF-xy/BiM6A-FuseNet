# v0d BiRNA-BERT Attention-and-FFN LoRA

This version keeps v0a unchanged except that LoRA targets `Wqkv`, `attention.output.dense`, `gated_layers`, and `wo`.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v0d_birna_lora_attention_ffn --dataset H_b --seed 42
CUDA_VISIBLE_DEVICES=1 python train.py --version v0d_birna_lora_attention_ffn --dataset H_k --seed 42
CUDA_VISIBLE_DEVICES=2 python train.py --version v0d_birna_lora_attention_ffn --dataset H_l --seed 42
```
