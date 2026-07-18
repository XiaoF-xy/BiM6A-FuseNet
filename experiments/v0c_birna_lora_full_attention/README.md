# v0c BiRNA-BERT Full-Attention LoRA

This version keeps v0a unchanged except that LoRA targets both `Wqkv` and `attention.output.dense`.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v0c_birna_lora_full_attention --dataset H_b --seed 42
CUDA_VISIBLE_DEVICES=1 python train.py --version v0c_birna_lora_full_attention --dataset H_k --seed 42
CUDA_VISIBLE_DEVICES=2 python train.py --version v0c_birna_lora_full_attention --dataset H_l --seed 42
```
