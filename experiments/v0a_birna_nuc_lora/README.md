# v0a BiRNA-BERT NUC LoRA

Pure BiRNA-BERT NUC single-branch baseline. It uses masked mean pooling and Wqkv LoRA without FiLM, BPE, handcrafted features, or MKE fusion.

```bash
python train.py --version v0a_birna_nuc_lora --dataset H_b --seed 42
```
