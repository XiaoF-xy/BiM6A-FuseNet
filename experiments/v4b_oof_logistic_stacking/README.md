# v4b OOF Logistic Stacking

This version performs no neural-network training. It uses `[p_handcrafted, p_birna]` from completed `v2c_mke_handcrafted_only_official4c` and `v0a_birna_nuc_lora` benchmark OOF predictions as inputs to an L2-regularized logistic regression.

Benchmark reporting uses a fold-preserving meta cross-fit. The final meta-classifier is fitted on all benchmark OOF rows and then applied once to the aligned independent five-model ensemble probabilities. Independent labels are used only for final metric calculation.

```bash
python train.py --version v4b_oof_logistic_stacking --dataset H_b --seed 42
```

No GPU or retained model checkpoint is required. The two base result directories must already exist under the selected `--outputs_root`.
