# v2c Official Handcrafted-Only MKE Design

## Purpose

Add one isolated control experiment, `v2c_mke_handcrafted_only_official4c`, to answer whether the public MKE-ResNet handcrafted model can reproduce its reported strength under BiM6A-FuseNet's strict evaluation protocol. This experiment contains no BiRNA-BERT branch and does not alter v1, v1b, v2a/v2b, or v3a/v3b.

## Source-of-truth decision

The paper and public repository disagree in several implementation details. This version follows the executable public model and feature code for architecture and input representation, while following the paper for the principal training schedule and the existing BiM6A-FuseNet pipeline for fair evaluation.

- Input features follow the repository: ONEHOT(4), chemical properties plus cumulative nucleotide frequency(4), EIIP(1), and ENAC(4), for 13 channels over 41 positions.
- The network follows `MKE-Resnet/models/custom_model_resnet.py`: four residual streams, branch-level ECA, merged residual blocks, dropout 0.3/0.3/0.85, and a 64 -> 32 -> 2 classifier.
- The public residual blocks use GroupNorm and GELU, so v2c preserves those operations.
- This version does not add the paper-described post-fusion multi-scale spatial MKE-ECA block. That block belongs to the separate v3 experiment family.
- The paper says BCE, but the public model produces two logits. v2c therefore uses two-logit cross-entropy, which is the mathematically compatible classification loss and matches the public executable model interface.

## Evaluation protocol

- Use only `benchmark.csv` to construct `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` folds.
- Train each fold with fold seeds 42 through 46.
- Select the best epoch by held-out benchmark ACC.
- Never use `independent_test.csv` for epoch or hyperparameter selection.
- Evaluate each selected fold model on the independent test set and average the five positive-class probabilities for the final soft-voting result.
- Export the existing complete metric, prediction, ROC, and PR artifacts.
- Delete each temporary fold checkpoint only after its validation and independent-test predictions and metrics have been verified on disk.

## Training configuration

- Optimizer: Adam
- Loss: CrossEntropyLoss over two logits
- Batch size: 64
- Initial learning rate: 1e-3
- Weight decay: 1e-5
- Maximum epochs: 100
- Learning-rate scheduler: ReduceLROnPlateau on benchmark validation loss, mode=min, patience=10
- Early stopping: 20 consecutive epochs without strict improvement in benchmark validation ACC
- Best checkpoint: highest benchmark validation ACC; ties keep the earlier epoch

The optimizer and stopping options are configuration-driven. Their defaults remain AdamW with no scheduler or early stopping so existing experiment commands and behavior stay unchanged.

## Isolation and naming

- Version: `v2c_mke_handcrafted_only_official4c`
- Plot label: `MKE-ResNet-official4c`
- New official-compatible feature and model modules are separate from the hash-protected v9a migration files and from the 12-channel v2/v3 fusion encoder.
- Outputs use `outputs/v2c_mke_handcrafted_only_official4c/<dataset>/seed_42/`.

## Acceptance criteria

- The chemical feature's fourth row is the per-position cumulative frequency of the current nucleotide, rounded to three decimals as in the public repository.
- The model accepts exactly `(batch, 13, 41)`, splits channels as 4/4/1/4, and returns `(batch, 2)`.
- v2c's dry-run command contains the official-only flag, 100 epochs, batch size 64, Adam, scheduler patience 10, early stopping patience 20, ACC selection, and no BiRNA/FiLM/LoRA flags.
- Existing v1-v3 commands remain unchanged.
- The full test suite passes.
