# BiRNA-BERT Single-Branch v0a/v0b Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add isolated pure BiRNA-BERT NUC experiments for Wqkv LoRA and full-model fine-tuning without changing any existing v1-v3/v2c behavior.

**Architecture:** A dedicated single-branch classifier loads the existing local BiRNA-BERT checkpoint, masks CLS/SEP/padding, mean-pools 41 nucleotide embeddings, and applies a shared 768→256→2 GELU head. Version configs choose either Wqkv LoRA or full fine-tuning; the existing strict five-fold runner handles ACC checkpoint selection, OOF export, five-model independent voting, plotting, and checkpoint deletion.

**Tech Stack:** Python 3.10, PyTorch, Transformers, PEFT, scikit-learn, pytest.

---

## File map

- Create `src/model_birna_single.py`: isolated shared v0a/v0b model and special-token-aware mean pooling.
- Create `experiments/v0a_birna_nuc_lora/`: v0a configuration and usage notes.
- Create `experiments/v0b_birna_nuc_fullft/`: v0b configuration and usage notes.
- Modify `configs/configarg.py`: register versions and add opt-in single-branch/warmup fields.
- Modify `scripts/train.py`: translate new config fields into runner flags.
- Modify `src/train_cv.py`: validate flags, instantiate the model, and create an opt-in batch warmup scheduler.
- Modify `src/training_control.py`: build constant-after-warmup scheduler.
- Modify `src/training_utils.py`: step an optional batch scheduler after each optimizer update.
- Modify `scripts/verify_portable.py`: include the new portable source/config files.
- Create `tests/test_birna_single_branch.py`: model masking, parameter policy, config, command, and scheduler isolation tests.
- Modify `tests/test_protocol.py` and `tests/test_model_parity.py`: protect old version behavior and portability expectations.

### Task 1: Add failing configuration and command tests

**Files:**
- Create: `tests/test_birna_single_branch.py`
- Modify: `tests/test_protocol.py`

- [ ] **Step 1: Write failing tests for v0a/v0b configs and launch commands**

```python
def test_v0a_is_pure_nuc_wqkv_lora():
    config = load_experiment_config("v0a_birna_nuc_lora", "H_b", seed=42)
    command = build_cv_command(config)
    assert config.model.use_birna_single_branch is True
    assert config.model.use_handcrafted_features is False
    assert config.model.use_film is False
    assert config.model.use_bpe_view is False
    assert config.model.use_lora is True
    assert config.model.freeze_backbone is True
    assert config.model.use_center_pooling is False
    assert config.training.epochs == 20
    assert config.training.batch_size == 32
    assert config.training.lr == pytest.approx(1e-4)
    assert config.training.warmup_ratio is None
    assert "--use_birna_single_branch" in command
    assert "--use_lora" in command
    assert "--use_film" not in command
    assert "--use_handcrafted_features" not in command


def test_v0b_is_pure_nuc_full_finetuning_with_warmup():
    config = load_experiment_config("v0b_birna_nuc_fullft", "H_b", seed=42)
    command = build_cv_command(config)
    assert config.model.use_birna_single_branch is True
    assert config.model.use_lora is False
    assert config.model.freeze_backbone is False
    assert config.training.epochs == 10
    assert config.training.batch_size == 64
    assert config.training.lr == pytest.approx(1e-6)
    assert config.training.warmup_ratio == pytest.approx(0.1)
    assert "--use_birna_single_branch" in command
    assert command[command.index("--warmup_ratio") + 1] == "0.1"
    assert "--freeze_backbone" not in command
    assert "--use_lora" not in command
```

- [ ] **Step 2: Add an old-version isolation assertion**

```python
for version in EXISTING_VERSIONS:
    config = load_experiment_config(version, "H_b", seed=42)
    command = build_cv_command(config)
    assert config.model.use_birna_single_branch is False
    assert config.training.warmup_ratio is None
    assert "--use_birna_single_branch" not in command
    assert "--warmup_ratio" not in command
```

- [ ] **Step 3: Run the focused tests and confirm they fail before implementation**

Run:

```bash
pytest -q tests/test_birna_single_branch.py tests/test_protocol.py
```

Expected: failure because the two versions and new dataclass fields do not exist.

### Task 2: Register the two isolated experiment configurations

**Files:**
- Create: `experiments/v0a_birna_nuc_lora/__init__.py`
- Create: `experiments/v0a_birna_nuc_lora/config_v0a.py`
- Create: `experiments/v0a_birna_nuc_lora/README.md`
- Create: `experiments/v0b_birna_nuc_fullft/__init__.py`
- Create: `experiments/v0b_birna_nuc_fullft/config_v0b.py`
- Create: `experiments/v0b_birna_nuc_fullft/README.md`
- Modify: `configs/configarg.py`
- Modify: `scripts/train.py`

- [ ] **Step 1: Add opt-in dataclass fields with old-version-safe defaults**

```python
class ModelConfig:
    use_birna_single_branch: bool = False


class TrainConfig:
    warmup_ratio: float | None = None
```

- [ ] **Step 2: Register both version modules**

```python
VERSION_CONFIG_MODULES = {
    "v0a_birna_nuc_lora": "experiments.v0a_birna_nuc_lora.config_v0a",
    "v0b_birna_nuc_fullft": "experiments.v0b_birna_nuc_fullft.config_v0b",
    # existing entries remain unchanged
}
```

- [ ] **Step 3: Add the v0a config**

```python
def get_overrides(dataset_name: str, seed: int) -> dict:
    return {
        "experiment": {
            "version_name": "v0a_birna_nuc_lora",
            "plot_label": "BiRNA-BERT-NUC-LoRA",
            "description": "Pure BiRNA-BERT NUC mean-pooling classifier with Wqkv LoRA.",
        },
        "model": {
            "use_birna_single_branch": True,
            "freeze_backbone": True,
            "use_center_pooling": False,
            "use_bpe_view": False,
            "use_film": False,
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": ["Wqkv"],
            "use_handcrafted_features": False,
        },
        "data": {"dataset_name": dataset_name, "sequence_length": 41},
        "training": {
            "seed": seed,
            "folds": 5,
            "selection_metric": "ACC",
            "epochs": 20,
            "batch_size": 32,
            "lr": 1e-4,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "max_length": 64,
            "warmup_ratio": None,
        },
    }
```

- [ ] **Step 4: Add the v0b config**

```python
def get_overrides(dataset_name: str, seed: int) -> dict:
    return {
        "experiment": {
            "version_name": "v0b_birna_nuc_fullft",
            "plot_label": "BiRNA-BERT-NUC-FullFT",
            "description": "Pure BiRNA-BERT NUC mean-pooling classifier with full fine-tuning.",
        },
        "model": {
            "use_birna_single_branch": True,
            "freeze_backbone": False,
            "use_center_pooling": False,
            "use_bpe_view": False,
            "use_film": False,
            "use_lora": False,
            "use_handcrafted_features": False,
        },
        "data": {"dataset_name": dataset_name, "sequence_length": 41},
        "training": {
            "seed": seed,
            "folds": 5,
            "selection_metric": "ACC",
            "epochs": 10,
            "batch_size": 64,
            "lr": 1e-6,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "max_length": 64,
            "warmup_ratio": 0.1,
        },
    }
```

- [ ] **Step 5: Pass only explicitly enabled flags from the launcher**

```python
if training.warmup_ratio is not None:
    command.extend(["--warmup_ratio", str(training.warmup_ratio)])
if model.use_birna_single_branch:
    command.extend(["--use_birna_single_branch", "--model_label", config.experiment.plot_label])
```

- [ ] **Step 6: Run configuration tests**

Run: `pytest -q tests/test_birna_single_branch.py tests/test_protocol.py`

Expected: config tests pass; model/scheduler tests added later may still fail.

### Task 3: Implement the shared masked-mean single-branch model

**Files:**
- Create: `src/model_birna_single.py`
- Test: `tests/test_birna_single_branch.py`

- [ ] **Step 1: Write a failing masked-mean pooling test**

The test must use a fake backbone returning `[B, L, 768]` embeddings and attention masks with unequal padding. It must assert that changing CLS, SEP, or padded embeddings does not change the pooled feature or final logits.

- [ ] **Step 2: Implement special-token-aware pooling**

```python
def nucleotide_content_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected attention_mask [B, L], got {tuple(attention_mask.shape)}")
    mask = attention_mask.bool().clone()
    if mask.size(1) < 3:
        raise ValueError("NUC input requires CLS, at least one nucleotide, and SEP.")
    mask[:, 0] = False
    active_lengths = attention_mask.long().sum(dim=1)
    if torch.any(active_lengths < 3):
        raise ValueError("Every NUC input requires CLS, at least one nucleotide, and SEP.")
    sep_indices = active_lengths - 1
    mask.scatter_(1, sep_indices.unsqueeze(1), False)
    return mask
```

- [ ] **Step 3: Implement the dedicated classifier**

```python
class BiRNASingleBranchClassifier(nn.Module):
    def __init__(self, model_dir, freeze_backbone, use_lora, lora_r=8,
                 lora_alpha=32, lora_dropout=0.05,
                 lora_target_modules=None, dropout=0.2):
        super().__init__()
        self.birna_model = load_birna_backbone(model_dir)
        self.use_lora = use_lora
        if use_lora:
            self.birna_model = apply_lora_to_birna(
                self.birna_model,
                lora_target_modules or ["Wqkv"],
                lora_r,
                lora_alpha,
                lora_dropout,
            )
        elif freeze_backbone:
            for parameter in self.birna_model.parameters():
                parameter.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        embeddings = self.birna_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits
        mask = nucleotide_content_mask(attention_mask)
        counts = mask.sum(dim=1)
        if torch.any(counts != 41):
            raise ValueError(f"Expected exactly 41 NUC content tokens, got {counts.tolist()}")
        pooled = (embeddings * mask.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(1)
        return self.classifier(pooled)
```

- [ ] **Step 4: Test LoRA/full-finetune parameter policy**

Use monkeypatching so the test does not load the 400+ MB real checkpoint. Assert the v0a fake base parameters are frozen and adapter/head parameters train, while v0b base/head parameters all train.

- [ ] **Step 5: Run model tests**

Run: `pytest -q tests/test_birna_single_branch.py`

Expected: all model and configuration tests pass.

### Task 4: Add opt-in full-finetuning warmup without changing old schedulers

**Files:**
- Modify: `src/training_control.py`
- Modify: `src/training_utils.py`
- Modify: `src/train_cv.py`
- Test: `tests/test_birna_single_branch.py`

- [ ] **Step 1: Write failing scheduler tests**

```python
def test_constant_warmup_scheduler_reaches_and_keeps_base_lr():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))], lr=1e-6)
    scheduler = build_constant_warmup_scheduler(optimizer, total_steps=10, warmup_ratio=0.2)
    observed = []
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])
    assert observed[1] == pytest.approx(1e-6)
    assert observed[-1] == pytest.approx(1e-6)
```

- [ ] **Step 2: Implement validated opt-in warmup**

```python
def build_constant_warmup_scheduler(optimizer, *, total_steps: int, warmup_ratio: float | None):
    if warmup_ratio is None:
        return None
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    warmup_steps = int(total_steps * warmup_ratio)
    if warmup_steps == 0:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, float(step + 1) / float(warmup_steps)),
    )
```

- [ ] **Step 3: Step the optional scheduler after optimizer updates**

```python
def train_one_epoch(..., batch_scheduler=None) -> float:
    loss.backward()
    optimizer.step()
    if batch_scheduler is not None:
        batch_scheduler.step()
```

- [ ] **Step 4: Add runner flags and incompatibility validation**

Add `--use_birna_single_branch` and `--warmup_ratio`. Reject combining the single branch with FiLM, BPE, handcrafted, MKE, projected, gated, or center pooling. Reject combining batch warmup with `ReduceLROnPlateau`.

- [ ] **Step 5: Instantiate the dedicated model and scheduler**

```python
if args.use_birna_single_branch:
    model = BiRNASingleBranchClassifier(**common_model_kwargs)

batch_scheduler = build_constant_warmup_scheduler(
    optimizer,
    total_steps=len(train_loader) * args.epochs,
    warmup_ratio=args.warmup_ratio,
)
```

Pass `batch_scheduler` to every `train_one_epoch` call. Existing versions pass `None` through the default config.

- [ ] **Step 6: Run scheduler and protocol tests**

Run: `pytest -q tests/test_birna_single_branch.py tests/test_protocol.py`

Expected: all focused tests pass.

### Task 5: Portability, documentation, and regression verification

**Files:**
- Modify: `scripts/verify_portable.py`
- Modify: `tests/test_model_parity.py`
- Modify: `README.md`
- Test: all tests

- [ ] **Step 1: Add required portable files**

Add the model and both version configs to `REQUIRED`, and assert them in `test_portable_check_requires_all_mke_variant_entrypoints` (renaming that test to cover all experiment entrypoints is allowed).

- [ ] **Step 2: Document launch commands**

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v0a_birna_nuc_lora --dataset H_b --seed 42
CUDA_VISIBLE_DEVICES=1 python train.py --version v0a_birna_nuc_lora --dataset H_k --seed 42
CUDA_VISIBLE_DEVICES=2 python train.py --version v0a_birna_nuc_lora --dataset H_l --seed 42
```

Document the equivalent `v0b_birna_nuc_fullft` commands and note its larger GPU memory requirement.

- [ ] **Step 3: Verify dry-run commands**

Run:

```bash
python train.py --version v0a_birna_nuc_lora --dataset H_b --seed 42 --dry_run
python train.py --version v0b_birna_nuc_fullft --dataset H_b --seed 42 --dry_run
```

Expected: v0a command contains LoRA and no handcrafted/FiLM flags; v0b contains full-finetuning warmup and no LoRA/freeze flags.

- [ ] **Step 4: Run the portable check and full test suite**

Run:

```bash
python scripts/verify_portable.py
pytest -q
```

Expected: portable check succeeds and all tests pass.

- [ ] **Step 5: Inspect the final diff for isolation**

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Expected: only the design/plan, new single-branch files, scoped configuration/runner changes, tests, portability list, and documentation are modified.

- [ ] **Step 6: Commit only after the complete versions pass verification**

```bash
git add docs/superpowers/specs/2026-07-18-birna-single-branch-design.md \
  docs/superpowers/plans/2026-07-18-birna-single-branch-implementation.md \
  src/model_birna_single.py src/training_control.py src/training_utils.py src/train_cv.py \
  configs/configarg.py scripts/train.py scripts/verify_portable.py README.md \
  experiments/v0a_birna_nuc_lora experiments/v0b_birna_nuc_fullft \
  tests/test_birna_single_branch.py tests/test_protocol.py tests/test_model_parity.py
git commit -m "feat: add pure BiRNA-BERT single-branch experiments"
```

Do not push until the user explicitly asks for GitHub publication.
