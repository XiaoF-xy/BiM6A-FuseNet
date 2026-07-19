# BiRNA-BERT Layer Mix and DoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independently runnable `v0f_birna_last4_scalar_mix` and `v0g_birna_nuc_dora` experiments without changing existing version behavior.

**Architecture:** Extend the shared pure-BiRNA classifier with two disabled-by-default switches. Scalar mix reads the custom backbone's packed last four encoder layers and pools exactly 41 nucleotide tokens; DoRA passes `use_dora=True` into the existing PEFT LoRA construction path. Version configs activate only one switch apiece while reusing the strict-CV pipeline.

**Tech Stack:** Python 3.10, PyTorch, Transformers 4.41.2, PEFT 0.11.1, pytest/unittest, existing versioned experiment launcher.

---

### Task 1: Define scalar-mix behavior with failing model tests

**Files:**
- Modify: `tests/test_birna_single_model.py`
- Modify: `src/model_birna_single.py`

- [ ] **Step 1: Write failing tests for scalar normalization and packed pooling**

Add tests that import `LastFourLayerScalarMix` and `masked_mean_packed_nucleotide_embeddings`, assert `softmax([-6,-6,-6,0]).sum() == 1`, verify a packed `[B*43,H]` tensor ignores CLS/SEP, and verify invalid token counts raise `ValueError`.

```python
mix = LastFourLayerScalarMix(hidden_size=2)
weights = mix.normalized_weights()
torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
assert weights[-1] > 0.99

attention_mask = torch.ones(2, 43, dtype=torch.long)
packed = torch.cat([sample_a, sample_b], dim=0)
pooled = masked_mean_packed_nucleotide_embeddings(packed, attention_mask)
torch.testing.assert_close(pooled, torch.tensor([[3.0, 3.0], [7.0, 7.0]]))
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/bim6a_pycache python -m unittest tests.test_birna_single_model -v
```

Expected: import failure because the new scalar-mix API does not exist.

- [ ] **Step 3: Implement packed pooling and scalar mix**

Add focused helpers to `src/model_birna_single.py`:

```python
def masked_mean_packed_nucleotide_embeddings(packed_embeddings, attention_mask):
    active_lengths = attention_mask.long().sum(dim=1)
    if packed_embeddings.size(0) != int(active_lengths.sum().item()):
        raise ValueError("Packed encoder output must contain one row per active token.")
    samples = torch.split(packed_embeddings, active_lengths.cpu().tolist(), dim=0)
    content = [sample[1:-1] for sample in samples]
    if any(tokens.size(0) != EXPECTED_NUCLEOTIDE_COUNT for tokens in content):
        raise ValueError("Expected exactly 41 NUC content tokens per sequence.")
    return torch.stack([tokens.mean(dim=0) for tokens in content])

class LastFourLayerScalarMix(nn.Module):
    def __init__(self, hidden_size=EXPECTED_HIDDEN_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        self.scalar_logits = nn.Parameter(torch.tensor([-6.0, -6.0, -6.0, 0.0]))

    def normalized_weights(self):
        return torch.softmax(self.scalar_logits, dim=0)

    def forward(self, hidden_states):
        selected = hidden_states[-4:]
        stacked = torch.stack(selected, dim=0)
        weights = self.normalized_weights().view(4, 1, 1)
        return torch.sum(stacked * weights, dim=0)
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the same unittest command. Expected: all scalar-mix and existing single-model tests pass.

### Task 2: Connect scalar mix and DoRA to the shared model

**Files:**
- Modify: `tests/test_birna_single_model.py`
- Modify: `src/model_birna_single.py`

- [ ] **Step 1: Write failing integration tests**

Add a fake custom backbone whose `.bert` method accepts `output_all_encoded_layers=True` and verify `BiRNASingleBranchClassifier(use_last4_scalar_mix=True)` returns `[B,2]`. Patch PEFT symbols and verify the new DoRA helper builds `LoraConfig(use_dora=True)`. The existing v9a LoRA helper remains unchanged and is still exercised by the pre-existing LoRA policy test.

```python
model = BiRNASingleBranchClassifier(
    model_dir="unused", freeze_backbone=False, use_lora=False,
    use_last4_scalar_mix=True,
)
assert model(input_ids=input_ids, attention_mask=attention_mask).shape == (2, 2)
```

- [ ] **Step 2: Run focused tests and verify RED**

Expected: constructor rejects the new scalar-mix option and the DoRA helper does not exist.

- [ ] **Step 3: Implement disabled-by-default switches**

Add a single-branch-only `apply_dora_to_birna` helper and construct:

```python
lora_config = LoraConfig(
    r=r,
    lora_alpha=alpha,
    target_modules=target_modules,
    lora_dropout=dropout,
    bias="none",
    use_dora=True,
)
```

Extend `BiRNASingleBranchClassifier` with `use_dora=False` and `use_last4_scalar_mix=False`. Select the untouched legacy `apply_lora_to_birna` helper when DoRA is false and the new helper only when DoRA is true. The scalar path unwraps PEFT with `get_base_model()` when available, calls the `.bert` method with `output_all_encoded_layers=True`, mixes the last four packed layers, and uses packed nucleotide pooling. The existing path remains byte-for-behavior equivalent.

- [ ] **Step 4: Run focused tests and verify GREEN**

Expected: all `tests.test_birna_single_model` tests pass.

### Task 3: Expose CLI/config switches and version configs

**Files:**
- Modify: `tests/test_birna_single_branch.py`
- Create: `tests/test_birna_new_versions.py`
- Modify: `configs/configarg.py`
- Modify: `scripts/train.py`
- Modify: `src/train_cv.py`
- Create: `experiments/v0f_birna_last4_scalar_mix/__init__.py`
- Create: `experiments/v0f_birna_last4_scalar_mix/config_v0f.py`
- Create: `experiments/v0f_birna_last4_scalar_mix/README.md`
- Create: `experiments/v0g_birna_nuc_dora/__init__.py`
- Create: `experiments/v0g_birna_nuc_dora/config_v0g.py`
- Create: `experiments/v0g_birna_nuc_dora/README.md`

- [ ] **Step 1: Write failing version-isolation tests**

Compare each candidate with v0a after removing only its intended flag:

```python
assert v0f.model.use_last4_scalar_mix is True
assert v0f.model.use_dora is False
assert v0g.model.use_last4_scalar_mix is False
assert v0g.model.use_dora is True
assert "--use_last4_scalar_mix" in build_cv_command(v0f)
assert "--use_dora" in build_cv_command(v0g)
```

Also assert all existing versions default both flags to false.

- [ ] **Step 2: Run version tests and verify RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/bim6a_pycache python -m pytest tests/test_birna_single_branch.py -q
```

Expected: unknown versions or missing model fields.

- [ ] **Step 3: Add model fields, CLI flags, validation, and configs**

Add to `ModelConfig`:

```python
use_last4_scalar_mix: bool = False
use_dora: bool = False
```

Register both config modules. Append their CLI flags only when enabled. In `validate_single_branch_options`, require each flag to be used only with pure single-branch LoRA. Pass both options from `train_one_fold` to `BiRNASingleBranchClassifier`.

Each new config copies v0a hyperparameters exactly and changes one model flag plus version metadata.

- [ ] **Step 4: Run version tests and verify GREEN**

Expected: all single-branch version tests pass.

### Task 4: Preserve portable-server migration and document commands

**Files:**
- Modify: `tests/test_model_parity.py`
- Modify: `scripts/verify_portable.py`
- Modify: `experiments/v0f_birna_last4_scalar_mix/README.md`
- Modify: `experiments/v0g_birna_nuc_dora/README.md`

- [ ] **Step 1: Write a failing portable requirement test**

Require both config paths in `scripts.verify_portable.REQUIRED`.

- [ ] **Step 2: Run the parity test and verify RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/bim6a_pycache python -m pytest tests/test_model_parity.py::test_portable_check_requires_all_mke_variant_entrypoints -q
```

Expected: missing v0f/v0g config paths.

- [ ] **Step 3: Register portable files and final commands**

Add both config files to `REQUIRED`. Document three-GPU background commands using `&`, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v0f_birna_last4_scalar_mix --dataset H_b --seed 42 &
CUDA_VISIBLE_DEVICES=1 python train.py --version v0f_birna_last4_scalar_mix --dataset H_k --seed 42 &
CUDA_VISIBLE_DEVICES=2 python train.py --version v0f_birna_last4_scalar_mix --dataset H_l --seed 42 &
```

- [ ] **Step 4: Run parity test and verify GREEN**

Expected: portable requirement test passes.

### Task 5: Full verification and implementation commit

**Files:**
- Verify all modified and created files

- [ ] **Step 1: Run focused and full tests**

```bash
PYTHONPYCACHEPREFIX=/tmp/bim6a_pycache python -m pytest tests/test_birna_single_model.py tests/test_birna_single_branch.py tests/test_model_parity.py -q
PYTHONPYCACHEPREFIX=/tmp/bim6a_pycache python -m pytest -q
```

Expected: zero failures. If PyTorch/pytest is unavailable, record that limitation and run dependency-free config regression checks instead.

- [ ] **Step 2: Compile changed Python files**

```bash
PYTHONPYCACHEPREFIX=/tmp/bim6a_pycache python -m py_compile src/model_birna_single.py src/train_cv.py scripts/train.py configs/configarg.py experiments/v0f_birna_last4_scalar_mix/config_v0f.py experiments/v0g_birna_nuc_dora/config_v0g.py
```

Expected: exit code 0.

- [ ] **Step 3: Verify both dry-run commands and portable package**

```bash
python train.py --version v0f_birna_last4_scalar_mix --dataset H_b --seed 42 --dry_run
python train.py --version v0g_birna_nuc_dora --dataset H_b --seed 42 --dry_run
python scripts/verify_portable.py
```

Expected: v0f command contains only `--use_last4_scalar_mix`; v0g contains only `--use_dora`; portable check prints `portable_check: OK`.

- [ ] **Step 4: Inspect diff and commit complete implementation**

```bash
git diff --check
git status --short
git add configs/configarg.py scripts/train.py scripts/verify_portable.py src/model_birna_single.py src/train_cv.py tests/test_birna_single_model.py tests/test_birna_single_branch.py tests/test_birna_new_versions.py tests/test_model_parity.py experiments/v0f_birna_last4_scalar_mix experiments/v0g_birna_nuc_dora docs/superpowers/plans/2026-07-19-birna-layer-mix-dora-implementation.md
git commit -m "feat: add BiRNA layer mix and DoRA experiments"
```

Expected: one implementation commit; no push.
