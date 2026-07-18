# BiRNA-BERT LoRA Target Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add isolated v0c and v0d pure BiRNA-BERT experiments that expand LoRA from Wqkv to full attention and then to attention plus FFN.

**Architecture:** Reuse `BiRNASingleBranchClassifier`, masked mean pooling, the classifier, and all v0a training settings. Only each experiment's `lora_target_modules` differs; a small pre-wrap validator ensures every configured suffix matches at least one linear backbone module.

**Tech Stack:** Python 3.10, PyTorch 2.3.1, Transformers 4.41.2, PEFT 0.11.1, pytest.

---

### Task 1: Specify configuration and target matching behavior

**Files:**
- Modify: `tests/test_birna_single_branch.py`
- Modify: `tests/test_birna_single_model.py`

- [ ] **Step 1: Write failing configuration tests**

Add assertions that v0c uses `['Wqkv', 'attention.output.dense']`, v0d additionally uses `gated_layers` and `wo`, and both otherwise match v0a's model/training configuration.

- [ ] **Step 2: Write failing target-validation tests**

Construct a small nested fake backbone with module names ending in the four supported suffixes. Assert the validator accepts the correct list and raises a descriptive error for an unmatched suffix.

- [ ] **Step 3: Run the focused tests and confirm RED**

Run: `pytest -q tests/test_birna_single_branch.py tests/test_birna_single_model.py`

Expected: failure because v0c/v0d are not registered and the target validator does not exist.

### Task 2: Add target validation and experiment entrypoints

**Files:**
- Modify: `src/model_birna_single.py`
- Modify: `configs/configarg.py`
- Create: `experiments/v0c_birna_lora_full_attention/__init__.py`
- Create: `experiments/v0c_birna_lora_full_attention/config_v0c.py`
- Create: `experiments/v0c_birna_lora_full_attention/README.md`
- Create: `experiments/v0d_birna_lora_attention_ffn/__init__.py`
- Create: `experiments/v0d_birna_lora_attention_ffn/config_v0d.py`
- Create: `experiments/v0d_birna_lora_attention_ffn/README.md`

- [ ] **Step 1: Implement suffix validation**

Before calling PEFT, collect named `nn.Linear` modules and require every configured target suffix to match at least one name. Include both missing targets and representative available names in the error.

- [ ] **Step 2: Add v0c configuration**

Copy v0a settings exactly and change only the version metadata and targets to `Wqkv,attention.output.dense`.

- [ ] **Step 3: Add v0d configuration**

Copy v0a settings exactly and change only the version metadata and targets to `Wqkv,attention.output.dense,gated_layers,wo`.

- [ ] **Step 4: Register both versions and document commands**

Add both modules to `BASE_VERSION_CONFIG_MODULES`; document the three-GPU H_b/H_k/H_l launch commands in each experiment README.

- [ ] **Step 5: Run focused tests and confirm GREEN**

Run: `pytest -q tests/test_birna_single_branch.py tests/test_birna_single_model.py`

Expected: all focused tests pass.

### Task 3: Portability, documentation, and regression verification

**Files:**
- Modify: `scripts/verify_portable.py`
- Modify: `tests/test_model_parity.py`
- Modify: `README.md`

- [ ] **Step 1: Require the new server entrypoints**

Add both config files to `REQUIRED` and assert them in the portability test.

- [ ] **Step 2: Add root usage documentation**

List v0c/v0d and their three-GPU launch commands without changing existing commands.

- [ ] **Step 3: Run dry-run command verification**

Run both versions with `--dry_run` and check that the emitted `--lora_target_modules` values exactly match their designs.

- [ ] **Step 4: Run project verification**

Run: `pytest -q`

Run: `python scripts/verify_portable.py`

Expected: tests pass and `portable_check: OK`.

- [ ] **Step 5: Commit the complete implementation**

Stage only the v0c/v0d implementation, tests, and documentation, then commit with `feat: expand BiRNA LoRA target modules`.
