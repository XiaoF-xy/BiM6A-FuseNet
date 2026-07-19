# BiRNA-BERT LoRA+ Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated `v0e_birna_nuc_loraplus` experiment that preserves v0a and trains LoRA A, LoRA B, and the classifier with learning rates `5e-5`, `8e-4`, and `1e-4`.

**Architecture:** Extend training configuration and CLI with an opt-in LoRA+ policy. A focused optimizer helper partitions all trainable named parameters into three mutually exclusive groups and delegates optimizer construction to the existing Adam/AdamW factory; all existing versions keep their current flat parameter iterable.

**Tech Stack:** Python 3.10, PyTorch 2.3.1, Transformers 4.41.2, PEFT 0.11.1, pytest/unittest.

---

### Task 1: Specify v0e configuration and command behavior

**Files:**
- Modify: `tests/test_birna_single_branch.py`
- Modify: `configs/configarg.py`
- Modify: `scripts/train.py`
- Create: `experiments/v0e_birna_nuc_loraplus/__init__.py`
- Create: `experiments/v0e_birna_nuc_loraplus/config_v0e.py`

- [ ] **Step 1: Write the failing configuration test**

Add a test that loads v0a and v0e, removes version metadata/output paths and the four LoRA+ training fields, and asserts every remaining model/training field is identical. Assert v0e uses `Wqkv`, A LR `5e-5`, B LR `8e-4`, classifier LR `1e-4`, and emits all three CLI flags.

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `pytest -q tests/test_birna_single_branch.py -k loraplus`

Expected: FAIL because v0e is not registered.

- [ ] **Step 3: Add opt-in configuration fields and v0e**

Add `use_loraplus: bool = False`, `lora_a_lr`, `lora_b_lr`, and `classifier_lr` optional fields to `TrainConfig`. Register a v0e module copied from v0a with only metadata and those four fields changed.

- [ ] **Step 4: Emit CLI flags only for v0e**

When `use_loraplus` is true, require all three learning rates and append `--use_loraplus`, `--lora_a_lr`, `--lora_b_lr`, and `--classifier_lr` to the CV command.

- [ ] **Step 5: Re-run the focused test and confirm GREEN**

Run: `pytest -q tests/test_birna_single_branch.py -k loraplus`

Expected: PASS.

### Task 2: Implement strict LoRA+ parameter grouping

**Files:**
- Create: `src/loraplus.py`
- Create: `tests/test_loraplus_grouping.py`
- Modify: `tests/test_single_branch_training.py`
- Modify: `src/training_control.py`

- [ ] **Step 1: Write failing grouping tests**

First test a PyTorch-independent partition function with dummy parameters whose trainable names contain `lora_A`, `lora_B`, and `classifier`. Then assert the returned optimizer has group names `lora_A`, `lora_B`, `classifier`, the configured learning rates, no duplicated parameter IDs, and complete coverage. Add tests that raise when a LoRA group is missing or a trainable parameter object is duplicated.

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m unittest tests.test_single_branch_training.SingleBranchTrainingTests -v`

Expected: import failure because `build_loraplus_optimizer` is absent.

- [ ] **Step 3: Implement the minimal helper**

Add `partition_loraplus_named_parameters` to the dependency-free `src/loraplus.py`, then add `build_loraplus_optimizer(named_parameters, name, lora_a_lr, lora_b_lr, classifier_lr, weight_decay)` to `training_control.py`. Partition only `requires_grad` parameters using `.lora_A.` and `.lora_B.` markers; treat remaining trainable parameters as the classifier group. Reject non-positive rates, empty groups, duplicates, or incomplete coverage, then call `build_optimizer` with named parameter-group dictionaries.

- [ ] **Step 4: Re-run tests and confirm GREEN**

Run: `python -m unittest tests.test_single_branch_training.SingleBranchTrainingTests -v`

Expected: all tests pass.

### Task 3: Connect LoRA+ to five-fold training

**Files:**
- Modify: `src/train_cv.py`
- Modify: `tests/test_single_branch_training.py`

- [ ] **Step 1: Add failing argument and validation tests**

Assert parser support for the four LoRA+ flags and reject LoRA+ unless pure single-branch LoRA is enabled and all three rates are positive.

- [ ] **Step 2: Run the validation tests and confirm RED**

Run: `python -m unittest tests.test_single_branch_training.SingleBranchTrainingTests -v`

Expected: parser/validation assertions fail because the flags are absent.

- [ ] **Step 3: Add CLI parsing, validation, and optimizer selection**

Parse the four flags, validate them before data loading, print the resolved LoRA+ policy, and call `build_loraplus_optimizer(model.named_parameters(), ...)` only when enabled. Otherwise preserve the existing `build_optimizer` call unchanged.

- [ ] **Step 4: Re-run tests and confirm GREEN**

Run: `python -m unittest tests.test_single_branch_training.SingleBranchTrainingTests -v`

Expected: all tests pass.

### Task 4: Add server entrypoint and verify regression safety

**Files:**
- Create: `experiments/v0e_birna_nuc_loraplus/README.md`
- Modify: `README.md`
- Modify: `scripts/verify_portable.py`
- Modify: `tests/test_model_parity.py`

- [ ] **Step 1: Add portability assertions**

Require `experiments/v0e_birna_nuc_loraplus/config_v0e.py` in the portable file list and parity test.

- [ ] **Step 2: Document the three-GPU command**

Document v0e and the H_b/H_k/H_l launch commands. State the three learning rates and that benchmark CV, not the independent set, decides whether v0e improves on v0a.

- [ ] **Step 3: Verify dry-run configuration**

Run: `python train.py --version v0e_birna_nuc_loraplus --dataset H_b --seed 42 --dry_run`

Expected: command contains Wqkv LoRA and exactly the three approved LoRA+ rates.

- [ ] **Step 4: Run focused and project regression checks**

Run: `pytest -q tests/test_birna_single_branch.py tests/test_model_parity.py`

Run: `python -m unittest tests.test_single_branch_training -v`

Run: `python scripts/verify_portable.py`

Expected: tests pass and `portable_check: OK`.

- [ ] **Step 5: Review diff and commit implementation**

Run `git diff --check` and inspect `git diff --stat`. Stage only v0e implementation, tests, documentation, and this plan, then commit with `feat: add BiRNA LoRA+ optimizer experiment`.
