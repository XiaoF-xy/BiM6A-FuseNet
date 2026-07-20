# v4 OOF Late-Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add runnable v4a weighted late fusion and v4b logistic stacking using leakage-safe benchmark OOF predictions and untouched independent-test predictions.

**Architecture:** Put probability alignment, fusion fitting, fold-preserving cross-fitting, and artifact generation in a focused `src/late_fusion.py` module. Add a small `scripts/fuse_predictions.py` CLI and dispatch the two non-neural versions from the existing `train.py` workflow before neural configuration loading.

**Tech Stack:** Python, NumPy, scikit-learn, existing metrics/reporting/plotting utilities, pytest.

---

### Task 1: Fusion core

**Files:**
- Create: `src/late_fusion.py`
- Create: `tests/test_late_fusion.py`

- [ ] Write failing tests for strict prediction alignment, deterministic ACC-based alpha fitting, logistic fitting, and five-fold meta cross-fitting.
- [ ] Run `pytest -q tests/test_late_fusion.py` and confirm failure because `src.late_fusion` does not exist.
- [ ] Implement the smallest pure functions needed by the tests.
- [ ] Run `pytest -q tests/test_late_fusion.py` and confirm all tests pass.

### Task 2: Artifact-producing fusion runner

**Files:**
- Modify: `src/late_fusion.py`
- Create: `scripts/fuse_predictions.py`
- Modify: `tests/test_late_fusion.py`

- [ ] Write a failing integration test that creates two synthetic five-fold result trees and expects the standard fusion artifacts.
- [ ] Run the integration test and confirm the missing runner behavior causes the failure.
- [ ] Implement source-tree validation, meta-OOF and independent prediction generation, JSON/CSV summaries, model metadata, and existing ROC/PR plotting.
- [ ] Run `pytest -q tests/test_late_fusion.py` and confirm all tests pass.

### Task 3: Existing command entry point

**Files:**
- Modify: `scripts/train.py`
- Modify: `tests/test_protocol.py`

- [ ] Write failing tests showing both v4 versions dispatch to fusion and a neural version still builds its original CV command.
- [ ] Run the focused tests and confirm the v4 dispatch test fails.
- [ ] Add the minimal early dispatch and optional `--outputs_root` argument without adding v4 to neural config modules.
- [ ] Run the focused tests and confirm they pass.

### Task 4: Documentation and verification

**Files:**
- Create: `experiments/v4a_oof_weighted_late_fusion/README.md`
- Create: `experiments/v4b_oof_logistic_stacking/README.md`
- Modify: `README.md`

- [ ] Document prerequisites, leakage-safe data flow, output files, and three-dataset commands.
- [ ] Run `pytest -q` and confirm the complete suite passes.
- [ ] Run v4a and v4b against the existing human-brain base outputs using an explicit output root and inspect the metrics/artifacts.
- [ ] Run `python train.py --version v1_baseline --dataset H_b --seed 42 --dry_run` and confirm the existing neural command remains unchanged.
