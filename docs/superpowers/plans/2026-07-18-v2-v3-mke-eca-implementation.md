# BiM6A-FuseNet v2/v3 MKE-ECA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the four approved experiments `v2a_mke_res_eca_native`, `v2b_mke_res_eca_proj256`, `v3a_full_mke_eca_native`, and `v3b_full_mke_eca_proj256` without changing v1/v1b behavior.

**Architecture:** A shared four-stream MKE handcrafted encoder consumes the existing ordered 12-channel tensor as ONEHOT(4), NCP(3), EIIP(1), and ENAC(4). One flag enables the post-fusion full MKE-ECA block for v3, and one explicit dimension policy selects either asymmetric 1536+128 fusion or aligned 256+256 fusion. Version configs contain no duplicated model implementation.

**Tech Stack:** Python 3.10, PyTorch, Transformers/PEFT, pytest/unittest, existing BiM6A-FuseNet five-fold runner.

**Implementation status (2026-07-18):** Tasks 1–5 are complete and verified. Task 6 is intentionally left uncommitted until the user requests a commit; no partial implementation commit or push has been made.

---

## File Structure

- Create `src/model_mke_handcrafted.py`: residual block, branch ECA, full MKE-ECA, four-stream encoder, and dimension-policy fusion head.
- Create `src/model_birna_mke.py`: BiRNA FiLM wrapper that connects the unchanged BiRNA feature extractor to the shared MKE classifier.
- Modify `src/train_cv.py`: CLI flags, validation, model selection, and diagnostics for the MKE encoder and dimension policy.
- Modify `configs/configarg.py`: four version registrations and explicit MKE/dimension-policy fields.
- Modify `scripts/train.py`: translate version configuration into CLI flags.
- Create four `experiments/<version>/config_*.py`, `README.md`, and `__init__.py` files.
- Create `tests/test_mke_handcrafted.py`: shape, attention, schema, policy, and forward tests.
- Modify `tests/test_protocol.py`: four version command/config tests and v1/v1b regression assertions.
- Modify `README.md`: version table and runnable examples.
- Update `docs/superpowers/specs/2026-07-17-v2-v3-mke-eca-design.md`: keep the approved design with the implementation commit.

### Task 1: Four-stream encoder contracts

**Files:**
- Create: `tests/test_mke_handcrafted.py`
- Create: `src/model_mke_handcrafted.py`

- [ ] **Step 1: Write failing tests for `MKEResidualBlock`, `ECA1D`, and `FourStreamMKEEncoder`.**

Tests instantiate an input tensor shaped `(2, 12, 41)`, require branch outputs `(2,32,20)`, `(2,32,20)`, `(2,16,20)`, `(2,32,20)`, require merged output `(2,64)`, and reject wrong channel count, length, or feature order.

- [ ] **Step 2: Run the focused tests and verify failure due to the missing module.**

Run: `python -m pytest tests/test_mke_handcrafted.py -v`

Expected: collection/import failure naming `model_mke_handcrafted`.

- [ ] **Step 3: Implement the minimum repository-faithful encoder.**

Implement GroupNorm/GELU residual blocks, adaptive ECA, fixed named slices `(0:4, 4:7, 7:8, 8:12)`, branch pooling, 112-channel merge, repository dropout values `0.3/0.3/0.85`, and the 64-dimensional output.

- [ ] **Step 4: Run the focused tests and verify they pass.**

Run: `python -m pytest tests/test_mke_handcrafted.py -v`

Expected: all Task 1 tests pass.

### Task 2: Full MKE-ECA attention

**Files:**
- Modify: `tests/test_mke_handcrafted.py`
- Modify: `src/model_mke_handcrafted.py`

- [ ] **Step 1: Write failing tests for `FullMKEECA`.**

Tests require input/output shape `(2,32,20)`, channel weights broadcastable as `(2,32,1)`, spatial weights as `(2,1,20)`, kernels `3/5/7`, values in `[0,1]`, and output equal to `Hc * ws` without an external `+H` path.

- [ ] **Step 2: Run the focused test and verify the missing-class failure.**

Run: `python -m pytest tests/test_mke_handcrafted.py -v`

Expected: failure because `FullMKEECA` is not defined.

- [ ] **Step 3: Implement channel SE and multi-scale spatial attention.**

Use reduction ratio 16 for 32 channels, three `Conv1d(2,1)` spatial branches with kernels 3/5/7, `Conv1d(3,1,1)` fusion, sigmoid weights, and no residual addition.

- [ ] **Step 4: Run the focused tests and verify they pass.**

Run: `python -m pytest tests/test_mke_handcrafted.py -v`

Expected: all attention tests pass.

### Task 3: Native and proj256 fusion classifiers

**Files:**
- Modify: `tests/test_mke_handcrafted.py`
- Modify: `src/model_mke_handcrafted.py`

- [ ] **Step 1: Write failing tests for both fusion policies.**

The native policy must map handcrafted `64->128`, keep BiRNA native width, feed 1664 inputs to the current checkpoint's classifier, and return `(batch,2)`. The aligned policy must project `1536->256` and `64->256`, feed 512 inputs to its classifier, and return `(batch,2)`. Invalid policies must raise `ValueError`.

- [ ] **Step 2: Run tests and verify failure because the fusion class is missing.**

Run: `python -m pytest tests/test_mke_handcrafted.py -v`

- [ ] **Step 3: Implement a shared fusion head and BiRNA classifier wrapper.**

Reuse `BiRNAFiLMLocalClassifier._build_film_features`; do not edit v1/v1b model modules. Select full MKE-ECA and fusion policy only through constructor arguments.

- [ ] **Step 4: Run tests and verify both policies pass.**

Run: `python -m pytest tests/test_mke_handcrafted.py -v`

### Task 4: Version configs and launcher wiring

**Files:**
- Modify: `configs/configarg.py`
- Modify: `scripts/train.py`
- Modify: `src/train_cv.py`
- Create: `experiments/v2a_mke_res_eca_native/{__init__.py,config_v2a.py,README.md}`
- Create: `experiments/v2b_mke_res_eca_proj256/{__init__.py,config_v2b.py,README.md}`
- Create: `experiments/v3a_full_mke_eca_native/{__init__.py,config_v3a.py,README.md}`
- Create: `experiments/v3b_full_mke_eca_proj256/{__init__.py,config_v3b.py,README.md}`
- Modify: `tests/test_protocol.py`

- [ ] **Step 1: Write failing config and dry-run command tests.**

Require all four versions to preserve strict five-fold evaluation, seed 42, ACC selection, v1 LoRA/FiLM/training settings, ordered feature list, and distinct output directories. Require v2 to disable full MKE-ECA, v3 to enable it, native to select `native`, and aligned to select `proj256`.

- [ ] **Step 2: Run protocol tests and verify unknown-version/flag failures.**

Run: `python -m pytest tests/test_protocol.py -v`

- [ ] **Step 3: Add explicit configuration fields and CLI flags.**

Add `use_mke_handcrafted`, `use_full_mke_eca`, and `fusion_dim_policy`; add corresponding `--use_mke_handcrafted`, `--use_full_mke_eca`, and `--fusion_dim_policy {native,proj256}` flags. Validate that MKE requires FiLM, handcrafted features, exact ordered features `onehot,ncp,eiip,enac`, and cannot combine with legacy gated/projected fusion flags.

- [ ] **Step 4: Register the four experiment configs and add launcher documentation.**

Each config changes only version metadata, MKE attention level, and dimension policy from the shared v1 hyperparameters.

- [ ] **Step 5: Run protocol tests and four dry runs.**

Run: `python -m pytest tests/test_protocol.py -v`

Run each: `python train.py --version <version> --dataset H_b --seed 42 --dry_run`

Expected: commands include the correct MKE/policy flags and do not launch training.

### Task 5: Regression, real forward verification, and documentation

**Files:**
- Modify: `README.md`
- Modify: `tests/test_model_parity.py` only if new-version assertions fit its existing responsibility.

- [ ] **Step 1: Add v1/v1b negative assertions before changing shared wiring.**

Require v1/v1b commands not to include MKE flags and keep their existing protected arguments.

- [ ] **Step 2: Run the regression assertions and verify they fail only if wiring leaks into old versions.**

Run: `python -m pytest tests/test_protocol.py tests/test_model_parity.py -v`

- [ ] **Step 3: Add concise README version and run-command documentation.**

Document the four experiment IDs, their dimension policies, and the unchanged output/reporting protocol.

- [ ] **Step 4: Run CPU structural tests and a real-weight batch-size-two forward pass for all four versions.**

Use two real benchmark sequences so BatchNorm1d is valid in training mode. Verify logits `(2,2)` and no changes to the copied pretrained files.

- [ ] **Step 5: Run the full verification suite.**

Run: `python -m pytest -v`

Run: `python scripts/verify_portable.py`

Run: `git diff --check`

Run: `git status --short`

Expected: all tests pass, portable verification passes, no whitespace errors, and only intended source/config/test/doc files are modified or created.

### Task 6: Commit only complete implementations

- [ ] **Step 1: Review the diff against every section of the approved design.**

- [ ] **Step 2: Commit v2a/v2b and shared infrastructure only when both are complete.**

Use commit message: `feat: add native and aligned MKE ResNet-ECA variants`

- [ ] **Step 3: Commit v3a/v3b only when both full MKE-ECA variants are complete.**

Use commit message: `feat: add native and aligned full MKE-ECA variants`

The user requested no partial implementation commits and no design-only commit. Do not push unless separately requested.
