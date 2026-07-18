# v2c Official Handcrafted-Only MKE Implementation Plan

**Goal:** Add a source-faithful 13-channel pure handcrafted MKE-ResNet experiment under the existing strict five-fold and independent soft-voting protocol without changing v1-v3.

**Architecture:** A dedicated feature module builds ONEHOT4 + CHEM4 + EIIP1 + ENAC4. A dedicated classifier reproduces the public four-stream GroupNorm/GELU residual network, branch ECA, merged residual stack, and 64 -> 32 -> 2 head. The shared CV runner selects it only through an explicit flag and receives opt-in Adam, ReduceLROnPlateau, and early-stopping settings.

## Task 1: Specify feature and model contracts with failing tests

- Create `tests/test_mke_official_handcrafted.py`.
- Test the CHEM4 cumulative-frequency values and 13-channel ordering.
- Test exact model split shapes, output shape, normalization/activation family, dropout values, and malformed input rejection.
- Run the focused tests and confirm they fail because the new modules do not exist.

## Task 2: Implement isolated official-compatible modules

- Create `src/mke_official_features.py` without modifying the hash-protected `src/handcrafted_features.py`.
- Create `src/model_mke_official.py` without modifying the existing 12-channel fusion encoder.
- Run the focused tests to green.

## Task 3: Specify and implement the v2c protocol wiring

- Extend `tests/test_protocol.py` with v2c configuration and dry-run assertions plus old-version isolation checks.
- Add opt-in training fields to `TrainConfig` with legacy-preserving defaults.
- Add `experiments/v2c_mke_handcrafted_only_official4c/config_v2c.py` and register it.
- Add explicit official-model, optimizer, scheduler, and early-stopping CLI options.
- Add a tokenizer-free handcrafted collator and skip loading BiRNA tokenizer for the pure official version.
- Instantiate Adam only for v2c; retain AdamW by default.
- Step ReduceLROnPlateau on validation loss and stop after 20 non-improving ACC epochs only when configured.
- Keep checkpoint selection, fold predictions, soft voting, plotting, and checkpoint deletion unchanged.

## Task 4: Documentation and portability

- Update README experiment and command documentation.
- Extend the portable file manifest if the verification script uses an explicit source list.
- Verify `python train.py --version v2c_mke_handcrafted_only_official4c --dataset H_b --seed 42 --dry_run`.

## Task 5: Verification

- Run focused feature/model tests.
- Run protocol/config tests.
- Run the complete test suite.
- Run portable verification and inspect `git diff --check` and `git status --short`.
- Do not commit or push unless the user explicitly requests it.
