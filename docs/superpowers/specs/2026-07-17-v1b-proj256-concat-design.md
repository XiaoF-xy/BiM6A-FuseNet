# BiM6A-FuseNet v1b Projected-Concatenation Design

## Objective

Create `v1b_proj256_concat` as the first controlled improvement over `v1_baseline`. The experiment tests whether independently projecting the heterogeneous BiRNA-BERT and handcrafted branches to the same 256-dimensional space improves feature fusion.

The experiment does not claim that a larger branch necessarily dominates. It tests whether dimensional alignment, branch-specific transformation, and normalization improve predictive performance.

## Version Boundary

- Existing baseline: `v1_baseline` / paper label `BiM6A-FuseNet-v1`
- New experiment: `v1b_proj256_concat` / paper label `BiM6A-FuseNet-v1b`
- Existing v1 model files and configuration remain unchanged.
- v1b uses the same datasets, folds, seeds, optimizer, epochs, checkpoint policy, metrics, soft voting, and plotting pipeline as v1.

## Architecture

The existing v1 feature extractors remain unchanged:

```text
BiRNA-BERT + FiLM + local CNN -> birna_feat (1536)
ONEHOT/NCP/EIIP/ENAC CNN      -> hand_feat (128)
```

v1b adds two independent projections:

```text
birna_feat (1536)
  -> Linear(1536, 256)
  -> LayerNorm(256)
  -> GELU
  -> Dropout(0.2)
  -> z_birna (256)

hand_feat (128)
  -> Linear(128, 256)
  -> LayerNorm(256)
  -> GELU
  -> Dropout(0.2)
  -> z_hand (256)
```

The aligned representations are concatenated and passed to a classifier that preserves the v1 classifier style:

```text
concat(z_birna, z_hand) (512)
  -> Linear(512, 256)
  -> ReLU
  -> Dropout(0.2)
  -> Linear(256, 2)
```

The implementation derives `birna_feature_dim` from the BiRNA hidden size and active FiLM pooling mode, and derives the handcrafted dimension from `handcrafted_output_dim`; it does not hardcode 1536 or 128 in the reusable model class.

## Controlled-Experiment Rationale

The alternative full redesign would also change the final classifier to LayerNorm, GELU, and Dropout 0.3. That alternative may improve or reduce raw performance, but it changes several variables simultaneously. It is deferred to a later version so v1b remains interpretable.

Consequently, v1b is the better first experiment even though neither approach can be guaranteed to achieve a higher score before training.

## Implementation Boundary

- Add a new projected-concatenation model class in a new source module; do not edit the hash-protected v1 architecture modules.
- Add a minimal model-selection flag to the shared training factory.
- Add `experiments/v1b_proj256_concat/config_v1b.py` by deriving v1 settings and enabling only projected concatenation.
- Register `v1b_proj256_concat` in the version registry and launcher.
- Keep the public command form:

```bash
python train.py --version v1b_proj256_concat --dataset H_b --seed 42
```

## Evaluation Protocol

The protocol remains identical to v1:

- `benchmark.csv` only for stratified five-fold training and validation
- split seed 42; fold training seeds 42–46
- 20 epochs per fold
- best epoch selected by held-out benchmark ACC
- five fold models evaluated on `independent_test.csv`
- final independent result from aligned five-model probability soft voting
- temporary fold checkpoint deleted only after complete result validation

## Verification

1. Verify the v1 architecture hashes remain unchanged.
2. Verify the v1b config differs from v1 only by version metadata and projected-concatenation enablement.
3. Unit-test projection dimensions: 1536→256, 128→256, concat 512, logits 2.
4. Unit-test a forward pass through the new fusion head.
5. Verify the generated v1b training command contains the projected-concatenation flag and all original v1 hyperparameters.
6. Run the complete existing test suite and a real BiRNA-BERT v1b smoke forward pass.

## Success Criteria

Implementation success means v1b is independently runnable and structurally correct. Scientific success is determined only after comparing v1 and v1b under the identical evaluation protocol; no performance improvement is assumed in advance.
