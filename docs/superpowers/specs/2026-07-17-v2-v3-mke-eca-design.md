# BiM6A-FuseNet v2/v3 MKE-ECA Design

## 1. Objective

Add a two-by-two controlled experiment after `v1_baseline` and `v1b_proj256_concat`. The two factors are handcrafted encoder strength and fusion dimension policy:

1. v2 replaces the current handcrafted multi-scale CNN with the four-stream ResNet-ECA encoder adapted to the project's existing 3-channel NCP encoding.
2. v3 keeps the complete v2 handcrafted encoder and adds the paper-described full MKE-ECA module after four-stream fusion.
3. Both v2 and v3 are implemented under two fusion dimension policies:
   - `native`: preserve the asymmetric v1 fusion contract, with a 1536-dimensional BiRNA representation and a 128-dimensional handcrafted representation.
   - `proj256`: independently project both branches to 256 dimensions before concatenation, matching the v1b fusion contract.

The experiment sequence is:

```text
v1 native concat ---------------------> v2a native -----------------> v3a native
                                         |                              |
dimension-policy comparison              |                              |
                                         |                              |
v1b aligned proj256 concat -----------> v2b proj256 ---------------> v3b proj256
```

The two parallel tracks preserve single-variable comparisons while testing whether stronger handcrafted features interact with dimension alignment.

## 2. Version Naming

| Stage | Experiment ID | Paper label |
|---|---|---|
| Existing asymmetric baseline | `v1_baseline` | BiM6A-FuseNet-v1 |
| Existing aligned baseline | `v1b_proj256_concat` | BiM6A-FuseNet-v1b |
| Four-stream ResNet-ECA, asymmetric dimensions | `v2a_mke_res_eca_native` | BiM6A-FuseNet-v2a |
| Four-stream ResNet-ECA, aligned dimensions | `v2b_mke_res_eca_proj256` | BiM6A-FuseNet-v2b |
| Full MKE-ECA, asymmetric dimensions | `v3a_full_mke_eca_native` | BiM6A-FuseNet-v3a |
| Full MKE-ECA, aligned dimensions | `v3b_full_mke_eca_proj256` | BiM6A-FuseNet-v3b |

All four new versions use the project-standard 3-channel NCP representation. The NCP choice is recorded in the resolved configuration and version documentation rather than repeated in every experiment ID.

## 3. Verified Repository/Paper Differences

The MKE repository and paper differ in two relevant ways:

- The paper defines NCP as `3 x 41`.
- The repository's `calculate_chem_properties` appends cumulative nucleotide frequency, producing `4 x 41`, and its second branch therefore uses `in_channels=4`.
- The paper describes post-fusion MKE-ECA as SE channel attention followed by multi-kernel spatial attention.
- The repository implements four branch-level ECA blocks but does not implement the complete paper-described post-fusion multi-kernel spatial attention.

Decisions for this project:

- Use the existing BiM6A NCP encoding with exactly 3 channels.
- Use the repository's four-stream residual/ECA backbone in v2, changing only the NCP branch input from 4 to 3.
- Retain the four branch-level ECA blocks in v3 and add the complete paper-described MKE-ECA after fusion.
- Do not add an external `+H` residual connection to the full MKE-ECA in the first v3 experiment.

## 4. Shared BiRNA Branch

All v2 and v3 variants reuse the complete v1/v1b BiRNA feature extractor without modification:

```text
BiRNA-BERT NUC + LoRA
  -> NUC global representation
  -> FiLM-modulated center-local representation
  -> birna_feat: 1536
```

The fusion policy is applied only after `birna_feat` is produced:

```text
native:  birna_feat remains 1536-dimensional
proj256: birna_feat -> Linear + LayerNorm + GELU + Dropout(0.2) -> z_birna: 256
```

The implementation must derive the BiRNA input dimension from the loaded model/configuration, as v1b already does, rather than hardcoding 1536 in the reusable model class.

## 5. v2 Handcrafted Branch

### 5.1 Input schema

The feature order and channel counts are fixed:

```text
ONEHOT: 4 x 41
NCP:    3 x 41
EIIP:   1 x 41
ENAC:   4 x 41
Total: 12 x 41
```

The encoder must validate the total channel count and split the existing handcrafted tensor using named, fixed channel boundaries. A mismatched feature order, channel count, or sequence length must raise a clear error rather than silently entering the wrong branch.

### 5.2 Parallel repository-style branches

```text
ONEHOT 4x41 -> ResBlock 4->64 -> ResBlock 64->32 -> MaxPool -> ECA -> 32x20
NCP    3x41 -> ResBlock 3->64 -> ResBlock 64->32 -> MaxPool -> ECA -> 32x20
EIIP   1x41 -> ResBlock 1->64 -> ResBlock 64->16 -> MaxPool -> ECA -> 16x20
ENAC   4x41 -> ResBlock 4->64 -> ResBlock 64->32 -> MaxPool -> ECA -> 32x20
```

Each residual block follows the repository implementation:

```text
Conv1d(kernel=3) -> GroupNorm -> GELU
-> Conv1d(kernel=3) -> GroupNorm
-> identity or Conv1d(kernel=1) shortcut
-> add -> GELU
```

Each branch-level ECA follows the repository implementation: temporal global average pooling, adaptive odd-kernel `Conv1d(1,1,k)` across the channel descriptor, Sigmoid, then channel-wise multiplication.

### 5.3 Four-stream fusion

```text
concat: (32 + 32 + 16 + 32) x 20 = 112 x 20
  -> Dropout(0.3)
  -> ResidualBlock 112->32
  -> MaxPool: 20->10
  -> Dropout(0.3)
  -> ResidualBlock 32->16
  -> ResidualBlock 16->16
  -> Flatten: 16x10 = 160
  -> Linear 160->64
  -> BatchNorm1d(64)
  -> ReLU
  -> Dropout(0.85)
  -> hand_feat: 64
```

The first v2 experiment retains all three repository dropout probabilities (`0.3`, `0.3`, and `0.85`). In PyTorch, `Dropout(0.85)` drops 85% of activations. A lower-dropout experiment, if needed after observing underfitting, must be a separately named future ablation and must not be mixed into v2.

### 5.4 Fusion dimension policies and classifiers

The repository-style handcrafted encoder returns `hand_feat_raw: 64`. Both policies use an explicit handcrafted adapter so that the encoder implementation is identical and only the target fusion width changes.

#### Native asymmetric policy (`v2a`, `v3a`)

```text
birna_feat: 1536 (unchanged)

hand_feat_raw: 64
  -> Linear + LayerNorm + GELU + Dropout(0.2)
  -> hand_feat: 128

concat: 1536 + 128 = 1664
  -> Linear 1664->256
  -> ReLU
  -> Dropout(0.2)
  -> Linear 256->2
```

This preserves the v1 asymmetric classifier input contract. The 64-to-128 handcrafted adapter is required because the repository-style MKE encoder naturally returns 64 features whereas the existing v1 handcrafted encoder returns 128.

#### Aligned policy (`v2b`, `v3b`)

```text
birna_feat: 1536
  -> Linear + LayerNorm + GELU + Dropout(0.2)
  -> z_birna: 256

hand_feat_raw: 64
  -> Linear + LayerNorm + GELU + Dropout(0.2)
  -> z_hand: 256

concat(z_birna, z_hand): 512
  -> Linear 512->256
  -> ReLU
  -> Dropout(0.2)
  -> Linear 256->2
```

Dimension policy must be an explicit configuration value such as `fusion_dim_policy = "native" | "proj256"`. It must not be inferred from the version name.

## 6. v3 Full MKE-ECA

Each v3 variant inherits every component and parameter from its corresponding v2 dimension policy. The only architectural addition is a full MKE-ECA block inserted after the first merged residual block and before the second max-pooling operation:

```text
four branch outputs
  -> concat: 112x20
  -> Dropout(0.3)
  -> ResidualBlock 112->32
  -> H: Bx32x20
  -> Full MKE-ECA
  -> MaxPool: 20->10
  -> remaining v2 handcrafted path
```

### 6.1 Channel attention

Use the paper-described squeeze-and-excitation operation with reduction ratio `r=16`:

```text
H: Bx32x20
  -> temporal GAP: Bx32
  -> Linear 32->2
  -> ReLU
  -> Linear 2->32
  -> Sigmoid
  -> wc: Bx32x1

Hc = H * wc
```

### 6.2 Multi-scale spatial attention

```text
channel mean(Hc): Bx1x20
channel max(Hc):  Bx1x20
  -> concat: Bx2x20

Conv1d(2,1,kernel=3,padding=1) -+
Conv1d(2,1,kernel=5,padding=2) -+-> concat: Bx3x20
Conv1d(2,1,kernel=7,padding=3) -+
  -> Conv1d(3,1,kernel=1)
  -> Sigmoid
  -> ws: Bx1x20

Hout = Hc * ws
```

No external identity addition is used:

```text
Hout != H + Hc * ws
```

This choice is paper-faithful and allows the attention module to suppress uninformative channels and positions. A stabilized residual-attention variant may be tested later under a separate version name.

## 7. Training and Evaluation Protocol

v1, v1b, v2a, v2b, v3a, and v3b must use an identical evaluation protocol. Within each dimension-policy track, all training hyperparameters must also remain identical:

- Same `benchmark.csv` and `independent_test.csv` files.
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on `benchmark.csv` only.
- Fold training seeds fixed at 42-46.
- 20 epochs per fold.
- Same batch size, learning rate, optimizer, weight decay, LoRA settings, and all BiRNA settings as v1b.
- Best epoch selected by held-out benchmark validation ACC.
- Independent test data never participates in epoch/model selection.
- Five best fold models produce positive-class probabilities on the independent set.
- Final independent result uses mean-probability soft voting and computes one final set of metrics.
- Preserve ACC, MCC, AUC, AUPRC, F1, Precision, Recall/Sensitivity, and Specificity outputs.
- Preserve prediction CSVs, ROC/PR coordinate CSVs, PNG/PDF figures, audit files, and resolved configuration.
- Delete each temporary fold checkpoint only after all corresponding predictions and metrics have been written and verified as readable.

Output directories must remain version-separated:

```text
outputs/v2a_mke_res_eca_native/<dataset>/seed_42/
outputs/v2b_mke_res_eca_proj256/<dataset>/seed_42/
outputs/v3a_full_mke_eca_native/<dataset>/seed_42/
outputs/v3b_full_mke_eca_proj256/<dataset>/seed_42/
```

## 8. Implementation Boundaries

- Do not modify the hash-protected v1/v9a architecture modules.
- Reuse the v1 native classifier contract for `native` variants and the v1b BiRNA projection/final projected-concat contract for `proj256` variants.
- Put reusable MKE residual/ECA/attention/handcrafted encoder components in focused new source modules.
- Implement one shared v2/v3 model path parameterized by handcrafted-attention level and fusion dimension policy; do not duplicate model code for the four versions.
- Select v2/v3 and the dimension policy through explicit configuration flags; do not infer architectures from directory names.
- v2a and v2b must not include full post-fusion MKE-ECA.
- v3a and v3b must retain the v2 branch-level ECA blocks and add exactly one post-fusion full MKE-ECA.
- Do not add gated fusion, extra FiLM blocks, MoE, auxiliary losses, alternate dropout, early stopping, or other unrelated changes.
- Feature-wise gated fusion is a future experiment and must use an aligned latent space. It is not part of v2 or v3.

## 9. Verification Requirements

### 9.1 Feature/schema tests

- Confirm real feature shapes are `(4,41)`, `(3,41)`, `(1,41)`, and `(4,41)`.
- Confirm the concatenated input is `(12,41)` and is split into the correct named branches.
- Reject incorrect feature order, channel count, and sequence length.

### 9.2 v2 structural tests

- Verify branch outputs are `32x20`, `32x20`, `16x20`, and `32x20`.
- Verify concatenation is `112x20`.
- Verify the merged path produces `32x10`, then `16x10`, then a 160-dimensional flattened vector.
- Verify the repository-style handcrafted encoder returns 64 dimensions under both policies.
- Verify the native adapter returns 128 dimensions, keeps BiRNA at its derived native dimension, and presents 1664 features to the classifier for the current 768-hidden-size checkpoint.
- Verify the aligned projections independently return 256 dimensions and present 512 features to the classifier.
- Verify all four final models return logits with shape `(batch,2)`.
- Verify v2a and v2b use identical handcrafted encoders before the dimension-policy adapters.

### 9.3 v3 attention tests

- Verify full MKE-ECA input/output shape is unchanged at `Bx32x20`.
- Verify `wc` is broadcastable as `Bx32x1` and `ws` as `Bx1x20`.
- Verify the spatial convolution branches use kernels 3, 5, and 7 and preserve sequence length 20.
- Verify the module output is `Hc * ws` and has no external `+H` identity path.
- Verify v2a and v3a differ only by full MKE-ECA enablement and version/output metadata.
- Verify v2b and v3b differ only by full MKE-ECA enablement and version/output metadata.
- Verify the `a`/`b` pair at each stage differs only in dimension-policy adapters, classifier input width, and version/output metadata.

### 9.4 Integration and regression tests

- Run a real forward pass using the copied BiRNA-BERT weight and benchmark samples for all four new versions.
- Verify dry-run commands retain five folds, 20 epochs, seed 42, ACC selection, LoRA, FiLM, and all v1b hyperparameters.
- Re-run the complete existing test suite and protected v1 source hashes.
- Confirm v1/v1b commands do not enable either new handcrafted architecture or a new dimension policy.
- Confirm output files still support the existing ROC/PR and paper-table workflow.

## 10. Implementation and Commit Sequence

Implementation is performed one complete version at a time:

1. Implement the shared v2 components plus both v2a/v2b configurations, launcher wiring, tests, documentation, and real-model verification.
2. Only after both v2 dimension policies are complete and verified, create one v2 implementation commit.
3. Implement the shared full MKE-ECA component plus both v3a/v3b configurations, tests, documentation, and real-model verification on top of v2.
4. Only after both v3 dimension policies are complete and verified, create one v3 implementation commit.

Do not create a design-only commit or intermediate partial implementation commits. This design file remains uncommitted until it is included with the first completed version, unless the user later gives different instructions.

## 11. Scientific Interpretation

The intended comparisons are:

- `v1 -> v2a`: does four-stream repository-style ResNet-ECA extraction improve the asymmetric/native-width fusion track?
- `v1b -> v2b`: does the same handcrafted encoder improvement help under aligned 256-dimensional fusion?
- `v2a -> v3a`: does full MKE-ECA improve the native-width track?
- `v2b -> v3b`: does full MKE-ECA improve the aligned track?
- `v2a <-> v2b` and `v3a <-> v3b`: does dimension alignment become beneficial after strengthening the handcrafted branch?

The four versions form a two-by-two factorial comparison:

```text
handcrafted attention: branch ECA only | full post-fusion MKE-ECA
fusion dimensions:     native/asymmetric | aligned proj256
```

The aligned track additionally establishes a stable interface for future feature-wise gating, additive fusion, Hadamard interaction, cross-branch consistency loss, and other operations that require equal-width latent representations. The native track remains necessary to measure whether the 1536-to-256 BiRNA bottleneck discards useful information.

No performance improvement is assumed in advance. Conclusions must be based on the identical cross-validation and independent-test protocol defined above.
