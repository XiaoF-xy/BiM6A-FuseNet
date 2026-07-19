# BiRNA-BERT Last-Four-Layer Mix and DoRA Design

## Goal

Add two isolated pure BiRNA-BERT experiments that may improve the current `v0a_birna_nuc_lora` baseline while keeping the MKE-comparable evaluation protocol unchanged.

The two experiments answer different questions:

- `v0f_birna_last4_scalar_mix`: does a learnable mixture of the last four BiRNA-BERT layers provide a better sequence representation than using only the final layer?
- `v0g_birna_nuc_dora`: does DoRA adapt the same Wqkv modules more effectively than ordinary LoRA?

## Fixed Experimental Protocol

Both versions preserve the v0a controls:

- data: the current MKE benchmark and independent-test files;
- benchmark evaluation: stratified five-fold cross-validation with seed 42;
- fold training seeds: 42 through 46;
- checkpoint selection: highest benchmark validation ACC within each fold;
- independent evaluation: soft voting over the five selected fold models;
- epochs: 20;
- batch size: 32;
- optimizer: AdamW;
- learning rate: `1e-4`;
- weight decay: `0.01`;
- warmup, early stopping, and time-varying scheduler: disabled;
- classifier: `768 -> 256 -> 2`, GELU, dropout 0.2;
- tokenization: NUC view with exactly 41 nucleotide tokens;
- LoRA/DoRA target: Wqkv only, rank 8, alpha 32, dropout 0.05.

Existing versions `v0a` through `v0e` must retain their current configuration and behavior.

## Version v0f: Last-Four-Layer Scalar Mix

### Architecture

The BiRNA-BERT encoder produces the outputs of transformer layers 9 through 12. Four trainable scalar logits are converted to normalized weights with softmax:

```text
h9, h10, h11, h12
       | softmax(a1, a2, a3, a4)
       v
H_mix = w1*h9 + w2*h10 + w3*h11 + w4*h12
       | remove CLS, SEP, and padding
       | mean over exactly 41 nucleotide tokens
       v
768 -> 256 -> 2
```

The scalar logits are initialized to `[-6, -6, -6, 0]`. This starts the model close to v0a's final-layer representation while still giving all four layers nonzero gradients. Only four additional trainable parameters are introduced.

### BiRNA-BERT compatibility

The bundled custom BiRNA-BERT `BertForMaskedLM` always returns `hidden_states=None`, even when `output_hidden_states=True`. Therefore v0f must not rely on the standard Transformers `MaskedLMOutput.hidden_states` field.

Instead, the single-branch classifier accesses the wrapped base model's `bert` module and calls it with `output_all_encoded_layers=True`. The custom encoder returns packed, unpadded per-layer tensors. The implementation must:

1. require at least four encoder-layer outputs;
2. require all selected layer tensors to have the same shape;
3. require the packed token count to equal `attention_mask.sum()`;
4. split the packed representation using each sample's active token count;
5. require every sample to contain CLS, exactly 41 nucleotide tokens, and SEP;
6. remove the first and last active token and mean-pool the remaining 41 tokens.

Any mismatch raises a descriptive `ValueError`; there is no silent fallback to the final layer.

### Isolation

A new model option, disabled by default, activates scalar mixing only for v0f. When the option is false, `BiRNASingleBranchClassifier.forward` continues to use the existing v0a path unchanged.

## Version v0g: Wqkv DoRA

### Architecture

v0g keeps v0a's final-layer masked-mean representation and classifier unchanged. Its only model difference from v0a is PEFT's DoRA decomposition on the same Wqkv projections:

```text
BiRNA-BERT final layer
       | Wqkv DoRA, r=8, alpha=32, dropout=0.05
       | masked mean over 41 nucleotides
       v
768 -> 256 -> 2
```

The project pins `peft==0.11.1`, whose `LoraConfig` supports `use_dora=True`. A new `use_dora` option is passed into `LoraConfig`; it defaults to false so all existing LoRA versions remain unchanged.

DoRA is mutually dependent on LoRA configuration: requesting DoRA without `use_lora` is invalid and must fail before training starts.

## Configuration and Launching

Create versioned experiment directories:

- `experiments/v0f_birna_last4_scalar_mix/`
- `experiments/v0g_birna_nuc_dora/`

Each directory contains `__init__.py`, a version config, and a README with three-GPU background commands. Outputs follow the existing central convention:

```text
outputs/<version>/<human_brain|human_kidney|human_liver>/seed_42/
```

The launcher exposes explicit flags for the two behaviors so `resolved_config.json` and saved checkpoint arguments record the exact experiment:

- `--use_last4_scalar_mix`
- `--use_dora`

No new terminal-log directory is required; users may redirect terminal output with `nohup` independently of experiment outputs.

## Testing

Tests must cover:

- scalar weights normalize to one and the initial state is final-layer dominant;
- packed last-four-layer mixing pools exactly 41 nucleotides and returns two logits;
- insufficient layers, mismatched packed shapes, or invalid token counts fail clearly;
- v0f changes only scalar-mix behavior relative to v0a;
- v0g changes only DoRA behavior relative to v0a;
- DoRA is forwarded as `use_dora=True` to PEFT and ordinary LoRA remains false;
- both version commands contain their own flag and do not leak flags into v0a-v0e or MKE fusion versions;
- portable-server verification requires both new experiment configs.

Verification includes focused unit tests, the full available test suite, Python compilation, dry-run commands, and the portable-folder check. If the local machine lacks PyTorch or pytest, dependency-free checks must still run and the unavailable checks must be reported rather than claimed.

## Success Criteria

Implementation success means both versions are independently runnable without changing prior versions and generate the same metrics, prediction CSVs, ROC/PR data, checkpoint deletion behavior, and independent soft-voting artifacts as the existing strict-CV pipeline.

Scientific success remains an empirical gate rather than an implementation guarantee. Run human brain first and compare against the v0a benchmark means:

- ACC: `0.7435`;
- MCC: `0.4908`.

Kidney and liver runs are worthwhile only after interpreting the human-brain result, although the code and launch commands support all three datasets immediately.
