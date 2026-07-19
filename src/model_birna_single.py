from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

try:
    from .model_birna_nuc import apply_lora_to_birna, load_birna_backbone
except ImportError:  # pragma: no cover - direct src/train_cv.py execution
    from model_birna_nuc import apply_lora_to_birna, load_birna_backbone


EXPECTED_NUCLEOTIDE_COUNT = 41
EXPECTED_HIDDEN_SIZE = 768


def apply_dora_to_birna(
    birna_model: nn.Module,
    target_modules: list[str],
    r: int,
    alpha: int,
    dropout: float,
) -> nn.Module:
    """Apply PEFT DoRA without changing the legacy v9a LoRA helper."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "PEFT with DoRA support is required. Install the pinned project dependencies "
            "with: pip install -r requirements.txt"
        ) from exc

    dora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        use_dora=True,
    )
    return get_peft_model(birna_model, dora_config)


def validate_lora_target_modules(
    model: nn.Module,
    target_modules: list[str],
) -> dict[str, list[str]]:
    """Require every configured suffix to match at least one linear backbone module."""
    linear_module_names = [
        name for name, module in model.named_modules() if name and isinstance(module, nn.Linear)
    ]
    matches = {
        target: [name for name in linear_module_names if name.endswith(target)]
        for target in target_modules
    }
    missing = [target for target, names in matches.items() if not names]
    if missing:
        available = ", ".join(linear_module_names[:20]) or "<none>"
        raise ValueError(
            "LoRA target modules did not match a linear BiRNA-BERT module: "
            f"{missing}. Available linear modules include: {available}"
        )
    return matches


def nucleotide_content_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return a mask containing only NUC tokens, excluding CLS, SEP, and padding."""
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected attention_mask with shape [B, L], got {tuple(attention_mask.shape)}")
    if attention_mask.size(1) < 3:
        raise ValueError("NUC input requires CLS, at least one nucleotide, and SEP.")

    active_lengths = attention_mask.long().sum(dim=1)
    if torch.any(active_lengths < 3):
        raise ValueError("Every NUC input requires CLS, at least one nucleotide, and SEP.")

    mask = attention_mask.bool().clone()
    mask[:, 0] = False
    sep_indices = active_lengths - 1
    mask.scatter_(1, sep_indices.unsqueeze(1), False)
    return mask


def masked_mean_nucleotide_embeddings(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool exactly 41 nucleotide embeddings for each sequence."""
    if embeddings.ndim != 3:
        raise ValueError(f"Expected embeddings with shape [B, L, H], got {tuple(embeddings.shape)}")
    if tuple(embeddings.shape[:2]) != tuple(attention_mask.shape):
        raise ValueError(
            "Embedding and attention-mask dimensions must match: "
            f"embeddings={tuple(embeddings.shape)}, attention_mask={tuple(attention_mask.shape)}"
        )

    content_mask = nucleotide_content_mask(attention_mask)
    content_counts = content_mask.sum(dim=1)
    if torch.any(content_counts != EXPECTED_NUCLEOTIDE_COUNT):
        raise ValueError(
            f"Expected exactly {EXPECTED_NUCLEOTIDE_COUNT} NUC content tokens per sequence, "
            f"got {content_counts.detach().cpu().tolist()}. Check NUC tokenization and max_length."
        )

    weighted_sum = (embeddings * content_mask.unsqueeze(-1)).sum(dim=1)
    return weighted_sum / content_counts.to(embeddings.dtype).unsqueeze(1)


def masked_mean_packed_nucleotide_embeddings(
    packed_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool 41 NUC tokens from the encoder's packed, unpadded output."""
    if packed_embeddings.ndim != 2:
        raise ValueError(
            "Expected packed embeddings with shape [active_tokens, hidden], "
            f"got {tuple(packed_embeddings.shape)}"
        )
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected attention_mask with shape [B, L], got {tuple(attention_mask.shape)}")

    active_lengths = attention_mask.long().sum(dim=1)
    if torch.any(active_lengths < 3):
        raise ValueError("Every NUC input requires CLS, at least one nucleotide, and SEP.")
    expected_packed_tokens = int(active_lengths.sum().item())
    if packed_embeddings.size(0) != expected_packed_tokens:
        raise ValueError(
            "Packed encoder output must contain one row per active token: "
            f"got {packed_embeddings.size(0)}, expected {expected_packed_tokens}."
        )

    samples = torch.split(packed_embeddings, active_lengths.detach().cpu().tolist(), dim=0)
    content_embeddings = [sample[1:-1] for sample in samples]
    content_counts = [sample.size(0) for sample in content_embeddings]
    if any(count != EXPECTED_NUCLEOTIDE_COUNT for count in content_counts):
        raise ValueError(
            f"Expected exactly {EXPECTED_NUCLEOTIDE_COUNT} NUC content tokens per sequence, "
            f"got {content_counts}. Check NUC tokenization and max_length."
        )
    return torch.stack([sample.mean(dim=0) for sample in content_embeddings], dim=0)


class LastFourLayerScalarMix(nn.Module):
    """Learn a softmax-normalized mixture of the final four encoder layers."""

    def __init__(self, hidden_size: int = EXPECTED_HIDDEN_SIZE):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.scalar_logits = nn.Parameter(torch.tensor([-6.0, -6.0, -6.0, 0.0]))

    def normalized_weights(self) -> torch.Tensor:
        return torch.softmax(self.scalar_logits, dim=0)

    def forward(self, hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(hidden_states) < 4:
            raise ValueError(f"Last-four-layer scalar mix requires at least 4 layers, got {len(hidden_states)}.")
        selected = list(hidden_states[-4:])
        expected_shape = tuple(selected[0].shape)
        if any(tuple(layer.shape) != expected_shape for layer in selected[1:]):
            raise ValueError(
                "Last-four-layer scalar mix requires every selected layer to have the same shape; "
                f"got {[tuple(layer.shape) for layer in selected]}."
            )
        if not expected_shape or expected_shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected selected layers ending in hidden_size={self.hidden_size}, got {expected_shape}."
            )

        stacked = torch.stack(selected, dim=0)
        weights = self.normalized_weights().to(dtype=stacked.dtype)
        weight_shape = (len(selected),) + (1,) * (stacked.ndim - 1)
        return (stacked * weights.view(weight_shape)).sum(dim=0)


class BiRNASingleBranchClassifier(nn.Module):
    """Pure BiRNA-BERT NUC classifier shared by LoRA and full fine-tuning."""

    def __init__(
        self,
        model_dir: Path | str,
        freeze_backbone: bool,
        use_lora: bool,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        dropout: float = 0.2,
        use_last4_scalar_mix: bool = False,
        use_dora: bool = False,
    ):
        super().__init__()
        if use_dora and not use_lora:
            raise ValueError("DoRA requires use_lora=True.")
        self.birna_model = load_birna_backbone(Path(model_dir))
        self.use_lora = bool(use_lora)
        self.use_last4_scalar_mix = bool(use_last4_scalar_mix)

        hidden_size = int(getattr(self.birna_model.config, "hidden_size", EXPECTED_HIDDEN_SIZE))
        if hidden_size != EXPECTED_HIDDEN_SIZE:
            raise ValueError(
                f"Pure BiRNA-BERT single-branch experiments require hidden_size={EXPECTED_HIDDEN_SIZE}, "
                f"got {hidden_size}."
            )

        if self.use_lora:
            targets = lora_target_modules or ["Wqkv"]
            validate_lora_target_modules(self.birna_model, targets)
            adapter_factory = apply_dora_to_birna if use_dora else apply_lora_to_birna
            self.birna_model = adapter_factory(
                birna_model=self.birna_model,
                target_modules=targets,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
        elif freeze_backbone:
            for parameter in self.birna_model.parameters():
                parameter.requires_grad = False

        self.scalar_mix = (
            LastFourLayerScalarMix(hidden_size=hidden_size)
            if self.use_last4_scalar_mix
            else None
        )

        self.classifier = nn.Sequential(
            nn.Linear(EXPECTED_HIDDEN_SIZE, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if attention_mask is None:
            raise ValueError("Pure BiRNA-BERT single-branch input requires attention_mask.")
        if self.scalar_mix is not None:
            base_model = self.birna_model
            get_base_model = getattr(base_model, "get_base_model", None)
            if callable(get_base_model):
                base_model = get_base_model()
            encoder = getattr(base_model, "bert", None)
            if encoder is None:
                raise ValueError(
                    "Last-four-layer scalar mix requires the bundled BiRNA-BERT model to expose .bert."
                )
            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_all_encoded_layers=True,
            )
            hidden_states = encoder_outputs[0]
            if not isinstance(hidden_states, (list, tuple)):
                raise ValueError(
                    "Last-four-layer scalar mix expected a list of BiRNA-BERT encoder-layer outputs."
                )
            mixed_embeddings = self.scalar_mix(hidden_states)
            pooled = masked_mean_packed_nucleotide_embeddings(mixed_embeddings, attention_mask)
            return self.classifier(pooled)

        outputs = self.birna_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        embeddings = outputs.logits
        if embeddings.size(-1) != EXPECTED_HIDDEN_SIZE:
            raise ValueError(
                f"Expected BiRNA-BERT embeddings ending in {EXPECTED_HIDDEN_SIZE}, "
                f"got {tuple(embeddings.shape)}"
            )
        pooled = masked_mean_nucleotide_embeddings(embeddings, attention_mask)
        return self.classifier(pooled)
