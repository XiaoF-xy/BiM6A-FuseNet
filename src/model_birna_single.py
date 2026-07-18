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
    ):
        super().__init__()
        self.birna_model = load_birna_backbone(Path(model_dir))
        self.use_lora = bool(use_lora)

        hidden_size = int(getattr(self.birna_model.config, "hidden_size", EXPECTED_HIDDEN_SIZE))
        if hidden_size != EXPECTED_HIDDEN_SIZE:
            raise ValueError(
                f"Pure BiRNA-BERT single-branch experiments require hidden_size={EXPECTED_HIDDEN_SIZE}, "
                f"got {hidden_size}."
            )

        if self.use_lora:
            targets = lora_target_modules or ["Wqkv"]
            validate_lora_target_modules(self.birna_model, targets)
            self.birna_model = apply_lora_to_birna(
                birna_model=self.birna_model,
                target_modules=targets,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
        elif freeze_backbone:
            for parameter in self.birna_model.parameters():
                parameter.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(EXPECTED_HIDDEN_SIZE, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if attention_mask is None:
            raise ValueError("Pure BiRNA-BERT single-branch input requires attention_mask.")
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
