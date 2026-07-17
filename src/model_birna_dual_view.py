from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model_birna_nuc import apply_lora_to_birna, load_birna_backbone


def mask_aware_mean_pool(emb: torch.Tensor, content_mask: torch.Tensor) -> torch.Tensor:
    token_mask = content_mask.unsqueeze(-1).to(dtype=emb.dtype, device=emb.device)
    return (emb * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1.0)


class BiRNADualViewClassifier(nn.Module):
    def __init__(
        self,
        model_dir: Path,
        freeze_backbone: bool = True,
        dropout: float = 0.2,
        center_index: int = 20,
        use_center_pooling: bool = True,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
    ):
        super().__init__()
        self.birna_model = load_birna_backbone(model_dir)
        self.use_lora = use_lora
        self.center_index = center_index
        self.use_center_pooling = use_center_pooling
        hidden_size = int(getattr(self.birna_model.config, "hidden_size", 768))
        if use_lora:
            self.birna_model = apply_lora_to_birna(
                birna_model=self.birna_model,
                target_modules=lora_target_modules or ["Wqkv"],
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
        classifier_input_size = hidden_size * (3 if use_center_pooling else 2)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )
        if freeze_backbone and not use_lora:
            for parameter in self.birna_model.parameters():
                parameter.requires_grad = False

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.birna_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        emb = outputs.logits
        if emb.ndim != 3:
            raise ValueError(f"Expected BiRNA-BERT token embeddings with shape [B, L, H], got: {tuple(emb.shape)}")
        return emb

    def forward(
        self,
        nuc_input_ids,
        nuc_attention_mask=None,
        nuc_token_type_ids=None,
        nuc_content_mask=None,
        bpe_input_ids=None,
        bpe_attention_mask=None,
        bpe_token_type_ids=None,
        bpe_content_mask=None,
    ):
        if bpe_input_ids is None:
            raise ValueError("Dual-view classifier requires bpe_input_ids from DualViewDataCollator.")
        if nuc_content_mask is None or bpe_content_mask is None:
            raise ValueError("Dual-view classifier requires nuc_content_mask and bpe_content_mask.")

        nuc_emb = self._encode(
            input_ids=nuc_input_ids,
            attention_mask=nuc_attention_mask,
            token_type_ids=nuc_token_type_ids,
        )
        if self.use_center_pooling and int(nuc_content_mask.sum(dim=1).min().item()) <= self.center_index:
            raise ValueError(
                "BiRNA-BERT NUC output is too short for 41bp center pooling: "
                f"min_content_token_count={int(nuc_content_mask.sum(dim=1).min().item())}, "
                f"center_index={self.center_index}. "
                "Check NUC tokenization and max_length."
            )
        nuc_mean = mask_aware_mean_pool(nuc_emb, nuc_content_mask)

        bpe_emb = self._encode(
            input_ids=bpe_input_ids,
            attention_mask=bpe_attention_mask,
            token_type_ids=bpe_token_type_ids,
        )
        if int(bpe_content_mask.sum(dim=1).min().item()) == 0:
            raise ValueError("BiRNA-BERT BPE output has no non-special tokens. Check BPE tokenization.")
        bpe_mean = mask_aware_mean_pool(bpe_emb, bpe_content_mask)

        if self.use_center_pooling:
            nuc_center = nuc_emb[:, self.center_index + 1, :]
            feat = torch.cat([nuc_mean, nuc_center, bpe_mean], dim=1)
        else:
            feat = torch.cat([nuc_mean, bpe_mean], dim=1)
        return self.classifier(feat)
