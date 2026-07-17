from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model_birna_dual_view import mask_aware_mean_pool
from model_birna_nuc import apply_lora_to_birna, load_birna_backbone


FILM_NUC_POOLING_MODES = {
    "center_mean",
    "full_mean",
    "center_cnn_mean",
    "full_cnn_mean",
    "full_mean_center_cnn_mean",
}


class FiLM(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim * 2),
        )

    def forward(self, global_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.mlp(global_feat)
        return torch.chunk(params, 2, dim=-1)


class BiRNAFiLMLocalClassifier(nn.Module):
    def __init__(
        self,
        model_dir: Path,
        freeze_backbone: bool = True,
        dropout: float = 0.2,
        center_index: int = 20,
        local_window_radius: int = 3,
        film_global_view: str = "bpe",
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        film_nuc_pooling: str = "center_mean",
        cnn_kernel_sizes: list[int] | None = None,
    ):
        super().__init__()
        if film_global_view not in {"bpe", "nuc"}:
            raise ValueError(f"film_global_view must be 'bpe' or 'nuc', got: {film_global_view}")
        if film_nuc_pooling not in FILM_NUC_POOLING_MODES:
            supported = ", ".join(sorted(FILM_NUC_POOLING_MODES))
            raise ValueError(f"film_nuc_pooling must be one of [{supported}], got: {film_nuc_pooling}")
        if local_window_radius < 0:
            raise ValueError(f"local_window_radius must be non-negative, got: {local_window_radius}")
        cnn_kernel_sizes = cnn_kernel_sizes or [3, 5, 7]
        if not cnn_kernel_sizes:
            raise ValueError("cnn_kernel_sizes must contain at least one kernel size.")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in cnn_kernel_sizes):
            raise ValueError(f"cnn_kernel_sizes must be positive odd integers, got: {cnn_kernel_sizes}")

        self.birna_model = load_birna_backbone(model_dir)
        self.use_lora = use_lora
        self.center_index = center_index
        self.local_window_radius = local_window_radius
        self.film_global_view = film_global_view
        self.film_nuc_pooling = film_nuc_pooling
        self.cnn_kernel_sizes = cnn_kernel_sizes
        hidden_size = int(getattr(self.birna_model.config, "hidden_size", 768))

        if use_lora:
            self.birna_model = apply_lora_to_birna(
                birna_model=self.birna_model,
                target_modules=lora_target_modules or ["Wqkv"],
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

        if self._uses_cnn_branch:
            self.cnn_layers = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=hidden_size,
                        out_channels=hidden_size,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    )
                    for kernel_size in cnn_kernel_sizes
                ]
            )
            self.cnn_activation = nn.GELU()
            self.cnn_dropout = nn.Dropout(dropout)
            self.cnn_projection = nn.Sequential(
                nn.Linear(hidden_size * len(cnn_kernel_sizes), hidden_size),
                nn.LayerNorm(hidden_size),
            )
        if self._uses_dual_local_branch:
            self.full_film = FiLM(input_dim=hidden_size, output_dim=hidden_size)
            self.center_cnn_film = FiLM(input_dim=hidden_size, output_dim=hidden_size)
            classifier_input_size = hidden_size * 3
        else:
            self.film = FiLM(input_dim=hidden_size, output_dim=hidden_size)
            classifier_input_size = hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

        if freeze_backbone and not use_lora:
            for parameter in self.birna_model.parameters():
                parameter.requires_grad = False

    @property
    def _uses_cnn_branch(self) -> bool:
        return self.film_nuc_pooling in {
            "center_cnn_mean",
            "full_cnn_mean",
            "full_mean_center_cnn_mean",
        }

    @property
    def _uses_center_window(self) -> bool:
        return self.film_nuc_pooling in {"center_mean", "center_cnn_mean", "full_mean_center_cnn_mean"}

    @property
    def _uses_dual_local_branch(self) -> bool:
        return self.film_nuc_pooling == "full_mean_center_cnn_mean"

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

    def _nuc_content_embeddings(self, nuc_emb: torch.Tensor, nuc_content_mask: torch.Tensor) -> torch.Tensor:
        token_counts = nuc_content_mask.sum(dim=1)
        expected_tokens = int(token_counts.min().item())
        if expected_tokens <= 0:
            raise ValueError("BiRNA-BERT NUC output has no non-special tokens. Check NUC tokenization.")
        if int(token_counts.max().item()) != expected_tokens:
            raise ValueError(
                "FiLM local pooling expects fixed-length NUC content tokens within a batch. "
                f"Observed min={expected_tokens}, max={int(token_counts.max().item())}."
            )
        return nuc_emb[:, 1 : 1 + expected_tokens, :]

    def _center_window_bounds(self, token_count: int) -> tuple[int, int]:
        local_start = self.center_index - self.local_window_radius
        local_end = self.center_index + self.local_window_radius + 1
        if local_start < 0 or token_count < local_end:
            raise ValueError(
                "BiRNA-BERT NUC output is too short for local FiLM pooling: "
                f"content_token_count={token_count}, local_window=[{local_start}, {local_end}). "
                "Check NUC tokenization, sequence length, max_length, and local_window_radius."
            )
        return local_start, local_end

    def _cnn_feature_map(self, nuc_token_emb: torch.Tensor) -> torch.Tensor:
        cnn_input = nuc_token_emb.transpose(1, 2)
        cnn_features = [conv_layer(cnn_input) for conv_layer in self.cnn_layers]
        cnn_map = torch.cat(cnn_features, dim=1)
        cnn_map = self.cnn_activation(cnn_map)
        return self.cnn_dropout(cnn_map)

    def _pool_center_cnn_branch(self, nuc_token_emb: torch.Tensor) -> torch.Tensor:
        token_count = nuc_token_emb.size(1)
        local_start, local_end = self._center_window_bounds(token_count)
        cnn_map = self._cnn_feature_map(nuc_token_emb)
        pooled = cnn_map[:, :, local_start:local_end].mean(dim=2)
        return self.cnn_projection(pooled)

    def _pool_film_nuc_branch(self, nuc_token_emb: torch.Tensor) -> torch.Tensor:
        token_count = nuc_token_emb.size(1)
        if self.film_nuc_pooling == "full_mean":
            return nuc_token_emb.mean(dim=1)

        if self.film_nuc_pooling == "center_mean":
            local_start, local_end = self._center_window_bounds(token_count)
            return nuc_token_emb[:, local_start:local_end, :].mean(dim=1)

        if self.film_nuc_pooling == "center_cnn_mean":
            return self._pool_center_cnn_branch(nuc_token_emb)
        elif self.film_nuc_pooling == "full_cnn_mean":
            cnn_map = self._cnn_feature_map(nuc_token_emb)
            pooled = cnn_map.mean(dim=2)
        else:
            raise ValueError(f"Unsupported film_nuc_pooling: {self.film_nuc_pooling}")
        return self.cnn_projection(pooled)

    def _build_film_features(
        self,
        nuc_input_ids=None,
        nuc_attention_mask=None,
        nuc_token_type_ids=None,
        nuc_content_mask=None,
        bpe_input_ids=None,
        bpe_attention_mask=None,
        bpe_token_type_ids=None,
        bpe_content_mask=None,
    ) -> torch.Tensor:
        if nuc_input_ids is None or nuc_content_mask is None:
            raise ValueError("FiLM classifier requires DualViewDataCollator outputs with nuc_input_ids and nuc_content_mask.")

        nuc_emb = self._encode(
            input_ids=nuc_input_ids,
            attention_mask=nuc_attention_mask,
            token_type_ids=nuc_token_type_ids,
        )
        nuc_token_emb = self._nuc_content_embeddings(nuc_emb, nuc_content_mask)

        if self.film_global_view == "bpe":
            if bpe_input_ids is None or bpe_content_mask is None:
                raise ValueError("BPE-global FiLM requires bpe_input_ids and bpe_content_mask.")
            bpe_emb = self._encode(
                input_ids=bpe_input_ids,
                attention_mask=bpe_attention_mask,
                token_type_ids=bpe_token_type_ids,
            )
            if int(bpe_content_mask.sum(dim=1).min().item()) == 0:
                raise ValueError("BiRNA-BERT BPE output has no non-special tokens. Check BPE tokenization.")
            h_global = mask_aware_mean_pool(bpe_emb, bpe_content_mask)
        else:
            h_global = mask_aware_mean_pool(nuc_emb, nuc_content_mask)

        if self._uses_dual_local_branch:
            h_full = nuc_token_emb.mean(dim=1)
            h_center_cnn = self._pool_center_cnn_branch(nuc_token_emb)

            gamma_full, beta_full = self.full_film(h_global)
            gamma_center, beta_center = self.center_cnn_film(h_global)
            h_full_mod = gamma_full * h_full + beta_full
            h_center_mod = gamma_center * h_center_cnn + beta_center
            feat = torch.cat([h_global, h_full_mod, h_center_mod], dim=1)
        else:
            h_local = self._pool_film_nuc_branch(nuc_token_emb)
            gamma, beta = self.film(h_global)
            h_mod = gamma * h_local + beta
            feat = torch.cat([h_global, h_mod], dim=1)
        return feat

    def forward(
        self,
        nuc_input_ids=None,
        nuc_attention_mask=None,
        nuc_token_type_ids=None,
        nuc_content_mask=None,
        bpe_input_ids=None,
        bpe_attention_mask=None,
        bpe_token_type_ids=None,
        bpe_content_mask=None,
    ):
        feat = self._build_film_features(
            nuc_input_ids=nuc_input_ids,
            nuc_attention_mask=nuc_attention_mask,
            nuc_token_type_ids=nuc_token_type_ids,
            nuc_content_mask=nuc_content_mask,
            bpe_input_ids=bpe_input_ids,
            bpe_attention_mask=bpe_attention_mask,
            bpe_token_type_ids=bpe_token_type_ids,
            bpe_content_mask=bpe_content_mask,
        )
        return self.classifier(feat)


class HandcraftedFeatureCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 12,
        cnn_channels: int = 64,
        output_dim: int = 128,
        kernel_sizes: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        kernel_sizes = kernel_sizes or [3, 5, 7]
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in kernel_sizes):
            raise ValueError(f"kernel_sizes must be positive odd integers, got: {kernel_sizes}")
        self.input_channels = input_channels
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=input_channels,
                    out_channels=cnn_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                )
                for kernel in kernel_sizes
            ]
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(cnn_channels * len(kernel_sizes), output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, handcrafted_features: torch.Tensor) -> torch.Tensor:
        if handcrafted_features.ndim != 3:
            raise ValueError(
                "handcrafted_features must have shape [B, C, L], "
                f"got: {tuple(handcrafted_features.shape)}"
            )
        if handcrafted_features.size(1) != self.input_channels:
            raise ValueError(
                f"Expected handcrafted_features with {self.input_channels} channels, "
                f"got: {handcrafted_features.size(1)}"
            )
        conv_maps = [conv(handcrafted_features) for conv in self.convs]
        features = torch.cat(conv_maps, dim=1)
        features = self.activation(features)
        features = self.dropout(features)
        pooled = features.mean(dim=2)
        return self.projection(pooled)


class BiRNAFiLMHandcraftedClassifier(BiRNAFiLMLocalClassifier):
    def __init__(
        self,
        model_dir: Path,
        freeze_backbone: bool = True,
        dropout: float = 0.2,
        center_index: int = 20,
        local_window_radius: int = 3,
        film_global_view: str = "nuc",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        film_nuc_pooling: str = "center_cnn_mean",
        cnn_kernel_sizes: list[int] | None = None,
        handcrafted_input_channels: int = 12,
        handcrafted_cnn_channels: int = 64,
        handcrafted_output_dim: int = 128,
    ):
        super().__init__(
            model_dir=model_dir,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
            center_index=center_index,
            local_window_radius=local_window_radius,
            film_global_view=film_global_view,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            film_nuc_pooling=film_nuc_pooling,
            cnn_kernel_sizes=cnn_kernel_sizes,
        )
        hidden_size = int(getattr(self.birna_model.config, "hidden_size", 768))
        birna_feature_dim = hidden_size * (3 if self._uses_dual_local_branch else 2)
        self.handcrafted_encoder = HandcraftedFeatureCNN(
            input_channels=handcrafted_input_channels,
            cnn_channels=handcrafted_cnn_channels,
            output_dim=handcrafted_output_dim,
            kernel_sizes=cnn_kernel_sizes,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(birna_feature_dim + handcrafted_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(
        self,
        nuc_input_ids=None,
        nuc_attention_mask=None,
        nuc_token_type_ids=None,
        nuc_content_mask=None,
        bpe_input_ids=None,
        bpe_attention_mask=None,
        bpe_token_type_ids=None,
        bpe_content_mask=None,
        handcrafted_features=None,
    ):
        if handcrafted_features is None:
            raise ValueError("BiRNAFiLMHandcraftedClassifier requires handcrafted_features from the data collator.")
        birna_feat = self._build_film_features(
            nuc_input_ids=nuc_input_ids,
            nuc_attention_mask=nuc_attention_mask,
            nuc_token_type_ids=nuc_token_type_ids,
            nuc_content_mask=nuc_content_mask,
            bpe_input_ids=bpe_input_ids,
            bpe_attention_mask=bpe_attention_mask,
            bpe_token_type_ids=bpe_token_type_ids,
            bpe_content_mask=bpe_content_mask,
        )
        hand_feat = self.handcrafted_encoder(handcrafted_features)
        return self.classifier(torch.cat([birna_feat, hand_feat], dim=1))


class BiRNAFiLMGatedHandcraftedClassifier(BiRNAFiLMHandcraftedClassifier):
    def __init__(
        self,
        model_dir: Path,
        freeze_backbone: bool = True,
        dropout: float = 0.2,
        center_index: int = 20,
        local_window_radius: int = 3,
        film_global_view: str = "nuc",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        film_nuc_pooling: str = "center_cnn_mean",
        cnn_kernel_sizes: list[int] | None = None,
        handcrafted_input_channels: int = 12,
        handcrafted_cnn_channels: int = 64,
        handcrafted_output_dim: int = 128,
        gated_fusion_dim: int = 256,
        gated_hidden_dim: int = 128,
    ):
        if gated_fusion_dim <= 0:
            raise ValueError(f"gated_fusion_dim must be positive, got: {gated_fusion_dim}")
        if gated_hidden_dim <= 0:
            raise ValueError(f"gated_hidden_dim must be positive, got: {gated_hidden_dim}")
        super().__init__(
            model_dir=model_dir,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
            center_index=center_index,
            local_window_radius=local_window_radius,
            film_global_view=film_global_view,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            film_nuc_pooling=film_nuc_pooling,
            cnn_kernel_sizes=cnn_kernel_sizes,
            handcrafted_input_channels=handcrafted_input_channels,
            handcrafted_cnn_channels=handcrafted_cnn_channels,
            handcrafted_output_dim=handcrafted_output_dim,
        )
        hidden_size = int(getattr(self.birna_model.config, "hidden_size", 768))
        birna_feature_dim = hidden_size * (3 if self._uses_dual_local_branch else 2)
        self.birna_projection = nn.Sequential(
            nn.Linear(birna_feature_dim, gated_fusion_dim),
            nn.LayerNorm(gated_fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.handcrafted_projection = nn.Sequential(
            nn.Linear(handcrafted_output_dim, gated_fusion_dim),
            nn.LayerNorm(gated_fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(gated_fusion_dim * 2, gated_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gated_hidden_dim, gated_fusion_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(gated_fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(
        self,
        nuc_input_ids=None,
        nuc_attention_mask=None,
        nuc_token_type_ids=None,
        nuc_content_mask=None,
        bpe_input_ids=None,
        bpe_attention_mask=None,
        bpe_token_type_ids=None,
        bpe_content_mask=None,
        handcrafted_features=None,
    ):
        if handcrafted_features is None:
            raise ValueError("BiRNAFiLMGatedHandcraftedClassifier requires handcrafted_features from the data collator.")
        birna_feat = self._build_film_features(
            nuc_input_ids=nuc_input_ids,
            nuc_attention_mask=nuc_attention_mask,
            nuc_token_type_ids=nuc_token_type_ids,
            nuc_content_mask=nuc_content_mask,
            bpe_input_ids=bpe_input_ids,
            bpe_attention_mask=bpe_attention_mask,
            bpe_token_type_ids=bpe_token_type_ids,
            bpe_content_mask=bpe_content_mask,
        )
        hand_feat = self.handcrafted_encoder(handcrafted_features)
        birna_proj = self.birna_projection(birna_feat)
        hand_proj = self.handcrafted_projection(hand_feat)
        gate = self.fusion_gate(torch.cat([birna_proj, hand_proj], dim=1))
        fused_feat = gate * birna_proj + (1.0 - gate) * hand_proj
        return self.classifier(fused_feat)


class HandcraftedOnlyClassifier(nn.Module):
    def __init__(
        self,
        handcrafted_input_channels: int = 12,
        handcrafted_cnn_channels: int = 64,
        handcrafted_output_dim: int = 128,
        cnn_kernel_sizes: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_lora = False
        self.handcrafted_encoder = HandcraftedFeatureCNN(
            input_channels=handcrafted_input_channels,
            cnn_channels=handcrafted_cnn_channels,
            output_dim=handcrafted_output_dim,
            kernel_sizes=cnn_kernel_sizes,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(handcrafted_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, handcrafted_features=None, **_):
        if handcrafted_features is None:
            raise ValueError("HandcraftedOnlyClassifier requires handcrafted_features from the data collator.")
        hand_feat = self.handcrafted_encoder(handcrafted_features)
        return self.classifier(hand_feat)
