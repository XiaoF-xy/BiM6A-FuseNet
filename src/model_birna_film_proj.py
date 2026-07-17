from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model_birna_film import BiRNAFiLMLocalClassifier, HandcraftedFeatureCNN


class ProjectedConcatFusionHead(nn.Module):
    """Project two heterogeneous branches to equal width before concatenation."""

    def __init__(
        self,
        birna_input_dim: int,
        hand_input_dim: int,
        projection_dim: int = 256,
        projection_dropout: float = 0.2,
        classifier_dropout: float = 0.2,
    ):
        super().__init__()
        self.birna_projection = nn.Sequential(
            nn.Linear(birna_input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(projection_dropout),
        )
        self.hand_projection = nn.Sequential(
            nn.Linear(hand_input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(projection_dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, 2),
        )

    def forward(self, birna_feat: torch.Tensor, hand_feat: torch.Tensor) -> torch.Tensor:
        z_birna = self.birna_projection(birna_feat)
        z_hand = self.hand_projection(hand_feat)
        return self.classifier(torch.cat([z_birna, z_hand], dim=1))


class BiRNAFiLMProjectedConcatClassifier(BiRNAFiLMLocalClassifier):
    """v1 feature extractors with independent 256-dimensional branch projections."""

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
        projection_dim: int = 256,
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
        del self.classifier
        hidden_size = int(getattr(self.birna_model.config, "hidden_size", 768))
        birna_feature_dim = hidden_size * (3 if self._uses_dual_local_branch else 2)
        self.handcrafted_encoder = HandcraftedFeatureCNN(
            input_channels=handcrafted_input_channels,
            cnn_channels=handcrafted_cnn_channels,
            output_dim=handcrafted_output_dim,
            kernel_sizes=cnn_kernel_sizes,
            dropout=dropout,
        )
        self.fusion_head = ProjectedConcatFusionHead(
            birna_input_dim=birna_feature_dim,
            hand_input_dim=handcrafted_output_dim,
            projection_dim=projection_dim,
            projection_dropout=dropout,
            classifier_dropout=dropout,
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
            raise ValueError(
                "BiRNAFiLMProjectedConcatClassifier requires handcrafted_features from the data collator."
            )
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
        return self.fusion_head(birna_feat, hand_feat)
