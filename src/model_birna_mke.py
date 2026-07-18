from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from model_birna_film import BiRNAFiLMLocalClassifier
from model_mke_handcrafted import MKEFeatureFusionClassifier, MKE_FEATURE_ORDER


class BiRNAFiLMMKEClassifier(BiRNAFiLMLocalClassifier):
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
        handcrafted_feature_names: Sequence[str] = MKE_FEATURE_ORDER,
        use_full_mke_eca: bool = False,
        fusion_dim_policy: str = "native",
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
        self.mke_classifier = MKEFeatureFusionClassifier(
            birna_input_dim=birna_feature_dim,
            feature_names=handcrafted_feature_names,
            dimension_policy=fusion_dim_policy,
            use_full_mke_eca=use_full_mke_eca,
            dropout=dropout,
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
            raise ValueError("BiRNAFiLMMKEClassifier requires handcrafted_features from the data collator.")
        birna_features = self._build_film_features(
            nuc_input_ids=nuc_input_ids,
            nuc_attention_mask=nuc_attention_mask,
            nuc_token_type_ids=nuc_token_type_ids,
            nuc_content_mask=nuc_content_mask,
            bpe_input_ids=bpe_input_ids,
            bpe_attention_mask=bpe_attention_mask,
            bpe_token_type_ids=bpe_token_type_ids,
            bpe_content_mask=bpe_content_mask,
        )
        return self.mke_classifier(birna_features, handcrafted_features)
