from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_birna_mke import BiRNAFiLMMKEClassifier  # noqa: E402


class FakeBiRNABackbone(torch.nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        batch, length = input_ids.shape
        positions = torch.arange(length, dtype=torch.float32, device=input_ids.device)
        embeddings = positions.view(1, length, 1).expand(batch, length, self.config.hidden_size)
        return SimpleNamespace(logits=embeddings * self.scale)


@pytest.mark.parametrize(
    ("dimension_policy", "use_full_mke_eca", "expected_input_dim"),
    [("native", False, 144), ("proj256", False, 512), ("native", True, 144), ("proj256", True, 512)],
)
def test_birna_mke_wrapper_forwards_film_and_handcrafted_features(
    dimension_policy,
    use_full_mke_eca,
    expected_input_dim,
):
    with patch("model_birna_film.load_birna_backbone", return_value=FakeBiRNABackbone()):
        model = BiRNAFiLMMKEClassifier(
            model_dir=ROOT / "pretrained" / "birna-bert-model",
            freeze_backbone=True,
            dropout=0.2,
            film_global_view="nuc",
            local_window_radius=3,
            use_lora=False,
            film_nuc_pooling="center_cnn_mean",
            cnn_kernel_sizes=[3, 5, 7],
            handcrafted_feature_names=["onehot", "ncp", "eiip", "enac"],
            use_full_mke_eca=use_full_mke_eca,
            fusion_dim_policy=dimension_policy,
        )
    model.eval()
    nuc_input_ids = torch.ones(2, 43, dtype=torch.long)
    nuc_content_mask = torch.zeros(2, 43, dtype=torch.bool)
    nuc_content_mask[:, 1:42] = True

    with torch.no_grad():
        logits = model(
            nuc_input_ids=nuc_input_ids,
            nuc_attention_mask=torch.ones_like(nuc_input_ids),
            nuc_content_mask=nuc_content_mask,
            handcrafted_features=torch.randn(2, 12, 41),
        )

    assert model.mke_classifier.fusion_head.classifier[0].in_features == expected_input_dim
    assert logits.shape == (2, 2)
