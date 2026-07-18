from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from src.model_birna_single import (
    BiRNASingleBranchClassifier,
    masked_mean_nucleotide_embeddings,
    nucleotide_content_mask,
)


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=768)
        self.base_weight = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        embeddings = input_ids.float().unsqueeze(-1).repeat(1, 1, 768)
        return SimpleNamespace(logits=embeddings * self.base_weight)


class BiRNASingleModelTests(unittest.TestCase):
    def test_content_mask_excludes_cls_sep_and_padding(self):
        attention_mask = torch.tensor(
            [
                [1] * 43 + [0, 0, 0],
                [1] * 43 + [0, 0, 0],
            ]
        )

        mask = nucleotide_content_mask(attention_mask)

        self.assertEqual(mask.sum(dim=1).tolist(), [41, 41])
        self.assertFalse(mask[:, 0].any())
        self.assertFalse(mask[:, 42:].any())

    def test_masked_mean_ignores_special_and_padded_embeddings(self):
        attention_mask = torch.tensor([[1] * 43 + [0, 0, 0]])
        embeddings = torch.zeros(1, 46, 2)
        embeddings[:, 1:42, :] = 3.0
        embeddings[:, 0, :] = 1000.0
        embeddings[:, 42:, :] = -1000.0

        pooled = masked_mean_nucleotide_embeddings(embeddings, attention_mask)

        torch.testing.assert_close(pooled, torch.full((1, 2), 3.0))

    def test_masked_mean_rejects_non_41nt_inputs(self):
        embeddings = torch.zeros(1, 42, 768)
        attention_mask = torch.ones(1, 42, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "exactly 41"):
            masked_mean_nucleotide_embeddings(embeddings, attention_mask)

    def test_lora_policy_freezes_base_and_trains_adapter_and_head(self):
        backbone = FakeBackbone()

        def fake_apply_lora(birna_model, **_kwargs):
            for parameter in birna_model.parameters():
                parameter.requires_grad = False
            birna_model.register_parameter("adapter_weight", nn.Parameter(torch.ones(1)))
            return birna_model

        with (
            patch("src.model_birna_single.load_birna_backbone", return_value=backbone),
            patch("src.model_birna_single.apply_lora_to_birna", side_effect=fake_apply_lora),
        ):
            model = BiRNASingleBranchClassifier(
                model_dir="unused",
                freeze_backbone=True,
                use_lora=True,
            )

        self.assertFalse(model.birna_model.base_weight.requires_grad)
        self.assertTrue(model.birna_model.adapter_weight.requires_grad)
        self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))

    def test_full_finetuning_trains_backbone_and_head(self):
        backbone = FakeBackbone()
        with patch("src.model_birna_single.load_birna_backbone", return_value=backbone):
            model = BiRNASingleBranchClassifier(
                model_dir="unused",
                freeze_backbone=False,
                use_lora=False,
            )

        self.assertTrue(all(parameter.requires_grad for parameter in model.birna_model.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))

    def test_forward_returns_two_logits(self):
        backbone = FakeBackbone()
        with patch("src.model_birna_single.load_birna_backbone", return_value=backbone):
            model = BiRNASingleBranchClassifier(
                model_dir="unused",
                freeze_backbone=False,
                use_lora=False,
            )
        model.eval()
        input_ids = torch.arange(43).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertEqual(tuple(logits.shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
