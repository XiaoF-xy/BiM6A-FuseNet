from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from src.model_birna_single import (
    BiRNASingleBranchClassifier,
    LastFourLayerScalarMix,
    apply_dora_to_birna,
    masked_mean_nucleotide_embeddings,
    masked_mean_packed_nucleotide_embeddings,
    nucleotide_content_mask,
    validate_lora_target_modules,
)


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=768)
        self.base_weight = nn.Parameter(torch.ones(1))
        self.Wqkv = nn.Linear(1, 3)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        embeddings = input_ids.float().unsqueeze(-1).repeat(1, 1, 768)
        return SimpleNamespace(logits=embeddings * self.base_weight)


class FakeTargetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wqkv = nn.Linear(4, 12)
        self.attention = nn.Module()
        self.attention.output = nn.Module()
        self.attention.output.dense = nn.Linear(4, 4)
        self.gated_layers = nn.Linear(4, 8)
        self.wo = nn.Linear(8, 4)


class FakePackedBert(nn.Module):
    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        output_all_encoded_layers=False,
    ):
        if not output_all_encoded_layers:
            raise AssertionError("Scalar-mix path must request all encoder layers.")
        active = input_ids[attention_mask.bool()].float().unsqueeze(-1).repeat(1, 768)
        return [active + float(index) for index in range(12)], None


class FakePackedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=768)
        self.bert = FakePackedBert()


class BiRNASingleModelTests(unittest.TestCase):
    def test_last_four_scalar_mix_starts_final_layer_dominant(self):
        scalar_mix = LastFourLayerScalarMix(hidden_size=2)

        weights = scalar_mix.normalized_weights()

        torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
        self.assertGreater(weights[-1].item(), 0.99)
        self.assertTrue(torch.all(weights > 0))

    def test_last_four_scalar_mix_combines_only_last_four_layers(self):
        scalar_mix = LastFourLayerScalarMix(hidden_size=2)
        hidden_states = [torch.full((3, 2), float(value)) for value in range(1, 7)]

        mixed = scalar_mix(hidden_states)
        expected = sum(
            weight * layer
            for weight, layer in zip(scalar_mix.normalized_weights(), hidden_states[-4:])
        )

        torch.testing.assert_close(mixed, expected)

    def test_packed_masked_mean_ignores_cls_and_sep(self):
        first = torch.full((43, 2), 3.0)
        second = torch.full((43, 2), 7.0)
        first[0], first[-1] = 1000.0, -1000.0
        second[0], second[-1] = 2000.0, -2000.0
        packed = torch.cat([first, second], dim=0)
        attention_mask = torch.ones(2, 43, dtype=torch.long)

        pooled = masked_mean_packed_nucleotide_embeddings(packed, attention_mask)

        torch.testing.assert_close(pooled, torch.tensor([[3.0, 3.0], [7.0, 7.0]]))

    def test_packed_masked_mean_rejects_non_41nt_inputs(self):
        packed = torch.zeros(42, 768)
        attention_mask = torch.ones(1, 42, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "exactly 41"):
            masked_mean_packed_nucleotide_embeddings(packed, attention_mask)

    def test_last_four_scalar_mix_rejects_mismatched_layer_shapes(self):
        scalar_mix = LastFourLayerScalarMix(hidden_size=2)
        hidden_states = [torch.zeros(3, 2) for _ in range(4)]
        hidden_states[-1] = torch.zeros(4, 2)

        with self.assertRaisesRegex(ValueError, "same shape"):
            scalar_mix(hidden_states)

    def test_scalar_mix_classifier_reads_packed_encoder_layers(self):
        backbone = FakePackedBackbone()
        with patch("src.model_birna_single.load_birna_backbone", return_value=backbone):
            model = BiRNASingleBranchClassifier(
                model_dir="unused",
                freeze_backbone=False,
                use_lora=False,
                use_last4_scalar_mix=True,
            )
        model.eval()
        input_ids = torch.arange(43).repeat(2, 1)
        attention_mask = torch.ones_like(input_ids)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_apply_dora_forwards_true_flag_to_peft(self):
        captured_configs = []

        class FakeLoraConfig:
            def __init__(self, **kwargs):
                captured_configs.append(kwargs)

        fake_peft = SimpleNamespace(
            LoraConfig=FakeLoraConfig,
            get_peft_model=lambda model, _config: model,
        )
        with patch.dict(sys.modules, {"peft": fake_peft}):
            apply_dora_to_birna(
                birna_model=FakeTargetBackbone(),
                target_modules=["Wqkv"],
                r=8,
                alpha=32,
                dropout=0.05,
            )

        self.assertEqual([config["use_dora"] for config in captured_configs], [True])

    def test_lora_target_validation_accepts_every_configured_suffix(self):
        matches = validate_lora_target_modules(
            FakeTargetBackbone(),
            ["Wqkv", "attention.output.dense", "gated_layers", "wo"],
        )

        self.assertEqual(
            set(matches),
            {"Wqkv", "attention.output.dense", "gated_layers", "wo"},
        )

    def test_lora_target_validation_rejects_an_unmatched_suffix(self):
        with self.assertRaisesRegex(ValueError, "missing_target"):
            validate_lora_target_modules(FakeTargetBackbone(), ["Wqkv", "missing_target"])

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
