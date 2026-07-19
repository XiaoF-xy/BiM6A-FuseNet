from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training_control import (  # noqa: E402
    build_constant_warmup_scheduler,
    build_loraplus_optimizer,
)
from training_utils import train_one_epoch  # noqa: E402
from train_cv import parse_args, validate_single_branch_options  # noqa: E402


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, features):
        return self.linear(features)


class CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class SingleBranchTrainingTests(unittest.TestCase):
    def test_loraplus_optimizer_uses_three_complete_parameter_groups(self):
        lora_a = nn.Parameter(torch.tensor(1.0))
        lora_b = nn.Parameter(torch.tensor(2.0))
        classifier = nn.Parameter(torch.tensor(3.0))
        optimizer = build_loraplus_optimizer(
            [
                ("birna_model.Wqkv.lora_A.default.weight", lora_a),
                ("birna_model.Wqkv.lora_B.default.weight", lora_b),
                ("classifier.0.weight", classifier),
            ],
            name="adamw",
            lora_a_lr=5e-5,
            lora_b_lr=8e-4,
            classifier_lr=1e-4,
            weight_decay=0.01,
        )

        self.assertEqual(
            [group["group_name"] for group in optimizer.param_groups],
            ["lora_A", "lora_B", "classifier"],
        )
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [5e-5, 8e-4, 1e-4],
        )
        parameter_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(parameter_ids), 3)
        self.assertEqual(len(set(parameter_ids)), 3)

    def test_loraplus_optimizer_rejects_non_positive_learning_rate(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            build_loraplus_optimizer(
                [
                    ("module.lora_A.default.weight", nn.Parameter(torch.tensor(1.0))),
                    ("module.lora_B.default.weight", nn.Parameter(torch.tensor(2.0))),
                    ("classifier.weight", nn.Parameter(torch.tensor(3.0))),
                ],
                name="adamw",
                lora_a_lr=0.0,
                lora_b_lr=8e-4,
                classifier_lr=1e-4,
                weight_decay=0.01,
            )

    def test_runner_parses_single_branch_and_warmup_flags(self):
        with patch.object(
            sys,
            "argv",
            ["train_cv.py", "--use_birna_single_branch", "--warmup_ratio", "0.1"],
        ):
            args = parse_args()

        self.assertTrue(args.use_birna_single_branch)
        self.assertEqual(args.warmup_ratio, 0.1)

    def test_runner_parses_loraplus_learning_rates(self):
        with patch.object(
            sys,
            "argv",
            [
                "train_cv.py",
                "--use_loraplus",
                "--lora_a_lr",
                "5e-5",
                "--lora_b_lr",
                "8e-4",
                "--classifier_lr",
                "1e-4",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.use_loraplus)
        self.assertEqual(args.lora_a_lr, 5e-5)
        self.assertEqual(args.lora_b_lr, 8e-4)
        self.assertEqual(args.classifier_lr, 1e-4)

    def test_loraplus_requires_pure_single_branch_lora(self):
        args = SimpleNamespace(
            use_loraplus=True,
            use_birna_single_branch=True,
            use_lora=False,
            lora_a_lr=5e-5,
            lora_b_lr=8e-4,
            classifier_lr=1e-4,
            use_film=False,
            use_bpe_view=False,
            use_handcrafted_features=False,
            handcrafted_only=False,
            use_mke_handcrafted=False,
            use_official_mke_handcrafted=False,
            use_projected_concat=False,
            use_gated_fusion=False,
            disable_center_pooling=True,
        )

        with self.assertRaisesRegex(ValueError, "requires.*--use_lora"):
            validate_single_branch_options(args)

    def test_single_branch_rejects_film_or_handcrafted_inputs(self):
        args = SimpleNamespace(
            use_birna_single_branch=True,
            use_film=True,
            use_bpe_view=False,
            use_handcrafted_features=False,
            handcrafted_only=False,
            use_mke_handcrafted=False,
            use_official_mke_handcrafted=False,
            use_projected_concat=False,
            use_gated_fusion=False,
            disable_center_pooling=True,
        )

        with self.assertRaisesRegex(ValueError, "pure NUC inputs"):
            validate_single_branch_options(args)

    def test_constant_warmup_reaches_and_keeps_base_lr(self):
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-6)
        scheduler = build_constant_warmup_scheduler(
            optimizer,
            total_steps=10,
            warmup_ratio=0.2,
        )
        observed = []

        for _ in range(10):
            optimizer.step()
            scheduler.step()
            observed.append(optimizer.param_groups[0]["lr"])

        self.assertAlmostEqual(observed[1], 1e-6)
        self.assertAlmostEqual(observed[-1], 1e-6)

    def test_train_one_epoch_steps_batch_scheduler_once_per_batch(self):
        model = TinyClassifier()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = CountingScheduler()
        criterion = nn.CrossEntropyLoss()
        loader = [
            {
                "features": torch.tensor([[1.0, 0.0]]),
                "labels": torch.tensor([1]),
                "sequences": ["A" * 41],
            },
            {
                "features": torch.tensor([[0.0, 1.0]]),
                "labels": torch.tensor([0]),
                "sequences": ["C" * 41],
            },
        ]

        train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=torch.device("cpu"),
            epoch=1,
            freeze_backbone=False,
            batch_scheduler=scheduler,
        )

        self.assertEqual(scheduler.steps, 2)


if __name__ == "__main__":
    unittest.main()
