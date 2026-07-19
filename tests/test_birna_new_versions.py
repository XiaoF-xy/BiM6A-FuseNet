from __future__ import annotations

import unittest
from dataclasses import asdict

from configs.configarg import load_experiment_config
from scripts.train import build_cv_command
from scripts.verify_portable import REQUIRED, ROOT


class BiRNANewVersionConfigTests(unittest.TestCase):
    def test_portable_check_requires_both_new_configs(self):
        required = {path.relative_to(ROOT).as_posix() for path in REQUIRED}

        self.assertIn("experiments/v0f_birna_last4_scalar_mix/config_v0f.py", required)
        self.assertIn("experiments/v0g_birna_nuc_dora/config_v0g.py", required)

    def test_v0f_changes_only_last_four_scalar_mix(self):
        self._assert_single_model_change(
            version="v0f_birna_last4_scalar_mix",
            changed_field="use_last4_scalar_mix",
            expected_flag="--use_last4_scalar_mix",
            absent_flag="--use_dora",
        )

    def test_v0g_changes_only_dora(self):
        self._assert_single_model_change(
            version="v0g_birna_nuc_dora",
            changed_field="use_dora",
            expected_flag="--use_dora",
            absent_flag="--use_last4_scalar_mix",
        )

    def _assert_single_model_change(
        self,
        version: str,
        changed_field: str,
        expected_flag: str,
        absent_flag: str,
    ) -> None:
        baseline = load_experiment_config("v0a_birna_nuc_lora", "H_b", seed=42)
        candidate = load_experiment_config(version, "H_b", seed=42)
        command = build_cv_command(candidate)

        baseline_model = asdict(baseline.model)
        candidate_model = asdict(candidate.model)
        self.assertFalse(baseline_model.pop(changed_field))
        self.assertTrue(candidate_model.pop(changed_field))
        self.assertEqual(candidate_model, baseline_model)

        baseline_training = asdict(baseline.training)
        candidate_training = asdict(candidate.training)
        baseline_training.pop("output_dir")
        candidate_training.pop("output_dir")
        self.assertEqual(candidate_training, baseline_training)
        self.assertIn(expected_flag, command)
        self.assertNotIn(absent_flag, command)


if __name__ == "__main__":
    unittest.main()
