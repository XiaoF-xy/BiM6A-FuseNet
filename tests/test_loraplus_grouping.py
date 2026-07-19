from __future__ import annotations

import unittest

from src.loraplus import partition_loraplus_named_parameters


class DummyParameter:
    def __init__(self, *, requires_grad: bool = True):
        self.requires_grad = requires_grad


class LoRAPlusGroupingTests(unittest.TestCase):
    def test_partitions_every_trainable_parameter_once(self):
        lora_a = DummyParameter()
        lora_b = DummyParameter()
        classifier = DummyParameter()
        frozen = DummyParameter(requires_grad=False)

        groups = partition_loraplus_named_parameters(
            [
                ("birna_model.layer.0.Wqkv.lora_A.default.weight", lora_a),
                ("birna_model.layer.0.Wqkv.lora_B.default.weight", lora_b),
                ("classifier.0.weight", classifier),
                ("birna_model.layer.0.Wqkv.base_layer.weight", frozen),
            ]
        )

        self.assertEqual(list(groups), ["lora_A", "lora_B", "classifier"])
        self.assertEqual(groups["lora_A"], [lora_a])
        self.assertEqual(groups["lora_B"], [lora_b])
        self.assertEqual(groups["classifier"], [classifier])
        grouped_ids = [id(parameter) for values in groups.values() for parameter in values]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))

    def test_rejects_missing_lora_group(self):
        with self.assertRaisesRegex(ValueError, "missing.*lora_B"):
            partition_loraplus_named_parameters(
                [
                    ("birna_model.layer.0.Wqkv.lora_A.default.weight", DummyParameter()),
                    ("classifier.0.weight", DummyParameter()),
                ]
            )

    def test_rejects_duplicate_trainable_parameter_objects(self):
        duplicate = DummyParameter()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            partition_loraplus_named_parameters(
                [
                    ("birna_model.layer.0.Wqkv.lora_A.default.weight", duplicate),
                    ("birna_model.layer.0.Wqkv.lora_B.default.weight", duplicate),
                    ("classifier.0.weight", DummyParameter()),
                ]
            )


if __name__ == "__main__":
    unittest.main()
