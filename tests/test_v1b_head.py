from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # The lightweight config-test environment does not include PyTorch.
    torch = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if torch is not None:
    from model_birna_film_proj import ProjectedConcatFusionHead  # noqa: E402


@unittest.skipIf(torch is None, "PyTorch is required for the fusion-head forward test.")
class ProjectedConcatFusionHeadTest(unittest.TestCase):
    def test_projection_dimensions_and_forward_shape(self):
        head = ProjectedConcatFusionHead(
            birna_input_dim=1536,
            hand_input_dim=128,
            projection_dim=256,
        )

        self.assertEqual(head.birna_projection[0].in_features, 1536)
        self.assertEqual(head.birna_projection[0].out_features, 256)
        self.assertEqual(head.hand_projection[0].in_features, 128)
        self.assertEqual(head.hand_projection[0].out_features, 256)
        self.assertEqual(head.classifier[0].in_features, 512)
        self.assertIsInstance(head.classifier[1], torch.nn.ReLU)
        self.assertEqual(head.classifier[2].p, 0.2)

        logits = head(torch.randn(4, 1536), torch.randn(4, 128))

        self.assertEqual(tuple(logits.shape), (4, 2))


if __name__ == "__main__":
    unittest.main()
