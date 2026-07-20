from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from configs.configarg import canonical_dataset_name  # noqa: E402
from late_fusion import run_fusion_experiment  # noqa: E402
from metrics_utils import format_metrics, json_safe_metrics  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse completed handcrafted and BiRNA strict-CV predictions without retraining."
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--method", required=True, choices=["weighted", "logistic"])
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--dataset", default="H_b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outputs_root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument(
        "--base_handcrafted_version",
        default="v2c_mke_handcrafted_only_official4c",
    )
    parser.add_argument("--base_birna_version", default="v0a_birna_nuc_lora")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed != 42:
        raise ValueError("v4 fusion requires strict-CV base predictions produced with seed 42.")
    dataset = canonical_dataset_name(args.dataset)
    payload = run_fusion_experiment(
        method=args.method,
        outputs_root=args.outputs_root,
        output_version=args.version,
        dataset=dataset,
        seed=args.seed,
        model_label=args.model_label,
        handcrafted_version=args.base_handcrafted_version,
        birna_version=args.base_birna_version,
    )
    print("benchmark meta-OOF result:")
    print(format_metrics(payload["benchmark_meta_oof_metrics"]))
    print("independent-test fusion result:")
    print(format_metrics(payload["independent_metrics"]))
    print("final fusion model:")
    print(json.dumps(json_safe_metrics(payload["fusion_model"]), indent=2, ensure_ascii=False))
    print(f"Saved fusion outputs to: {payload['output_dir']}")


if __name__ == "__main__":
    main()
