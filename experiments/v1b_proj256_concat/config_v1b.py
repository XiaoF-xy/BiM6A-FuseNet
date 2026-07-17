from __future__ import annotations

from experiments.v1_baseline.config_v1 import get_overrides as get_v1_overrides


def get_overrides(dataset_name: str, seed: int) -> dict:
    overrides = get_v1_overrides(dataset_name=dataset_name, seed=seed)
    overrides["experiment"] = {
        "version_name": "v1b_proj256_concat",
        "description": (
            "v1 with independent 256-dimensional BiRNA and handcrafted "
            "projections followed by controlled concat fusion."
        ),
    }
    overrides["model"] = {
        **overrides["model"],
        "use_projected_concat": True,
    }
    return overrides
