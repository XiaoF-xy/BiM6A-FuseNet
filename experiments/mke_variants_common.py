from __future__ import annotations

from experiments.v1_baseline.config_v1 import get_overrides as get_v1_overrides


def get_mke_overrides(
    dataset_name: str,
    seed: int,
    *,
    version_name: str,
    plot_label: str,
    description: str,
    use_full_mke_eca: bool,
    fusion_dim_policy: str,
) -> dict:
    overrides = get_v1_overrides(dataset_name=dataset_name, seed=seed)
    overrides["experiment"] = {
        "version_name": version_name,
        "plot_label": plot_label,
        "description": description,
    }
    overrides["model"] = {
        **overrides["model"],
        "use_mke_handcrafted": True,
        "use_full_mke_eca": use_full_mke_eca,
        "fusion_dim_policy": fusion_dim_policy,
        "use_projected_concat": False,
        "use_gated_fusion": False,
    }
    return overrides
