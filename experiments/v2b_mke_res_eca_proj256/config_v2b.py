from __future__ import annotations

from experiments.mke_variants_common import get_mke_overrides


def get_overrides(dataset_name: str, seed: int) -> dict:
    return get_mke_overrides(
        dataset_name,
        seed,
        version_name="v2b_mke_res_eca_proj256",
        plot_label="BiM6A-FuseNet-v2b",
        description="Four-stream MKE ResNet-ECA handcrafted encoder with aligned 256+256 concat fusion.",
        use_full_mke_eca=False,
        fusion_dim_policy="proj256",
    )
