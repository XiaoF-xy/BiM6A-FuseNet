from __future__ import annotations

from experiments.mke_variants_common import get_mke_overrides


def get_overrides(dataset_name: str, seed: int) -> dict:
    return get_mke_overrides(
        dataset_name,
        seed,
        version_name="v2a_mke_res_eca_native",
        plot_label="BiM6A-FuseNet-v2a",
        description="Four-stream MKE ResNet-ECA handcrafted encoder with native 1536+128 concat fusion.",
        use_full_mke_eca=False,
        fusion_dim_policy="native",
    )
