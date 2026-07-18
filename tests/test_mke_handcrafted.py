from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_mke_handcrafted import (  # noqa: E402
    ECA1D,
    FourStreamMKEEncoder,
    FullMKEECA,
    MKEFeatureFusionClassifier,
    MKEFusionHead,
    MKEResidualBlock,
)


EXPECTED_FEATURE_ORDER = ("onehot", "ncp", "eiip", "enac")


def test_residual_block_changes_channels_without_changing_sequence_length():
    block = MKEResidualBlock(in_channels=4, out_channels=32)

    output = block(torch.randn(2, 4, 41))

    assert output.shape == (2, 32, 41)


def test_eca_preserves_shape_and_scales_channels():
    block = ECA1D(channels=32)
    inputs = torch.randn(2, 32, 20)

    attention = block.attention(inputs)
    output = block(inputs)

    assert output.shape == inputs.shape
    assert attention.shape == (2, 32, 1)
    assert torch.all(attention >= 0)
    assert torch.all(attention <= 1)


def test_four_stream_encoder_uses_expected_branch_shapes_and_returns_64_features():
    encoder = FourStreamMKEEncoder(feature_names=EXPECTED_FEATURE_ORDER)
    features = torch.randn(2, 12, 41)
    onehot, ncp, eiip, enac = encoder.split_features(features)

    onehot_out = encoder.onehot_branch(onehot)
    ncp_out = encoder.ncp_branch(ncp)
    eiip_out = encoder.eiip_branch(eiip)
    enac_out = encoder.enac_branch(enac)
    output = encoder(features)

    assert onehot_out.shape == (2, 32, 20)
    assert ncp_out.shape == (2, 32, 20)
    assert eiip_out.shape == (2, 16, 20)
    assert enac_out.shape == (2, 32, 20)
    assert output.shape == (2, 64)
    assert encoder.merged_channels == 112


@pytest.mark.parametrize("shape", [(2, 11, 41), (2, 12, 40)])
def test_four_stream_encoder_rejects_invalid_input_shape(shape):
    encoder = FourStreamMKEEncoder(feature_names=EXPECTED_FEATURE_ORDER)

    with pytest.raises(ValueError, match="expected handcrafted input"):
        encoder(torch.randn(*shape))


def test_four_stream_encoder_rejects_wrong_feature_order():
    with pytest.raises(ValueError, match="feature order"):
        FourStreamMKEEncoder(feature_names=("ncp", "onehot", "eiip", "enac"))


def test_full_mke_eca_preserves_shape_and_uses_channel_then_spatial_multiplication():
    block = FullMKEECA(channels=32, reduction=16)
    inputs = torch.randn(2, 32, 20)

    channel_weights = block.channel_attention(inputs)
    channel_scaled = inputs * channel_weights
    spatial_weights = block.spatial_attention(channel_scaled)
    output = block(inputs)

    assert channel_weights.shape == (2, 32, 1)
    assert spatial_weights.shape == (2, 1, 20)
    assert output.shape == inputs.shape
    assert torch.all(channel_weights >= 0)
    assert torch.all(channel_weights <= 1)
    assert torch.all(spatial_weights >= 0)
    assert torch.all(spatial_weights <= 1)
    assert torch.allclose(output, channel_scaled * spatial_weights)


def test_full_mke_eca_uses_three_five_and_seven_spatial_kernels():
    block = FullMKEECA(channels=32)

    assert [conv.kernel_size[0] for conv in block.spatial_convs] == [3, 5, 7]
    assert block.spatial_fusion.in_channels == 3
    assert block.spatial_fusion.out_channels == 1


def test_v3_encoder_enables_one_full_mke_eca_without_changing_output_shape():
    encoder = FourStreamMKEEncoder(
        feature_names=EXPECTED_FEATURE_ORDER,
        use_full_mke_eca=True,
    )

    output = encoder(torch.randn(2, 12, 41))

    assert isinstance(encoder.full_mke_eca, FullMKEECA)
    assert output.shape == (2, 64)


def test_native_fusion_keeps_birna_width_and_adapts_handcrafted_to_128():
    head = MKEFusionHead(
        birna_input_dim=1536,
        hand_input_dim=64,
        dimension_policy="native",
    )

    logits = head(torch.randn(3, 1536), torch.randn(3, 64))

    assert isinstance(head.birna_projection, torch.nn.Identity)
    assert head.hand_projection[0].in_features == 64
    assert head.hand_projection[0].out_features == 128
    assert head.classifier[0].in_features == 1664
    assert logits.shape == (3, 2)


def test_proj256_fusion_aligns_both_branches_before_concat():
    head = MKEFusionHead(
        birna_input_dim=1536,
        hand_input_dim=64,
        dimension_policy="proj256",
    )

    logits = head(torch.randn(3, 1536), torch.randn(3, 64))

    assert head.birna_projection[0].in_features == 1536
    assert head.birna_projection[0].out_features == 256
    assert head.hand_projection[0].in_features == 64
    assert head.hand_projection[0].out_features == 256
    assert head.classifier[0].in_features == 512
    assert logits.shape == (3, 2)


def test_fusion_rejects_unknown_dimension_policy():
    with pytest.raises(ValueError, match="dimension policy"):
        MKEFusionHead(
            birna_input_dim=1536,
            hand_input_dim=64,
            dimension_policy="unknown",
        )


@pytest.mark.parametrize(
    ("dimension_policy", "use_full_mke_eca", "expected_classifier_input"),
    [("native", False, 1664), ("proj256", False, 512), ("native", True, 1664), ("proj256", True, 512)],
)
def test_shared_feature_fusion_classifier_covers_all_four_architectures(
    dimension_policy,
    use_full_mke_eca,
    expected_classifier_input,
):
    classifier = MKEFeatureFusionClassifier(
        birna_input_dim=1536,
        feature_names=EXPECTED_FEATURE_ORDER,
        dimension_policy=dimension_policy,
        use_full_mke_eca=use_full_mke_eca,
    )

    logits = classifier(torch.randn(2, 1536), torch.randn(2, 12, 41))

    assert classifier.handcrafted_encoder.use_full_mke_eca is use_full_mke_eca
    assert classifier.fusion_head.classifier[0].in_features == expected_classifier_input
    assert logits.shape == (2, 2)
