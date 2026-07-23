from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mke_official_features import (  # noqa: E402
    OFFICIAL_MKE_FEATURE_ORDER,
    chemical4_encode,
    official_mke_feature_matrix,
)
from dataset_utils import SequenceSample  # noqa: E402
from model_mke_official import OfficialMKEClassifier, OfficialResidualBlock  # noqa: E402
from train_cv import make_loader  # noqa: E402
from training_control import EarlyStopping, build_optimizer, build_plateau_scheduler  # noqa: E402
from training_utils import OfficialMKEDataCollator  # noqa: E402


def test_chemical4_appends_repository_cumulative_frequency():
    encoded = chemical4_encode("AACA")

    assert encoded.shape == (4, 4)
    np.testing.assert_array_equal(
        encoded[:3],
        np.asarray(
            [
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(encoded[3], [1.0, 1.0, 0.333, 0.75], atol=1e-6)


def test_official_feature_matrix_has_fixed_4_4_1_4_order():
    sequence = "ACGT" * 10 + "A"
    matrix = official_mke_feature_matrix(sequence)

    assert OFFICIAL_MKE_FEATURE_ORDER == ("onehot", "chemical4", "eiip", "enac")
    assert matrix.shape == (13, 41)
    np.testing.assert_array_equal(matrix[0:4], official_mke_feature_matrix(sequence)[0:4])
    np.testing.assert_array_equal(matrix[4:8], chemical4_encode(sequence))


def test_official_model_splits_13_channels_and_returns_two_logits():
    model = OfficialMKEClassifier(sequence_length=41)
    features = torch.randn(2, 13, 41)

    onehot, chemical4, eiip, enac = model.split_features(features)
    logits = model(handcrafted_features=features)

    assert onehot.shape == (2, 4, 41)
    assert chemical4.shape == (2, 4, 41)
    assert eiip.shape == (2, 1, 41)
    assert enac.shape == (2, 4, 41)
    assert logits.shape == (2, 2)


def test_official_model_matches_public_normalization_activation_and_dropout_contract():
    model = OfficialMKEClassifier(sequence_length=41)

    assert isinstance(model.res_block1_1, OfficialResidualBlock)
    assert isinstance(model.res_block1_1.bn1, torch.nn.GroupNorm)
    assert isinstance(model.res_block1_1.activation, torch.nn.GELU)
    assert model.res_block2_1.conv1.in_channels == 4
    assert model.fc1.in_features == 160
    assert model.fc_final_1.in_features == 64
    assert model.fc_final_1.out_features == 32
    assert model.fc_final_2.out_features == 2
    assert model.dropout1.p == pytest.approx(0.3)
    assert model.dropout2.p == pytest.approx(0.3)
    assert model.dropout3.p == pytest.approx(0.85)


@pytest.mark.parametrize("shape", [(2, 12, 41), (2, 13, 40)])
def test_official_model_rejects_wrong_combined_input_shape(shape):
    model = OfficialMKEClassifier(sequence_length=41)

    with pytest.raises(ValueError, match="expected handcrafted input"):
        model(handcrafted_features=torch.randn(*shape))


def test_official_collator_builds_features_without_a_tokenizer():
    collator = OfficialMKEDataCollator()
    batch = collator([
        {"sequence": "A" * 41, "label": 1},
        {"sequence": "C" * 41, "label": 0},
    ])

    assert set(batch) == {"handcrafted_features", "labels", "sequences"}
    assert batch["handcrafted_features"].shape == (2, 13, 41)
    assert batch["labels"].tolist() == [1, 0]


def test_shared_training_loader_drops_only_incomplete_tail_batches():
    samples = [SequenceSample(sequence="A" * 41, label=index % 2) for index in range(65)]

    train_loader = make_loader(
        samples,
        tokenizer=None,
        max_length=64,
        batch_size=64,
        shuffle=False,
        use_bpe_view=False,
        use_official_mke_handcrafted=True,
        drop_last=True,
    )
    evaluation_loader = make_loader(
        samples,
        tokenizer=None,
        max_length=64,
        batch_size=64,
        shuffle=False,
        use_bpe_view=False,
        use_official_mke_handcrafted=True,
        drop_last=False,
    )

    assert [len(batch["labels"]) for batch in train_loader] == [64]
    assert [len(batch["labels"]) for batch in evaluation_loader] == [64, 1]


def test_v2c_training_controls_use_adam_plateau_scheduler_and_acc_early_stopping():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = build_optimizer([parameter], name="adam", lr=1e-3, weight_decay=1e-5)
    scheduler = build_plateau_scheduler(optimizer, patience=10, factor=0.1)
    early_stopping = EarlyStopping(patience=2)

    assert isinstance(optimizer, torch.optim.Adam)
    assert scheduler.patience == 10
    assert scheduler.factor == pytest.approx(0.1)
    assert early_stopping.update(improved=True) is False
    assert early_stopping.update(improved=False) is False
    assert early_stopping.update(improved=False) is True
