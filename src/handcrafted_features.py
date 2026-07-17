from __future__ import annotations

import numpy as np


BASES = ("A", "C", "G", "T")
BASE_TO_INDEX = {base: index for index, base in enumerate(BASES)}
SUPPORTED_FEATURES = ("onehot", "ncp", "eiip", "enac")
FEATURE_CHANNELS = {
    "onehot": 4,
    "ncp": 3,
    "eiip": 1,
    "enac": 4,
}

NCP_VALUES = {
    "A": (1.0, 1.0, 1.0),
    "C": (0.0, 1.0, 0.0),
    "G": (1.0, 0.0, 0.0),
    "T": (0.0, 0.0, 1.0),
}

EIIP_VALUES = {
    "A": 0.1260,
    "C": 0.1340,
    "G": 0.0806,
    "T": 0.1335,
}


def onehot_encode(sequence: str) -> np.ndarray:
    sequence = _normalize_sequence(sequence)
    features = np.zeros((4, len(sequence)), dtype=np.float32)
    for position, base in enumerate(sequence):
        features[BASE_TO_INDEX[base], position] = 1.0
    return features


def ncp_encode(sequence: str) -> np.ndarray:
    sequence = _normalize_sequence(sequence)
    return np.asarray([NCP_VALUES[base] for base in sequence], dtype=np.float32).T


def eiip_encode(sequence: str) -> np.ndarray:
    sequence = _normalize_sequence(sequence)
    return np.asarray([[EIIP_VALUES[base] for base in sequence]], dtype=np.float32)


def enac_encode(sequence: str, window_size: int = 5) -> np.ndarray:
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(f"window_size must be a positive odd integer, got: {window_size}")
    onehot = onehot_encode(sequence)
    radius = window_size // 2
    padded = np.pad(onehot, ((0, 0), (radius, radius)), mode="constant", constant_values=0.0)
    features = np.zeros_like(onehot, dtype=np.float32)
    for position in range(onehot.shape[1]):
        features[:, position] = padded[:, position:position + window_size].mean(axis=1)
    return features


def parse_feature_names(feature_names: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if feature_names is None:
        return list(SUPPORTED_FEATURES)
    if isinstance(feature_names, str):
        parsed = [item.strip().lower() for item in feature_names.split(",") if item.strip()]
    else:
        parsed = [str(item).strip().lower() for item in feature_names if str(item).strip()]
    if not parsed:
        raise ValueError("At least one handcrafted feature must be selected.")
    invalid = [name for name in parsed if name not in SUPPORTED_FEATURES]
    if invalid:
        supported = ", ".join(SUPPORTED_FEATURES)
        raise ValueError(f"Unsupported handcrafted feature(s): {invalid}. Supported features: {supported}")
    duplicates = sorted({name for name in parsed if parsed.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate handcrafted feature(s): {duplicates}")
    return parsed


def handcrafted_channel_count(feature_names: str | list[str] | tuple[str, ...] | None = None) -> int:
    return sum(FEATURE_CHANNELS[name] for name in parse_feature_names(feature_names))


def handcrafted_feature_matrix(
    sequence: str,
    enac_window_size: int = 5,
    feature_names: str | list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    sequence = _normalize_sequence(sequence)
    encoders = {
        "onehot": lambda: onehot_encode(sequence),
        "ncp": lambda: ncp_encode(sequence),
        "eiip": lambda: eiip_encode(sequence),
        "enac": lambda: enac_encode(sequence, window_size=enac_window_size),
    }
    selected_features = [encoders[name]() for name in parse_feature_names(feature_names)]
    return np.concatenate(selected_features, axis=0).astype(np.float32)


def _normalize_sequence(sequence: str) -> str:
    normalized = str(sequence).upper().replace("U", "T")
    invalid_bases = sorted(set(normalized) - set(BASES))
    if invalid_bases:
        raise ValueError(f"Handcrafted features require A/C/G/T bases only, got invalid bases: {invalid_bases}")
    return normalized
