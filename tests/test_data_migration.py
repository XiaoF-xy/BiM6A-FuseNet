from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "m6a_41nt"
SOURCE_ROOT = ROOT.parent / "BiRNA_m6A" / "data" / "m6A_41bp"

DATASETS = {
    "H_b": ("human_brain", "Human_Brain", 9210, 9208),
    "H_k": ("human_kidney", "Human_Kidney", 9148, 9146),
    "H_l": ("human_liver", "Human_Liver", 5268, 5268),
    "M_b": ("mouse_brain", "Mouse_brain", 16050, 16050),
    "M_h": ("mouse_heart", "Mouse_heart", 4402, 4400),
    "M_k": ("mouse_kidney", "Mouse_kidney", 7906, 7904),
    "M_l": ("mouse_liver", "Mouse_liver", 8266, 8266),
    "M_t": ("mouse_testis", "Mouse_test", 9414, 9412),
    "R_b": ("rat_brain", "rat_brain", 4704, 4702),
    "R_k": ("rat_kidney", "rat_kidney", 6866, 6864),
    "R_l": ("rat_liver", "rat_liver", 3524, 3524),
}


def read_rows(path: Path) -> list[tuple[str, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [(row["sequence"], int(row["label"])) for row in csv.DictReader(handle)]


def test_semantic_dataset_files_are_complete_and_valid():
    for _alias, (canonical, _source, benchmark_count, independent_count) in DATASETS.items():
        benchmark = read_rows(DATA_ROOT / canonical / "benchmark.csv")
        independent = read_rows(DATA_ROOT / canonical / "independent_test.csv")

        assert len(benchmark) == benchmark_count
        assert len(independent) == independent_count
        assert Counter(label for _, label in benchmark)[0] == benchmark_count // 2
        assert Counter(label for _, label in independent)[0] == independent_count // 2
        assert all(len(sequence) == 41 and sequence[20] == "A" for sequence, _ in benchmark + independent)
        assert all(set(sequence) <= set("ACGT") for sequence, _ in benchmark + independent)


def test_manifest_matches_all_dataset_files():
    with (DATA_ROOT / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = {row["dataset_id"]: row for row in csv.DictReader(handle)}

    assert set(manifest) == set(DATASETS)
    for alias, (canonical, _source, benchmark_count, independent_count) in DATASETS.items():
        row = manifest[alias]
        assert row["canonical_name"] == canonical
        assert int(row["benchmark_total"]) == benchmark_count
        assert int(row["independent_total"]) == independent_count
        assert int(row["sequence_length"]) == 41
        assert row["validation_status"] == "valid"


@pytest.mark.skipif(not SOURCE_ROOT.exists(), reason="BiRNA_m6A source tree is not available")
def test_migrated_rows_exactly_match_birna_m6a_source():
    for _alias, (canonical, source_name, _benchmark_count, _independent_count) in DATASETS.items():
        assert read_rows(DATA_ROOT / canonical / "benchmark.csv") == read_rows(
            SOURCE_ROOT / source_name / "train.csv"
        )
        assert read_rows(DATA_ROOT / canonical / "independent_test.csv") == read_rows(
            SOURCE_ROOT / source_name / "test.csv"
        )
