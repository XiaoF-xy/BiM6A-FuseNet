from __future__ import annotations

import sys
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_audit import audit_dataset  # noqa: E402


def test_data_audit_records_counts_hashes_duplicates_and_inherited_overlap():
    audit = audit_dataset(ROOT / "data" / "m6a_41nt" / "human_brain")
    assert audit["validation_status"] == "valid"
    assert audit["benchmark"]["total"] == 9210
    assert audit["independent_test"]["total"] == 9208
    assert len(audit["benchmark"]["sha256"]) == 64
    assert audit["cross_split_overlap"]["policy"] == "reported_not_removed_for_source_comparability"
    assert audit["cross_split_overlap"]["shared_sequences"] >= 1


def test_data_audit_rejects_manifest_count_mismatch(tmp_path: Path):
    data_dir = tmp_path / "example"
    data_dir.mkdir()
    sequence = "C" * 20 + "A" + "C" * 20
    for name in ("benchmark.csv", "independent_test.csv"):
        with (data_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sequence", "label"])
            writer.writeheader()
            writer.writerows([{"sequence": sequence, "label": 0}, {"sequence": sequence, "label": 1}])
    with (tmp_path / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["canonical_name", "benchmark_total", "independent_total"])
        writer.writeheader()
        writer.writerow({"canonical_name": "example", "benchmark_total": 4, "independent_total": 2})

    import pytest

    with pytest.raises(ValueError, match="do not match manifest"):
        audit_dataset(data_dir)
