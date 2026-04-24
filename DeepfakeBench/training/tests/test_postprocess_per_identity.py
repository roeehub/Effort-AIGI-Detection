"""Tests for arena/postprocess_per_identity.py (WS-P2.b).

Verifies the reducer preserves fp/tp counts end-to-end: the sum of
``real_fpr * n_real`` across identities must equal the suite-level fp count,
and similarly for fakes.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REDUCER = REPO_ROOT / "arena" / "postprocess_per_identity.py"


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "method",
        "label",
        "video_id",
        "avg_video_prob",
        "prediction",
        "group_key",
        "family_key",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def test_per_identity_fp_tp_roundtrip(tmp_path: Path) -> None:
    # Synthetic suite: 3 identities x 2 real/2 fake each, handpicked probs.
    tau = 0.5
    rows: list[dict] = []
    # identity A: 2 real (one flagged), 2 fake (both caught)
    rows += [
        {"method": "real", "label": 0, "video_id": "A_r0", "avg_video_prob": "0.10", "prediction": 0, "group_key": "A", "family_key": "real"},
        {"method": "real", "label": 0, "video_id": "A_r1", "avg_video_prob": "0.70", "prediction": 1, "group_key": "A", "family_key": "real"},
        {"method": "fake", "label": 1, "video_id": "A_f0", "avg_video_prob": "0.80", "prediction": 1, "group_key": "A", "family_key": "df40_fake"},
        {"method": "fake", "label": 1, "video_id": "A_f1", "avg_video_prob": "0.90", "prediction": 1, "group_key": "A", "family_key": "df40_fake"},
    ]
    # identity B: 2 real (none flagged), 2 fake (one caught)
    rows += [
        {"method": "real", "label": 0, "video_id": "B_r0", "avg_video_prob": "0.05", "prediction": 0, "group_key": "B", "family_key": "real"},
        {"method": "real", "label": 0, "video_id": "B_r1", "avg_video_prob": "0.15", "prediction": 0, "group_key": "B", "family_key": "real"},
        {"method": "fake", "label": 1, "video_id": "B_f0", "avg_video_prob": "0.30", "prediction": 0, "group_key": "B", "family_key": "df40_fake"},
        {"method": "fake", "label": 1, "video_id": "B_f1", "avg_video_prob": "0.60", "prediction": 1, "group_key": "B", "family_key": "df40_fake"},
    ]
    # identity C: 2 real (both flagged), 2 fake (none caught) — pathological
    rows += [
        {"method": "real", "label": 0, "video_id": "C_r0", "avg_video_prob": "0.85", "prediction": 1, "group_key": "C", "family_key": "real"},
        {"method": "real", "label": 0, "video_id": "C_r1", "avg_video_prob": "0.95", "prediction": 1, "group_key": "C", "family_key": "real"},
        {"method": "fake", "label": 1, "video_id": "C_f0", "avg_video_prob": "0.10", "prediction": 0, "group_key": "C", "family_key": "df40_fake"},
        {"method": "fake", "label": 1, "video_id": "C_f1", "avg_video_prob": "0.40", "prediction": 0, "group_key": "C", "family_key": "df40_fake"},
    ]
    report = tmp_path / "dev_real_rlp6_04_videos_report.csv"
    _write_csv(report, rows)

    out = tmp_path / "per_identity.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(REDUCER),
            "--reports",
            str(report),
            "--threshold",
            str(tau),
            "--out",
            str(out),
            "--verify",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"reducer failed: stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )

    with out.open() as f:
        out_rows = list(csv.DictReader(f))
    assert len(out_rows) == 3, out_rows
    by_group = {r["group_key"]: r for r in out_rows}

    # identity A: 2 real, 1 flagged -> fpr=0.5, 2 fake, both caught -> recall=1.0
    assert float(by_group["A"]["real_fpr"]) == 0.5
    assert float(by_group["A"]["fake_recall"]) == 1.0
    # identity B: 0/2 real flagged -> fpr=0, 1/2 fake -> recall=0.5
    assert float(by_group["B"]["real_fpr"]) == 0.0
    assert float(by_group["B"]["fake_recall"]) == 0.5
    # identity C: 2/2 real -> fpr=1.0, 0/2 fake -> recall=0
    assert float(by_group["C"]["real_fpr"]) == 1.0
    assert float(by_group["C"]["fake_recall"]) == 0.0

    # Sanity: aggregate over identities matches suite aggregate.
    # Suite fp = 1+0+2 = 3; fake_tp = 2+1+0 = 3; n_real = 6; n_fake = 6
    expected_real_fpr = 3 / 6
    expected_fake_recall = 3 / 6
    agg_real_fpr = sum(float(r["real_fpr"]) * float(r["n_real"]) for r in out_rows) / sum(
        float(r["n_real"]) for r in out_rows
    )
    agg_fake_recall = sum(
        float(r["fake_recall"]) * float(r["n_fake"]) for r in out_rows
    ) / sum(float(r["n_fake"]) for r in out_rows)
    assert abs(agg_real_fpr - expected_real_fpr) < 1e-9
    assert abs(agg_fake_recall - expected_fake_recall) < 1e-9

    # --verify must print OK to stderr
    assert "-> OK" in result.stderr, result.stderr


def test_verify_flag_detects_bogus_threshold(tmp_path: Path) -> None:
    """Sanity: with a threshold change the aggregate still equals the per-group
    sum (counts are a function of (rows, tau) and the reducer uses the same
    rows, so they must agree)."""
    rows = [
        {"method": "real", "label": 0, "video_id": "x", "avg_video_prob": "0.60", "prediction": 1, "group_key": "G", "family_key": "real"},
        {"method": "fake", "label": 1, "video_id": "y", "avg_video_prob": "0.40", "prediction": 0, "group_key": "G", "family_key": "df40_fake"},
    ]
    report = tmp_path / "s_c_videos_report.csv"
    _write_csv(report, rows)
    out = tmp_path / "per_identity.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(REDUCER),
            "--reports", str(report),
            "--threshold", "0.9",
            "--out", str(out),
            "--verify",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "-> OK" in result.stderr
