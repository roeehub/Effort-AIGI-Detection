"""Tests for analysis/compare_teams_pools.py — local Teams-pool diagnostic."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest  # noqa: F401  (used by future tests)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "analysis" / "compare_teams_pools.py"


def test_cli_help_runs():
    """--help should exit 0 and mention all required CLI flags."""
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert out.returncode == 0, out.stderr
    for flag in ("--checkpoint", "--videos-per-group", "--frames-per-video",
                 "--output-dir", "--skip-pass", "--cache-dir", "--seed"):
        assert flag in out.stdout, f"missing flag {flag}"


def test_output_dir_layout(tmp_path):
    """Importing the module + calling _make_run_dir produces a timestamped subdir."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    run_dir = ctp._make_run_dir(tmp_path)
    assert run_dir.exists()
    assert run_dir.parent == tmp_path
    assert len(run_dir.name) >= 15


def test_sample_video_ids_is_deterministic():
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    pool = [f"vid_{i:04d}" for i in range(1000)]
    a = ctp._sample_video_ids(pool, n=50, seed=737)
    b = ctp._sample_video_ids(pool, n=50, seed=737)
    c = ctp._sample_video_ids(pool, n=50, seed=738)
    assert a == b
    assert a != c
    assert len(set(a)) == 50
    assert all(v in pool for v in a)


def test_sample_video_ids_handles_undersized_pool():
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    pool = [f"vid_{i:02d}" for i in range(10)]
    out = ctp._sample_video_ids(pool, n=50, seed=737)
    assert len(out) == 10
    assert sorted(out) == sorted(pool)


def test_pool_definitions_complete():
    """All 4 groups defined with bucket + label."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    groups = ctp.GROUP_DEFINITIONS
    expected = {
        "teams_ood_real", "teams_ood_fake",
        "proper_visomaster_teams_real", "proper_visomaster_teams_fake",
    }
    assert set(groups.keys()) == expected
    for g in groups.values():
        assert g.label in ("real", "fake")
        assert len(g.buckets) >= 1


def test_parse_gs_uri():
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    assert ctp._parse_gs_uri("gs://my-bucket/some/prefix") == ("my-bucket", "some/prefix")
    assert ctp._parse_gs_uri("gs://my-bucket") == ("my-bucket", "")


def test_compute_image_stats_synthetic():
    """A pure-grey 100x100 frame: brightness=0.5, contrast=0, sharpness=0."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    import numpy as np
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    stats = ctp._image_stats_from_array(img, file_size_bytes=10000)
    assert 0.45 < stats["brightness"] < 0.55
    assert stats["contrast"] < 0.01
    assert stats["sharpness"] < 0.5
    assert stats["resolution_w"] == 100
    assert stats["resolution_h"] == 100
    assert stats["bytes_per_pixel"] == pytest.approx(1.0, abs=1e-6)


def test_pairwise_ks_distance_basic():
    """KS-distance between identical samples is 0; between disjoint is 1."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    import numpy as np
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    c = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert ctp._ks_distance(a, b) == 0.0
    assert ctp._ks_distance(a, c) == 1.0


def test_ks_distance_handles_nan():
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    import numpy as np
    a = np.array([1.0, 2.0, float("nan"), 4.0])
    b = np.array([1.0, 2.0, 4.0, 5.0])
    result = ctp._ks_distance(a, b)
    assert 0.0 <= result <= 1.0


def test_face_geometry_no_face_returns_nan_row():
    """A pure-grey image has no face — should return face_detected=False."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    import numpy as np
    import math
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    geom = ctp._face_geometry_from_array(img)
    assert geom["face_detected"] is False
    assert math.isnan(geom["face_bbox_area_ratio"])
    assert math.isnan(geom["yaw_deg"])


def test_preprocess_image_shape_and_range():
    """Preprocess should output a (3, 224, 224) tensor with CLIP normalization."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    import numpy as np
    img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    t = ctp._preprocess_image(img)
    assert tuple(t.shape) == (3, 224, 224)
    assert -3.0 < float(t.mean()) < 3.0


def test_preprocess_image_all_mid_grey():
    """A grey image at 128 should normalize to a predictable small value."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    import numpy as np
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    t = ctp._preprocess_image(img)
    expected_r = (128/255 - 0.48145466) / 0.26862954
    assert abs(float(t[0].mean()) - expected_r) < 1e-4


def test_hard_sample_selection_buckets():
    """Selection produces 3 buckets (wrong/uncertain/right) per group."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    import pandas as pd
    rng_rows = []
    for g in ("teams_ood_real", "proper_visomaster_teams_fake"):
        label = "real" if "_real" in g else "fake"
        for i in range(200):
            rng_rows.append({
                "group": g, "label": label,
                "video_id": f"v{i:03d}", "frame_path": f"/tmp/{g}_{i}.jpg",
                "fake_prob": float((i % 100) / 100.0),
            })
    df = pd.DataFrame(rng_rows)
    sel = ctp._select_hard_samples(df, per_bucket=5, seed=737)
    for g in ("teams_ood_real", "proper_visomaster_teams_fake"):
        sub = sel[sel["group"] == g]
        for bucket in ("confidently_wrong", "uncertain", "confidently_right"):
            assert (sub["bucket"] == bucket).sum() <= 5
        # Each group should have at least one bucket populated.
        assert len(sub) > 0


def test_hard_sample_selection_empty_df():
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    import pandas as pd
    df = pd.DataFrame(columns=["group", "label", "video_id", "frame_path", "fake_prob"])
    sel = ctp._select_hard_samples(df, per_bucket=5, seed=737)
    assert len(sel) == 0


def test_html_report_renders_with_minimal_inputs(tmp_path):
    """Empty inputs should still produce a parseable HTML with all sections."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "thumbnails").mkdir()
    html_path = ctp.build_html_report(
        run_dir=run_dir,
        stats_section={"per_group": {}, "ks": {}, "feature_cols": [], "n_frames": 0},
        geometry_section={"per_group": {}, "ks": {}, "feature_cols": [], "n_frames": 0},
        model_section={"per_group": {}, "centroid_distances": {}, "n_frames": 0,
                       "embedding_dim": 0},
        gallery_section={"selected": 0, "rows": [], "per_bucket": 0},
    )
    assert html_path.exists()
    html = html_path.read_text()
    for section in ("Executive summary", "Image stats", "Face geometry",
                    "Model section", "Failures gallery"):
        assert section in html


def test_main_with_skip_all_passes_runs_clean(tmp_path, monkeypatch):
    """--skip-pass {stats,geometry,model,gallery} should produce empty report."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    import pandas as pd
    monkeypatch.setattr(ctp, "collect_frames",
                        lambda **kw: pd.DataFrame(
                            columns=["group", "label", "video_id", "frame_path"]))
    rc = ctp.main([
        "--output-dir", str(tmp_path),
        "--skip-pass", "stats",
        "--skip-pass", "geometry",
        "--skip-pass", "model",
        "--skip-pass", "gallery",
    ])
    assert rc == 0
    runs = list(tmp_path.iterdir())
    assert len(runs) == 1
    assert (runs[0] / "report.html").exists()
    assert (runs[0] / "stats.json").exists()


def test_html_report_renders_with_data(tmp_path):
    """A populated report should show the KS table with sorted entries."""
    sys.path.insert(0, str(REPO_ROOT))
    from analysis import compare_teams_pools as ctp
    run_dir = tmp_path / "run2"
    run_dir.mkdir()
    (run_dir / "thumbnails").mkdir()
    stats = {
        "per_group": {
            "teams_ood_real": {"brightness": {"mean": 0.45, "median": 0.5,
                                              "p5": 0.2, "p95": 0.7, "n": 100}},
            "proper_visomaster_teams_real": {
                "brightness": {"mean": 0.60, "median": 0.6, "p5": 0.3,
                               "p95": 0.85, "n": 100}},
        },
        "ks": {"teams_ood_real__vs__proper_visomaster_teams_real":
               {"brightness": 0.35}},
        "feature_cols": ["brightness"],
        "n_frames": 200,
    }
    html_path = ctp.build_html_report(
        run_dir=run_dir, stats_section=stats,
        geometry_section={"per_group": {}, "ks": {}, "feature_cols": [],
                          "n_frames": 0},
        model_section={"per_group": {}, "centroid_distances": {}, "n_frames": 0,
                       "embedding_dim": 0},
        gallery_section={"selected": 0, "rows": [], "per_bucket": 0},
    )
    html = html_path.read_text()
    assert "0.350" in html
    assert "brightness" in html
    assert 'class="hi"' in html  # ks=0.35 > 0.2 should be highlighted
