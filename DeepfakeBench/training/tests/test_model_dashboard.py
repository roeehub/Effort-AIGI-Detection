from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from viewer.model_dashboard import ModelDashboard, frame_key
from viewer import server


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _build_fixture(tmp_path: Path) -> ModelDashboard:
    image = tmp_path / "frames" / "frame_a.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    # Minimal valid JPEG.
    image.write_bytes(
        bytes.fromhex(
            "ffd8ffe000104a46494600010101006000600000ffdb004300"
            "080606070605080707070909080a0c140d0c0b0b0c1912130f"
            "141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c3031"
            "3434341f27393d38323c2e333432ffc0000b0800010001010111"
            "00ffc40014000100000000000000000000000000000000000000"
            "ffda0008010100003f00d2cf20ffd9"
        )
    )

    _write(
        tmp_path / "analysis/scorecard_fixture/checkpoint_summary.csv",
        "checkpoint_key,checkpoint_path,selected_threshold,lockbox_real_fpr,teams_fake_all_dev__fake_recall,promotion_rank\n"
        "RUN_A,gs://ckpt-a,0.7,0.02,0.5,1\n",
    )
    _write(
        tmp_path / "analysis/scorecard_fixture/selected_threshold_scorecard.csv",
        "checkpoint_key,suite_name,n_videos,real_fpr,fake_recall,mean_prob,p90_prob\n"
        "RUN_A,teams_real_all_dev,10,0.02,,0.1,0.2\n"
        "RUN_A,teams_fake_all_dev,12,,0.5,0.8,0.9\n",
    )
    _write(
        tmp_path / "analysis/scorecard_fixture/scorecard.json",
        json.dumps({"scorecard_rows": [{"checkpoint_key": "RUN_A", "suite_name": "teams_fake_all_dev"}]}),
    )
    _write(
        tmp_path / "analysis/scorecard_fixture/promotion_contract.json",
        json.dumps({"contract": {"target_fake_recall_min": 0.7}}),
    )
    _write(
        tmp_path / "analysis/manifold/triptych_coords_tsne.csv",
        "gcs_uri,local_path,label,method,identity_key,video_id,clip_capture_mode,face_pixel_area,checkpoint,tsne_x,tsne_y,prob_fake\n"
        f"gs://bucket/a.jpg,{image},fake,m1,id1,vid1,webcam,12000,RUN_ALIAS,1.0,2.0,0.99\n",
    )
    _write(
        tmp_path / "analysis/face/run_a_invariance.csv",
        "frame_id,frame_path,tightness,prob_fake,predicted_label\n"
        f"f1,{image},0.7,0.1,real\n"
        f"f1,{image},1.0,0.9,fake\n",
    )
    _write(
        tmp_path / "analysis/domain/summary.json",
        json.dumps({
            "results": {
                "RUN_ALIAS": {
                    "classes": [1, 2],
                    "macro_ovr_auc": 0.99,
                    "confusion_matrix": [[5, 1], [0, 6]],
                }
            }
        }),
    )
    manifest = {
        "runs": [{
            "id": "run_a",
            "label": "Run A",
            "checkpoint": "gs://ckpt-a",
            "checkpoint_key": "RUN_A",
            "aliases": ["RUN_ALIAS"],
            "artifacts": {
                "scorecard_dir": "analysis/scorecard_fixture",
                "manifold_tsne": "analysis/manifold/triptych_coords_tsne.csv",
                "manifold_checkpoint": "RUN_ALIAS",
                "face_size_invariance": "analysis/face/run_a_invariance.csv",
                "domain_probe_summary": "analysis/domain/summary.json",
                "domain_probe_key": "RUN_ALIAS",
            },
        }],
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest))
    return ModelDashboard(training_dir=tmp_path, manifest_path=manifest_path)


def test_frame_key_prefers_gcs_uri():
    a = frame_key({"gcs_uri": "gs://bucket/frame.jpg", "frame_path": "/tmp/other.jpg"})
    b = frame_key({"gcs_uri": "gs://bucket/frame.jpg"})
    c = frame_key({"frame_path": "/tmp/other.jpg"})
    assert a == b
    assert a != c


def test_dashboard_indexes_fixture_artifacts(tmp_path):
    dashboard = _build_fixture(tmp_path)
    runs = dashboard.list_runs()
    assert runs[0]["id"] == "run_a"
    assert runs[0]["evidence"]["scorecard"] is True
    assert runs[0]["evidence"]["manifold"] is True

    scorecard = dashboard.scorecard("run_a")
    assert scorecard["checkpoint_summary"]["selected_threshold"] == 0.7
    assert len(scorecard["selected_threshold_rows"]) == 2

    manifold = dashboard.manifold("run_a")
    assert len(manifold["points"]) == 1
    assert manifold["points"][0]["score"] is None
    assert "not checkpoint-specific" in manifold["points"][0]["score_note"]

    face = dashboard.face_size_invariance("run_a")
    assert face["summary"]["flip_rate"] == 1.0
    assert face["frames"][0]["image_url"]

    domain = dashboard.domain_probe("run_a")
    assert domain["result"]["macro_ovr_auc"] == 0.99


def test_model_dashboard_api_smoke(tmp_path):
    dashboard = _build_fixture(tmp_path)
    server._model_dashboard = dashboard
    client = server.app.test_client()

    assert client.get("/api/model-runs").status_code == 200
    summary = client.get("/api/model-runs/run_a/summary").get_json()
    assert summary["run"]["id"] == "run_a"

    manifold = client.get("/api/model-runs/run_a/manifold?color=face_area_bucket").get_json()
    assert manifold["points"][0]["color_value"] == "10-30k"

    frames = client.get("/api/model-runs/run_a/frames?kind=sensitive").get_json()
    assert frames["available"] is True

    server._model_dashboard = None
