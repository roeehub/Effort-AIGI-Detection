"""Read-only artifact index for the local model diagnostics dashboard."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


TRAINING_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "model_dashboard_runs.yaml"


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_int(value: Any) -> Optional[int]:
    f = _safe_float(value)
    return int(f) if f is not None else None


def _coerce_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        if "." not in text and "e" not in text.lower():
            return int(text)
        return float(text)
    except ValueError:
        return value


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [
            {k: _coerce_scalar(v) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _frame_source(row: Dict[str, Any]) -> Optional[str]:
    for key in ("gcs_uri", "frame_path", "blob_path", "local_path", "frame_id"):
        value = row.get(key)
        if value:
            return str(value).replace("\\", "/")
    return None


def frame_key(row: Dict[str, Any]) -> Optional[str]:
    explicit = row.get("_frame_key")
    if explicit:
        return str(explicit)
    source = _frame_source(row)
    if not source:
        return None
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:20]


def _variant_frame_key(parent_key: str, tightness: Any) -> str:
    value = _safe_float(tightness)
    suffix = f"{value:.4f}" if value is not None else str(tightness)
    return hashlib.sha1(f"{parent_key}|tightness={suffix}".encode("utf-8")).hexdigest()[:20]


def _face_area_bucket(value: Any) -> str:
    area = _safe_float(value)
    if area is None:
        return "unknown"
    if area < 10_000:
        return "<10k"
    if area < 30_000:
        return "10-30k"
    if area < 50_000:
        return "30-50k"
    if area < 75_000:
        return "50-75k"
    if area < 100_000:
        return "75-100k"
    return ">=100k"


def _label_text(value: Any) -> str:
    if value in (1, "1", "fake", "Fake", "FAKE"):
        return "fake"
    if value in (0, "0", "real", "Real", "REAL"):
        return "real"
    return str(value) if value is not None else "unknown"


def _stratified_sample_points(
    points: List[Dict[str, Any]],
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if len(points) <= limit:
        return points
    strata: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for point in points:
        key = (
            str(point.get("label") or "unknown"),
            str(point.get("method") or "unknown"),
            str(point.get("capture_mode") or "unknown"),
            str(point.get("face_area_bucket") or "unknown"),
        )
        strata[key].append(point)

    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    buckets = list(strata.values())
    for bucket in buckets:
        rng.shuffle(bucket)
        selected.append(bucket.pop())
        if len(selected) >= limit:
            return selected

    remaining_slots = limit - len(selected)
    total_remaining = sum(len(bucket) for bucket in buckets)
    if total_remaining <= 0 or remaining_slots <= 0:
        return selected[:limit]

    extras: List[Dict[str, Any]] = []
    for bucket in buckets:
        if not bucket:
            continue
        share = max(1, round(len(bucket) / total_remaining * remaining_slots))
        extras.extend(bucket[:share])
    rng.shuffle(extras)
    selected.extend(extras[:remaining_slots])
    return selected[:limit]


@dataclass
class ModelRun:
    id: str
    label: str
    run_id: str = ""
    checkpoint: str = ""
    checkpoint_key: str = ""
    aliases: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    recipe: str = ""
    trainer_metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    notes: List[Dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "ModelRun":
        aliases = list(row.get("aliases") or [])
        if row.get("checkpoint_key") and row["checkpoint_key"] not in aliases:
            aliases.append(row["checkpoint_key"])
        if row.get("id") and row["id"] not in aliases:
            aliases.append(row["id"])
        return cls(
            id=row["id"],
            label=row.get("label", row["id"]),
            run_id=row.get("run_id", ""),
            checkpoint=row.get("checkpoint", ""),
            checkpoint_key=row.get("checkpoint_key", ""),
            aliases=aliases,
            tags=list(row.get("tags") or []),
            recipe=row.get("recipe", ""),
            trainer_metrics=dict(row.get("trainer_metrics") or {}),
            artifacts=dict(row.get("artifacts") or {}),
            notes=list(row.get("notes") or []),
        )

    def to_dict(self, evidence: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "run_id": self.run_id,
            "checkpoint": self.checkpoint,
            "checkpoint_key": self.checkpoint_key,
            "aliases": self.aliases,
            "tags": self.tags,
            "recipe": self.recipe,
            "trainer_metrics": self.trainer_metrics,
            "evidence": evidence or {},
        }


class ModelDashboard:
    """Indexes local analysis artifacts without launching any jobs."""

    def __init__(
        self,
        training_dir: Path = TRAINING_DIR,
        manifest_path: Path = DEFAULT_MANIFEST,
    ):
        self.training_dir = Path(training_dir)
        self.manifest_path = Path(manifest_path)
        self.runs: Dict[str, ModelRun] = {}
        self._frame_paths: Dict[Tuple[str, str], Path] = {}
        self._frame_rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._scorecard_cache: Dict[str, Dict[str, Any]] = {}
        self._manifold_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._face_cache: Dict[str, Dict[str, Any]] = {}
        self._domain_cache: Dict[str, Dict[str, Any]] = {}
        self._lockbox_note_cache: Optional[List[Dict[str, str]]] = None
        self._load_manifest()

    def _load_manifest(self) -> None:
        data = yaml.safe_load(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
        self.runs = {
            run.id: run
            for run in (ModelRun.from_dict(row) for row in data.get("runs", []))
        }

    def _path(self, value: Optional[str]) -> Optional[Path]:
        if not value:
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.training_dir / path).resolve()

    def _artifact_path(self, run: ModelRun, key: str) -> Optional[Path]:
        value = run.artifacts.get(key)
        if isinstance(value, str):
            return self._path(value)
        return None

    def _artifact_paths(self, run: ModelRun, key: str) -> List[Path]:
        value = run.artifacts.get(key)
        if isinstance(value, str):
            p = self._path(value)
            return [p] if p else []
        if isinstance(value, list):
            return [p for p in (self._path(str(v)) for v in value) if p]
        return []

    def _resolve_image_path(self, row: Dict[str, Any]) -> Optional[Path]:
        for key in ("local_path", "frame_path"):
            value = row.get(key)
            if not value:
                continue
            path = Path(str(value))
            if not path.is_absolute():
                path = self.training_dir / path
            path = path.resolve()
            if path.exists() and path.is_file():
                return path
        return None

    def _register_frame(self, run_id: str, row: Dict[str, Any]) -> Optional[str]:
        key = frame_key(row)
        if not key:
            return None
        normalized = dict(row)
        normalized["frame_key"] = key
        self._frame_rows[(run_id, key)] = normalized
        image_path = self._resolve_image_path(row)
        if image_path:
            self._frame_paths[(run_id, key)] = image_path
        return key

    def _run(self, run_id: str) -> ModelRun:
        if run_id not in self.runs:
            raise KeyError(f"Unknown model run: {run_id}")
        return self.runs[run_id]

    def _evidence(self, run: ModelRun) -> Dict[str, bool]:
        score_dir = self._artifact_path(run, "scorecard_dir")
        domain = self._artifact_path(run, "domain_probe_summary")
        return {
            "scorecard": bool(score_dir and (score_dir / "checkpoint_summary.csv").exists()),
            "manifold": bool(self._artifact_path(run, "manifold_tsne")),
            "face_size_invariance": bool(self._artifact_path(run, "face_size_invariance")),
            "domain_probe": bool(domain and domain.exists() and run.artifacts.get("domain_probe_key")),
            "per_frame_scores": bool(self._artifact_paths(run, "per_frame_scores")),
        }

    def list_runs(self) -> List[Dict[str, Any]]:
        return [run.to_dict(self._evidence(run)) for run in self.runs.values()]

    def summary(self, run_id: str) -> Dict[str, Any]:
        run = self._run(run_id)
        scorecard = self.scorecard(run_id)
        face = self.face_size_invariance(run_id)
        domain = self.domain_probe(run_id)
        notes = self._evidence_notes(run, face, domain)
        return {
            "run": run.to_dict(self._evidence(run)),
            "scorecard_brief": scorecard.get("checkpoint_summary"),
            "face_size_brief": face.get("summary"),
            "domain_probe_brief": domain.get("result"),
            "evidence_notes": notes,
        }

    def scorecard(self, run_id: str) -> Dict[str, Any]:
        if run_id in self._scorecard_cache:
            return self._scorecard_cache[run_id]
        run = self._run(run_id)
        score_dir = self._artifact_path(run, "scorecard_dir")
        if not score_dir or not score_dir.exists():
            result = {"available": False, "reason": "No local scorecard artifact configured"}
            self._scorecard_cache[run_id] = result
            return result

        checkpoint_rows = _read_csv_rows(score_dir / "checkpoint_summary.csv")
        selected_rows = _read_csv_rows(score_dir / "selected_threshold_scorecard.csv")
        wide_rows = _read_csv_rows(score_dir / "scorecard.wide.csv")
        contract = _read_json(score_dir / "promotion_contract.json").get("contract", {})
        scorecard_json = _read_json(score_dir / "scorecard.json")

        checkpoint_row = self._pick_checkpoint_row(run, checkpoint_rows)
        checkpoint_key = checkpoint_row.get("checkpoint_key") if checkpoint_row else run.checkpoint_key

        if checkpoint_key:
            selected_rows = [r for r in selected_rows if r.get("checkpoint_key") == checkpoint_key]
            wide_rows = [r for r in wide_rows if r.get("checkpoint_key") == checkpoint_key]
            scorecard_at_0p5 = [
                r for r in scorecard_json.get("scorecard_rows", [])
                if r.get("checkpoint_key") == checkpoint_key
            ]
        else:
            scorecard_at_0p5 = scorecard_json.get("scorecard_rows", [])

        result = {
            "available": bool(checkpoint_row or selected_rows or scorecard_at_0p5),
            "artifact_dir": str(score_dir.relative_to(self.training_dir)) if score_dir.is_relative_to(self.training_dir) else str(score_dir),
            "contract": contract,
            "checkpoint_summary": checkpoint_row,
            "selected_threshold_rows": selected_rows,
            "scorecard_at_0p5": scorecard_at_0p5,
            "wide_rows": wide_rows,
        }
        self._scorecard_cache[run_id] = result
        return result

    def _pick_checkpoint_row(self, run: ModelRun, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        aliases = set(run.aliases)
        if run.checkpoint_key:
            aliases.add(run.checkpoint_key)
        for row in rows:
            if row.get("checkpoint_key") in aliases:
                return row
        if run.checkpoint:
            for row in rows:
                if row.get("checkpoint_path") == run.checkpoint:
                    return row
        return rows[0] if len(rows) == 1 else None

    def manifold(
        self,
        run_id: str,
        reducer: str = "tsne",
        color: str = "label",
        limit: int = 800,
        seed: int = 737,
    ) -> Dict[str, Any]:
        cache_key = (run_id, reducer)
        if cache_key in self._manifold_cache:
            base = self._manifold_cache[cache_key]
        else:
            base = self._load_manifold(run_id, reducer)
            self._manifold_cache[cache_key] = base
        base_points = base.get("points", [])
        sampled_points = _stratified_sample_points(base_points, limit=max(1, min(int(limit), 5000)), seed=seed)
        points = []
        for row in sampled_points:
            out = dict(row)
            out["color_value"] = self._color_value(row, color)
            points.append(out)
        return {
            "available": base.get("available", False),
            "reason": base.get("reason"),
            "reducer": reducer,
            "color": color,
            "points": points,
            "total_points": len(base_points),
            "returned_points": len(points),
            "sampling": {
                "strategy": "deterministic stratified sample",
                "limit": max(1, min(int(limit), 5000)),
                "seed": seed,
                "strata": ["label", "method", "capture_mode", "face_area_bucket"],
            },
            "color_values": sorted({str(p.get("color_value", "unknown")) for p in points}),
            "artifact": base.get("artifact"),
        }

    def _load_manifold(self, run_id: str, reducer: str) -> Dict[str, Any]:
        run = self._run(run_id)
        artifact_key = f"manifold_{reducer}"
        path = self._artifact_path(run, artifact_key) or self._artifact_path(run, "manifold_tsne")
        if not path or not path.exists():
            return {"available": False, "reason": "No local manifold artifact configured", "points": []}
        aliases = set(run.aliases)
        checkpoint_alias = run.artifacts.get("manifold_checkpoint")
        if checkpoint_alias:
            aliases.add(str(checkpoint_alias))

        rows = _read_csv_rows(path)
        points = []
        for row in rows:
            if aliases and row.get("checkpoint") not in aliases:
                continue
            key = self._register_frame(run_id, row)
            gcs_uri = row.get("gcs_uri") or row.get("frame_path") or ""
            # Image URL: prefer local registered frame, fall back to GCS proxy for gs:// URIs
            if key and (run_id, key) in self._frame_paths:
                image_url = f"/api/model-runs/{run_id}/frame/{key}"
            elif isinstance(gcs_uri, str) and gcs_uri.startswith("gs://"):
                bp = gcs_uri[len("gs://"):]
                if "/" in bp:
                    bucket, blob_path = bp.split("/", 1)
                    image_url = f"/api/frame/{bucket}/{blob_path}"
                else:
                    image_url = ""
            else:
                image_url = ""
            points.append({
                "frame_key": key,
                "x": _safe_float(row.get(f"{reducer}_x") or row.get("tsne_x") or row.get("umap_x")),
                "y": _safe_float(row.get(f"{reducer}_y") or row.get("tsne_y") or row.get("umap_y")),
                "manifold_x": _safe_float(row.get(f"{reducer}_x") or row.get("tsne_x") or row.get("umap_x")),
                "manifold_y": _safe_float(row.get(f"{reducer}_y") or row.get("tsne_y") or row.get("umap_y")),
                "label": _label_text(row.get("label")),
                "method": row.get("method") or "unknown",
                "identity_key": row.get("identity_key") or "",
                "video_id": row.get("video_id") or "",
                "gcs_uri": gcs_uri,
                "capture_mode": row.get("clip_capture_mode") or "unknown",
                "face_area_bucket": _face_area_bucket(row.get("face_pixel_area")),
                "face_pixel_area": _safe_float(row.get("face_pixel_area")),
                "sharpness_laplacian": _safe_float(row.get("sharpness_laplacian")),
                "caught_by_subset": row.get("caught_by_subset") or "",
                "local_image": bool(key and (run_id, key) in self._frame_paths),
                "image_url": image_url,
                "score": None,
                "score_note": "not available: triptych prob_fake metadata is not checkpoint-specific",
                "artifact_source": "manifold",
                "manifold_status": "in current t-SNE sample",
            })
        points = [p for p in points if p["x"] is not None and p["y"] is not None]
        return {
            "available": bool(points),
            "artifact": str(path.relative_to(self.training_dir)) if path.is_relative_to(self.training_dir) else str(path),
            "points": points,
        }

    def _color_value(self, row: Dict[str, Any], color: str) -> str:
        if color == "method":
            return str(row.get("method") or "unknown")
        if color == "capture_mode":
            return str(row.get("capture_mode") or "unknown")
        if color == "face_area_bucket":
            return str(row.get("face_area_bucket") or "unknown")
        if color == "score_bin":
            return "score unavailable"
        if color == "correctness":
            return "correctness unavailable"
        if color == "caught_by_subset":
            return str(row.get("caught_by_subset") or "unknown")
        return str(row.get("label") or "unknown")

    def frames(self, run_id: str, kind: str = "sensitive", limit: int = 48) -> Dict[str, Any]:
        if kind == "sensitive":
            face = self.face_size_invariance(run_id)
            frames = list(face.get("frames", []))
            frames.sort(key=lambda r: (r.get("delta_prob") or 0, r.get("flip")), reverse=True)
            return {
                "available": bool(frames),
                "kind": kind,
                "score_rows_available": True,
                "frames": frames[:limit],
            }
        if kind in {"false_positive", "false_negative", "near_threshold", "high_confidence"}:
            rows = self._per_frame_scores(run_id, kind, limit)
            return {
                "available": bool(rows),
                "kind": kind,
                "score_rows_available": bool(rows),
                "reason": None if rows else "No local per-frame score reports configured for this run",
                "frames": rows,
            }

        manifold = self.manifold(run_id)
        rows = manifold.get("points", [])
        return {
            "available": bool(rows),
            "kind": "manifold",
            "score_rows_available": False,
            "frames": rows[:limit],
        }

    def _per_frame_scores(self, run_id: str, kind: str, limit: int) -> List[Dict[str, Any]]:
        run = self._run(run_id)
        paths = self._artifact_paths(run, "per_frame_scores")
        if not paths:
            return []
        threshold = _safe_float((self.scorecard(run_id).get("checkpoint_summary") or {}).get("selected_threshold")) or 0.5
        rows = []
        for path in paths:
            for row in _read_csv_rows(path):
                prob = _safe_float(row.get("prob_fake") or row.get("frame_prob"))
                label = _label_text(row.get("label"))
                if prob is None:
                    continue
                if kind == "false_positive" and not (label == "real" and prob >= threshold):
                    continue
                if kind == "false_negative" and not (label == "fake" and prob < threshold):
                    continue
                if kind == "near_threshold":
                    row["threshold_distance"] = abs(prob - threshold)
                key = self._register_frame(run_id, row)
                gcs_uri = row.get("frame_path") or row.get("gcs_uri") or row.get("blob_path") or ""
                if (run_id, key) in self._frame_paths:
                    image_url = f"/api/model-runs/{run_id}/frame/{key}"
                elif isinstance(gcs_uri, str) and gcs_uri.startswith("gs://"):
                    bucket_and_path = gcs_uri[len("gs://"):]
                    if "/" in bucket_and_path:
                        bucket, blob_path = bucket_and_path.split("/", 1)
                        image_url = f"/api/frame/{bucket}/{blob_path}"
                    else:
                        image_url = ""
                else:
                    image_url = ""
                rows.append({
                    "frame_key": key,
                    "label": label,
                    "method": row.get("method") or row.get("pool") or "unknown",
                    "prob_fake": prob,
                    "threshold": threshold,
                    "image_url": image_url,
                    "gcs_uri": gcs_uri,
                })
        if kind == "near_threshold":
            rows.sort(key=lambda r: r.get("threshold_distance", 999))
        else:
            rows.sort(key=lambda r: abs((r.get("prob_fake") or 0) - threshold), reverse=True)
        return rows[:limit]

    def face_size_invariance(self, run_id: str) -> Dict[str, Any]:
        if run_id in self._face_cache:
            return self._face_cache[run_id]
        run = self._run(run_id)
        path = self._artifact_path(run, "face_size_invariance")
        if not path or not path.exists():
            result = {"available": False, "reason": "No local face-size invariance artifact configured"}
            self._face_cache[run_id] = result
            return result

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in _read_csv_rows(path):
            fid = str(row.get("frame_id") or _frame_source(row) or "unknown")
            grouped[fid].append(row)

        frames = []
        deltas = []
        flips = 0
        for fid, rows in grouped.items():
            rows.sort(key=lambda r: _safe_float(r.get("tightness")) or 0)
            probs = [_safe_float(r.get("prob_fake")) for r in rows]
            probs = [p for p in probs if p is not None]
            if not probs:
                continue
            labels = {_label_text(r.get("predicted_label")) for r in rows}
            flip = len(labels) > 1
            flips += int(flip)
            delta = max(probs) - min(probs)
            deltas.append(delta)
            first = rows[0]
            native = min(
                rows,
                key=lambda r: abs((_safe_float(r.get("tightness")) or 1.0) - 1.0),
            )
            method = first.get("pool") or Path(str(first.get("frame_path", fid))).parent.name
            frame_record = {
                "frame_id": fid,
                "label": "unknown",
                "method": method,
                "frame_path": first.get("frame_path"),
                "local_path": first.get("local_path"),
                "gcs_uri": first.get("gcs_uri"),
                "blob_path": first.get("blob_path"),
                "flip": flip,
                "delta_prob": delta,
                "min_prob": min(probs),
                "max_prob": max(probs),
                "native_prob_fake": _safe_float(native.get("prob_fake")),
                "native_predicted_label": _label_text(native.get("predicted_label")),
                "native_tightness": _safe_float(native.get("tightness")),
                "artifact_source": "face_size_invariance",
                "manifold_status": "not in manifold sample unless this frame also appears in triptych artifacts",
            }
            parent_key = frame_key(frame_record)
            variant_points = []
            for r in rows:
                tightness = _safe_float(r.get("tightness"))
                variant_record = {
                    "_frame_key": _variant_frame_key(parent_key or fid, tightness),
                    "parent_frame_key": parent_key,
                    "frame_id": fid,
                    "label": "unknown",
                    "method": method,
                    "frame_path": r.get("frame_path") or first.get("frame_path"),
                    "local_path": r.get("local_path") or first.get("local_path"),
                    "gcs_uri": r.get("gcs_uri") or first.get("gcs_uri"),
                    "blob_path": r.get("blob_path") or first.get("blob_path"),
                    "crop_tightness": tightness,
                    "tightness": tightness,
                    "prob_fake": _safe_float(r.get("prob_fake")),
                    "predicted_label": _label_text(r.get("predicted_label")),
                    "threshold": 0.5,
                    "flip": flip,
                    "delta_prob": delta,
                    "min_prob": min(probs),
                    "max_prob": max(probs),
                    "native_prob_fake": _safe_float(native.get("prob_fake")),
                    "native_predicted_label": _label_text(native.get("predicted_label")),
                    "native_tightness": _safe_float(native.get("tightness")),
                    "artifact_source": "face_size_variant",
                    "manifold_status": "not in manifold sample unless this frame also appears in triptych artifacts",
                }
                variant_key = self._register_frame(run_id, variant_record)
                variant_points.append({
                    "frame_key": variant_key,
                    "parent_frame_key": parent_key,
                    "frame_id": fid,
                    "tightness": tightness,
                    "crop_tightness": tightness,
                    "prob_fake": _safe_float(r.get("prob_fake")),
                    "predicted_label": _label_text(r.get("predicted_label")),
                    "image_url": f"/api/model-runs/{run_id}/frame/{variant_key}" if variant_key and (run_id, variant_key) in self._frame_paths else "",
                })
            frame_record["points"] = variant_points
            key = self._register_frame(run_id, frame_record)
            frame_record["frame_key"] = key
            for point in variant_points:
                point["parent_frame_key"] = key
            frame_record["image_url"] = f"/api/model-runs/{run_id}/frame/{key}" if key and (run_id, key) in self._frame_paths else ""
            frames.append(frame_record)

        n = len(frames)
        summary = {
            "n_frames": n,
            "flip_count": flips,
            "flip_rate": flips / n if n else None,
            "median_abs_delta": statistics.median(deltas) if deltas else None,
            "artifact": str(path.relative_to(self.training_dir)) if path.is_relative_to(self.training_dir) else str(path),
        }
        result = {"available": True, "summary": summary, "frames": frames}
        self._face_cache[run_id] = result
        return result

    def domain_probe(self, run_id: str) -> Dict[str, Any]:
        if run_id in self._domain_cache:
            return self._domain_cache[run_id]
        run = self._run(run_id)
        path = self._artifact_path(run, "domain_probe_summary")
        probe_key = run.artifacts.get("domain_probe_key")
        if not path or not path.exists() or not probe_key:
            result = {"available": False, "reason": "No local domain probe artifact configured"}
            self._domain_cache[run_id] = result
            return result
        data = _read_json(path)
        result = data.get("results", {}).get(str(probe_key))
        if not result:
            out = {"available": False, "reason": f"No domain-probe result for key {probe_key}"}
        else:
            out = {
                "available": True,
                "artifact": str(path.relative_to(self.training_dir)) if path.is_relative_to(self.training_dir) else str(path),
                "probe_key": probe_key,
                "result": result,
            }
        self._domain_cache[run_id] = out
        return out

    def frame_detail(self, run_id: str, key: str) -> Dict[str, Any]:
        if (run_id, key) not in self._frame_rows:
            # Populate common indexes lazily before giving up.
            self.manifold(run_id)
            self.face_size_invariance(run_id)
            self.frames(run_id, kind="sensitive")
        row = self._frame_rows.get((run_id, key))
        if not row:
            raise KeyError(f"Unknown frame key for run {run_id}: {key}")
        return {
            "frame_key": key,
            "metadata": row,
            "image_url": f"/api/model-runs/{run_id}/frame/{key}" if (run_id, key) in self._frame_paths else "",
        }

    def frame_path(self, run_id: str, key: str) -> Path:
        if (run_id, key) not in self._frame_paths:
            self.frame_detail(run_id, key)
        path = self._frame_paths.get((run_id, key))
        if not path:
            raise KeyError(f"No local image available for {run_id}/{key}")
        return path

    def _evidence_notes(
        self,
        run: ModelRun,
        face: Dict[str, Any],
        domain: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        notes = list(run.notes)
        notes.extend(self._lockbox_data_notes())
        if face.get("available"):
            summary = face.get("summary") or {}
            rate = summary.get("flip_rate")
            delta = summary.get("median_abs_delta")
            if rate is not None:
                notes.append({
                    "title": "Crop-Size Sensitivity",
                    "status": "measured",
                    "text": f"Flip rate {rate:.1%}; median |delta prob_fake| {delta:.3f} on the local tightness sweep.",
                })
        if domain.get("available"):
            result = domain.get("result") or {}
            auc = result.get("macro_ovr_auc")
            if auc is not None:
                notes.append({
                    "title": "Domain Separability",
                    "status": "measured",
                    "text": f"Frozen-feature domain macro-OVR AUC {auc:.4f}; values near 1.0 mean the domain manifold remains linearly separable.",
                })
        return notes

    def _lockbox_data_notes(self) -> List[Dict[str, str]]:
        if self._lockbox_note_cache is not None:
            return self._lockbox_note_cache
        notes: List[Dict[str, str]] = []
        summary_path = self.training_dir / "analysis/modern_lockbox_v2_2026-04-27/p8a_lockbox_v2_summary.json"
        if summary_path.exists():
            data = _read_json(summary_path)
            real = data.get("lockbox_real") or {}
            fake = data.get("lockbox_fake") or {}
            filt = data.get("filter_definition") or {}
            drop_modes = ", ".join(filt.get("drop_clip_capture_mode_in") or [])
            notes.append({
                "title": "Lockbox Data Hygiene",
                "status": "warning",
                "text": (
                    f"Modern_lockbox_v2 keeps {real.get('n_v2_recommended')}/{real.get('n_baseline')} real "
                    f"and {fake.get('n_v2_recommended')}/{fake.get('n_baseline')} fake lockbox frames. "
                    f"It drops capture modes {{{drop_modes}}}, face_area_ratio < {filt.get('min_face_area_ratio')}, "
                    "pose-extreme, and no-face frames; is_likely_screen_capture was deliberately not used "
                    "because it over-drops fake lockbox rows."
                ),
            })
            notes.append({
                "title": "Lockbox Composition Caveat",
                "status": "warning",
                "text": (
                    "The 2026-04-27 lockbox tags are a diagnostic substrate, not a clean production proxy: "
                    "839 frames, five real identities / two fake capture groups, with webcam and phone-screen "
                    "mode carrying much of the false-positive story. Read weird selected frames through the "
                    "modern_v2 and eval-substrate hygiene caveats before treating them as model-only failures."
                ),
            })
        self._lockbox_note_cache = notes
        return notes
