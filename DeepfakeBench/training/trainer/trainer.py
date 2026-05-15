import math
import os
import sys
import time
import gc
import contextlib
import hashlib

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import random
from collections import OrderedDict
import numpy as np  # noqa
from tqdm import tqdm  # noqa
import torch  # noqa
import torch.nn.functional as F  # noqa - Added Jan 10, 2026 for ArcFace diagnostics
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa
from metrics.utils import get_test_metrics, metrics_at_threshold  # noqa
from torch.cuda.amp import autocast, GradScaler  # noqa
import wandb  # noqa
from collections import defaultdict
from dataset.dataloaders import load_and_process_video, collate_fn  # noqa
from torchdata.datapipes.iter import IterableWrapper, Mapper, Filter  # noqa
from google.cloud import storage  # noqa
from google.api_core import exceptions  # noqa
import gc
import csv
import tempfile
import shutil
from datetime import datetime
from sklearn.metrics import confusion_matrix
from utils.grouping import infer_group_and_family
from loss.anchor_aware_penalty import AnchorAwarePenalty

# Import trainer mixins for modular functionality
from trainer.mixins import (
    CheckpointingMixin,
    EarlyStoppingMixin,
    GroupDROMixin,
    CurriculumMixin,
    ArcFaceMixin,
    ValidationMixin,
    ReportingMixin,
    StabilityRegMixin,
    CanaryProbeMixin,
)

FFpp_pool = ['FaceForensics++', 'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Trainer Is Using device: {device}")


def _to_plain_dict(raw):
    """Coerce wandb.Config / dict / dict-like to a plain dict; {} on None.

    The earlier ``isinstance(raw, dict)`` guard silently dropped wandb.Config
    sub-objects (they are not dict subclasses), which made nested config
    blocks like ``anchor_aware``, ``face_scale_jitter`` and ``periodic_saves``
    appear empty at runtime even when the yaml was correct.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw) or {}
    if hasattr(raw, 'keys') and callable(raw.keys):
        try:
            return {k: raw[k] for k in raw.keys()}
        except Exception:
            return {}
    return {}


def _per_video_jitter_stats(frame_probs_np):
    """A1 helper — per-video frame-to-frame score jitter.

    Mirrors ``tools/teams_frame_policy_analysis.py`` (per-video ``mean_jitter``
    and ``max_jitter``) so training-time and post-hoc numbers agree.

    Args:
        frame_probs_np: numpy array of shape ``[B, T]``.

    Returns:
        List of dicts (one per video) with keys ``mean``, ``max`` and the raw
        ``diffs`` list. Videos with <2 frames are skipped.
    """
    out = []
    if frame_probs_np is None or frame_probs_np.ndim != 2:
        return out
    B = frame_probs_np.shape[0]
    for b in range(B):
        fp = frame_probs_np[b]
        if len(fp) < 2:
            continue
        diffs = np.abs(np.diff(fp))
        out.append({
            "mean": float(np.mean(diffs)),
            "max": float(np.max(diffs)),
            "diffs": diffs.astype(float),
        })
    return out


def _aggregate_jitter_across_videos(per_video_list, spike_threshold=0.3):
    """A1 helper — aggregate per-video jitter stats for one method.

    Returns dict with ``mean``, ``max``, ``p95``, ``spike_rate_{threshold}``,
    ``n_diffs`` and ``all_diffs`` (np.ndarray, concatenated across videos —
    used for the gated W&B histogram).
    """
    if not per_video_list:
        return {}
    per_video_mean = [v["mean"] for v in per_video_list]
    per_video_max = [v["max"] for v in per_video_list]
    diffs_arrays = [v["diffs"] for v in per_video_list if v["diffs"].size > 0]
    if not diffs_arrays:
        return {}
    all_diffs = np.concatenate(diffs_arrays)
    spike_count = int(np.sum(all_diffs > float(spike_threshold)))
    return {
        "mean": float(np.mean(per_video_mean)),
        "max": float(np.max(per_video_max)),
        "p95": float(np.percentile(all_diffs, 95)),
        f"spike_rate_{str(spike_threshold).replace('.', 'p')}": float(
            spike_count / max(1, all_diffs.size)
        ),
        "n_diffs": int(all_diffs.size),
        "all_diffs": all_diffs,
    }


# A9: value_composite — deployment-hierarchy-aligned readout metric.
# Real pools for the FPR operating point (§1.4 of the R13 Packet 3 plan).
_VALUE_COMPOSITE_REAL_POOLS = (
    "df40_real",
    "external_youtube_avspeech_real",
    "zoom_vcd_real",
    "teams_ood_real",
    "proper_clean_real",
    "proper_teams_real",
)
# df40 training-distribution methods — excluded from "other fakes" TPR (§8.5).
_VALUE_COMPOSITE_DF40_TRAINING_FAKES = frozenset(
    {
        "simswap", "facedancer", "blendface", "e4s",
        "inswap", "mobileswap", "uniface",
    }
)
# Method names used in stability term.
_VALUE_COMPOSITE_STABILITY_JITTER_METHODS = (
    "teams_ood_fake",
    "teams_ood_real",
    "external_youtube_avspeech",
)


def _fpr_at_threshold(preds, labels, thresh):
    """FPR for a single real pool: (#preds>=thresh among negatives) / #negatives."""
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    mask_negative = labels == 0
    n_neg = int(mask_negative.sum())
    if n_neg == 0:
        return None
    flagged = int(np.sum(preds[mask_negative] >= thresh))
    return float(flagged) / float(n_neg)


def _tpr_at_threshold(preds, labels, thresh):
    """TPR for a single fake pool: (#preds>=thresh among positives) / #positives."""
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    mask_positive = labels == 1
    n_pos = int(mask_positive.sum())
    if n_pos == 0:
        return None
    flagged = int(np.sum(preds[mask_positive] >= thresh))
    return float(flagged) / float(n_pos)


def _find_threshold_for_mean_fpr(
    real_pools,
    target_mean_fpr=0.02,
    max_pool_fpr=0.04,
    tol=1e-4,
    max_iter=50,
):
    """Bisection on ``τ`` for A9 step 1 — mean_FPR == target AND max_FPR ≤ ceiling.

    Args:
        real_pools: dict ``{pool_name: {'preds': np.array, 'labels': np.array}}``.
        target_mean_fpr: target mean false-positive rate across pools.
        max_pool_fpr: per-pool worst-case ceiling.
        tol: tolerance on mean FPR for bisection convergence.
        max_iter: maximum bisection iterations.

    Returns:
        ``(tau, mean_fpr_at_tau, max_fpr_at_tau, per_pool_fpr_at_tau)``; if no τ
        satisfies both constraints, returns ``(None, mean, max, per_pool)`` with
        observed values at the best-effort τ (used only for diagnostic logging).
    """
    usable = {k: v for k, v in real_pools.items()
              if v and len(v.get("labels", [])) > 0 and int(np.sum(np.asarray(v["labels"]) == 0)) > 0}
    if not usable:
        return None, None, None, {}

    def _mean_and_max(tau):
        fprs = {}
        for name, pool in usable.items():
            fpr = _fpr_at_threshold(pool["preds"], pool["labels"], tau)
            if fpr is not None:
                fprs[name] = fpr
        if not fprs:
            return None, None, {}
        return float(np.mean(list(fprs.values()))), float(np.max(list(fprs.values()))), fprs

    lo, hi = 0.0, 1.0
    best = None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        mean_fpr, max_fpr, per_pool = _mean_and_max(mid)
        if mean_fpr is None:
            return None, None, None, {}
        best = (mid, mean_fpr, max_fpr, per_pool)
        if abs(mean_fpr - target_mean_fpr) < tol:
            break
        # FPR is monotonically non-increasing in τ. If observed mean > target,
        # raise τ (move lo up); if observed mean < target, lower τ (move hi down).
        if mean_fpr > target_mean_fpr:
            lo = mid
        else:
            hi = mid
    tau, mean_fpr, max_fpr, per_pool = best
    if max_fpr is not None and max_fpr > max_pool_fpr:
        return None, mean_fpr, max_fpr, per_pool
    return tau, mean_fpr, max_fpr, per_pool


def _compute_value_composite(
    real_pools,
    teams_fake_pools,
    other_fake_pools,
    stability_jitter_max=0.0,
    target_mean_fpr=0.02,
    max_pool_fpr=0.04,
):
    """A9 — deployment-hierarchy-aligned value composite (readout only).

    Returns a dict with ``value_composite`` (float or NaN), ``tau``,
    ``mean_fpr``, ``max_fpr``, ``max_fpr_at_mean_02``, and per-sub-component
    TPR / stability terms. When the max-FPR gate trips, ``value_composite`` is
    NaN and ``value_composite_blocked_by = "worst_pool_fpr"`` (or
    ``"insufficient_real_pools"`` if no τ could be found).
    """
    tau, mean_fpr, max_fpr, per_pool_fpr = _find_threshold_for_mean_fpr(
        real_pools,
        target_mean_fpr=target_mean_fpr,
        max_pool_fpr=max_pool_fpr,
    )
    out = {
        "tau": tau,
        "mean_fpr": mean_fpr,
        "max_fpr": max_fpr,
        "max_fpr_at_mean_02": max_fpr,
        "per_pool_fpr": per_pool_fpr,
        "stability": float(max(0.0, 1.0 - max(0.0, min(1.0, stability_jitter_max)))),
        "teams_fakes_tpr": None,
        "other_fakes_tpr": None,
        "value_composite": float("nan"),
        "value_composite_blocked_by": None,
    }

    if tau is None:
        if not per_pool_fpr:
            out["value_composite_blocked_by"] = "insufficient_real_pools"
        else:
            out["value_composite_blocked_by"] = "worst_pool_fpr"
        return out

    teams_tprs = []
    for name, pool in (teams_fake_pools or {}).items():
        if not pool or not len(pool.get("labels", [])):
            continue
        tpr = _tpr_at_threshold(pool["preds"], pool["labels"], tau)
        if tpr is not None:
            teams_tprs.append(tpr)
    other_tprs = []
    for name, pool in (other_fake_pools or {}).items():
        if not pool or not len(pool.get("labels", [])):
            continue
        if name in _VALUE_COMPOSITE_DF40_TRAINING_FAKES:
            continue
        tpr = _tpr_at_threshold(pool["preds"], pool["labels"], tau)
        if tpr is not None:
            other_tprs.append(tpr)

    teams_mean = float(np.mean(teams_tprs)) if teams_tprs else None
    other_mean = float(np.mean(other_tprs)) if other_tprs else None
    out["teams_fakes_tpr"] = teams_mean
    out["other_fakes_tpr"] = other_mean

    if teams_mean is None and other_mean is None:
        out["value_composite_blocked_by"] = "no_fake_pools"
        return out

    teams_component = 0.6 * (teams_mean if teams_mean is not None else 0.0)
    other_component = 0.3 * (other_mean if other_mean is not None else 0.0)
    stab_component = 0.1 * out["stability"]

    active_weight = 0.1 + (0.6 if teams_mean is not None else 0.0) + (0.3 if other_mean is not None else 0.0)
    out["value_composite"] = float(
        (teams_component + other_component + stab_component) / active_weight
    )
    return out


def _compute_per_bucket_recall_fpr(
    method_preds,
    method_labels,
    real_source_names,
    threshold=0.5,
    log_prefix="mid_eval",
):
    """W&B Block B: per-dataset-bucket recall/FPR for mid-training observability.

    Consumes the same per-method prediction/label dicts the trainer already
    builds during validation (`method_preds`, `method_labels` in
    ``_run_validation``); emits a flat dict suitable for direct merge into
    ``wandb_log_dict``. Read-only with respect to existing data flow.

    Keys emitted:
      - Fake bucket (``method`` not in ``real_source_names``):
          ``{log_prefix}/{method}/recall_fake``
      - Real bucket (``method`` in ``real_source_names``):
          ``{log_prefix}/{method}/recall_real``
          ``{log_prefix}/{method}/fpr``

    Buckets with zero samples are silently skipped — no NaN poisoning, no
    raise. Pure numpy at logging time, so the helper carries zero autograd
    risk (the eval loop is already ``inference=True`` + ``setEval()``).

    Args:
        method_preds: ``{bucket_name: list-or-array of float probabilities}``.
        method_labels: ``{bucket_name: list-or-array of {0, 1} labels}``.
        real_source_names: iterable of bucket names that are negative-class
            real pools (matched against ``method_preds`` keys).
        threshold: τ for the binarization step (default 0.5).
        log_prefix: namespace prefix for emitted W&B keys.

    Returns:
        Dict ``{wandb_key: float}`` ready to merge into ``wandb_log_dict``.
    """
    real_set = set(real_source_names or [])
    out = {}
    for bucket, preds in method_preds.items():
        labels = method_labels.get(bucket, [])
        preds_arr = np.asarray(preds, dtype=float)
        labels_arr = np.asarray(labels, dtype=int)
        if preds_arr.size == 0 or labels_arr.size == 0:
            continue
        if preds_arr.size != labels_arr.size:
            # Defensive: shape mismatch shouldn't happen given the trainer's
            # parallel append pattern, but skip rather than raise so a logging
            # path never blows up training.
            continue

        flagged = preds_arr >= float(threshold)
        if bucket in real_set:
            n_neg = int((labels_arr == 0).sum())
            if n_neg == 0:
                continue
            n_flagged_neg = int(flagged[labels_arr == 0].sum())
            fpr = float(n_flagged_neg) / float(n_neg)
            out[f"{log_prefix}/{bucket}/fpr"] = float(fpr)
            out[f"{log_prefix}/{bucket}/recall_real"] = float(1.0 - fpr)
        else:
            n_pos = int((labels_arr == 1).sum())
            if n_pos == 0:
                continue
            n_flagged_pos = int(flagged[labels_arr == 1].sum())
            recall_fake = float(n_flagged_pos) / float(n_pos)
            out[f"{log_prefix}/{bucket}/recall_fake"] = float(recall_fake)
    return out


def _load_capture_mode_lookup(parquet_path):
    """Load `clip_capture_mode` tags from a tag parquet, return a flat dict
    `{gcs_uri: capture_mode_string}`.

    Disabled-state contract: returning ``{}`` instead of raising lets the call
    site `wandb_log_dict.update(...)` against an empty result without a
    try/except. ``None`` path or missing file → ``{}``. The lockbox tags
    parquet is the canonical source (memory
    ``project_lockbox_fpr_dominated_by_webcam_mode.md``).
    """
    if parquet_path is None:
        return {}
    try:
        import os
        if not os.path.exists(parquet_path):
            return {}
        import pandas as pd
        df = pd.read_parquet(parquet_path, columns=["gcs_uri", "clip_capture_mode"])
        return {
            str(uri): str(mode)
            for uri, mode in zip(df["gcs_uri"].tolist(), df["clip_capture_mode"].tolist())
            if uri is not None
        }
    except Exception:
        return {}


def _compute_per_capture_mode_recall_fpr(
    method_preds,
    method_labels,
    method_paths,
    capture_mode_lookup,
    real_source_names,
    threshold=0.5,
    log_prefix="mid_eval/per_capture_mode",
):
    """Per-capture-mode recall/FPR for mid-training observability.

    Why this lives here: lockbox FPR is ~10× higher in webcam mode than in
    studio modes (memory ``project_lockbox_fpr_dominated_by_webcam_mode.md``).
    Block B's per-bucket panels show *which dataset* is leaking; this helper
    shows *which capture mode within a dataset* is leaking. Same threshold,
    same numerics, finer cut.

    Trainer feeds three parallel structures (one entry per validation video):
      - ``method_preds[bucket][i]`` — averaged video probability
      - ``method_labels[bucket][i]`` — {0, 1} label
      - ``method_paths[bucket][i]`` — representative frame path (any frame in
        the video; capture_mode is video-level)

    Paths absent from the lookup count under ``unknown`` so a quiet coverage
    drop is visible rather than silent. Buckets present in ``method_preds``
    but missing from ``method_paths`` are skipped (defensive).

    Keys emitted (one per ``mode`` ∈ values of the lookup ∪ {"unknown"}):
      - ``{log_prefix}/<mode>/recall_fake`` — fraction of fake-bucket samples
        flagged at τ
      - ``{log_prefix}/<mode>/fpr`` — fraction of real-bucket samples flagged
        at τ
      - ``{log_prefix}/<mode>/recall_real`` — ``1 - fpr``

    Pure numpy at logging time → zero autograd risk.
    """
    real_set = set(real_source_names or [])
    lookup = capture_mode_lookup or {}
    # mode -> {"fake": [...], "real": [...]} of float probs
    by_mode = {}
    for bucket, preds in method_preds.items():
        labels = method_labels.get(bucket, [])
        paths = method_paths.get(bucket, []) if method_paths else []
        if not paths:
            continue
        n = min(len(preds), len(labels), len(paths))
        if n == 0:
            continue
        is_real_bucket = bucket in real_set
        for i in range(n):
            mode = lookup.get(paths[i], "unknown")
            slot = by_mode.setdefault(mode, {"fake": [], "real": []})
            if is_real_bucket:
                slot["real"].append(float(preds[i]))
            else:
                slot["fake"].append(float(preds[i]))

    out = {}
    for mode, slots in by_mode.items():
        real_probs = np.asarray(slots["real"], dtype=float)
        fake_probs = np.asarray(slots["fake"], dtype=float)
        if real_probs.size > 0:
            n_flagged = int((real_probs >= float(threshold)).sum())
            fpr = float(n_flagged) / float(real_probs.size)
            out[f"{log_prefix}/{mode}/fpr"] = float(fpr)
            out[f"{log_prefix}/{mode}/recall_real"] = float(1.0 - fpr)
        if fake_probs.size > 0:
            n_flagged = int((fake_probs >= float(threshold)).sum())
            recall_fake = float(n_flagged) / float(fake_probs.size)
            out[f"{log_prefix}/{mode}/recall_fake"] = float(recall_fake)
    return out


def _safe_wandb_table_key(log_prefix: str, suffix: str) -> str:
    # wandb wraps Table keys into artifact names like
    # 'run-<8charid>-<sanitized_key>-<32charhash>' (sanitization strips '/').
    # That's ~46 chars of overhead against wandb's 128-char artifact-name limit,
    # leaving ~82 for the key itself. Reserve 70 as a safety margin.
    full_key = f"{log_prefix}/{suffix}"
    sanitized_len = len(full_key.replace("/", ""))
    if sanitized_len <= 70:
        return full_key
    h = hashlib.blake2s(log_prefix.encode("utf-8"), digest_size=4).hexdigest()
    short_prefix = f"{log_prefix[:30]}_{h}"
    return f"{short_prefix}/{suffix}"


class Trainer(
    CheckpointingMixin,
    EarlyStoppingMixin,
    GroupDROMixin,
    CurriculumMixin,
    ArcFaceMixin,
    ValidationMixin,
    ReportingMixin,
    StabilityRegMixin,
    CanaryProbeMixin,
):
    """
    Main trainer class for DeepfakeBench training.
    
    Composes functionality from multiple mixins:
    - CheckpointingMixin: Model saving, top-N checkpoints, GCS upload
    - EarlyStoppingMixin: Patience-based early stopping
    - GroupDROMixin: Distributionally Robust Optimization
    - CurriculumMixin: Lesson gates and curriculum learning
    - ArcFaceMixin: ArcFace head parameter annealing
    - ValidationMixin: Validation state management
    - ReportingMixin: Report generation and GCS upload
    """
    def __init__(
            self,
            config,
            model,
            optimizer,
            scheduler,
            logger,
            val_in_dist_loader,
            val_holdout_loader,
            metric_scoring='auc',
            wandb_run=None,
            ood_loader=None,
            ood_heldout_loader=None,
            test_loader=None,
            use_group_dro=False  # Argument to activate the feature
    ):
        if config is None or model is None or logger is None:
            raise ValueError("config, model, and logger must be provided")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.metric_scoring = metric_scoring
        self.wandb_run = wandb_run
        self.val_in_dist_loader = val_in_dist_loader
        self.val_holdout_loader = val_holdout_loader
        self.ood_loader = ood_loader  # Optional OOD loader (A10: monitored partition only)
        self.ood_heldout_loader = ood_heldout_loader  # A10: held-out slice for final_eval only
        self.test_loader = test_loader  # A2: 5% test slice for final_eval only
        self.unified_val_loader = None  # To cache the efficient loader

        # --- Initialize mixins ---
        # Checkpointing: manages model checkpoints and GCS uploads
        self.init_checkpointing()
        
        # Early stopping: patience-based training termination
        self.init_early_stopping()

        # --- Step-based training control ---
        self.max_train_steps = self.config.get('max_train_steps', None)
        self.evaluate_every_steps = self.config.get('evaluate_every_steps', None)
        if self.max_train_steps:
            self.logger.info(f"✅ Training will stop at a maximum of {self.max_train_steps} total steps.")
        if self.evaluate_every_steps:
            self.logger.info(
                f"✅ Evaluation will run every {self.evaluate_every_steps} steps, overriding epoch frequency.")

        # --- OOD monitoring cadence controls ---
        # Defaults preserve prior behavior: run on every validation call.
        # A8: prefer nested ood_monitoring.{first_ood_step, ood_cadence} when
        # present (either at top level or under combined_paired), fall back to
        # flat keys. Nested form makes FT vs scratch explicit in yaml.
        self.ood_monitoring_enabled = bool(self.config.get('ood_monitoring_enabled', True))

        def _nested_ood_knob(key_nested, key_flat, default):
            """Resolve ood_monitoring.{key_nested} with sensible fallbacks."""
            combined_cfg = (self.config.get('combined_paired') or {}) \
                if isinstance(self.config.get('combined_paired'), dict) else {}
            nested_cp = (combined_cfg.get('ood_monitoring') or {}) \
                if isinstance(combined_cfg.get('ood_monitoring'), dict) else {}
            top_ood = (self.config.get('ood_monitoring') or {}) \
                if isinstance(self.config.get('ood_monitoring'), dict) else {}
            for container in (top_ood, nested_cp):
                if key_nested in container and container[key_nested] is not None:
                    return container[key_nested]
            flat_val = self.config.get(key_flat)
            return flat_val if flat_val is not None else default

        self.ood_monitoring_start_step = int(
            _nested_ood_knob('first_ood_step', 'ood_monitoring_start_step', 0) or 0
        )
        cfg_ood_every = _nested_ood_knob('ood_cadence', 'ood_monitoring_every_steps', None)
        if cfg_ood_every is None:
            if self.evaluate_every_steps and self.evaluate_every_steps > 0:
                self.ood_monitoring_every_steps = int(self.evaluate_every_steps)
            else:
                self.ood_monitoring_every_steps = 1
        else:
            self.ood_monitoring_every_steps = int(cfg_ood_every)
            if self.ood_monitoring_every_steps <= 0:
                self.ood_monitoring_every_steps = 1
        self._last_ood_monitor_step = None
        self._ood_warmup_logged = False
        self.logger.info(
            "OOD monitoring cadence: enabled=%s first_ood_step=%d ood_cadence=%d",
            self.ood_monitoring_enabled,
            self.ood_monitoring_start_step,
            self.ood_monitoring_every_steps,
        )

        # A9 value_composite — config-driven gate + stability jitter aggregator.
        # Defaults match the packet 3 hardcoded values (0.02 / 0.04 / "max") so
        # runs without the new config block are bit-identical to pre-packet-3.5
        # trainers. Packet 3.5 yamls set {0.03, 0.05, "p95"} via this block.
        vc_cfg = (self.config.get("value_composite") or {})
        self._vc_target_mean_fpr = float(vc_cfg.get("target_mean_fpr", 0.02))
        self._vc_max_pool_fpr = float(vc_cfg.get("max_pool_fpr", 0.04))
        self._vc_stability_jitter_stat = str(vc_cfg.get("stability_jitter_stat", "max"))
        if self._vc_stability_jitter_stat not in ("max", "p95", "mean"):
            self.logger.warning(
                "Unknown value_composite.stability_jitter_stat=%r; falling back to 'max'.",
                self._vc_stability_jitter_stat,
            )
            self._vc_stability_jitter_stat = "max"
        self.logger.info(
            "value_composite config: target_mean_fpr=%.4f max_pool_fpr=%.4f stability_jitter_stat=%s",
            self._vc_target_mean_fpr,
            self._vc_max_pool_fpr,
            self._vc_stability_jitter_stat,
        )

        # Initialize AMP scaler for mixed precision training
        self.scaler = GradScaler()
        self.gradient_clip_val = self.config.get('gradient_clip_val')
        if self.gradient_clip_val:
            self.logger.info(f"✅ Gradient clipping enabled with max norm: {self.gradient_clip_val}")
        self.speed_up()

        self.log_dir = self.wandb_run.dir if self.wandb_run else './logs'
        # These are now only used for 'per_method' strategy, but are initialized here
        self.real_method_iters = {}
        self.fake_method_iters = {}

        # Group-DRO: distributionally robust optimization
        self.use_group_dro = use_group_dro
        if self.use_group_dro:
            self.init_group_dro()

        # Curriculum learning: lesson gate functionality
        self.init_curriculum()
        
        # ArcFace: parameter annealing (if using ArcFace head)
        self.init_arcface()

        # Stability regularisation: perturbation consistency loss
        self.init_stability_reg()

        # Canary probe: in-training deployment-quality monitoring.
        # No-op unless config.canary_probe.enabled=true.
        self.init_canary_probe()

        # Anchor-aware penalty: push prob_fake on false-flag pools toward target.
        # _to_plain_dict() handles wandb.Config sub-objects (not dict subclasses).
        anchor_cfg = _to_plain_dict(self.config.get('anchor_aware'))
        self.anchor_aware_penalty = AnchorAwarePenalty(
            config=anchor_cfg,
            anchor_cache_dir=getattr(self, 'anchor_cache_dir', '~/.cache/anchor_pools'),
            logger=self.logger,
        )

        # Face scale-jitter — module-level config consumed by collate_fns.
        from data.augmentations.face_scale_jitter import set_face_scale_jitter_config
        fsj_cfg = _to_plain_dict(self.config.get('face_scale_jitter'))
        set_face_scale_jitter_config(
            enabled=bool(fsj_cfg.get('enabled', False)),
            scale_limit=float(fsj_cfg.get('scale_limit', 0.0)),
            logger=self.logger,
        )

        # Fourier band-amplitude aug — module-level config consumed by collate_fns.
        # Operates on the post-resize 224x224 frame; bands 8-13 randomized,
        # bands 5-6 preserved per Probe 6 GREEN verdict.
        from data.augmentations.fourier_band_aug import set_fourier_aug_config
        fourier_cfg = _to_plain_dict(self.config.get('fourier_aug'))
        set_fourier_aug_config(fourier_cfg, logger=self.logger)

        # Resolution-chain aug — module-level config consumed by collate_fns.
        # Random downsample->upsample chain per frame; targets the 2026-05-15
        # CPU-probe finding that source-resolution dominates score swing on
        # real cohorts (P8A range_p50=0.97 over 20 resolution-chain variants).
        from data.augmentations.resolution_chain_aug import set_resolution_chain_aug_config
        rca_cfg = _to_plain_dict(self.config.get('resolution_chain_aug'))
        set_resolution_chain_aug_config(
            enabled=bool(rca_cfg.get('enabled', False)),
            p_apply=float(rca_cfg.get('p_apply', 0.5)),
            down_sizes=rca_cfg.get('down_sizes'),
            kernels=rca_cfg.get('kernels'),
            logger=self.logger,
        )

        # Method-domain mode — flip combined_paired iterators to emit 12-class
        # method-conditional GRL labels when the yaml config indicates Phase 3
        # operation (quality_domain_count >= 12 with use_quality_domain_head).
        # Default OFF (legacy 4-class). See data.sources.method_domain_map.
        try:
            from data.sources.combined_paired import set_method_domain_mode
            quality_domain_count = int(self.config.get('quality_domain_count', 4))
            use_qdh = bool(self.config.get('use_quality_domain_head', False))
            method_mode = use_qdh and quality_domain_count >= 12
            set_method_domain_mode(enabled=method_mode, log=self.logger)
        except Exception as exc:
            # Non-fatal: legacy yamls without these fields fall through to default off.
            self.logger.warning(
                "Method-domain mode setup failed (non-fatal, defaulting to legacy 4-class): %s",
                exc,
            )

    # --- Group-DRO methods are now provided by GroupDROMixin ---
    # The mixin provides: init_group_dro(), calculate_group_dro_loss(), get_group_dro_stats()

    def _update_arcface_s(self, step_cnt):
        """Anneals the 's' parameter of the ArcFace head if configured."""
        model_instance = self.model.module if isinstance(self.model, DDP) else self.model

        # Check if the model is an EffortDetector and has annealing configured
        if not hasattr(model_instance, 'use_arcface_head') or not model_instance.use_arcface_head:
            return
        if not hasattr(model_instance, 'anneal_steps') or model_instance.anneal_steps <= 0:
            return

        anneal_steps = model_instance.anneal_steps

        # Get the device from the existing buffer to ensure device consistency
        target_device = model_instance.head.s.device

        if step_cnt <= anneal_steps:
            # Linear annealing
            progress = step_cnt / anneal_steps
            current_s_float = model_instance.s_start + progress * (model_instance.s_end - model_instance.s_start)

            # Convert the float to a tensor on the correct device
            current_s_tensor = torch.tensor(current_s_float, device=target_device)
            model_instance.head.s = current_s_tensor
        else:
            # Ensure s is fixed at s_end after annealing is complete
            current_s_float = model_instance.s_end

            # Only update if necessary to avoid redundant operations
            if model_instance.head.s.item() != current_s_float:
                current_s_tensor = torch.tensor(current_s_float, device=target_device)
                model_instance.head.s = current_s_tensor

        # Log the change periodically during the annealing phase
        log_progress_steps = self.config.get('wandb', {}).get('log_progress_steps', 50)
        if self.wandb_run and step_cnt <= anneal_steps and (
                step_cnt % log_progress_steps == 0 or step_cnt == anneal_steps):
            self.wandb_run.log({'train/arcface_s': model_instance.head.s, 'train/step': step_cnt})

    def _update_lambda_reg(self, step_cnt):
        """Anneals lambda_reg (orthogonal constraint) if configured."""
        model_instance = self.model.module if isinstance(self.model, DDP) else self.model
        
        # Check if the model supports lambda_reg annealing
        if not hasattr(model_instance, 'update_lambda_reg'):
            return
        if not hasattr(model_instance, 'lambda_reg_anneal_steps') or model_instance.lambda_reg_anneal_steps <= 0:
            return
        
        # Update lambda_reg via model method
        current_lambda = model_instance.update_lambda_reg(step_cnt)
        
        # Log the change periodically during the annealing phase
        log_progress_steps = self.config.get('wandb', {}).get('log_progress_steps', 50)
        anneal_steps = model_instance.lambda_reg_anneal_steps
        if self.wandb_run and step_cnt <= anneal_steps and (
                step_cnt % log_progress_steps == 0 or step_cnt == anneal_steps):
            self.wandb_run.log({'train/lambda_reg': current_lambda, 'train/step': step_cnt})

    def _update_quality_domain_lambda(self, step_cnt):
        """Anneal gradient-reversal lambda for the quality-domain head (DANN sigmoid schedule)."""
        model_instance = self.model.module if isinstance(self.model, DDP) else self.model
        if not getattr(model_instance, 'use_quality_head', False):
            return

        total_steps = self.config.get('total_training_steps', 65000)
        progress = min(step_cnt / max(total_steps, 1), 1.0)
        # Sigmoid schedule: 0→~1 with inflection at 50% of training
        lambda_val = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
        model_instance.quality_head.set_lambda(lambda_val)

        log_progress_steps = self.config.get('wandb', {}).get('log_progress_steps', 50)
        if self.wandb_run and step_cnt % log_progress_steps == 0:
            self.wandb_run.log({
                'train/quality_grl_lambda': lambda_val,
                'train/step': step_cnt,
            })

    def _update_multi_axis_grl_lambda(self, step_cnt):
        """Update gradient-reversal lambda for the multi-axis GRL block.

        Supports two schedules via `multi_axis_grl.lambda_schedule`:

        - "linear_warmup_flat" (DEFAULT, used by T4): ramp 0 → λ_max
          linearly over `lambda_warmup_steps`, flat at λ_max thereafter.

        - "triangular_cyclic" (T5-A, 2026-05-11): same linear warmup
          0 → λ_max over `lambda_warmup_steps`, then triangular cycles
          between 0 and λ_max with period `lambda_cycle_steps`. Addresses
          the T4 oscillation finding (analysis/cpu_diagnostics_2026-05-10
          /outputs/L11_inv_mean_with_t4.csv): inv_mean is movable but
          encoder oscillates back to high-shortcut state under flat λ.
          Cycling re-applies pressure each time the encoder drifts.
        """
        model_instance = self.model.module if isinstance(self.model, DDP) else self.model
        if not getattr(model_instance, 'use_multi_axis_grl', False):
            return

        cfg = self.config.get('multi_axis_grl', {}) or {}
        lambda_max = float(cfg.get('lambda_max', 1.0))
        warmup_steps = int(cfg.get('lambda_warmup_steps', 500))
        schedule = str(cfg.get('lambda_schedule', 'linear_warmup_flat'))

        if warmup_steps <= 0:
            warmup_done = True
            lambda_warmup = lambda_max
        elif step_cnt <= warmup_steps:
            warmup_done = False
            lambda_warmup = lambda_max * (step_cnt / warmup_steps)
        else:
            warmup_done = True
            lambda_warmup = lambda_max

        if schedule == 'triangular_cyclic' and warmup_done:
            cycle_steps = int(cfg.get('lambda_cycle_steps', 1500))
            if cycle_steps <= 0:
                lambda_val = lambda_max  # fallback
            else:
                # Triangular wave starting at λ_max immediately post-warmup,
                # descending to 0 by half-period, then back to λ_max by full
                # period. Continues indefinitely.
                phase = (step_cnt - warmup_steps) % cycle_steps
                half = cycle_steps / 2.0
                if phase <= half:
                    # λ_max → 0 over first half
                    lambda_val = lambda_max * (1.0 - phase / half)
                else:
                    # 0 → λ_max over second half
                    lambda_val = lambda_max * ((phase - half) / half)
        else:
            lambda_val = lambda_warmup

        model_instance.multi_axis_grl_block.set_lambda(lambda_val)

        log_progress_steps = self.config.get('wandb', {}).get('log_progress_steps', 50)
        if self.wandb_run and step_cnt % log_progress_steps == 0:
            self.wandb_run.log({
                'train/multi_axis_grl_lambda': lambda_val,
                'train/step': step_cnt,
            })

    def _check_collapse_warning(self, predictions, data_dict, step_cnt):
        """
        Early warning system for model collapse.
        
        Added: Jan 10, 2026 - Critical diagnostic for ArcFace + SVD training
        
        Detects when model is outputting near-constant predictions, which indicates
        collapse into a degenerate solution (predicting all real or all fake).
        
        Returns:
            dict: Warning indicators and detailed collapse metrics
        """
        collapse_metrics = {}
        
        if 'raw_logits' not in predictions:
            return collapse_metrics
            
        raw_logits = predictions['raw_logits'].detach()
        probs = predictions['prob'].detach()
        labels = data_dict['label']
        
        # Handle expanded labels for video batches
        if raw_logits.shape[0] > labels.shape[0]:
            B = labels.shape[0]
            T = raw_logits.shape[0] // B
            labels = labels.repeat_interleave(T)
        
        # 1. Logit separation between classes
        logit_diff = raw_logits[:, 1] - raw_logits[:, 0]  # fake - real
        logit_std = logit_diff.std().item()
        logit_range = (logit_diff.max() - logit_diff.min()).item()
        
        # 2. Per-class logit statistics
        real_mask = (labels == 0)
        fake_mask = (labels == 1)
        
        if real_mask.any() and fake_mask.any():
            real_logit_mean = logit_diff[real_mask].mean().item()
            fake_logit_mean = logit_diff[fake_mask].mean().item()
            class_separation = fake_logit_mean - real_logit_mean  # Should be positive
            
            collapse_metrics['train/collapse/real_logit_diff_mean'] = real_logit_mean
            collapse_metrics['train/collapse/fake_logit_diff_mean'] = fake_logit_mean
            collapse_metrics['train/collapse/class_separation'] = class_separation
        
        # 3. Probability spread (should NOT be near 0)
        prob_spread = probs.std().item()
        prob_entropy = -((probs * (probs + 1e-8).log()) + ((1 - probs) * (1 - probs + 1e-8).log())).mean().item()
        
        collapse_metrics['train/collapse/logit_std'] = logit_std
        collapse_metrics['train/collapse/logit_range'] = logit_range
        collapse_metrics['train/collapse/prob_spread'] = prob_spread
        collapse_metrics['train/collapse/prob_entropy'] = prob_entropy
        
        # 4. Check for constant output (CRITICAL WARNING)
        is_constant_output = logit_std < 0.01
        collapse_metrics['train/collapse/is_constant_output'] = float(is_constant_output)
        
        if is_constant_output and step_cnt > 100:  # Only warn after warmup
            self.logger.warning(
                f"⚠️ COLLAPSE WARNING at step {step_cnt}: "
                f"Logit std={logit_std:.6f}, range={logit_range:.6f}. "
                f"Model may be outputting near-constant predictions!"
            )
            
            # Store collapse detection for potential intervention
            if not hasattr(self, '_collapse_warning_count'):
                self._collapse_warning_count = 0
            self._collapse_warning_count += 1
            collapse_metrics['train/collapse/warning_count'] = self._collapse_warning_count
        
        return collapse_metrics
    
    def _collect_arcface_diagnostics(self, step_cnt):
        """
        Collect ArcFace head diagnostics including gradient flow.
        
        Added: Jan 10, 2026 - For debugging ArcFace-induced collapse
        
        Returns:
            dict: ArcFace weight norms, gradient norms, and scale parameter
        """
        arcface_metrics = {}
        model_ref = self.model.module if isinstance(self.model, DDP) else self.model
        
        if not hasattr(model_ref, 'head') or not hasattr(model_ref.head, 's'):
            return arcface_metrics
        
        head = model_ref.head
        
        # Current scale
        current_s = head.s.item() if hasattr(head.s, 'item') else float(head.s)
        arcface_metrics['train/arcface/scale'] = current_s
        
        # Weight statistics
        if hasattr(head, 'weight'):
            weight = head.weight.detach()
            arcface_metrics['train/arcface/weight_norm'] = weight.norm().item()
            
            # Class center separation (cosine distance between real and fake centers)
            if weight.shape[0] == 2:  # Binary classification
                real_center = F.normalize(weight[0:1], dim=1)
                fake_center = F.normalize(weight[1:2], dim=1)
                center_cosine = (real_center @ fake_center.T).item()
                arcface_metrics['train/arcface/center_cosine_sim'] = center_cosine
                # Ideally should be negative (opposite directions)
            
            # Gradient statistics (if available)
            if head.weight.grad is not None:
                grad = head.weight.grad
                arcface_metrics['train/arcface/weight_grad_norm'] = grad.norm().item()
                arcface_metrics['train/arcface/weight_grad_max'] = grad.abs().max().item()
        
        return arcface_metrics
    
    def _collect_svd_residual_stats(self):
        """
        Collects statistics from SVDResidualLinear layers for diagnostic logging.
        
        Added: Jan 4, 2026 (Task D3)
        Updated: Jan 10, 2026 - Added gradient flow diagnostics
        
        This helps diagnose the 'params_with_grad' drop observed in LAION B16 training,
        by tracking whether S_residual values are approaching zero (dying gradients).
        
        Returns:
            dict: Statistics including min/max/mean of S_residual across all layers,
                  and count of layers with very small S_residual values.
        """
        model_ref = self.model.module if isinstance(self.model, DDP) else self.model
        
        s_residual_values = []
        s_residual_mins = []
        s_residual_maxs = []
        layer_count = 0
        near_zero_count = 0
        
        # Threshold for considering S_residual as "near zero"
        near_zero_threshold = 1e-6
        
        for name, module in model_ref.named_modules():
            # Check for SVDResidualLinear (works for both HuggingFace and OpenCLIP)
            if hasattr(module, 'S_residual') and module.S_residual is not None:
                s_vals = module.S_residual.detach()
                layer_count += 1
                
                s_min = s_vals.min().item()
                s_max = s_vals.max().item()
                s_mean = s_vals.mean().item()
                
                s_residual_mins.append(s_min)
                s_residual_maxs.append(s_max)
                s_residual_values.append(s_mean)
                
                # Count layers where S_residual is very small (potential dying gradient)
                if s_vals.abs().max().item() < near_zero_threshold:
                    near_zero_count += 1
        
        if layer_count == 0:
            return {}
        
        stats = {
            'svd/S_residual_min': min(s_residual_mins),
            'svd/S_residual_max': max(s_residual_maxs),
            'svd/S_residual_mean': sum(s_residual_values) / len(s_residual_values),
            'svd/layer_count': layer_count,
            'svd/near_zero_layers': near_zero_count,
        }
        
        # Add per-layer detailed stats if requested (expensive, so optional)
        # This can be enabled via config if needed for deep debugging
        if self.config.get('log_svd_per_layer', False):
            for i, (s_min, s_max, s_mean) in enumerate(zip(s_residual_mins, s_residual_maxs, s_residual_values)):
                stats[f'svd/layer_{i}/S_min'] = s_min
                stats[f'svd/layer_{i}/S_max'] = s_max
        
        return stats

    def _check_lesson_gate(self, all_val_metrics: dict):
        """Checks if the curriculum lesson's gate conditions have been met."""
        self.logger.info("--- Checking Lesson Gate conditions...")

        # A helper to safely retrieve nested metric values
        def get_metric(metric_name, dataset):
            # This check is important: if config keys are missing, metric_name or dataset will be None.
            if not metric_name or not dataset:
                return None
            return all_val_metrics.get(dataset, {}).get(metric_name)

        # 1. Guardrail Check
        guardrail_ok = True
        guard_conf = self.gate_guardrail_config
        if guard_conf.get('enabled'):
            # FIX: Use .get() for safe dictionary access instead of ['key']
            guard_metric = get_metric(guard_conf.get('metric'), guard_conf.get('dataset'))
            if guard_metric is not None:
                if self.gate_guardrail_start_value is None:
                    self.gate_guardrail_start_value = guard_metric
                    self.logger.info(f"Guardrail initialized: {guard_conf.get('metric')} = {guard_metric:.4f}")

                drop = self.gate_guardrail_start_value - guard_metric
                if drop > guard_conf.get('max_drop', 0.01):
                    guardrail_ok = False
                    self.logger.warning(
                        f"🚨 GUARDRAIL FAILED: {guard_conf.get('metric')} dropped by {drop:.4f} (max allowed: {guard_conf.get('max_drop')})")
            else:
                self.logger.warning(
                    f"Guardrail check skipped: metric '{guard_conf.get('metric')}' not found for dataset '{guard_conf.get('dataset')}'. Check your lesson_gate config."
                )

        # 2. Threshold Checks
        all_thresholds_met = True
        for check in self.gate_checks:
            # FIX: Use .get() for safe dictionary access
            metric_val = get_metric(check.get('metric'), check.get('dataset'))
            if metric_val is None:
                all_thresholds_met = False
                self.logger.warning(
                    f"Gate check skipped: metric '{check.get('metric')}' not found for dataset '{check.get('dataset')}'.")
                break

            op = check.get('comparison')
            thresh = check.get('threshold')

            # FIX: Ensure comparison and threshold keys exist
            if op is None or thresh is None:
                all_thresholds_met = False
                self.logger.warning(
                    f"Gate check skipped: malformed check config (missing 'comparison' or 'threshold'): {check}")
                break

            passed = (op == 'ge' and metric_val >= thresh) or (op == 'le' and metric_val <= thresh)

            if not passed:
                all_thresholds_met = False
                self.logger.info(
                    f"Gate check FAILED: {check.get('dataset')}/{check.get('metric')} ({metric_val:.4f}) did not meet {op} {thresh}")
                break
            else:
                self.logger.info(
                    f"Gate check PASSED: {check.get('dataset')}/{check.get('metric')} ({metric_val:.4f}) met {op} {thresh}")

        # 3. Plateau Check (based on the primary validation metric)
        plateau_met = False
        plateau_conf = self.gate_plateau_config
        if plateau_conf.get('enabled') and all_thresholds_met:  # Only check plateau if thresholds are met
            primary_metric_val = all_val_metrics.get('val_holdout', {}).get('overall', {}).get(self.metric_scoring)
            if primary_metric_val is not None:
                self.gate_primary_metric_history.append(primary_metric_val)
                patience = plateau_conf.get('patience', 2)

                if len(self.gate_primary_metric_history) >= patience:
                    recent_history = self.gate_primary_metric_history[-patience:]
                    best_recent = max(recent_history)
                    improvement = best_recent - self.gate_primary_metric_history[-patience]

                    if improvement < plateau_conf.get('min_delta', 0.001):
                        plateau_met = True
                        self.logger.info(
                            f"✅ Plateau condition MET: Improvement ({improvement:.4f}) is less than min_delta over last {patience} evals.")

        # 4. Final Decision
        if guardrail_ok and all_thresholds_met and plateau_met:
            self.early_stop_triggered = True
            self.logger.critical("✅✅✅ LESSON GATE PASSED! All conditions met. Triggering stop.")
        else:
            self.logger.info("--- Lesson Gate conditions not yet met. Continuing training.")

    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config['ddp']:
            self.model = DDP(self.model, device_ids=[self.config['local_rank']], find_unused_parameters=True,
                             output_device=self.config['local_rank'])

    def setTrain(self):
        self.model.train()

    def setEval(self):
        self.model.eval()

    def _upload_to_gcs(self, local_path, gcs_path):
        """Uploads a local file to a GCS path."""
        try:
            storage_client = storage.Client()
            bucket_name = gcs_path.split('gs://', 1)[1].split('/', 1)[0]
            blob_name = gcs_path.split(f'gs://{bucket_name}/', 1)[1]
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            self.logger.info(f"Uploading checkpoint to GCS: {gcs_path}")
            blob.upload_from_filename(local_path)
            self.logger.info(f"✅ SUCCESS: Uploaded to {gcs_path}")
            return True
        except exceptions.GoogleAPICallError as e:
            self.logger.error(f"FAILED to upload to GCS. Check permissions. Error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during GCS upload: {e}")
            return False

    def _delete_from_gcs(self, gcs_path):
        """Deletes a blob from a given GCS path."""
        if not gcs_path:
            return
        try:
            storage_client = storage.Client()
            bucket_name = gcs_path.split('gs://', 1)[1].split('/', 1)[0]
            blob_name = gcs_path.split(f'gs://{bucket_name}/', 1)[1]
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            if blob.exists():
                self.logger.info(f"Deleting old GCS checkpoint: {gcs_path}")
                blob.delete()
                self.logger.info(f"✅ SUCCESS: Deleted {gcs_path}")
            else:
                self.logger.warning(f"Attempted to delete non-existent GCS blob: {gcs_path}")
        except Exception as e:
            self.logger.error(f"Failed to delete GCS blob {gcs_path}. Error: {e}")

    def load_ckpt(self, model_path, validate):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            
            # Handle both old (state_dict only) and new (complete checkpoint) formats
            if isinstance(saved, dict) and 'state_dict' in saved:
                # New format with configuration
                state_dict = saved['state_dict']
                model_config = saved.get('model_config', {})

                if validate:
                    # Validate critical configuration parameters
                    self._validate_model_config(model_config, model_path)
                
                # Restore dynamic parameters if available
                if model_config.get('use_arcface_head', False) and 'current_arcface_s' in model_config:
                    model_instance = self.model.module if self.config['ddp'] else self.model
                    if hasattr(model_instance, 'head') and hasattr(model_instance.head, 's'):
                        current_s = model_config['current_arcface_s']
                        model_instance.head.s.data.fill_(current_s)
                        self.logger.info(f"Restored ArcFace s parameter to: {current_s}")
                
                self.logger.info(f"Loaded checkpoint from epoch {saved.get('epoch', 'unknown')} "
                               f"with AUC: {saved.get('auc', 'unknown'):.4f}")
            else:
                # Old format (state_dict only) - issue warning
                state_dict = saved
                self.logger.warning(f"Loading old checkpoint format from {model_path}. "
                                   "Configuration validation not possible.")
            
            # Load state dict with module prefix handling
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v

            # NEW (P17): when intermediate_layer is set, the head dim differs
            # from the checkpoint's head dim (768 vs 512). Drop head.* keys so
            # the head re-initializes fresh. Backward-compatible: behavior is
            # unchanged when intermediate_layer is None.
            backbone_cfg = self.config.get('backbone', {}) if isinstance(self.config, dict) else {}
            if backbone_cfg.get('intermediate_layer') is not None:
                head_keys = [k for k in list(new_state_dict.keys()) if k.startswith('head.')]
                for k in head_keys:
                    del new_state_dict[k]
                if head_keys:
                    self.logger.info(
                        f"P17 intermediate_layer={backbone_cfg['intermediate_layer']}: "
                        f"dropped {len(head_keys)} head.* keys from checkpoint "
                        f"(head dim changed; head will re-initialize)"
                    )

            # Pre-flight check: detect size mismatches before load_state_dict
            # This gives a clearer error message (e.g., wrong backbone checkpoint)
            model_state = self.model.state_dict()
            mismatches = []
            for key in new_state_dict:
                if key in model_state and new_state_dict[key].shape != model_state[key].shape:
                    mismatches.append(
                        f"  {key}: checkpoint={list(new_state_dict[key].shape)} "
                        f"vs model={list(model_state[key].shape)}"
                    )
            if mismatches:
                mismatch_str = "\n".join(mismatches)
                self.logger.error(
                    f"❌ CHECKPOINT SIZE MISMATCH loading {model_path}!\n"
                    f"This usually means the checkpoint was trained with a different backbone "
                    f"(e.g., ViT-L-14 vs ViT-B-16). Mismatched parameters:\n{mismatch_str}"
                )
                raise RuntimeError(
                    f"Checkpoint size mismatch: the checkpoint at {model_path} is incompatible "
                    f"with the current model architecture. {len(mismatches)} parameter(s) have "
                    f"different shapes. Check that gcs_base_checkpoint points to the correct "
                    f"backbone variant."
                )
            
            self.model.load_state_dict(new_state_dict, strict=False)
            
            # Validate model checksum if available (skip if curriculum learning might modify parameters)
            train_arcface = self.config.get('train_arcface', True)
            if isinstance(saved, dict) and 'model_checksum' in saved and not train_arcface:
                expected_checksum = saved['model_checksum']
                actual_checksum = self.compute_model_checksum()
                if expected_checksum == actual_checksum:
                    self.logger.info(f'✅ Model checksum validated: {actual_checksum}')
                else:
                    self.logger.error(f'❌ Model checksum mismatch! Expected: {expected_checksum}, Got: {actual_checksum}')
                    raise ValueError("Model checksum validation failed - model state may be corrupted")
            elif isinstance(saved, dict) and 'model_checksum' in saved and train_arcface:
                self.logger.info("⚠️  Skipping checksum validation - curriculum learning (train_arcface=True) may modify parameters")
            else:
                self.logger.warning("⚠️  No checksum available for validation (old checkpoint format)")
            
            self.logger.info(f'Model loaded from {model_path}')
        else:
            raise FileNotFoundError(f"=> no model found at '{model_path}'")

    def _validate_model_config(self, saved_config, checkpoint_path):
        """Validate that the saved model configuration matches current configuration."""
        if not saved_config:
            self.logger.warning("No model configuration found in checkpoint. Skipping validation.")
            return
        
        # Critical parameters that must match exactly
        critical_params = [
            'model_name', 'use_arcface_head', 
            'use_focal_loss', 'focal_loss_gamma', 'focal_loss_alpha', 'rank'
        ]
        
        # ArcFace parameters that can be overridden during curriculum learning
        arcface_params = ['arcface_s', 'arcface_m']
        
        # Check if ArcFace curriculum learning is enabled
        train_arcface = self.config.get('train_arcface', True)
        
        mismatches = []
        for param in critical_params:
            saved_val = saved_config.get(param)
            current_val = self.config.get(param)
            
            if saved_val != current_val:
                mismatches.append(f"{param}: saved={saved_val}, current={current_val}")
        
        # Only validate ArcFace parameters if curriculum learning is disabled
        if not train_arcface:
            for param in arcface_params:
                saved_val = saved_config.get(param)
                current_val = self.config.get(param)
                
                if saved_val != current_val:
                    mismatches.append(f"{param}: saved={saved_val}, current={current_val}")
        else:
            # Log that we're allowing ArcFace parameter override
            arcface_overrides = []
            for param in arcface_params:
                saved_val = saved_config.get(param)
                current_val = self.config.get(param)
                if saved_val != current_val:
                    arcface_overrides.append(f"{param}: saved={saved_val}, will_override_to={current_val}")
            
            if arcface_overrides:
                self.logger.info("ArcFace curriculum learning enabled - allowing parameter overrides:")
                for override in arcface_overrides:
                    self.logger.info(f"   - {override}")
        
        if mismatches:
            error_msg = (f"Critical configuration mismatch detected when loading {checkpoint_path}:\n" + 
                        "\n".join([f"  - {mm}" for mm in mismatches]))
            self.logger.error(error_msg)
            raise ValueError(f"Model configuration mismatch. {error_msg}")
        
        validation_mode = "strict" if not train_arcface else "curriculum learning enabled"
        self.logger.info(f"✅ Model configuration validation passed ({validation_mode}).")

    def compute_model_checksum(self):
        """Compute a checksum of the model's current state for validation."""
        import hashlib
        
        # Get model state dict
        model_state = self.model.module.state_dict() if self.config['ddp'] else self.model.state_dict()
        
        # Create a deterministic string representation
        checksum_data = []
        for key in sorted(model_state.keys()):
            tensor = model_state[key]
            # Convert to numpy for consistent hashing across devices
            tensor_np = tensor.detach().cpu().numpy()
            checksum_data.append(f"{key}:{tensor_np.shape}:{tensor_np.sum():.10f}")
        
        # Add critical config parameters
        config_items = [
            f"use_arcface_head:{self.config.get('use_arcface_head', False)}",
            f"arcface_s:{self.config.get('arcface_s', 30.0)}",
            f"arcface_m:{self.config.get('arcface_m', 0.35)}",
            f"rank:{self.config.get('rank', 1023)}",
        ]
        checksum_data.extend(config_items)
        
        # Create hash
        combined_str = "|".join(checksum_data)
        checksum = hashlib.sha256(combined_str.encode()).hexdigest()[:16]
        
        return checksum

    def save_ckpt(self, epoch, auc, eer, ckpt_prefix='ckpt', step=None):
        """
        Saves model checkpoint locally, uploads to GCS with a prefix, and cleans up.
        Returns the GCS path of the uploaded file.
        
        Args:
            epoch: The epoch number
            auc: The AUC score
            eer: The EER score
            ckpt_prefix: Prefix for the checkpoint name (e.g., 'first_best', 'top_n')
            step: Optional step count to use in naming instead of epoch (for top_n checkpoints)
        """
        gcs_config = self.config.get('checkpointing')
        if not gcs_config or not gcs_config.get('gcs_prefix'):
            self.logger.warning("GCS checkpointing not configured. Skipping upload.")
            return None

        model_name = self.config.get('model_name', 'model')
        date_str = time.strftime("%Y%m%d")
        
        # Use step count for top_n checkpoints, epoch for others
        if step is not None:
            ckpt_name = f"{ckpt_prefix}_{model_name}_{date_str}_step{step}_auc{auc:.4f}_eer{eer:.4f}.pth"
        else:
            ckpt_name = f"{ckpt_prefix}_{model_name}_{date_str}_ep{epoch}_auc{auc:.4f}_eer{eer:.4f}.pth"

        local_save_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(local_save_dir, exist_ok=True)
        local_save_path = os.path.join(local_save_dir, ckpt_name)

        model_state = self.model.module.state_dict() if self.config['ddp'] else self.model.state_dict()
        
        # Save complete checkpoint with configuration for exact reconstruction
        checkpoint = {
            'state_dict': model_state,
            'model_config': {
                'model_name': self.config.get('model_name'),
                'use_arcface_head': self.config.get('use_arcface_head', False),
                'arcface_s': self.config.get('arcface_s', 30.0),
                'arcface_m': self.config.get('arcface_m', 0.35),
                's_start': self.config.get('s_start'),
                's_end': self.config.get('s_end'),
                'anneal_steps': self.config.get('anneal_steps', 0),
                'use_focal_loss': self.config.get('use_focal_loss', False),
                'focal_loss_gamma': self.config.get('focal_loss_gamma', 2.0),
                'focal_loss_alpha': self.config.get('focal_loss_alpha'),
                'lambda_reg': self.config.get('lambda_reg', 1.0),
                'rank': self.config.get('rank', 1023),
            },
            'epoch': epoch,
            'auc': auc,
            'eer': eer,
            'training_step': getattr(self, 'current_step', None),
        }

        gcs_assets = self.config.get('gcs_assets') or {}
        checkpoint['model_config'].update({
            'backbone': self.config.get('backbone', {}),
            'backbone_path': self.config.get('backbone_path'),
            'backbone_name': self.config.get('backbone_name'),
            'backbone_config': self.config.get('backbone_config'),
            'gcs_assets': {
                'clip_backbone': gcs_assets.get('clip_backbone'),
            },
            'mean': self.config.get('mean'),
            'std': self.config.get('std'),
            'metadata_version': 2,
            'metadata_updated_at_utc': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        
        # If using ArcFace with annealing, save current s value
        if self.config.get('use_arcface_head', False):
            model_instance = self.model.module if self.config['ddp'] else self.model
            if hasattr(model_instance, 'head') and hasattr(model_instance.head, 's'):
                checkpoint['model_config']['current_arcface_s'] = float(model_instance.head.s)
        
        # Add model checksum for validation
        checkpoint['model_checksum'] = self.compute_model_checksum()
        
        torch.save(checkpoint, local_save_path)
        self.logger.info(f"💾 Saved checkpoint with checksum: {checkpoint['model_checksum']}")

        gcs_prefix = gcs_config['gcs_prefix']
        # <<< NEW: Ensure prefix ends with a slash for robust path joining
        if not gcs_prefix.endswith('/'):
            gcs_prefix += '/'
        run_id = self.wandb_run.id if self.wandb_run else "local_run"
        # <<< MODIFIED: Use os.path.join and then replace for cross-platform safety, though simple concatenation is fine for GCS.
        full_gcs_path = gcs_prefix + f"{run_id}/{ckpt_name}"

        upload_success = self._upload_to_gcs(local_save_path, full_gcs_path)

        try:
            os.remove(local_save_path)
        except OSError as e:
            self.logger.warning(f"Could not delete local temporary checkpoint: {e}")

        return full_gcs_path if upload_success else None

    def train_step(self, data_dict):
        with autocast():
            predictions = self.model(data_dict)
            if type(self.model) is DDP:
                losses = self.model.module.get_losses(data_dict, predictions)
            else:
                losses = self.model.get_losses(data_dict, predictions)

        # Fast-fail: detect NaN/Inf loss before wasting GPU hours
        if torch.isnan(losses['overall']) or torch.isinf(losses['overall']):
            raise RuntimeError(
                f"NaN/Inf loss detected in train_step "
                f"(global_step={getattr(self, 'global_step', '?')}). "
                "Aborting early. Check: learning rate, data pipeline, model init."
            )

        self.optimizer.zero_grad()
        self.scaler.scale(losses['overall']).backward()

        if hasattr(self, 'gradient_clip_val') and self.gradient_clip_val:
            self.scaler.unscale_(self.optimizer)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler is not None:
            self.scheduler.step()

        return losses, predictions

    def _current_total_grad_norm(self) -> float:
        """Measure the current total grad norm with a single device sync."""
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
        if torch.is_tensor(total_norm):
            return float(total_norm.item())
        return float(total_norm)

    def _next_batch_from_group(self, method_name, loaders, iters):
        """
        Gets the next batch from a specific method's dataloader.
        Manages a dictionary of iterators, creating or resetting them as needed.

        Args:
            method_name (str): The method to get a batch from.
            loaders (LazyDataLoaderManager): The manager holding all dataloader objects.
            iters (dict): The dictionary holding the active iterators for this group (real or fake).

        Returns:
            dict: The data dictionary for the next batch.
        """
        # 1. If we've never created an iterator for this method, create one.
        if method_name not in iters:
            loader = loaders[method_name]
            iters[method_name] = iter(loader)

        # 2. Try to get the next batch from the existing iterator.
        try:
            data_dict = next(iters[method_name])
        # 3. If the iterator is exhausted, create a new one and get the first batch.
        #    This allows us to loop over smaller datasets multiple times within one epoch.
        except StopIteration:
            loader = loaders[method_name]
            iters[method_name] = iter(loader)
            data_dict = next(iters[method_name])

        return data_dict

    def _train_per_method_step(self, real_loaders, fake_loaders, real_method_names, fake_method_names, real_weights,
                               fake_weights):
        """Helper to contain the logic for getting one 'per_method' batch."""
        chosen_fake_method = random.choices(fake_method_names, weights=fake_weights, k=1)[0]
        fake_data_dict = self._next_batch_from_group(chosen_fake_method, fake_loaders, self.fake_method_iters)

        chosen_real_method = random.choices(real_method_names, weights=real_weights, k=1)[0]
        real_data_dict = self._next_batch_from_group(chosen_real_method, real_loaders, self.real_method_iters)

        data_dict = {}
        for key in fake_data_dict.keys():
            f_val, r_val = fake_data_dict[key], real_data_dict[key]
            if torch.is_tensor(f_val) and torch.is_tensor(r_val):
                data_dict[key] = torch.cat((f_val, r_val), dim=0)
            elif isinstance(f_val, list):
                data_dict[key] = f_val + (r_val if r_val is not None else [])
            else:
                data_dict[key] = f_val

        batch_size = data_dict.get('label', torch.tensor([])).shape[0]
        if batch_size == 0: return None

        shuffle_indices = torch.randperm(batch_size)
        for key in data_dict.keys():
            if torch.is_tensor(data_dict[key]):
                data_dict[key] = data_dict[key][shuffle_indices]
            elif isinstance(data_dict[key], list):
                data_dict[key] = [data_dict[key][i] for i in shuffle_indices.tolist()]

        return data_dict

    def train_epoch(
            self,
            train_loader,  # Now a single argument for the training data
            epoch,
            train_videos,  # Used for calculating epoch length
            val_method_loaders=None
    ):
        self.logger.info(f"===> Epoch[{epoch + 1}] start!")
        strategy = self.config.get('dataloader_params', {}).get('strategy', 'per_method')

        if strategy == 'property_balancing':
            dl_params = self.config.get('dataloader_params', {})
            gpu_batch_size = dl_params.get('frames_per_batch', 64)
            # A logical batch is always composed of 32 unique videos.
            # Total logical frames = 32 videos * frames_per_video
            # Total GPU batches for one logical step = Total logical frames / frames_per_gpu_batch
            frames_per_video = dl_params.get('frames_per_video', 2)
            accumulation_steps = max(1, round((32.0 * frames_per_video) / gpu_batch_size))
            # `train_videos` is a list of frames (dicts) for this strategy
            total_items = len(train_videos)
            epoch_len = math.ceil(total_items / gpu_batch_size) if total_items > 0 else 0
        elif strategy == 'frame_level':
            gpu_batch_size = self.config.get('dataloader_params', {}).get('frames_per_batch')
            total_frames = sum(len(v.frame_paths) for v in train_videos)
            epoch_len = math.ceil(total_frames / gpu_batch_size) if total_frames > 0 else 0
            accumulation_steps = 1  # No accumulation for this strategy
        elif strategy == 'deeplive':
            # DeepLive uses IterableDataset with paired real/fake frames
            # train_videos is a list of DeepLiveSample objects
            dl_params = self.config.get('dataloader_params', {})
            gpu_batch_size = dl_params.get('frames_per_batch', 32)
            frames_per_sample = dl_params.get('frames_per_video', 8)
            # Each sample produces frames_per_sample * 2 (real + fake) frames
            total_frames = len(train_videos) * frames_per_sample * 2
            epoch_len = math.ceil(total_frames / gpu_batch_size) if total_frames > 0 else 0
            accumulation_steps = 1  # No accumulation for this strategy
        elif strategy == 'df40_paired':
            # DF40 Paired uses IterableDataset with paired real/fake frames (similar to deeplive)
            # train_videos is a list of DF40PairedSample objects
            dl_params = self.config.get('dataloader_params', {})
            df40_config = self.config.get('df40_paired', {})
            gpu_batch_size = dl_params.get('frames_per_batch', 32)
            frames_per_sample = dl_params.get('frames_per_video', 8)
            
            # Check if identity-balanced sampling is enabled
            identity_balanced = df40_config.get('identity_balanced_sampling', True)
            
            if identity_balanced:
                # With identity-balanced sampling, we sample ONE method per identity per epoch
                # So epoch length is based on unique identities, not total pairs
                unique_identities = set(s.target_identity for s in train_videos)
                num_samples_per_epoch = len(unique_identities)
                self.logger.info(f"DF40 identity-balanced sampling: {num_samples_per_epoch} unique identities (from {len(train_videos)} pairs)")
            else:
                # Legacy: iterate over all pairs
                num_samples_per_epoch = len(train_videos)
            
            # Each sample produces frames_per_sample * 2 (real + fake) frames
            total_frames = num_samples_per_epoch * frames_per_sample * 2
            epoch_len = math.ceil(total_frames / gpu_batch_size) if total_frames > 0 else 0
            accumulation_steps = 1  # No accumulation for this strategy
        elif strategy == 'combined_paired':
            # Combined Paired uses IterableDataset with DF40, DeepLive, and/or VisoMaster samples
            # train_videos is a list of UnifiedPairedSample objects
            dl_params = self.config.get('dataloader_params', {})
            combined_config = self.config.get('combined_paired', {})
            gpu_batch_size = dl_params.get('frames_per_batch', 32)
            frames_per_sample = dl_params.get('frames_per_video', 8)
            
            # Check if identity-balanced sampling is enabled
            identity_balanced = combined_config.get('identity_balanced_sampling', True)
            
            if identity_balanced:
                # With identity-balanced sampling, we sample ONE method per identity per epoch
                unique_identities = set(s.identity for s in train_videos)
                num_samples_per_epoch = len(unique_identities)
                self.logger.info(f"Combined identity-balanced sampling: {num_samples_per_epoch} unique identities (from {len(train_videos)} samples)")
            else:
                num_samples_per_epoch = len(train_videos)
            
            # Each paired sample produces frames_per_sample * 2 (real + fake) frames.
            # Unpaired real samples produce frames_per_sample * 1 (real only).
            n_paired = sum(1 for s in train_videos if not getattr(s, 'is_unpaired_real', False))
            n_unpaired = sum(1 for s in train_videos if getattr(s, 'is_unpaired_real', False))
            if identity_balanced:
                paired_ids = set(
                    s.identity for s in train_videos if not getattr(s, 'is_unpaired_real', False)
                )
                unpaired_ids = set(
                    s.identity for s in train_videos if getattr(s, 'is_unpaired_real', False)
                )
                n_paired = len(paired_ids)
                n_unpaired = len(unpaired_ids)
            total_frames = (n_paired * frames_per_sample * 2) + (n_unpaired * frames_per_sample)
            epoch_len = math.ceil(total_frames / gpu_batch_size) if total_frames > 0 else 0
            if n_unpaired > 0:
                self.logger.info(
                    f"Combined samples: {n_paired} paired + {n_unpaired} unpaired real -> {total_frames} total frames"
                )
            accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        elif strategy == 'visomaster':
            # Standalone VisoMaster uses IterableDataset (same pattern as combined_paired)
            dl_params = self.config.get('dataloader_params', {})
            viso_config = self.config.get('visomaster', {})
            gpu_batch_size = dl_params.get('frames_per_batch', 32)
            frames_per_sample = dl_params.get('frames_per_video', 8)
            identity_balanced = viso_config.get('identity_balanced_sampling', True)
            
            if identity_balanced:
                unique_identities = set(f"visomaster_{s.identity}" for s in train_videos)
                num_samples_per_epoch = len(unique_identities)
                self.logger.info(f"VisoMaster identity-balanced sampling: {num_samples_per_epoch} unique identities (from {len(train_videos)} samples)")
            else:
                num_samples_per_epoch = len(train_videos)
            
            total_frames = num_samples_per_epoch * frames_per_sample * 2
            epoch_len = math.ceil(total_frames / gpu_batch_size) if total_frames > 0 else 0
            accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        else:  # Handles 'per_method' and 'video_level'
            effective_batch_size = self.config.get('dataloader_params', {}).get('videos_per_batch')
            total_train_videos = len(train_videos)
            epoch_len = math.ceil(total_train_videos / effective_batch_size) if total_train_videos > 0 else 0
            accumulation_steps = 1  # No accumulation for these strategies

        self.logger.info(f"Training strategy: '{strategy}', Epoch length: {epoch_len} steps")
        if accumulation_steps > 1:
            self.logger.info(f"Using gradient accumulation with {accumulation_steps} steps.")

        # --- PRECISE EVALUATION SCHEDULING ---
        evaluation_frequency = self.config.get('data_params', {}).get('evaluation_frequency', 1)
        if evaluation_frequency <= 0: evaluation_frequency = 1
        eval_steps = set()
        # Only calculate epoch-based steps if step-based evaluation is not active
        if self.evaluate_every_steps is None or self.evaluate_every_steps <= 0:
            if epoch_len > 0:
                interval = max(1, epoch_len // evaluation_frequency)
                for i in range(1, evaluation_frequency): eval_steps.add(i * interval)
                eval_steps.add(epoch_len)
            self.logger.info(f"Scheduled evaluation at epoch steps: {sorted(list(eval_steps))}")
        else:
            # eval_steps remains empty; we will check step_cnt directly in the loop
            pass

        step_cnt = epoch * epoch_len
        epoch_start_time = time.time()

        # --- Conditional Training Loop based on strategy ---
        if strategy == 'per_method':
            # This logic remains self-contained as it doesn't use gradient accumulation
            real_source_names = self.config['methods']['use_real_sources']
            all_method_names = train_loader.keys()
            real_method_names = [m for m in all_method_names if m in real_source_names]
            fake_method_names = [m for m in all_method_names if m not in real_source_names]
            real_video_counts, fake_video_counts = defaultdict(int), defaultdict(int)
            for v in train_videos:
                (real_video_counts if v.method in real_source_names else fake_video_counts)[v.method] += 1
            total_real_videos, total_fake_videos = sum(real_video_counts.values()), sum(fake_video_counts.values())
            real_weights = [real_video_counts[m] / total_real_videos for m in
                            real_method_names] if total_real_videos > 0 else []
            fake_weights = [fake_video_counts[m] / total_fake_videos for m in
                            fake_method_names] if total_fake_videos > 0 else []

            pbar = tqdm(range(epoch_len), desc=f"EPOCH (per_method): {epoch + 1}/{self.config['nEpochs']}")
            for iteration in pbar:
                data_dict = self._train_per_method_step(train_loader, train_loader, real_method_names,
                                                        fake_method_names, real_weights, fake_weights)
                if data_dict is None:
                    continue
                # This strategy uses the original, single-step logic.
                self._run_train_step(data_dict, step_cnt, epoch, epoch_len, epoch_start_time)

                # --- Evaluation and Stop Condition Check ---
                step_cnt += 1
                should_evaluate_now = False

                if self.evaluate_every_steps and self.evaluate_every_steps > 0:
                    if step_cnt > 0 and step_cnt % self.evaluate_every_steps == 0:
                        should_evaluate_now = True
                else:  # Fallback to epoch-based frequency
                    if (iteration + 1) in eval_steps:
                        should_evaluate_now = True

                if should_evaluate_now:
                    self._run_validation(epoch, iteration, step_cnt)

                    # Check for max steps termination
                if self.max_train_steps and step_cnt >= self.max_train_steps:
                    self.logger.critical(f"MAX STEPS REACHED: {step_cnt}/{self.max_train_steps}. Stopping training.")
                    if not should_evaluate_now and self.evaluate_every_steps and self.evaluate_every_steps > 0:
                        next_eval_step = math.ceil(step_cnt / self.evaluate_every_steps) * self.evaluate_every_steps
                        steps_until_next_eval = next_eval_step - step_cnt
                        threshold = self.evaluate_every_steps / 1.5
                        if steps_until_next_eval <= threshold:
                            self.logger.info(
                                f"Running one final evaluation before stopping (steps until next eval {steps_until_next_eval} <= threshold {threshold:.1f}).")
                            self._run_validation(epoch, iteration, step_cnt)
                    self.early_stop_triggered = True

                if self.early_stop_triggered:
                    break

        elif strategy in ['video_level', 'frame_level', 'property_balancing', 'deeplive', 'df40_paired', 'combined_paired']:
            # Set epoch on the dataset for identity-balanced sampling (if supported)
            # This allows the dataset to vary random method selection per epoch
            if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'set_epoch'):
                train_loader.dataset.set_epoch(epoch)
                self.logger.info(f"Set epoch {epoch} on train_loader.dataset for identity-balanced sampling")
            
            pbar = tqdm(train_loader, desc=f"EPOCH ({strategy}): {epoch + 1}/{self.config['nEpochs']}", total=epoch_len)
            self.optimizer.zero_grad()  # Zero gradients at the start of the epoch
            
            # Flag for first-batch gradient diagnostics
            _first_backward_logged = False

            for i, data_dict in enumerate(pbar):
                if i >= epoch_len: break

                self._update_arcface_s(step_cnt)
                self._update_lambda_reg(step_cnt)
                self._update_quality_domain_lambda(step_cnt)
                self._update_multi_axis_grl_lambda(step_cnt)

                is_final_accumulation_step = (i + 1) % accumulation_steps == 0
                is_ddp = type(self.model) is DDP
                # Use DDP's no_sync context manager to avoid redundant gradient all-reduce calls.
                # This is a significant speed-up for DDP with gradient accumulation.
                context = self.model.no_sync() if is_ddp and not is_final_accumulation_step and accumulation_steps > 1 else contextlib.nullcontext()

                with context:
                    self.setTrain()
                    for key in data_dict.keys():
                        if isinstance(data_dict[key], torch.Tensor): data_dict[key] = data_dict[key].to(
                            self.model.device)
                    # --- FORWARD PASS ---
                    with autocast():
                        predictions = self.model(data_dict)
                        loss_fn_owner = self.model.module if type(self.model) is DDP else self.model

                        if self.use_group_dro:
                            # PREREQUISITE #1: Your model's get_losses must support reduction='none'
                            per_sample_losses_dict = loss_fn_owner.get_losses(
                                data_dict, predictions, reduction='none'
                            )
                            per_sample_loss = per_sample_losses_dict['overall']

                            # PREREQUISITE #2: The data_dict must contain 'method_id'
                            # Uses GroupDROMixin.calculate_group_dro_loss()
                            losses = self.calculate_group_dro_loss(data_dict, per_sample_loss)
                            # Preserve diagnostic scalars from the per-sample dict
                            # (pair_rank_loss, cls_loss, corr_penalty_loss, etc.) so they
                            # reach the W&B log loop below. Without this, every run with
                            # use_group_dro=True silently drops these scalars from history.
                            for k, v in per_sample_losses_dict.items():
                                if k != 'overall':
                                    losses[k] = v
                        else:
                            # Original behavior
                            losses = loss_fn_owner.get_losses(data_dict, predictions)

                    # --- Stability regularization loss ---
                    stability_loss = self.compute_stability_loss(
                        self.model, data_dict, predictions
                    )
                    losses['overall'] = losses['overall'] + stability_loss
                    losses['stability'] = stability_loss.detach()

                    # --- Anchor-aware penalty (false-flag real pools) ---
                    anchor_loss = self.anchor_aware_penalty.compute(
                        self.model, data_dict["image"].device,
                    )
                    losses['overall'] = losses['overall'] + anchor_loss
                    losses['anchor_aware'] = anchor_loss.detach()

                    # Store unscaled loss for accurate logging
                    unscaled_loss = losses['overall'].clone().detach()

                    # Fast-fail: detect NaN/Inf loss before wasting GPU hours
                    if torch.isnan(unscaled_loss) or torch.isinf(unscaled_loss):
                        raise RuntimeError(
                            f"NaN/Inf loss detected at step {step_cnt} "
                            f"(epoch {epoch}). "
                            "Aborting early. Check: learning rate, data pipeline, model init."
                        )

                    # Scale loss for accumulation
                    if accumulation_steps > 1:
                        losses['overall'] = losses['overall'] / accumulation_steps
                    # --- BACKWARD PASS ---
                    self.scaler.scale(losses['overall']).backward()

                # --- OPTIMIZER STEP (conditional) ---
                if is_final_accumulation_step or accumulation_steps == 1:
                    # Unscale gradients first (required for clipping and gradient logging)
                    self.scaler.unscale_(self.optimizer)
                    
                    # === FIRST STEP DIAGNOSTIC: Show gradient health on first optimizer step ===
                    if not _first_backward_logged and epoch == self.config.get('start_epoch', 0):
                        _first_backward_logged = True
                        model_ref = self.model.module if is_ddp else self.model
                        
                        total_grad_norm = 0.0
                        num_params_with_grad = 0
                        params_without_grad = []
                        params_with_tiny_grad = []
                        params_with_big_grad = []
                        
                        for name, param in model_ref.named_parameters():
                            if param.requires_grad:
                                if param.grad is not None:
                                    grad_norm = param.grad.data.norm(2).item()
                                    total_grad_norm += grad_norm ** 2
                                    num_params_with_grad += 1
                                    if grad_norm < 1e-8:
                                        params_with_tiny_grad.append((name, grad_norm))
                                    elif grad_norm > 10:
                                        params_with_big_grad.append((name, grad_norm))
                                else:
                                    params_without_grad.append(name)
                        
                        total_grad_norm = total_grad_norm ** 0.5
                        
                        self.logger.info("=" * 70)
                        self.logger.info("🔍 FIRST OPTIMIZER STEP GRADIENT DIAGNOSTIC")
                        self.logger.info("=" * 70)
                        self.logger.info(f"   Total gradient norm: {total_grad_norm:.6e}")
                        self.logger.info(f"   Parameters with gradients: {num_params_with_grad}")
                        self.logger.info(f"   Parameters WITHOUT gradients (requires_grad=True but grad=None):")
                        for pname in params_without_grad[:10]:
                            self.logger.info(f"      ❌ {pname}")
                        if len(params_without_grad) > 10:
                            self.logger.info(f"      ... and {len(params_without_grad) - 10} more")
                        
                        if params_with_tiny_grad:
                            self.logger.info(f"   Parameters with TINY gradients (<1e-8):")
                            for pname, gnorm in params_with_tiny_grad[:10]:
                                self.logger.info(f"      ⚠️ {pname}: {gnorm:.2e}")
                        
                        if params_with_big_grad:
                            self.logger.info(f"   Parameters with LARGE gradients (>10):")
                            for pname, gnorm in params_with_big_grad[:10]:
                                self.logger.info(f"      🔥 {pname}: {gnorm:.2e}")
                        
                        # Also log loss and prediction stats
                        probs = predictions['prob'].detach()
                        labels = data_dict['label']
                        self.logger.info(f"   First batch stats:")
                        self.logger.info(f"      Loss: {unscaled_loss.item():.4f}")
                        self.logger.info(f"      Prob mean: {probs.mean().item():.4f}, std: {probs.std().item():.4f}")
                        self.logger.info(f"      Label dist: {(labels==0).sum().item()} real, {(labels==1).sum().item()} fake")
                        self.logger.info("=" * 70)
                    
                    if hasattr(self, 'gradient_clip_val') and self.gradient_clip_val:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    
                    # Compute gradient health metrics before they're cleared
                    model_ref = self.model.module if is_ddp else self.model
                    _grad_norm = self._current_total_grad_norm()
                    _num_params_with_grad = sum(1 for p in model_ref.parameters() if p.grad is not None)
                    # Store for later logging
                    self._last_grad_norm = _grad_norm
                    self._last_params_with_grad = _num_params_with_grad

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if self.scheduler is not None:
                        self.scheduler.step()

                    self.optimizer.zero_grad()

                # --- LOGGING (every GPU batch, using unscaled loss) ---
                if self.wandb_run and self.config['local_rank'] == 0:
                    log_dict = {"train/step": step_cnt, "epoch": epoch + 1}
                    log_dict[f'train/loss/overall'] = unscaled_loss.item()
                    for name, value in losses.items():
                        if name == 'overall':
                            continue  # Already logged the unscaled version

                        # If the value is a scalar tensor, log it with .item()
                        if value.numel() == 1:
                            log_dict[f'train/loss/{name}'] = value.item()
                        # If it's a multi-element tensor (like group_weights), log it as a histogram
                        else:
                            # wandb.Histogram is the perfect tool for this
                            log_dict[f'train/diagnostic/{name}'] = wandb.Histogram(value.detach().cpu())

                    batch_metrics = self.model.module.get_train_metrics(data_dict,
                                                                        predictions) if is_ddp else self.model.get_train_metrics(
                        data_dict, predictions)
                    for name, value in batch_metrics.items(): log_dict[f'train/metric/{name}'] = value
                    
                    # === LEARNING VISIBILITY: Key metrics for tracking actual learning ===
                    model_ref = self.model.module if is_ddp else self.model
                    
                    # 1. ArcFace scale parameter (s) - tracks annealing progress
                    if hasattr(model_ref, 'head') and hasattr(model_ref.head, 's'):
                        current_s = model_ref.head.s.item() if hasattr(model_ref.head.s, 'item') else float(model_ref.head.s)
                        log_dict['train/arcface/s'] = current_s
                    
                    # 2. Learning rate tracking
                    if self.optimizer and len(self.optimizer.param_groups) > 0:
                        log_dict['train/lr'] = self.optimizer.param_groups[0]['lr']
                    
                    # 3. Prediction confidence (how decisive is the model?)
                    probs = predictions['prob'].detach()
                    log_dict['train/confidence/mean'] = probs.mean().item()
                    log_dict['train/confidence/std'] = probs.std().item()
                    # Fraction of confident predictions (>0.7 or <0.3)
                    confident_mask = (probs > 0.7) | (probs < 0.3)
                    log_dict['train/confidence/fraction_confident'] = confident_mask.float().mean().item()
                    
                    # 4. Class balance in predictions (should be ~0.5 for balanced data)
                    pred_fake_ratio = (probs > 0.5).float().mean().item()
                    log_dict['train/pred_balance/fake_ratio'] = pred_fake_ratio
                    
                    # 5. Raw logit statistics (key for ArcFace debugging)
                    if 'raw_logits' in predictions:
                        raw_logits = predictions['raw_logits'].detach()
                        logit_diff = raw_logits[:, 1] - raw_logits[:, 0]  # fake - real
                        log_dict['train/logits/diff_mean'] = logit_diff.mean().item()
                        log_dict['train/logits/diff_std'] = logit_diff.std().item()
                        
                        # 5b. Per-class logit statistics (Added Jan 10, 2026)
                        labels = data_dict['label']
                        if raw_logits.shape[0] > labels.shape[0]:
                            B = labels.shape[0]
                            T = raw_logits.shape[0] // B
                            labels = labels.repeat_interleave(T)
                        
                        real_mask = (labels == 0)
                        fake_mask = (labels == 1)
                        if real_mask.any():
                            log_dict['train/logits/real_mean'] = logit_diff[real_mask].mean().item()
                        if fake_mask.any():
                            log_dict['train/logits/fake_mean'] = logit_diff[fake_mask].mean().item()
                    
                    # 6. Collapse early warning system (Added Jan 10, 2026)
                    collapse_metrics = self._check_collapse_warning(predictions, data_dict, step_cnt)
                    log_dict.update(collapse_metrics)
                    
                    # 7. ArcFace detailed diagnostics (Added Jan 10, 2026)
                    arcface_metrics = self._collect_arcface_diagnostics(step_cnt)
                    log_dict.update(arcface_metrics)

                    log_progress_steps = self.config.get('wandb', {}).get('log_progress_steps', 50)
                    if (i % log_progress_steps == 0) or (i == epoch_len - 1):
                        time_elapsed = time.time() - epoch_start_time
                        steps_per_sec = (i + 1) / time_elapsed if time_elapsed > 0 else 0
                        log_dict['train/steps_per_sec'] = steps_per_sec
                        log_dict['train/probabilities'] = wandb.Histogram(predictions['prob'].detach().cpu().numpy())
                        if epoch_len > 0:
                            progress_pct = ((i + 1) / epoch_len) * 100
                            log_dict['train/epoch_progress'] = progress_pct
                            if steps_per_sec > 0:
                                time_remaining_sec = (epoch_len - (i + 1)) / steps_per_sec
                                log_dict['train/epoch_eta_min'] = time_remaining_sec / 60
                        
                        # 6. Gradient health - use cached values from before optimizer.zero_grad()
                        if hasattr(self, '_last_grad_norm') and hasattr(self, '_last_params_with_grad'):
                            if self._last_params_with_grad > 0:
                                log_dict['train/grad_norm'] = self._last_grad_norm
                                log_dict['train/params_with_grad'] = self._last_params_with_grad
                            else:
                                # Log zero to indicate NO gradients flowing - critical warning sign!
                                log_dict['train/grad_norm'] = 0.0
                                log_dict['train/params_with_grad'] = 0
                        
                        # 7. SVD Residual diagnostics (added Jan 4, 2026 - Task D3)
                        # Track S_residual values to diagnose params_with_grad drops
                        svd_stats = self._collect_svd_residual_stats()
                        if svd_stats:
                            log_dict.update(svd_stats)
                        
                        self.wandb_run.log(log_dict)
                        self.logger.info(
                            f"Epoch {epoch + 1}/{self.config['nEpochs']} | Step {i + 1}/{epoch_len} ({progress_pct:.1f}%) | Loss: {unscaled_loss.item():.4f} | Speed: {steps_per_sec:.2f} it/s")
                    else:
                        self.wandb_run.log({"train/loss/overall": unscaled_loss.item(), "train/step": step_cnt})

                # --- Evaluation and Stop Condition Check ---
                step_cnt += 1
                should_evaluate_now = False

                if self.evaluate_every_steps and self.evaluate_every_steps > 0:
                    if step_cnt > 0 and step_cnt % self.evaluate_every_steps == 0:
                        should_evaluate_now = True
                else:  # Fallback to epoch-based frequency
                    if (i + 1) in eval_steps:
                        should_evaluate_now = True

                if should_evaluate_now:
                    self._run_validation(epoch, i, step_cnt)

                    # Check for max steps termination
                if self.max_train_steps and step_cnt >= self.max_train_steps:
                    self.logger.critical(f"MAX STEPS REACHED: {step_cnt}/{self.max_train_steps}. Stopping training.")
                    if not should_evaluate_now and self.evaluate_every_steps and self.evaluate_every_steps > 0:
                        next_eval_step = math.ceil(step_cnt / self.evaluate_every_steps) * self.evaluate_every_steps
                        steps_until_next_eval = next_eval_step - step_cnt
                        threshold = self.evaluate_every_steps / 1.5
                        if steps_until_next_eval <= threshold:
                            self.logger.info(
                                f"Running one final evaluation before stopping (steps until next eval {steps_until_next_eval} <= threshold {threshold:.1f}).")
                            self._run_validation(epoch, i, step_cnt)
                    self.early_stop_triggered = True

                if self.early_stop_triggered:
                    break

            # Perform a final optimizer step for any remaining gradients at the end of the epoch
            try:
                # 'i' is the index of the last item processed by the loop.
                num_iterations_run = i + 1
            except NameError:
                num_iterations_run = 0

            # Perform a final optimizer step for any remaining gradients at the end of the epoch.
            if num_iterations_run > 0 and num_iterations_run % accumulation_steps != 0 and accumulation_steps > 1:
                self.logger.info("Performing final optimizer step for dangling gradients at end of epoch.")

                try:
                    # --- ATTEMPT THE OPTIMIZER STEP ---
                    if hasattr(self, 'gradient_clip_val') and self.gradient_clip_val:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if self.scheduler is not None:
                        self.scheduler.step()

                    # On success, zero the gradients for the next epoch.
                    self.optimizer.zero_grad()

                except AssertionError as e:
                    # --- FAILSAFE: CATCH THE ERROR ---
                    self.logger.error(
                        "!!! FAILSAFE TRIGGERED: Caught AssertionError during final optimizer step. "
                        "This indicates a logic error in gradient accumulation handling. "
                        f"Error: {e}. "
                        "Skipping this optimizer step and discarding stale gradients to prevent a crash."
                    )
                    # Manually discard the stale gradients that caused the error.
                    self.optimizer.zero_grad()
                    if hasattr(self.scaler, '_per_optimizer_states'):
                        self.scaler._per_optimizer_states.clear()
        else:
            raise ValueError(f"Unsupported training strategy: {strategy}")

        self.logger.info(f"===> Epoch[{epoch + 1}] finished in {(time.time() - epoch_start_time) / 60:.2f} minutes.")

    def _run_train_step(self, data_dict, step_cnt, epoch, epoch_len, epoch_start_time):  # Add new args
        """Helper to avoid code duplication in the training loop."""
        self._update_arcface_s(step_cnt)
        self._update_lambda_reg(step_cnt)
        self._update_quality_domain_lambda(step_cnt)
        self._update_multi_axis_grl_lambda(step_cnt)
        self.setTrain()
        for key in data_dict.keys():
            if isinstance(data_dict[key], torch.Tensor): data_dict[key] = data_dict[key].to(self.model.device)

        losses, predictions = self.train_step(data_dict)

        if self.wandb_run and self.config['local_rank'] == 0:
            log_dict = {"train/step": step_cnt, "epoch": epoch + 1}
            for name, value in losses.items():
                log_dict[f'train/loss/{name}'] = value.item()

            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            for name, value in batch_metrics.items():
                log_dict[f'train/metric/{name}'] = value

            # === ENHANCED DIAGNOSTICS (Added Jan 10, 2026) ===
            # 1. Collapse early warning
            collapse_metrics = self._check_collapse_warning(predictions, data_dict, step_cnt)
            log_dict.update(collapse_metrics)
            
            # 2. ArcFace diagnostics
            arcface_metrics = self._collect_arcface_diagnostics(step_cnt)
            log_dict.update(arcface_metrics)
            
            # 3. Per-class logit statistics
            if 'raw_logits' in predictions:
                raw_logits = predictions['raw_logits'].detach()
                logit_diff = raw_logits[:, 1] - raw_logits[:, 0]
                log_dict['train/logits/diff_mean'] = logit_diff.mean().item()
                log_dict['train/logits/diff_std'] = logit_diff.std().item()

            # --- DETAILED PROGRESS LOGGING (REPLACES PBAR) ---
            log_progress_steps = self.config.get('wandb', {}).get('log_progress_steps', 50)
            current_iter_in_epoch = step_cnt - (epoch * epoch_len)

            # Log every N steps or on the very last step of the epoch
            if (current_iter_in_epoch % log_progress_steps == 0) or (current_iter_in_epoch == epoch_len - 1):
                time_elapsed = time.time() - epoch_start_time
                steps_per_sec = (current_iter_in_epoch + 1) / time_elapsed if time_elapsed > 0 else 0

                log_dict['train/steps_per_sec'] = steps_per_sec
                log_dict['train/probabilities'] = wandb.Histogram(predictions['prob'].detach().cpu().numpy())

                progress_pct = 0.0
                if epoch_len > 0:
                    progress_pct = ((current_iter_in_epoch + 1) / epoch_len) * 100
                    log_dict['train/epoch_progress'] = progress_pct

                    if steps_per_sec > 0:
                        time_remaining_sec = (epoch_len - (current_iter_in_epoch + 1)) / steps_per_sec
                        log_dict['train/epoch_eta_min'] = time_remaining_sec / 60

                # Log to WandB
                self.wandb_run.log(log_dict)

                # Log a simple text line to the logger for basic feedback
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config['nEpochs']} | "
                    f"Step {current_iter_in_epoch + 1}/{epoch_len} "
                    f"({progress_pct:.1f}%) | "
                    f"Loss: {losses['overall'].item():.4f} | "
                    f"Speed: {steps_per_sec:.2f} it/s"
                )
            else:
                # For other steps, just log the minimal required info to not miss loss spikes
                self.wandb_run.log({"train/loss/overall": losses['overall'].item(), "train/step": step_cnt})

    def _run_validation(self, epoch, iteration, step_cnt):
        """
        Helper to run the full validation suite and check the lesson gate conditions.
        """
        if self.config['local_rank'] != 0:
            return

        self.logger.info(f"\n===> Evaluation at epoch {epoch + 1}, step {step_cnt}")

        all_val_metrics = {}

        # 1. Run In-Distribution Validation
        if self.val_in_dist_loader:
            in_dist_metrics = self.test_epoch(
                epoch=epoch,
                step_cnt=step_cnt,
                validation_loader=self.val_in_dist_loader,
                log_prefix="val_in_dist",
                is_primary_metric=False
            )
            all_val_metrics['val_in_dist'] = in_dist_metrics

        # 2. Run Holdout Validation
        if self.val_holdout_loader:
            holdout_metrics = self.test_epoch(
                epoch=epoch,
                step_cnt=step_cnt,
                validation_loader=self.val_holdout_loader,
                log_prefix="val_holdout",
                is_primary_metric=True  # This is the primary metric set for checkpointing
            )
            all_val_metrics['val_holdout'] = holdout_metrics

        # 3. Extract in-distribution EER threshold — this is the SINGLE
        #    operating-point threshold that all other splits are measured against.
        indist_threshold = None
        indist_m = all_val_metrics.get('val_in_dist')
        if indist_m and 'overall' in indist_m:
            indist_threshold = indist_m['overall'].get('eer_threshold')
            if indist_threshold is not None and indist_threshold > 0:
                self.logger.info(
                    f"In-dist EER threshold: {indist_threshold:.4f} — "
                    "will apply to holdout & OOD for unified evaluation"
                )

        # 4. Compute at-indist-threshold metrics for val_holdout
        holdout_m = all_val_metrics.get('val_holdout')
        if (
            indist_threshold is not None
            and indist_threshold > 0
            and holdout_m
            and 'all_preds' in holdout_m
            and len(holdout_m['all_preds']) > 0
        ):
            at_indist = metrics_at_threshold(
                holdout_m['all_preds'], holdout_m['all_labels'], indist_threshold
            )
            if at_indist and self.wandb_run:
                log_dict = {'train/step': step_cnt}
                for k, v in at_indist.items():
                    log_dict[f'val_holdout/at_indist/{k}'] = v
                log_dict['val_holdout/at_indist/threshold'] = indist_threshold
                self.wandb_run.log(log_dict)
                self.logger.info(
                    f"val_holdout @ indist threshold {indist_threshold:.4f}: "
                    f"acc={at_indist['acc']:.4f}  f1={at_indist['f1']:.4f}  "
                    f"fpr={at_indist['fpr']:.4f}  fnr={at_indist['fnr']:.4f}"
                )

        # 5. Compute unified threshold across all validation pools (legacy metric)
        unified_preds_parts = []
        unified_labels_parts = []
        for key in ('val_in_dist', 'val_holdout'):
            m = all_val_metrics.get(key)
            if m and 'all_preds' in m and len(m['all_preds']) > 0:
                unified_preds_parts.append(m['all_preds'])
                unified_labels_parts.append(m['all_labels'])
        if unified_preds_parts:
            combined_preds = np.concatenate(unified_preds_parts)
            combined_labels = np.concatenate(unified_labels_parts)
            if len(np.unique(combined_labels)) > 1:
                unified_metrics = get_test_metrics(combined_preds, combined_labels)
                if unified_metrics and self.wandb_run:
                    self.wandb_run.log({
                        'unified/eer_threshold': unified_metrics.get('eer_threshold', 0),
                        'unified/eer': unified_metrics.get('eer', 0),
                        'unified/f1': unified_metrics.get('f1_at_eer', 0),
                        'unified/auc': unified_metrics.get('auc', 0),
                        'train/step': step_cnt,
                    })
                    self.logger.info(
                        f"Unified threshold: {unified_metrics.get('eer_threshold', 0):.4f}, "
                        f"EER: {unified_metrics.get('eer', 0):.4f}, "
                        f"F1@EER: {unified_metrics.get('f1_at_eer', 0):.4f}"
                    )

        # Free raw predictions from returned metrics to save memory
        for m in all_val_metrics.values():
            m.pop('all_preds', None)
            m.pop('all_labels', None)

        # 6. Check Lesson Gate
        if self.gate_enabled:
            self._check_lesson_gate(all_val_metrics)

        # 7. Run OOD monitoring (does not affect gating)
        ood_auc = self._run_ood_monitoring(epoch, step_cnt, indist_threshold=indist_threshold)

        # 8. OOD-composite checkpointing (R12+)
        # If enabled, save a separate checkpoint ranked by hmean(holdout_auc, ood_auc).
        # This checkpoint list is independent of the holdout-only top-N list.
        if (
            self.ood_composite_enabled
            and ood_auc is not None
            and ood_auc > 0
        ):
            holdout_m = all_val_metrics.get('val_holdout')
            holdout_auc = holdout_m['overall'].get('auc') if holdout_m and 'overall' in holdout_m else None
            holdout_eer = holdout_m['overall'].get('eer') if holdout_m and 'overall' in holdout_m else None
            if holdout_auc and holdout_auc > 0:
                # Harmonic mean — penalizes large divergence between holdout and OOD
                composite = 2.0 * holdout_auc * ood_auc / (holdout_auc + ood_auc)

                # Log to W&B
                if self.wandb_run:
                    self.wandb_run.log({
                        'val_primary/ood_composite': composite,
                        'val_primary/ood_auc_for_composite': ood_auc,
                        'val_primary/holdout_auc_for_composite': holdout_auc,
                        # A7 piece 1: headline summary/ duplicate.
                        'summary/val_primary/ood_composite': composite,
                        'train/step': step_cnt,
                    })

                self.logger.info(
                    f"OOD composite: {composite:.4f} "
                    f"(holdout_auc={holdout_auc:.4f}, ood_auc={ood_auc:.4f})"
                )

                is_composite_improvement = composite > self.best_ood_composite
                if is_composite_improvement:
                    self.logger.info(
                        f"🎯 OOD COMPOSITE IMPROVED! {composite:.4f} "
                        f"(prev best: {self.best_ood_composite:.4f})"
                    )
                    self.best_ood_composite = composite
                    self.best_ood_composite_step = step_cnt
                    # A2: cache CPU state dict so final_eval can reload without
                    # hitting GCS.
                    try:
                        self._best_ood_composite_state_dict_cpu = {
                            k: v.detach().to('cpu').clone()
                            for k, v in (
                                self.model.module if self.config.get('ddp') else self.model
                            ).state_dict().items()
                        }
                    except Exception as cache_err:
                        self.logger.warning(
                            f"Failed to cache best_ood_composite state dict: {cache_err}"
                        )

                    if self.wandb_run:
                        self.wandb_run.summary['best_ood_composite/metric'] = composite
                        self.wandb_run.summary['best_ood_composite/holdout_auc'] = holdout_auc
                        self.wandb_run.summary['best_ood_composite/ood_auc'] = ood_auc
                        self.wandb_run.summary['best_ood_composite/step'] = step_cnt
                        # A7 piece 1: headline best-checkpoint step.
                        self.wandb_run.summary['summary/best_checkpoint/step'] = step_cnt

                    if self.config.get('save_ckpt', True):
                        is_top = (
                            len(self.ood_composite_top_n) < self.ood_composite_top_n_size
                            or composite > self.ood_composite_top_n[-1]['metric']
                        )
                        if is_top:
                            gcs_path = self.save_ckpt(
                                epoch=epoch + 1,
                                auc=holdout_auc,
                                eer=holdout_eer,
                                ckpt_prefix='ood_composite',
                                step=step_cnt,
                            )
                            if gcs_path:
                                self.ood_composite_top_n.append({
                                    'metric': composite,
                                    'holdout_auc': holdout_auc,
                                    'ood_auc': ood_auc,
                                    'epoch': epoch + 1,
                                    'step': step_cnt,
                                    'gcs_path': gcs_path,
                                })
                                self.ood_composite_top_n.sort(
                                    key=lambda x: x['metric'], reverse=True
                                )
                                if len(self.ood_composite_top_n) > self.ood_composite_top_n_size:
                                    worst = self.ood_composite_top_n.pop()
                                    self._delete_from_gcs(worst['gcs_path'])

                                if self.wandb_run:
                                    self.wandb_run.summary['best_ood_composite/gcs_path'] = (
                                        self.ood_composite_top_n[0]['gcs_path']
                                    )

        # Track locals for best_value_composite block (holdout_m is defined
        # earlier in this same method).
        all_val_metrics_snapshot = all_val_metrics

        # --- Anchor-pool monitor (Phase 0.4 methodology fix) -----------------
        # Per-step inference on ~180 cached real Dor/Roee frames spanning 6
        # pools. ``value_composite`` does NOT see anchor false-positives —
        # this monitor does. Output is logged to W&B and used to GATE
        # ``value_composite_*.pth`` checkpoint writes below. Wrapped in a
        # try/except so it can NEVER crash a long training run.
        anchor_metrics = None
        if (
            getattr(self, 'anchor_monitor_enabled', False)
            and not getattr(self, '_anchor_monitor_disabled_after_failures', False)
        ):
            try:
                from analysis.teams_pool_rescore import compute_anchor_metrics

                model_for_anchor = (
                    self.model.module if self.config.get('ddp') else self.model
                )
                was_training = model_for_anchor.training
                model_for_anchor.eval()
                anchor_device = next(model_for_anchor.parameters()).device
                anchor_metrics = compute_anchor_metrics(
                    model=model_for_anchor,
                    device=anchor_device,
                    anchor_cache_dir=self.anchor_cache_dir,
                )
                if was_training:
                    model_for_anchor.train()

                self.last_anchor_composite = (
                    float(anchor_metrics['composite'])
                    if anchor_metrics['composite'] == anchor_metrics['composite']  # not NaN
                    else None
                )
                if self.last_anchor_composite is not None:
                    if self.last_anchor_composite > self.best_anchor_composite:
                        self.best_anchor_composite = self.last_anchor_composite
                        self.best_anchor_composite_step = step_cnt

                self.logger.info(
                    "anchor monitor: composite=%.4f anchor_mean=%.4f "
                    "max_correct_real=%.4f spread_mean=%.4f n_frames=%d",
                    anchor_metrics['composite'],
                    anchor_metrics['anchor_mean'],
                    anchor_metrics['max_correct_real_mean'],
                    anchor_metrics['spread_mean'],
                    anchor_metrics['n_frames_total'],
                )

                if self.wandb_run:
                    log_dict_anchor = {
                        'anchor/composite': anchor_metrics['composite'],
                        'anchor/anchor_mean': anchor_metrics['anchor_mean'],
                        'anchor/anchor_frac_gt_0_9': anchor_metrics['anchor_frac_gt_0_9'],
                        'anchor/max_correct_real_mean': anchor_metrics['max_correct_real_mean'],
                        'anchor/spread_mean': anchor_metrics['spread_mean'],
                        'anchor/n_frames_total': anchor_metrics['n_frames_total'],
                        'train/step': step_cnt,
                    }
                    for pool_name, mean_val in anchor_metrics['per_pool_mean'].items():
                        log_dict_anchor[f'anchor/pool_{pool_name}_mean'] = mean_val
                    for pool_name, frac_val in anchor_metrics['per_pool_frac_gt_0_9'].items():
                        log_dict_anchor[f'anchor/pool_{pool_name}_frac_gt_0_9'] = frac_val
                    self.wandb_run.log(log_dict_anchor)
                    if self.last_anchor_composite is not None:
                        self.wandb_run.summary['best_anchor/composite'] = (
                            self.best_anchor_composite
                        )
                        self.wandb_run.summary['best_anchor/step'] = (
                            self.best_anchor_composite_step
                        )
            except Exception as anchor_err:
                self._anchor_monitor_failures += 1
                self.logger.warning(
                    "anchor monitor failed (attempt %d): %s",
                    self._anchor_monitor_failures, anchor_err,
                )
                if self._anchor_monitor_failures >= 3:
                    self._anchor_monitor_disabled_after_failures = True
                    self.logger.warning(
                        "anchor monitor disabled after %d consecutive failures.",
                        self._anchor_monitor_failures,
                    )
                anchor_metrics = None

        # A2: parallel tracking for best_value_composite. Readout-only — does
        # NOT change packet-3 checkpoint selection (see §8.7). Guards: only run
        # if A9 produced a non-NaN value_composite and value_composite_enabled.
        vc_state = getattr(self, '_last_value_composite', None)
        if (
            self.value_composite_enabled
            and vc_state is not None
            and vc_state.get('value_composite') is not None
            and vc_state['value_composite'] == vc_state['value_composite']  # not NaN
        ):
            vc_metric = float(vc_state['value_composite'])
            is_vc_improvement = vc_metric > self.best_value_composite

            # Anchor-gated checkpoint logic (Phase 0.4 methodology fix).
            # ``value_composite`` is blind to anchor-pool false-flag behaviour,
            # so a pure-VC gate routinely promotes deployment-broken weights.
            # New rule:
            #   save IFF (vc improved AND anchor/composite did not regress
            #            beyond ``anchor_regression_tolerance``)
            #         OR (anchor/composite improved alone, even with no VC win)
            # When the monitor is unavailable (disabled, errored, NaN), fall
            # back to legacy behaviour (vc improvement only) for backward
            # compatibility — runs without the monitor are bit-identical.
            anchor_now = getattr(self, 'last_anchor_composite', None)
            anchor_best = getattr(self, 'best_anchor_composite', -1e9)
            tol = getattr(self, 'anchor_regression_tolerance', 0.02)

            anchor_available = anchor_now is not None
            is_anchor_improvement = (
                anchor_available and anchor_now > anchor_best
            )
            anchor_did_not_regress_meaningfully = (
                (not anchor_available)
                or (anchor_now >= anchor_best - tol)
            )

            should_save_ckpt = False
            save_reason = None
            if anchor_available:
                if is_vc_improvement and anchor_did_not_regress_meaningfully:
                    should_save_ckpt = True
                    save_reason = "vc_improved_anchor_held"
                elif is_anchor_improvement:
                    should_save_ckpt = True
                    save_reason = "anchor_improved"
            else:
                # Legacy / fallback path — preserves previous behaviour exactly.
                if is_vc_improvement:
                    should_save_ckpt = True
                    save_reason = "vc_improved_no_anchor"

            # Always track best VC (readout-preserving) — independent of save.
            if is_vc_improvement:
                holdout_m_vc = all_val_metrics_snapshot.get('val_holdout') if all_val_metrics_snapshot else None
                holdout_auc_vc = None
                holdout_eer_vc = None
                if holdout_m_vc and 'overall' in holdout_m_vc:
                    holdout_auc_vc = holdout_m_vc['overall'].get('auc')
                    holdout_eer_vc = holdout_m_vc['overall'].get('eer')

                self.logger.info(
                    f"🎯 VALUE COMPOSITE IMPROVED! {vc_metric:.4f} "
                    f"(prev best: {self.best_value_composite:.4f}) "
                    f"[anchor_now={anchor_now} anchor_best={anchor_best:.4f} "
                    f"save={should_save_ckpt} reason={save_reason}]"
                )
                self.best_value_composite = vc_metric
                self.best_value_composite_step = step_cnt
                try:
                    self._best_value_composite_state_dict_cpu = {
                        k: v.detach().to('cpu').clone()
                        for k, v in (
                            self.model.module if self.config.get('ddp') else self.model
                        ).state_dict().items()
                    }
                except Exception as cache_err:
                    self.logger.warning(
                        f"Failed to cache best_value_composite state dict: {cache_err}"
                    )

                if self.wandb_run:
                    self.wandb_run.summary['best_value_composite/metric'] = vc_metric
                    self.wandb_run.summary['best_value_composite/step'] = step_cnt
                    if holdout_auc_vc is not None:
                        self.wandb_run.summary['best_value_composite/holdout_auc'] = holdout_auc_vc
                    self.wandb_run.summary['summary/best_value_composite/step'] = step_cnt
            else:
                # No VC improvement — but we may still need locals for the
                # save path (e.g. anchor_improved alone case).
                holdout_m_vc = all_val_metrics_snapshot.get('val_holdout') if all_val_metrics_snapshot else None
                holdout_auc_vc = None
                holdout_eer_vc = None
                if holdout_m_vc and 'overall' in holdout_m_vc:
                    holdout_auc_vc = holdout_m_vc['overall'].get('auc')
                    holdout_eer_vc = holdout_m_vc['overall'].get('eer')

            if (
                should_save_ckpt
                and self.config.get('save_ckpt', True)
                and holdout_auc_vc is not None
            ):
                # Top-N admission rule:
                # - For VC-driven saves, keep the legacy "must beat the worst
                #   in top_n by metric" rule.
                # - For anchor-only saves, ALWAYS admit and rank by anchor
                #   composite — otherwise an anchor improvement with a low
                #   VC value would never persist (defeating the methodology
                #   fix). The list is heterogeneous after this point but
                #   final_eval reads ``best_value_composite/gcs_path`` from
                #   the leader, which is set explicitly below.
                if save_reason == "anchor_improved":
                    is_top_vc = True
                else:
                    is_top_vc = (
                        len(self.value_composite_top_n) < self.value_composite_top_n_size
                        or vc_metric > self.value_composite_top_n[-1]['metric']
                    )
                if is_top_vc:
                    if self.wandb_run:
                        self.wandb_run.log({
                            'anchor/save_triggered': 1.0,
                            'anchor/save_reason_vc_held': float(save_reason == "vc_improved_anchor_held"),
                            'anchor/save_reason_anchor_only': float(save_reason == "anchor_improved"),
                            'anchor/save_reason_legacy': float(save_reason == "vc_improved_no_anchor"),
                            'train/step': step_cnt,
                        })
                    self.logger.info(
                        f"value_composite ckpt save triggered: reason={save_reason} "
                        f"vc_metric={vc_metric:.4f} anchor_now={anchor_now}"
                    )
                    gcs_path_vc = self.save_ckpt(
                        epoch=epoch + 1,
                        auc=holdout_auc_vc,
                        eer=holdout_eer_vc if holdout_eer_vc is not None else 0.0,
                        ckpt_prefix='value_composite',
                        step=step_cnt,
                    )
                    if gcs_path_vc:
                        self.value_composite_top_n.append({
                            'metric': vc_metric,
                            'holdout_auc': holdout_auc_vc,
                            'anchor_composite': anchor_now,
                            'save_reason': save_reason,
                            'epoch': epoch + 1,
                            'step': step_cnt,
                            'gcs_path': gcs_path_vc,
                        })
                        # Sort key: prefer anchor-improvement saves (they
                        # encode the methodology fix), then rank by metric
                        # within each tier. Within-tier metric is anchor
                        # composite for anchor saves, VC for VC saves —
                        # higher always better.
                        def _vc_sort_key(entry):
                            is_anchor = (
                                entry.get('save_reason') == 'anchor_improved'
                            )
                            ac = entry.get('anchor_composite')
                            ac = ac if ac is not None else float('-inf')
                            m = entry.get('metric', float('-inf'))
                            # Tuple: anchor-tier flag first (1 > 0), then a
                            # within-tier score, then step as a stable tiebreak.
                            within = ac if is_anchor else m
                            return (1 if is_anchor else 0, within, entry.get('step', 0))

                        self.value_composite_top_n.sort(
                            key=_vc_sort_key, reverse=True
                        )
                        if len(self.value_composite_top_n) > self.value_composite_top_n_size:
                            worst_vc = self.value_composite_top_n.pop()
                            self._delete_from_gcs(worst_vc['gcs_path'])
                        if self.wandb_run:
                            self.wandb_run.summary['best_value_composite/gcs_path'] = (
                                self.value_composite_top_n[0]['gcs_path']
                            )

        # Periodic step-based saves — guaranteed checkpoints at fixed steps
        # regardless of metric improvement. Added 2026-04-28 after the P11
        # overnight runs (2026-04-27) revealed that all metric-gated saves
        # (top_n by AUC, ood_composite, value_composite) saturate at step 1000
        # while the underlying anchor/recall metrics continue evolving through
        # step 6000+. Without this trigger, those late-stage improvements are
        # not persisted.
        # _to_plain_dict() unwraps wandb.Config sub-objects that the prior
        # isinstance(dict) guard silently dropped.
        _ps_raw = self.config.get('periodic_saves')
        periodic_cfg = _to_plain_dict(_ps_raw)
        _save_ckpt_ok = self.config.get('save_ckpt', True)
        _step_list = periodic_cfg.get('step_list') or []
        self.logger.info(
            f"periodic_saves diagnostic: step_cnt={step_cnt} "
            f"raw_type={type(_ps_raw).__name__} "
            f"resolved_keys={list(periodic_cfg.keys()) if periodic_cfg else []} "
            f"enabled={periodic_cfg.get('enabled', False)} "
            f"save_ckpt_ok={_save_ckpt_ok} "
            f"step_list={_step_list} "
            f"step_in_list={step_cnt in _step_list}"
        )
        if periodic_cfg.get('enabled', False) and _save_ckpt_ok:
            step_list = _step_list
            if step_cnt in step_list:
                holdout_m_p = (
                    all_val_metrics_snapshot.get('val_holdout')
                    if all_val_metrics_snapshot else None
                )
                holdout_auc_p = None
                holdout_eer_p = None
                if holdout_m_p and 'overall' in holdout_m_p:
                    holdout_auc_p = holdout_m_p['overall'].get('auc')
                    holdout_eer_p = holdout_m_p['overall'].get('eer')
                if holdout_auc_p is not None:
                    self.logger.info(
                        f"periodic_save triggered at step={step_cnt} "
                        f"(holdout_auc={holdout_auc_p:.4f})"
                    )
                    try:
                        gcs_path_p = self.save_ckpt(
                            epoch=epoch + 1,
                            auc=holdout_auc_p,
                            eer=holdout_eer_p if holdout_eer_p is not None else 0.0,
                            ckpt_prefix='periodic',
                            step=step_cnt,
                        )
                        if gcs_path_p and self.wandb_run:
                            self.wandb_run.summary[
                                f'periodic_saves/step_{step_cnt}/gcs_path'
                            ] = gcs_path_p
                            self.wandb_run.summary[
                                f'periodic_saves/step_{step_cnt}/holdout_auc'
                            ] = float(holdout_auc_p)
                    except Exception as save_err:
                        self.logger.warning(
                            f"periodic save failed at step {step_cnt}: {save_err}"
                        )

        # Canary probe: in-training deployment-quality signal.
        # No-op unless config.canary_probe.enabled=true. Bulletproof — never raises.
        self._run_canary_probe(step_cnt)

    @torch.no_grad()
    def test_epoch(self, epoch, step_cnt, validation_loader, log_prefix: str, is_primary_metric: bool,
                   generate_detailed_reports: bool = False, run_name: str = None,
                   output_gcs_folder: str = None, output_filename_prefix: str = None):
        """
        Performs a full evaluation on a given validation dataloader. This function is now
        general-purpose and can be used for any validation set.

        Args:
            epoch (int): The current epoch number.
            step_cnt (int): The current global training step.
            validation_loader (LazyDataLoaderManager): The dataloader to evaluate.
            log_prefix (str): The prefix for WandB logs (e.g., "val_in_dist", "val_holdout").
            is_primary_metric (bool): If True, the results from this run will be used for
                                      checkpointing and early stopping decisions.
            generate_detailed_reports (bool): If True, collects detailed per-frame and
                                              per-video data and uploads CSV/TXT reports to GCS.
            run_name (str): Optional custom name with generate_detailed_reports, used in reports to identify the run.
            output_gcs_folder (str): Optional GCS folder to write reports to (e.g., 'gs://bucket/path/folder').
                                     If specified, appends to existing folder instead of creating new timestamped one.
            output_filename_prefix (str): Optional prefix for report filenames (e.g., 'target_source_').
        """
        self.setEval()

        # Calculate total videos from the provided loader, not a stored class attribute.
        total_videos = sum(len(v) for v in validation_loader.videos_by_method.values())
        if total_videos == 0:
            self.logger.warning(f"No validation videos found for loader with prefix '{log_prefix}'. Skipping.")
            return {}  # Return an empty dict to avoid errors

        self.logger.info(f"--- Starting validation for '{log_prefix}' set ({total_videos} videos)...")
        val_start_time = time.time()
        videos_processed = 0

        method_labels = defaultdict(list)
        method_preds = defaultdict(list)
        # Per-video representative path, parallel to method_preds/method_labels.
        # Consumed by _compute_per_capture_mode_recall_fpr below.
        method_paths = defaultdict(list)
        # A1 mirror: per-video jitter on val/holdout paths, same helper as OOD.
        method_jitter_per_video_val = defaultdict(list)
        all_preds, all_labels = [], []
        all_losses = []
        method_id_to_name = getattr(validation_loader, "method_id_to_name", {}) or {}

        # --- NEW: Initialize lists for detailed reporting if flag is enabled ---
        if generate_detailed_reports:
            frame_report_data = []
            video_report_data = []

        # --- SANITY CHECK: Initialize data collection for model consistency verification ---
        sanity_check_data = []  # Will store first 2 frames per method for verification

        # Iterate through methods from the provided loader.
        for method in validation_loader.keys():
            loader = validation_loader[method]
            num_videos_in_method = len(validation_loader.videos_by_method[method])

            self.logger.info(f"Validating method: {method} ({num_videos_in_method} videos) for '{log_prefix}' set")

            batch_count = 0
            for data_dict in loader:
                batch_count += 1
                # Move tensors to the correct device
                for key, value in data_dict.items():
                    if isinstance(value, torch.Tensor):
                        data_dict[key] = value.to(self.model.device)

                if data_dict['image'].shape[0] == 0 or data_dict['image'].dim() != 5: continue

                B, T = data_dict['image'].shape[:2]
                batch_method_names = [method] * B
                batch_method_ids = data_dict.get("method_id")
                if (
                    isinstance(batch_method_ids, torch.Tensor)
                    and batch_method_ids.numel() == B
                    and method_id_to_name
                ):
                    method_ids_list = batch_method_ids.detach().cpu().tolist()
                    batch_method_names = [
                        method_id_to_name.get(int(method_id), method)
                        for method_id in method_ids_list
                    ]
                
                # Debug logging to track potential duplicate processing
                if generate_detailed_reports and batch_count % 10 == 0:
                    self.logger.info(f"  Processing batch {batch_count} for method '{method}', B={B}, T={T}")
                predictions = self.model(data_dict, inference=True)
                video_probs = predictions['prob'].view(B, T).mean(dim=1)
                
                # --- SANITY CHECK: Collect first 2 frames from first batch of each method ---
                if batch_count == 1:  # Only from the first batch
                    try:
                        frame_probs = predictions['prob'].view(B, T)  # Shape: [B, T]
                        for video_idx in range(min(1, B)):  # Only first video
                            # Safely get video_id with bounds checking
                            if 'video_id' in data_dict and len(data_dict['video_id']) > video_idx:
                                video_id = str(data_dict['video_id'][video_idx])
                            else:
                                video_id = f"unknown_video_{video_idx}"
                            
                            for frame_idx in range(min(2, T)):  # Only first 2 frames
                                # Safely get frame path
                                frame_path = f"video_{video_id}_frame_{frame_idx}"
                                if ('frame_paths' in data_dict and 
                                    data_dict['frame_paths'] is not None and 
                                    len(data_dict['frame_paths']) > video_idx):
                                    try:
                                        if isinstance(data_dict['frame_paths'][video_idx], list) and len(data_dict['frame_paths'][video_idx]) > frame_idx:
                                            frame_path = data_dict['frame_paths'][video_idx][frame_idx]
                                        elif not isinstance(data_dict['frame_paths'][video_idx], list):
                                            frame_path = str(data_dict['frame_paths'][video_idx])
                                    except (IndexError, TypeError):
                                        pass  # Keep default frame_path
                                
                                # Safely get label
                                if 'label' in data_dict and len(data_dict['label']) > video_idx:
                                    label = int(data_dict['label'][video_idx].cpu())
                                else:
                                    label = -1  # Unknown label
                                
                                sanity_check_data.append({
                                    'method': batch_method_names[video_idx] if video_idx < len(batch_method_names) else method,
                                    'video_id': video_id,
                                    'frame_idx': frame_idx,
                                    'frame_path': frame_path,
                                    'probability': float(frame_probs[video_idx, frame_idx].cpu()),
                                    'label': label,
                                    'epoch': epoch + 1,
                                    'step': step_cnt
                                })
                    except Exception as e:
                        self.logger.warning(f"Failed to collect sanity check data for method {method}: {e}")
                        # Continue validation without crashing

                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)
                all_losses.append(losses['overall'].item())

                labels_np = data_dict['label'].cpu().numpy()
                probs_np = video_probs.cpu().numpy()

                # A1 mirror: per-video jitter — attribute each video to its
                # per-sample method_name so sub-family splits stay separate.
                try:
                    frame_probs_np = predictions['prob'].view(B, T).detach().cpu().numpy()
                    for idx, method_name in enumerate(batch_method_names):
                        video_stats = _per_video_jitter_stats(
                            frame_probs_np[idx:idx + 1]
                        )
                        method_jitter_per_video_val[method_name].extend(video_stats)
                except Exception as jitter_err:
                    self.logger.debug(
                        f"val jitter compute failed for '{log_prefix}' method {method}: {jitter_err}"
                    )

                all_labels.extend(labels_np)
                all_preds.extend(probs_np)
                for idx, method_name in enumerate(batch_method_names):
                    method_labels[method_name].append(labels_np[idx])
                    method_preds[method_name].append(probs_np[idx])
                    # Representative frame path (any frame; capture_mode is
                    # video-level). Tolerate missing/empty frame_paths.
                    rep_path = None
                    fp = data_dict.get('frame_paths') if isinstance(data_dict, dict) else None
                    if fp is not None and idx < len(fp):
                        entry = fp[idx]
                        if isinstance(entry, list) and entry:
                            rep_path = str(entry[0])
                        elif entry is not None and not isinstance(entry, list):
                            rep_path = str(entry)
                    method_paths[method_name].append(rep_path)
                videos_processed += data_dict['image'].shape[0]

                # --- NEW: Collect detailed data for reports if flag is enabled ---
                if generate_detailed_reports:
                    frame_level_probs = predictions['prob'].view(B, T)
                    for i in range(B):  # Iterate over each video in the batch
                        video_id = data_dict['video_id'][i]
                        method_name = batch_method_names[i] if i < len(batch_method_names) else method

                        label = labels_np[i]
                        avg_prob = probs_np[i]
                        prediction = 1 if avg_prob >= 0.5 else 0
                        is_correct = 1 if prediction == label else 0
                        group_key, family_key = infer_group_and_family(
                            label=label,
                            method=method_name,
                            source=None,
                        )

                        # Append data for the video-level report
                        video_report_data.append([
                            method_name, label, video_id, avg_prob, prediction, is_correct, group_key, family_key
                        ])

                        # Append data for the frame-level report
                        for j in range(T):  # Iterate over each frame in the video
                            frame_path = data_dict['frame_paths'][i][j]
                            frame_prob = frame_level_probs[i, j].item()
                            frame_report_data.append([
                                method_name, label, video_id, frame_path, frame_prob, group_key, family_key
                            ])

            if generate_detailed_reports:
                self.logger.info(
                    f"  Finished method '{method}'. Processed {batch_count} batches.")
            else:
                self.logger.info(
                    f"  Finished method '{method}'. Total progress: {videos_processed}/{total_videos} videos.")

        if not all_labels:
            self.logger.error(f"Validation failed for '{log_prefix}': No data was processed.")
            return {}  # Return an empty dict

        total_val_time = time.time() - val_start_time
        self.logger.info(f"Validation for '{log_prefix}' finished in {total_val_time:.2f}s. Calculating metrics...")

        # --- NEW: Generate and upload reports if the flag was set ---
        if generate_detailed_reports:
            try:
                # Deduplicate frame data by method+frame_path (keep the first occurrence)
                seen_frame_keys = set()
                deduplicated_frame_data = []
                for row in frame_report_data:
                    method = row[0]      # method is at index 0
                    frame_path = row[3]  # frame_path is at index 3
                    unique_key = f"{method}_{frame_path}"  # Create unique key per method
                    if unique_key not in seen_frame_keys:
                        seen_frame_keys.add(unique_key)
                        deduplicated_frame_data.append(row)
                
                # Deduplicate video data by method+video_id (keep the first occurrence)
                seen_video_keys = set()
                deduplicated_video_data = []
                for row in video_report_data:
                    method = row[0]   # method is at index 0
                    video_id = row[2] # video_id is at index 2
                    unique_key = f"{method}_{video_id}"  # Create unique key per method
                    if unique_key not in seen_video_keys:
                        seen_video_keys.add(unique_key)
                        deduplicated_video_data.append(row)
                
                self.logger.info(f"Deduplication: Frame data reduced from {len(frame_report_data)} to {len(deduplicated_frame_data)} entries")
                self.logger.info(f"Deduplication: Video data reduced from {len(video_report_data)} to {len(deduplicated_video_data)} entries")
                
                self._generate_and_upload_reports(
                    log_prefix,
                    deduplicated_frame_data,
                    deduplicated_video_data,
                    all_preds,
                    all_labels,
                    method_preds,
                    method_labels,
                    generate_detailed_reports,
                    run_name=run_name,
                    output_gcs_folder=output_gcs_folder,
                    output_filename_prefix=output_filename_prefix,
                )
                self.logger.info("✅ Detailed reports generated and uploaded successfully.")
            except Exception as e:
                self.logger.error(f"❌ Error generating detailed reports: {e}")
                self.logger.error("Continuing with metric calculation...")

        self.logger.info(f"--- Calculating overall performance for '{log_prefix}' ---")
        try:
            overall_metrics = get_test_metrics(np.array(all_preds), np.array(all_labels))
        except Exception as e:
            self.logger.error(f"❌ Error calculating test metrics: {e}")
            overall_metrics = {
                'loss': -1.0,
                'acc': -1.0, 
                'auc': -1.0,
                'eer': -1.0,
                'eer_threshold': -1.0,
                'ap': -1.0,
                'error': str(e)
            }

        # Use the log_prefix for all WandB metrics to create separate charts.
        wandb_log_dict = {f"{log_prefix}/epoch": epoch + 1, "train/step": step_cnt}

        if all_losses:
            avg_val_loss = np.mean(all_losses)
            wandb_log_dict[f'{log_prefix}/overall/loss'] = avg_val_loss
            self.logger.info(f"Overall {log_prefix} loss: {avg_val_loss:.4f}")

        for name, value in overall_metrics.items():
            if name not in ['pred', 'label']:
                wandb_log_dict[f'{log_prefix}/overall/{name}'] = value
                self.logger.info(f"Overall {log_prefix} {name}: {value:.4f}")

        # A7 piece 1: summary/ namespace — duplicate val_holdout headline AUC
        # into the summary panel so the default filter view surfaces it.
        if log_prefix == 'val_holdout' and 'auc' in overall_metrics:
            wandb_log_dict[f'summary/val_holdout/auc'] = overall_metrics['auc']

        wandb_log_dict[f'{log_prefix}/probabilities'] = wandb.Histogram(np.array(all_preds))

        # A1 mirror: emit per-method jitter family on val / val_holdout paths.
        val_histogram_gate = (
            step_cnt is not None
            and step_cnt > 0
            and int(step_cnt) % 2500 == 0
            and self.wandb_run is not None
        )
        val_jitter_summary_cache = {}
        for method_name, per_video_list in method_jitter_per_video_val.items():
            agg_v = _aggregate_jitter_across_videos(per_video_list, spike_threshold=0.3)
            if not agg_v:
                continue
            val_jitter_summary_cache[method_name] = agg_v
            wandb_log_dict[f'{log_prefix}/score_jitter/{method_name}'] = agg_v['mean']
            wandb_log_dict[f'{log_prefix}/score_jitter_max/{method_name}'] = agg_v['max']
            wandb_log_dict[f'{log_prefix}/score_jitter_p95/{method_name}'] = agg_v['p95']
            wandb_log_dict[f'{log_prefix}/score_jitter_spike_rate_0p3/{method_name}'] = \
                agg_v.get('spike_rate_0p3', 0.0)
            if val_histogram_gate and agg_v.get('all_diffs') is not None and agg_v['all_diffs'].size > 0:
                try:
                    wandb_log_dict[f'{log_prefix}/score_jitter_hist/{method_name}'] = \
                        wandb.Histogram(np.clip(agg_v['all_diffs'], 0.0, 1.0))
                except Exception as hist_err:
                    self.logger.debug(f"val jitter histogram log failed for {method_name}: {hist_err}")
        if val_jitter_summary_cache:
            setattr(self, f"_last_{log_prefix}_jitter_summary", val_jitter_summary_cache)

        # --- THIS IS THE CRUCIAL CONTROL BLOCK ---
        # All checkpointing and early stopping logic is now conditional on this being the
        # designated primary validation set (i.e., the holdout set).
        if is_primary_metric:
            current_metric = overall_metrics.get(self.metric_scoring)
            if current_metric is None:
                self.logger.warning(
                    f"Primary metric '{self.metric_scoring}' not found. Skipping checkpointing and early stopping check.")
            else:
                is_improvement = current_metric > self.best_val_metric + self.early_stopping_min_delta
                if is_improvement:
                    self.logger.info(
                        f"🚀 PRIMARY METRIC IMPROVED! New best {self.metric_scoring}: {current_metric:.4f} (previously {self.best_val_metric:.4f})")
                    self.best_val_metric = current_metric
                    self.best_val_epoch = epoch + 1
                    self.epochs_without_improvement = 0

                    if self.wandb_run:
                        self.wandb_run.summary['best/epoch'] = self.best_val_epoch
                        self.wandb_run.summary['best/metric'] = self.best_val_metric
                        self.wandb_run.summary['best/auc'] = overall_metrics.get('auc', 0)
                        self.wandb_run.summary['best/eer'] = overall_metrics.get('eer', 0)
                        self.wandb_run.summary['best/eer_threshold'] = overall_metrics.get('eer_threshold', 0)
                        self.wandb_run.summary['best/acc'] = overall_metrics.get('acc', 0)
                        # FPR operating points at best epoch
                        for fpr_key in ['tpr_at_fpr1pct', 'tpr_at_fpr2pct', 'tpr_at_fpr5pct',
                                        'thresh_at_fpr1pct', 'thresh_at_fpr2pct', 'thresh_at_fpr5pct']:
                            if fpr_key in overall_metrics:
                                self.wandb_run.summary[f'best/{fpr_key}'] = overall_metrics[fpr_key]

                    if self.config.get('save_ckpt', True):
                        self.logger.info(f"✅ Saving new best checkpoint to GCS (Epoch {epoch + 1})...")
                        if self.first_best_gcs_path is None:
                            new_gcs_path = self.save_ckpt(epoch=epoch + 1, auc=overall_metrics.get('auc'),
                                                          eer=overall_metrics.get('eer'), ckpt_prefix='first_best')
                            if new_gcs_path:
                                self.first_best_gcs_path = new_gcs_path

                        is_top_n = len(self.top_n_checkpoints) < self.top_n_size or current_metric > \
                                   self.top_n_checkpoints[-1]['metric']
                        if is_top_n:
                            new_gcs_path = self.save_ckpt(epoch=epoch + 1, auc=overall_metrics.get('auc'),
                                                          eer=overall_metrics.get('eer'), ckpt_prefix='top_n',
                                                          step=step_cnt)
                            if new_gcs_path:
                                self.top_n_checkpoints.append(
                                    {'metric': current_metric, 'epoch': epoch + 1, 'gcs_path': new_gcs_path})
                                self.top_n_checkpoints.sort(key=lambda x: x['metric'], reverse=True)
                                if len(self.top_n_checkpoints) > self.top_n_size:
                                    worst_ckpt = self.top_n_checkpoints.pop()
                                    self._delete_from_gcs(worst_ckpt['gcs_path'])

                    if self.wandb_run:
                        self.wandb_run.summary['overall_best_ckpt_gcs'] = self.top_n_checkpoints[0]['gcs_path']
                        self.wandb_run.summary[
                            'bottom_line'] = f"best_epoch={self.best_val_epoch} AUC={self.wandb_run.summary['best/auc']:.4f}"

                else:  # No improvement
                    if self.early_stopping_enabled:
                        self.epochs_without_improvement += 1
                        self.logger.warning(
                            f"No primary metric improvement for {self.epochs_without_improvement}/{self.early_stopping_patience} epochs. "
                            f"Current {self.metric_scoring}: {current_metric:.4f}, Best: {self.best_val_metric:.4f}"
                        )

                if self.early_stopping_enabled and self.epochs_without_improvement >= self.early_stopping_patience:
                    self.early_stop_triggered = True
                    self.logger.critical(
                        f"🚨 EARLY STOPPING TRIGGERED! No improvement in '{self.metric_scoring}' for {self.early_stopping_patience} epochs.")

        # Log the tracked best metric state regardless of which validation set is running for consistent tracking.
        # We namespace it to make it clear this is the state of the *primary* validation metric.
        wandb_log_dict['val_primary/best_metric'] = self.best_val_metric
        wandb_log_dict['val_primary/best_epoch'] = self.best_val_epoch
        if self.early_stopping_enabled:
            wandb_log_dict['val_primary/epochs_without_improvement'] = self.epochs_without_improvement

        # --- Calculate and log MEANINGFUL per-method and derived metrics ---
        per_method_aucs = []
        real_source_names = self.config.get('dataset_methods', {}).get('use_real_sources', [])

        # Create a temporary dictionary for method metrics to build the table
        method_table_metrics = {}

        # 1. Pool all real predictions and labels from the validation set
        all_real_preds = []
        all_real_labels = []
        for method_name in real_source_names:
            if method_name in method_preds:
                all_real_preds.extend(method_preds[method_name])
                all_real_labels.extend(method_labels[method_name])

        # 2. Calculate TRUE per-method accuracy for each FAKE method (threshold-based)
        for method in sorted(method_preds.keys()):
            # Only calculate for fake methods
            if method not in real_source_names:
                method_preds_array = np.array(method_preds[method])
                method_labels_array = np.array(method_labels[method])
                
                # Calculate threshold-based accuracy: how many fake videos are correctly classified as fake (>=0.5)
                predictions_binary = (method_preds_array >= 0.5).astype(int)
                correct_predictions = np.sum(predictions_binary == method_labels_array)
                per_method_accuracy = correct_predictions / len(method_labels_array) if len(method_labels_array) > 0 else 0.0
                
                # Store simplified metrics (only accuracy for fake methods)
                method_table_metrics[method] = {'acc': per_method_accuracy, 'n_samples': len(method_labels_array)}
                
                # Log only the meaningful accuracy metric
                wandb_log_dict[f'{log_prefix}/method/{method}/acc'] = per_method_accuracy
                
                self.logger.info(f"Method '{method}' per-method accuracy: {per_method_accuracy:.4f} ({correct_predictions}/{len(method_labels_array)})")

        # 3. Calculate TRUE per-method accuracy for each REAL method (threshold-based)
        for method in sorted(method_preds.keys()):
            if method in real_source_names:
                method_preds_array = np.array(method_preds[method])
                method_labels_array = np.array(method_labels[method])
                
                # Calculate threshold-based accuracy: how many real videos are correctly classified as real (<0.5)
                predictions_binary = (method_preds_array >= 0.5).astype(int)
                correct_predictions = np.sum(predictions_binary == method_labels_array)
                per_method_accuracy = correct_predictions / len(method_labels_array) if len(method_labels_array) > 0 else 0.0
                
                # Store simplified metrics (only accuracy for real methods)
                method_table_metrics[method] = {'acc': per_method_accuracy, 'n_samples': len(method_labels_array)}
                
                # Log only the meaningful accuracy metric
                wandb_log_dict[f'{log_prefix}/method/{method}/acc'] = per_method_accuracy
                
                self.logger.info(f"Method '{method}' per-method accuracy: {per_method_accuracy:.4f} ({correct_predictions}/{len(method_labels_array)})")

        # --- W&B Block B: per-dataset-bucket recall/FPR for mid-training ---
        # Adds keys of the form `{log_prefix}/per_bucket/<dataset>/recall_fake`
        # (fake buckets) or `{log_prefix}/per_bucket/<dataset>/{recall_real,fpr}`
        # (real buckets) at the same cadence as the existing eval cycle.
        # Read-only on `method_preds`/`method_labels`; pure numpy at log time.
        try:
            per_bucket_metrics = _compute_per_bucket_recall_fpr(
                method_preds=method_preds,
                method_labels=method_labels,
                real_source_names=real_source_names,
                threshold=0.5,
                log_prefix=f"{log_prefix}/per_bucket",
            )
            wandb_log_dict.update(per_bucket_metrics)
            if per_bucket_metrics:
                self.logger.info(
                    f"Logged {len(per_bucket_metrics)} per-bucket recall/FPR metrics under '{log_prefix}/per_bucket/'"
                )
        except Exception as block_b_err:
            # Observability MUST NEVER take down training. Log and continue.
            self.logger.warning(
                f"W&B Block B per-bucket logging failed for '{log_prefix}': {block_b_err}"
            )

        # --- Per-capture-mode mid-eval (companion to Block B) ---
        # Reads `mid_eval_capture_mode_parquet` from config (top-level, not
        # nested — avoids the wandb-flattens-nested-dicts bug).
        # Disabled when the config key is unset / file missing.
        try:
            parquet_path = self.config.get('mid_eval_capture_mode_parquet')
            # hasattr — not truthiness — to distinguish "not loaded yet" from
            # "loaded but empty" (so we don't re-read the parquet every cycle
            # when the config key is unset).
            if not hasattr(self, '_capture_mode_lookup_cache'):
                self._capture_mode_lookup_cache = _load_capture_mode_lookup(parquet_path)
            lookup = self._capture_mode_lookup_cache
            if lookup:
                per_mode_metrics = _compute_per_capture_mode_recall_fpr(
                    method_preds=method_preds,
                    method_labels=method_labels,
                    method_paths=method_paths,
                    capture_mode_lookup=lookup,
                    real_source_names=real_source_names,
                    threshold=0.5,
                    log_prefix=f"{log_prefix}/per_capture_mode",
                )
                wandb_log_dict.update(per_mode_metrics)
                if per_mode_metrics:
                    self.logger.info(
                        f"Logged {len(per_mode_metrics)} per-capture-mode metrics under '{log_prefix}/per_capture_mode/'"
                    )
        except Exception as capmode_err:
            self.logger.warning(
                f"Per-capture-mode logging failed for '{log_prefix}': {capmode_err}"
            )

        # Create and log a simplified W&B Table with only meaningful metrics
        # --- Weakest-method tracking ---
        real_method_accs = {
            m: metrics['acc'] for m, metrics in method_table_metrics.items()
            if m in real_source_names and metrics.get('acc') is not None
        }
        fake_method_accs = {
            m: metrics['acc'] for m, metrics in method_table_metrics.items()
            if m not in real_source_names and metrics.get('acc') is not None
        }
        if real_method_accs:
            worst_real = min(real_method_accs.items(), key=lambda x: x[1])
            wandb_log_dict[f'{log_prefix}/weakest/real_method'] = worst_real[0]
            wandb_log_dict[f'{log_prefix}/weakest/real_acc'] = worst_real[1]
            self.logger.info(f"Weakest real method: {worst_real[0]} ({worst_real[1]:.4f})")
        if fake_method_accs:
            worst_fake = min(fake_method_accs.items(), key=lambda x: x[1])
            wandb_log_dict[f'{log_prefix}/weakest/fake_method'] = worst_fake[0]
            wandb_log_dict[f'{log_prefix}/weakest/fake_acc'] = worst_fake[1]
            self.logger.info(f"Weakest fake method: {worst_fake[0]} ({worst_fake[1]:.4f})")

        if self.wandb_run:
            columns = ["epoch", "method", "acc", "n_samples"]
            table_data = []
            for method, metrics in method_table_metrics.items():
                table_data.append([
                    epoch + 1, method, metrics.get('acc'), metrics.get('n_samples')
                ])
            wandb_log_dict[_safe_wandb_table_key(log_prefix, "method_table")] = wandb.Table(columns=columns, data=table_data)

        # Calculate macro accuracy: Unweighted average of per-method accuracies
        method_accuracies = [metrics.get('acc') for metrics in method_table_metrics.values() if metrics.get('acc') is not None]
        if method_accuracies:
            macro_accuracy = np.mean(method_accuracies)
            wandb_log_dict[f'{log_prefix}/derived/macro_accuracy'] = macro_accuracy
            self.logger.info(f"Macro per-method accuracy: {macro_accuracy:.4f}")
        else:
            macro_accuracy = None

        # Real-Real AUC calculation can remain the same
        real_preds_for_auc = []
        real_labels_for_auc = []
        for method_name in real_source_names:
            if method_name in method_preds:
                real_preds_for_auc.extend(method_preds[method_name])
                real_labels_for_auc.extend(method_labels[method_name])

        if len(real_labels_for_auc) > 1 and len(np.unique(real_labels_for_auc)) > 1:
            real_real_metrics = get_test_metrics(np.array(real_preds_for_auc), np.array(real_labels_for_auc))
            real_real_auc = real_real_metrics.get('auc', 0.0)
            wandb_log_dict[f'{log_prefix}/derived/real_real_auc'] = real_real_auc
        else:
            real_real_auc = None

        # --- SANITY CHECK: Log verification data as W&B table ---
        if self.wandb_run and sanity_check_data:
            sanity_check_columns = ['method', 'video_id', 'frame_idx', 'frame_path', 'probability', 'label', 'epoch', 'step']
            sanity_check_table_data = [[row[col] for col in sanity_check_columns] for row in sanity_check_data]
            wandb_log_dict[_safe_wandb_table_key(log_prefix, "sanity_check_predictions")] = wandb.Table(
                columns=sanity_check_columns,
                data=sanity_check_table_data
            )
            
            self.logger.info(f"✅ Logged {len(sanity_check_data)} sanity check predictions for verification")
            
            # Also log a summary for quick reference
            sanity_summary = {}
            for row in sanity_check_data:
                method_key = f"{row['method']}_frame_{row['frame_idx']}"
                sanity_summary[f"{log_prefix}/sanity/{method_key}"] = row['probability']
            wandb_log_dict.update(sanity_summary)

        if self.wandb_run: self.wandb_run.log(wandb_log_dict)

        returned_metrics = {
            'overall': overall_metrics,
            'per_method': method_table_metrics,
            'macro_accuracy': macro_accuracy if 'macro_accuracy' in locals() else None,
            'real_real_auc': real_real_auc,
            'all_preds': np.array(all_preds),
            'all_labels': np.array(all_labels),
        }

        del method_labels, method_preds, overall_metrics, all_losses
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info(f"===> Evaluation for '{log_prefix}' Done!")
        return returned_metrics

    def run_final_eval(self):
        """A2: dual-checkpoint final evaluation at training end.

        Reloads each cached "best" state dict in turn and evaluates it on:
          - the 5% test slice (``self.test_loader``), if present
          - the A10 held-out OOD slice (``self.ood_heldout_loader``), if present
          - the val_holdout loader (for a frozen end-of-training readout)

        Results are logged under ``final_eval/by_ood_composite/*`` and
        ``final_eval/by_value_composite/*``. When only one cached checkpoint
        exists (e.g. ``best_value_composite`` never improved), the other block
        is skipped. ``final_eval/agree/same_step`` records whether the two
        criteria picked the same training step — a strong signal they agree.

        Readout-only: packet-3 selection is still ``best_ood_composite``. The
        by_value_composite block exists so packet 4 can decide whether to
        switch selection metrics without paying for a retroactive re-eval.
        """
        if self.config['local_rank'] != 0:
            return
        if self.wandb_run is None:
            self.logger.info("run_final_eval skipped: no wandb_run.")
            return

        cached = []
        if self._best_ood_composite_state_dict_cpu is not None:
            cached.append(("ood_composite", self._best_ood_composite_state_dict_cpu,
                            self.best_ood_composite_step, self.best_ood_composite))
        if self._best_value_composite_state_dict_cpu is not None:
            cached.append(("value_composite", self._best_value_composite_state_dict_cpu,
                            self.best_value_composite_step, self.best_value_composite))

        if not cached:
            self.logger.info("run_final_eval skipped: no cached best checkpoints.")
            return

        self.logger.info("=" * 70)
        self.logger.info(
            "🏁 Running A2 final_eval over %d cached checkpoint(s): %s",
            len(cached),
            [c[0] for c in cached],
        )
        self.logger.info("=" * 70)

        # Snapshot the current (last-step) state dict so we can restore it
        # after final_eval mutates model weights.
        inner_model = self.model.module if self.config.get('ddp') else self.model
        try:
            current_state = {k: v.detach().to('cpu').clone()
                             for k, v in inner_model.state_dict().items()}
        except Exception as snap_err:
            self.logger.warning(f"run_final_eval: state snapshot failed, aborting: {snap_err}")
            return

        final_eval_summary = {}
        try:
            for crit_label, state_dict_cpu, best_step, best_metric in cached:
                self.logger.info(
                    f"--- final_eval(by_{crit_label}): reloading step={best_step} metric={best_metric:.4f}"
                )
                try:
                    inner_model.load_state_dict(
                        {k: v.to(self.model.device) for k, v in state_dict_cpu.items()},
                        strict=True,
                    )
                except Exception as load_err:
                    self.logger.warning(
                        f"run_final_eval: failed to load by_{crit_label}: {load_err}"
                    )
                    continue

                # A2 eval pass on val_holdout.
                if self.val_holdout_loader is not None:
                    try:
                        m_holdout = self.test_epoch(
                            epoch=-1,
                            step_cnt=best_step if best_step and best_step > 0 else -1,
                            validation_loader=self.val_holdout_loader,
                            log_prefix=f"final_eval/by_{crit_label}/val_holdout",
                            is_primary_metric=False,
                        )
                        if m_holdout and 'overall' in m_holdout:
                            auc_v = m_holdout['overall'].get('auc')
                            if auc_v is not None:
                                final_eval_summary[f"final_eval/by_{crit_label}/val_holdout/auc"] = auc_v
                    except Exception as ev_err:
                        self.logger.warning(f"final_eval val_holdout({crit_label}) failed: {ev_err}")

                # A2 eval pass on the 5% test slice.
                if self.test_loader is not None:
                    try:
                        m_test = self.test_epoch(
                            epoch=-1,
                            step_cnt=best_step if best_step and best_step > 0 else -1,
                            validation_loader=self.test_loader,
                            log_prefix=f"final_eval/by_{crit_label}/test",
                            is_primary_metric=False,
                        )
                        if m_test and 'overall' in m_test:
                            auc_t = m_test['overall'].get('auc')
                            if auc_t is not None:
                                final_eval_summary[f"final_eval/by_{crit_label}/test/auc"] = auc_t
                                if crit_label == "ood_composite":
                                    # A7 piece 1 headline.
                                    final_eval_summary["summary/final_eval/test/auc"] = auc_t
                    except Exception as ev_err:
                        self.logger.warning(f"final_eval test({crit_label}) failed: {ev_err}")

                # A10 held-out OOD pass — temporarily swap ood_loader.
                if self.ood_heldout_loader is not None:
                    saved_ood = self.ood_loader
                    self.ood_loader = self.ood_heldout_loader
                    try:
                        heldout_auc = self.ood_monitoring_epoch(
                            epoch=-1,
                            step_cnt=best_step if best_step and best_step > 0 else -1,
                            indist_threshold=None,
                        )
                        if heldout_auc is not None:
                            final_eval_summary[
                                f"final_eval/by_{crit_label}/heldout_ood/auc"
                            ] = heldout_auc
                    except Exception as ev_err:
                        self.logger.warning(f"final_eval heldout({crit_label}) failed: {ev_err}")
                    finally:
                        self.ood_loader = saved_ood

            # agree/same_step flag — most informative when both criteria caught.
            if len(cached) == 2:
                same_step = bool(cached[0][2] == cached[1][2] and cached[0][2] > 0)
                final_eval_summary["final_eval/agree/same_step"] = same_step
                final_eval_summary["final_eval/agree/by_ood_composite_step"] = cached[0][2]
                final_eval_summary["final_eval/agree/by_value_composite_step"] = cached[1][2]
                self.logger.info(
                    f"final_eval agreement: same_step={same_step} "
                    f"ood_composite_step={cached[0][2]} value_composite_step={cached[1][2]}"
                )

            if self.wandb_run and final_eval_summary:
                for k, v in final_eval_summary.items():
                    try:
                        self.wandb_run.summary[k] = v
                    except Exception as summary_err:
                        self.logger.debug(f"summary write failed for {k}: {summary_err}")
        finally:
            # Always restore the last-step state so downstream code doesn't
            # silently operate on a reloaded checkpoint.
            try:
                inner_model.load_state_dict(
                    {k: v.to(self.model.device) for k, v in current_state.items()},
                    strict=True,
                )
                self.logger.info("run_final_eval: restored last-step model state.")
            except Exception as rest_err:
                self.logger.error(
                    f"run_final_eval: FAILED to restore last-step state: {rest_err}. "
                    "Model left on the last reloaded checkpoint."
                )

    def _run_ood_monitoring(self, epoch, step_cnt, indist_threshold=None):
        """Helper to run the OOD monitoring loop.

        Returns:
            float | None: Overall OOD AUC if monitoring ran, else None.
        """
        if self.ood_loader is None or self.config['local_rank'] != 0:
            return None
        if not self.ood_monitoring_enabled:
            return None
        if step_cnt < self.ood_monitoring_start_step:
            if not self._ood_warmup_logged:
                self.logger.info(
                    "OOD monitoring warmup active: first run at step %d",
                    self.ood_monitoring_start_step,
                )
                self._ood_warmup_logged = True
            return None

        steps_since_start = step_cnt - self.ood_monitoring_start_step
        if steps_since_start % self.ood_monitoring_every_steps != 0:
            return None
        if self._last_ood_monitor_step == step_cnt:
            return None

        self._last_ood_monitor_step = step_cnt
        self.logger.info(f"\n===> OOD Monitoring at epoch {epoch + 1}, step {step_cnt}")
        return self.ood_monitoring_epoch(epoch, step_cnt, indist_threshold=indist_threshold)

    @torch.no_grad()
    def ood_monitoring_epoch(self, epoch, step_cnt, indist_threshold=None):
        """
        Runs evaluation on the OOD set. Logs metrics with an 'ood/' prefix.

        Returns the overall OOD AUC so callers can use it for composite
        checkpointing (R12+).  Does NOT trigger checkpointing itself.

        Args:
            indist_threshold: If provided, also logs ``ood/at_indist/*`` metrics
                evaluated at this fixed threshold (the val_in_dist EER threshold).

        Returns:
            float | None: Overall OOD AUC, or None if evaluation failed.
        """
        self.setEval()

        total_videos = sum(len(v_list) for v_list in self.ood_loader.videos_by_method.values())
        if total_videos == 0:
            self.logger.warning("OOD loader is configured but contains no videos. Skipping.")
            return None

        method_labels = defaultdict(list)
        method_preds = defaultdict(list)
        # A1: per-video jitter — store per-video dicts (mean/max/diffs) instead
        # of a single scalar. Aggregation happens at log time.
        method_jitter_per_video = defaultdict(list)
        all_preds, all_labels = [], []

        self.logger.info(f"Starting OOD Monitoring for {total_videos} videos...")
        ood_start_time = time.time()
        videos_processed = 0

        for method in self.ood_loader.keys():
            loader = self.ood_loader[method]

            for data_dict in loader:
                for key, value in data_dict.items():
                    if isinstance(value, torch.Tensor): data_dict[key] = value.to(self.model.device)
                if data_dict['image'].shape[0] == 0 or data_dict['image'].dim() != 5: continue

                B, T = data_dict['image'].shape[:2]
                predictions = self.model(data_dict, inference=True)
                video_probs = predictions['prob'].view(B, T).mean(dim=1)

                # A1: per-video frame-to-frame score jitter
                frame_probs = predictions['prob'].view(B, T).detach().cpu().numpy()
                method_jitter_per_video[method].extend(
                    _per_video_jitter_stats(frame_probs)
                )

                labels_np = data_dict['label'].cpu().numpy()
                probs_np = video_probs.cpu().numpy()

                all_labels.extend(labels_np)
                all_preds.extend(probs_np)
                method_labels[method].extend(labels_np)
                method_preds[method].extend(probs_np)

                B = data_dict['image'].shape[0]
                videos_processed += B

            self.logger.info(
                f"  ... OOD method {method} done. Total progress: {videos_processed}/{total_videos} videos.")

        total_ood_time = time.time() - ood_start_time
        self.logger.info(f"OOD Monitoring finished in {total_ood_time:.2f}s. Calculating metrics...")

        if not all_labels:
            self.logger.error("OOD Monitoring failed: No data was processed.")
            return

        overall_metrics = get_test_metrics(np.array(all_preds), np.array(all_labels))
        wandb_log_dict = {"ood/epoch": epoch + 1, "train/step": step_cnt}

        for name, value in overall_metrics.items():
            if name not in ['pred', 'label']:
                wandb_log_dict[f'ood/overall/{name}'] = value
                self.logger.info(f"OOD Overall {name}: {value:.4f}")

        # A7 piece 1: summary/ namespace — headline OOD AUC.
        if 'auc' in overall_metrics:
            wandb_log_dict['summary/ood/overall/auc'] = overall_metrics['auc']

        # Log the probability distribution histogram
        wandb_log_dict['ood/probabilities'] = wandb.Histogram(np.array(all_preds))

        # Metrics Per Method
        method_metrics = {}
        for method in method_preds.keys():
            # Check if there are any labels for this method to avoid errors
            if not method_labels[method]:
                continue

            labels_for_method = np.array(method_labels[method])
            preds_for_method = np.array(method_preds[method])

            # All labels for a given method group (e.g., 'tiktok' from real videos) should be the same.
            # We use the first label to determine if it's a real (0) or fake (1) group.
            label_type = '_real' if labels_for_method[0] == 0 else '_fake'
            method_key = f"{method}{label_type}"

            method_metrics[method] = get_test_metrics(preds_for_method, labels_for_method)

            for name, value in method_metrics[method].items():
                if name in ['acc', 'auc', 'eer']:
                    wandb_log_dict[f'ood/method/{method_key}/{name}'] = value

        # A1: per-method score jitter — mean / max / p95 / spike_rate_0p3 + gated histogram.
        histogram_gate = (
            step_cnt is not None
            and step_cnt > 0
            and int(step_cnt) % 2500 == 0
            and self.wandb_run is not None
        )
        ood_jitter_summary_cache = {}
        for method, per_video_list in method_jitter_per_video.items():
            agg = _aggregate_jitter_across_videos(per_video_list, spike_threshold=0.3)
            if not agg:
                continue
            ood_jitter_summary_cache[method] = agg
            wandb_log_dict[f'ood/score_jitter/{method}'] = agg['mean']
            wandb_log_dict[f'ood/score_jitter_max/{method}'] = agg['max']
            wandb_log_dict[f'ood/score_jitter_p95/{method}'] = agg['p95']
            wandb_log_dict[f'ood/score_jitter_spike_rate_0p3/{method}'] = agg.get(
                'spike_rate_0p3', 0.0
            )
            self.logger.info(
                f"OOD jitter {method}: mean={agg['mean']:.4f} max={agg['max']:.4f} "
                f"p95={agg['p95']:.4f} spike_rate_0p3={agg.get('spike_rate_0p3', 0.0):.4f} "
                f"n={agg['n_diffs']}"
            )
            if histogram_gate and agg.get('all_diffs') is not None and agg['all_diffs'].size > 0:
                try:
                    wandb_log_dict[f'ood/score_jitter_hist/{method}'] = wandb.Histogram(
                        np.clip(agg['all_diffs'], 0.0, 1.0)
                    )
                except Exception as hist_err:
                    self.logger.debug(f"Histogram log failed for {method}: {hist_err}")
            # A7: headline summary/ keys for the two most decision-driving
            # jitter numbers (teams_ood_fake). Stays quiet for other methods.
            if method == 'teams_ood_fake':
                wandb_log_dict['summary/ood/score_jitter_max/teams_ood_fake'] = agg['max']
                wandb_log_dict['summary/ood/score_jitter_spike_rate_0p3/teams_ood_fake'] = \
                    agg.get('spike_rate_0p3', 0.0)
        self._last_ood_jitter_summary = ood_jitter_summary_cache

        # A3 / A3b: stress-family aggregate AUCs.
        # Group methods by prefix: "ood_lighting_stress_*" -> lighting family,
        # "ood_spatial_stress_*" -> spatial family. For each family, emit the
        # per-preset AUC plus an overall aggregate for the summary/ panel.
        stress_families = {
            "ood_lighting_stress": "ood_lighting_stress_",
            "ood_spatial_stress": "ood_spatial_stress_",
        }
        for family_log_key, family_prefix in stress_families.items():
            family_methods = [
                m for m in method_preds.keys() if str(m).startswith(family_prefix)
            ]
            if not family_methods:
                continue
            family_all_preds = []
            family_all_labels = []
            for m in family_methods:
                if not method_labels[m]:
                    continue
                labels_arr = np.asarray(method_labels[m])
                preds_arr = np.asarray(method_preds[m])
                family_all_preds.append(preds_arr)
                family_all_labels.append(labels_arr)
                # Per-preset AUC when the method has both real and fake samples
                # within it (or it's part of a paired real+fake stress set).
                if len(np.unique(labels_arr)) > 1:
                    try:
                        m_metrics = get_test_metrics(preds_arr, labels_arr)
                        if 'auc' in m_metrics:
                            preset_token = str(m)[len(family_prefix):]
                            wandb_log_dict[
                                f"{family_log_key}/{preset_token}/auc"
                            ] = m_metrics['auc']
                    except Exception as preset_err:
                        self.logger.debug(
                            f"preset AUC failed for {m}: {preset_err}"
                        )
            if family_all_preds:
                merged_preds = np.concatenate(family_all_preds)
                merged_labels = np.concatenate(family_all_labels)
                if len(np.unique(merged_labels)) > 1:
                    try:
                        fam_metrics = get_test_metrics(merged_preds, merged_labels)
                        if 'auc' in fam_metrics:
                            wandb_log_dict[f"{family_log_key}/overall/auc"] = fam_metrics['auc']
                            # A7 piece 1 headline.
                            wandb_log_dict[f"summary/{family_log_key}/auc"] = fam_metrics['auc']
                    except Exception as fam_err:
                        self.logger.debug(
                            f"stress family AUC failed ({family_log_key}): {fam_err}"
                        )

        # A9: value_composite (deployment-aligned readout; does NOT drive
        # checkpoint selection in packet 3). Assembled from whatever real/fake
        # pools are present in the OOD loader. Missing pools (e.g. A6 enhanced-
        # proper lanes, deferred) are logged once as a warning and skipped.
        try:
            real_pools_for_vc = {}
            teams_fake_pools_for_vc = {}
            other_fake_pools_for_vc = {}

            for m, preds_list in method_preds.items():
                labels_np = np.asarray(method_labels[m])
                preds_np = np.asarray(preds_list)
                if labels_np.size == 0:
                    continue
                is_real = int(labels_np[0]) == 0
                pool_blob = {"preds": preds_np, "labels": labels_np}

                # Real-pool routing. Method names follow the OOD yaml
                # conventions. We accept common variants.
                if is_real:
                    m_norm = str(m).lower()
                    if "teams_ood_real" in m_norm:
                        real_pools_for_vc["teams_ood_real"] = pool_blob
                    elif "external_youtube_avspeech" in m_norm:
                        real_pools_for_vc["external_youtube_avspeech_real"] = pool_blob
                    elif "zoom_vcd_real" in m_norm or "vcd_real" in m_norm:
                        real_pools_for_vc["zoom_vcd_real"] = pool_blob
                    elif "proper_clean_real" in m_norm or m_norm.endswith("proper_clean"):
                        real_pools_for_vc["proper_clean_real"] = pool_blob
                    elif "proper_teams_real" in m_norm or m_norm.endswith("proper_teams"):
                        real_pools_for_vc["proper_teams_real"] = pool_blob
                    elif "df40_real" in m_norm:
                        real_pools_for_vc["df40_real"] = pool_blob
                else:
                    m_norm = str(m).lower()
                    if "teams_ood_fake" in m_norm or "teams_fake" in m_norm or "visomaster_teams" in m_norm:
                        teams_fake_pools_for_vc[m] = pool_blob
                    elif "deeplive" in m_norm:
                        teams_fake_pools_for_vc[m] = pool_blob
                    else:
                        other_fake_pools_for_vc[m] = pool_blob

            stat_key = self._vc_stability_jitter_stat
            stab_candidates = [
                ood_jitter_summary_cache.get(k, {}).get(stat_key, 0.0)
                for k in _VALUE_COMPOSITE_STABILITY_JITTER_METHODS
            ]
            stab_max = max(stab_candidates) if stab_candidates else 0.0

            vc = _compute_value_composite(
                real_pools=real_pools_for_vc,
                teams_fake_pools=teams_fake_pools_for_vc,
                other_fake_pools=other_fake_pools_for_vc,
                stability_jitter_max=float(stab_max),
                target_mean_fpr=self._vc_target_mean_fpr,
                max_pool_fpr=self._vc_max_pool_fpr,
            )
            # Self-identify which metric definition produced this row so
            # downstream retro-scoring / W&B comparisons are unambiguous.
            wandb_log_dict["value_composite_stability_stat"] = stat_key
            wandb_log_dict["value_composite_target_mean_fpr"] = self._vc_target_mean_fpr
            wandb_log_dict["value_composite_max_pool_fpr"] = self._vc_max_pool_fpr
            wandb_log_dict["value_composite"] = vc["value_composite"] \
                if vc["value_composite"] == vc["value_composite"] else float("nan")
            if vc.get("tau") is not None:
                wandb_log_dict["value_composite_tau"] = vc["tau"]
            if vc.get("mean_fpr") is not None:
                wandb_log_dict["value_composite_mean_fpr"] = vc["mean_fpr"]
            if vc.get("max_fpr_at_mean_02") is not None:
                wandb_log_dict["value_composite_max_fpr_at_mean_02"] = vc["max_fpr_at_mean_02"]
            if vc.get("value_composite_blocked_by"):
                wandb_log_dict["value_composite_blocked_by"] = vc["value_composite_blocked_by"]
            if vc.get("teams_fakes_tpr") is not None:
                wandb_log_dict["value_composite_teams_fakes_tpr"] = vc["teams_fakes_tpr"]
            if vc.get("other_fakes_tpr") is not None:
                wandb_log_dict["value_composite_other_fakes_tpr"] = vc["other_fakes_tpr"]
            wandb_log_dict["value_composite_stability"] = vc["stability"]

            # A7 piece 1: headline summary/ duplicate.
            if vc["value_composite"] == vc["value_composite"]:
                wandb_log_dict["summary/value_composite"] = vc["value_composite"]

            self.logger.info(
                "value_composite: val=%s blocked=%s tau=%s mean_fpr=%s max_fpr=%s "
                "teams_tpr=%s other_tpr=%s stability=%.4f "
                "(real_pools=%s teams_pools=%s other_pools=%s)",
                vc["value_composite"],
                vc.get("value_composite_blocked_by"),
                vc.get("tau"),
                vc.get("mean_fpr"),
                vc.get("max_fpr"),
                vc.get("teams_fakes_tpr"),
                vc.get("other_fakes_tpr"),
                vc["stability"],
                sorted(real_pools_for_vc.keys()),
                sorted(teams_fake_pools_for_vc.keys()),
                sorted(other_fake_pools_for_vc.keys()),
            )
            self._last_value_composite = vc
        except Exception as vc_err:
            self.logger.warning(f"value_composite computation failed: {vc_err}")

        # --- At-indist-threshold metrics for OOD ---
        if indist_threshold is not None and indist_threshold > 0:
            ood_at_indist = metrics_at_threshold(
                np.array(all_preds), np.array(all_labels), indist_threshold
            )
            if ood_at_indist:
                for k, v in ood_at_indist.items():
                    wandb_log_dict[f'ood/at_indist/{k}'] = v
                wandb_log_dict['ood/at_indist/threshold'] = indist_threshold
                self.logger.info(
                    f"OOD @ indist threshold {indist_threshold:.4f}: "
                    f"acc={ood_at_indist['acc']:.4f}  f1={ood_at_indist['f1']:.4f}  "
                    f"fpr={ood_at_indist['fpr']:.4f}  fnr={ood_at_indist['fnr']:.4f}"
                )

        if self.wandb_run: self.wandb_run.log(wandb_log_dict)

        # Capture OOD AUC before cleanup so callers can use it for
        # composite checkpointing (R12+).
        ood_auc = overall_metrics.get('auc')

        del all_preds, all_labels, method_labels, method_preds, method_jitter_per_video, overall_metrics
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info("===> OOD Monitoring Done!")
        return ood_auc

    def _generate_and_upload_reports(self, log_prefix, frame_data, video_data, all_preds, all_labels, method_preds,
                                     method_labels, generate_detailed_reports=False, run_name="",
                                     output_gcs_folder: str = None, output_filename_prefix: str = None):
        """
        Generates and uploads detailed CSV and TXT reports to GCS.
        
        Args:
            output_gcs_folder: Optional GCS folder to write reports to (e.g., 'gs://bucket/path/folder').
                              If specified, appends to existing folder instead of creating new timestamped one.
            output_filename_prefix: Optional prefix for report filenames (e.g., 'target_source_').
        """
        self.logger.info("Generating detailed validation reports...")
        local_temp_dir = tempfile.mkdtemp()
        
        # Determine filename prefix
        prefix = output_filename_prefix or ""
        
        # Track what we successfully generated
        files_generated = []
        expected_files = 4

        def _resolve_group_family_from_row(row):
            if len(row) >= 8:
                return row[6], row[7]
            return infer_group_and_family(label=row[1], method=row[0], source=None)

        normalized_video_rows = []

        try:
            # 1. --- Create Frame-level CSV ---
            try:
                frame_filename = f'{prefix}frames_report.csv'
                frame_csv_path = os.path.join(local_temp_dir, frame_filename)
                frame_rows = []
                for row in frame_data:
                    if len(row) >= 7:
                        frame_rows.append([row[0], row[1], row[2], row[3], row[4], row[5], row[6]])
                    else:
                        group_key, family_key = _resolve_group_family_from_row(row)
                        frame_rows.append([row[0], row[1], row[2], row[3], row[4], group_key, family_key])
                with open(frame_csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['method', 'label', 'video_id', 'frame_path', 'frame_prob', 'group_key', 'family_key'])
                    writer.writerows(frame_rows)
                self.logger.info(f"Frame report generated with {len(frame_rows)} entries.")
                files_generated.append((frame_filename, frame_csv_path))
            except Exception as e:
                self.logger.error(f"Failed to generate frame report: {e}")

            # 2. --- Create Video-level CSV ---
            try:
                video_filename = f'{prefix}videos_report.csv'
                video_csv_path = os.path.join(local_temp_dir, video_filename)
                for row in video_data:
                    if len(row) >= 8:
                        normalized_video_rows.append([row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]])
                    else:
                        group_key, family_key = _resolve_group_family_from_row(row)
                        normalized_video_rows.append([
                            row[0], row[1], row[2], row[3], row[4], row[5], group_key, family_key
                        ])
                with open(video_csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'method', 'label', 'video_id', 'avg_video_prob', 'prediction', 'is_correct',
                        'group_key', 'family_key'
                    ])
                    writer.writerows(normalized_video_rows)
                self.logger.info(f"Video report generated with {len(normalized_video_rows)} entries.")
                files_generated.append((video_filename, video_csv_path))
            except Exception as e:
                self.logger.error(f"Failed to generate video report: {e}")

            # 3. --- Create Per-Group Metrics CSV ---
            group_metrics_rows = []
            try:
                group_filename = f'{prefix}group_metrics.csv'
                group_csv_path = os.path.join(local_temp_dir, group_filename)
                group_to_rows = defaultdict(list)
                for row in normalized_video_rows:
                    group_to_rows[row[6]].append(row)

                for group_key in sorted(group_to_rows.keys()):
                    rows = group_to_rows[group_key]
                    labels = np.array([int(r[1]) for r in rows], dtype=np.int64)
                    preds = np.array([int(r[4]) for r in rows], dtype=np.int64)
                    probs = np.array([float(r[3]) for r in rows], dtype=np.float32)

                    tn = int(np.sum((labels == 0) & (preds == 0)))
                    fp = int(np.sum((labels == 0) & (preds == 1)))
                    fn = int(np.sum((labels == 1) & (preds == 0)))
                    tp = int(np.sum((labels == 1) & (preds == 1)))

                    n_videos = len(rows)
                    accuracy = (tp + tn) / n_videos if n_videos > 0 else 0.0
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

                    mean_prob = float(np.mean(probs)) if probs.size > 0 else 0.0
                    p50_prob = float(np.percentile(probs, 50)) if probs.size > 0 else 0.0
                    p90_prob = float(np.percentile(probs, 90)) if probs.size > 0 else 0.0
                    family_key = rows[0][7] if len(rows[0]) >= 8 else "unknown"

                    group_metrics_rows.append([
                        group_key,
                        family_key,
                        n_videos,
                        accuracy,
                        tp,
                        tn,
                        fp,
                        fn,
                        tpr,
                        fpr,
                        fnr,
                        mean_prob,
                        p50_prob,
                        p90_prob,
                    ])

                with open(group_csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'group_key', 'family_key', 'n_videos', 'accuracy', 'tp', 'tn', 'fp', 'fn',
                        'tpr', 'fpr', 'fnr', 'mean_prob', 'p50_prob', 'p90_prob'
                    ])
                    writer.writerows(group_metrics_rows)

                self.logger.info(f"Group metrics report generated with {len(group_metrics_rows)} groups.")
                files_generated.append((group_filename, group_csv_path))
            except Exception as e:
                self.logger.error(f"Failed to generate group metrics report: {e}")

            # 4. --- Create Summary TXT file ---
            try:
                summary_filename = f'{prefix}summary_report.txt'
                summary_txt_path = os.path.join(local_temp_dir, summary_filename)
                with open(summary_txt_path, 'w') as f:
                    f.write(f"Validation Summary Report for: {log_prefix}\n")
                    f.write(f"Run Name: {run_name}\n")
                    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 40 + "\n")
                    f.write("Overall Performance\n")
                    f.write("-" * 40 + "\n")

                    # Calculate overall confusion matrix
                    # Note: fake=1 (positive), real=0 (negative)
                    
                    # When detailed reports are enabled, use video-level aggregated data for summary
                    if generate_detailed_reports and normalized_video_rows:
                        # Extract video-level predictions and labels from the detailed report data
                        video_labels = [row[1] for row in normalized_video_rows]  # Column 1 is label
                        video_preds = [row[4] for row in normalized_video_rows]   # Column 4 is prediction (binary)
                        
                        tn, fp, fn, tp = confusion_matrix(video_labels, video_preds, labels=[0, 1]).ravel()
                        total = len(video_labels)
                        acc = (tp + tn) / total if total > 0 else 0
                        f.write(f"Total Videos: {total}\n")
                    else:
                        # Fallback to frame-level data (original behavior)
                        overall_preds_binary = (np.array(all_preds) >= 0.5).astype(int)
                        tn, fp, fn, tp = confusion_matrix(all_labels, overall_preds_binary, labels=[0, 1]).ravel()
                        total = tn + fp + fn + tp
                        acc = (tp + tn) / total if total > 0 else 0
                        f.write(f"Total Videos: {total}\n")
                    f.write(f"Accuracy: {acc:.4f}\n")
                    f.write(f"True Positives (Correctly identified Fake): {tp}\n")
                    f.write(f"True Negatives (Correctly identified Real): {tn}\n")
                    f.write(f"False Positives (Real misclassified as Fake): {fp}\n")
                    f.write(f"False Negatives (Fake misclassified as Real): {fn}\n\n")

                    f.write("=" * 40 + "\n")
                    f.write("Per-Method Performance\n")
                    f.write("-" * 40 + "\n")

                    # Calculate per-method performance using deduplicated video data
                    method_video_data = defaultdict(list)
                    for row in normalized_video_rows:  # video_data is deduplicated
                        method = row[0]  # Column 0 is method
                        method_video_data[method].append(row)
                    
                    for method in sorted(method_video_data.keys()):
                        method_rows = method_video_data[method]
                        if not method_rows:
                            continue
                            
                        # Extract data from deduplicated video rows
                        labels = [row[1] for row in method_rows]      # Column 1 is label
                        predictions = [row[4] for row in method_rows] # Column 4 is prediction (binary)
                        
                        labels = np.array(labels)
                        predictions = np.array(predictions)

                        # Determine method type
                        is_real_method = (labels[0] == 0)
                        method_type = "REAL" if is_real_method else "FAKE"

                        f.write(f"----- Method: {method} ({method_type}) -----\n")
                        f.write(f"Total Videos: {len(labels)}\n")

                        # Calculate accuracy using deduplicated data
                        if len(np.unique(labels)) == 1:
                            correct_predictions = np.sum(labels == predictions)
                            accuracy = correct_predictions / len(labels)
                            f.write(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{len(labels)} correct)\n\n")
                        else:  # Mixed labels (rare case)
                            m_tn, m_fp, m_fn, m_tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
                            m_total = m_tn + m_fp + m_fn + m_tp
                            m_acc = (m_tp + m_tn) / m_total if m_total > 0 else 0
                            f.write(f"Accuracy: {m_acc:.4f}\n")
                            f.write(f"  TN: {m_tn}, FP: {m_fp}, FN: {m_fn}, TP: {m_tp}\n\n")

                    f.write("=" * 40 + "\n")
                    f.write("Per-Group Performance\n")
                    f.write("-" * 40 + "\n")
                    if group_metrics_rows:
                        best_group = max(group_metrics_rows, key=lambda row: row[3])
                        worst_group = min(group_metrics_rows, key=lambda row: row[3])
                        for row in group_metrics_rows:
                            f.write(
                                f"{row[0]} (family={row[1]}): "
                                f"n={row[2]}, acc={row[3]:.4f}, tpr={row[8]:.4f}, "
                                f"fpr={row[9]:.4f}, fnr={row[10]:.4f}\n"
                            )
                        gap = best_group[3] - worst_group[3]
                        f.write("\n")
                        f.write(
                            f"Best-group accuracy: {best_group[0]} = {best_group[3]:.4f}\n"
                        )
                        f.write(
                            f"Worst-group accuracy: {worst_group[0]} = {worst_group[3]:.4f}\n"
                        )
                        f.write(f"Best-vs-worst group accuracy gap: {gap:.4f}\n")
                    else:
                        f.write("No group-level rows were available.\n")
                files_generated.append((summary_filename, summary_txt_path))
            except Exception as e:
                self.logger.error(f"Failed to generate summary report: {e}")

            # 4. --- Upload files to GCS ---
            # Use custom folder if provided, otherwise create timestamped folder
            if output_gcs_folder:
                # Use provided folder path (e.g., 'gs://bucket/path/folder')
                gcs_base = output_gcs_folder.rstrip('/')
                self.logger.info(f"Appending reports to existing folder: {gcs_base}")
            else:
                # Create new timestamped folder
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                gcs_folder = f"{timestamp}_{log_prefix}"
                gcs_base = f"gs://training-job-outputs/test_results/{gcs_folder}"
                self.logger.info(f"Creating new results folder: {gcs_base}")

            # Only upload files that were successfully generated
            for filename, local_path in files_generated:
                try:
                    gcs_path = f"{gcs_base}/{filename}"
                    self._upload_to_gcs(local_path, gcs_path)
                except Exception as e:
                    self.logger.error(f"Failed to upload {filename} to GCS: {e}")
            
            if files_generated:
                self.logger.info(f"Successfully processed {len(files_generated)}/{expected_files} report files.")
            else:
                self.logger.warning("No report files were successfully generated.")

        finally:
            # 5. --- Clean up local temporary directory ---
            try:
                shutil.rmtree(local_temp_dir)
            except Exception as e:
                self.logger.error(f"Failed to clean up temporary directory: {e}")

    def run_validation_on_demand(
            self,
            validation_loader,
            log_prefix: str = "on_demand_validation",
            generate_detailed_reports: bool = False,
            run_name: str = None,
            output_gcs_folder: str = None,
            output_filename_prefix: str = None,
    ) -> dict:
        """
        Runs a full validation pass on a provided dataloader.

        This method is designed for post-training analysis. It assumes that the
        model held by the Trainer instance has already been loaded with the
        desired checkpoint weights. It reuses the standard `test_epoch` logic
        to ensure the validation process is identical to the one used during
        training.

        Args:
            validation_loader: A fully configured LazyDataLoaderManager instance
                               containing the videos to evaluate.
            log_prefix: A string to prefix all WandB logs, allowing for clear
                        separation of different validation runs.
            generate_detailed_reports (bool): If True, generates and uploads
                                              detailed frame, video, and summary
                                              reports to GCS for deep analysis.
            run_name: Optional name for the report summary, useful for identifying
                      different runs in GCS.
            output_gcs_folder (str): Optional GCS folder to write reports to (e.g., 'gs://bucket/path/folder').
                                     If specified, appends to existing folder instead of creating new timestamped one.
            output_filename_prefix (str): Optional prefix for report filenames (e.g., 'target_source_').

        Returns:
            A dictionary containing the calculated metrics for this validation run.
        """
        self.logger.info(f"--- Starting on-demand validation for '{log_prefix}' ---")
        if generate_detailed_reports:
            self.logger.info("Detailed report generation is ENABLED.")
            if output_gcs_folder:
                self.logger.info(f"Output folder: {output_gcs_folder}")
            if output_filename_prefix:
                self.logger.info(f"Filename prefix: {output_filename_prefix}")

        # Reuse the existing test_epoch logic completely.
        # We pass dummy values for epoch/step and critically set
        # is_primary_metric=False to prevent any checkpointing or
        # early stopping state changes from being triggered.
        # The `generate_detailed_reports` flag will activate the new logic
        # inside the `test_epoch` method.
        try:
            metrics = self.test_epoch(
                epoch=-1,
                step_cnt=-1,
                validation_loader=validation_loader,
                log_prefix=log_prefix,
                is_primary_metric=False,
                generate_detailed_reports=generate_detailed_reports,
                run_name=run_name,
                output_gcs_folder=output_gcs_folder,
                output_filename_prefix=output_filename_prefix,
            )
        except Exception as e:
            self.logger.error(f"Error during test_epoch: {e}")
            # Return a minimal metrics dict so the caller can continue
            metrics = {
                'overall': {
                    'loss': -1.0,
                    'acc': -1.0,
                    'auc': -1.0,
                    'eer': -1.0,
                    'ap': -1.0,
                    'error': str(e)
                }
            }
            # Still try to log the error but don't crash completely
            self.logger.error("Returning default metrics to prevent total failure.")

        overall_auc = metrics.get('overall', {}).get('auc', 'N/A')
        # Fix formatting to handle non-float values properly
        if isinstance(overall_auc, (int, float)) and overall_auc >= 0:
            auc_str = f"{overall_auc:.4f}"
        else:
            auc_str = str(overall_auc)
        self.logger.info(
            f"--- On-demand validation for '{log_prefix}' complete. Overall AUC: {auc_str} ---")
        return metrics
