"""Canary probe mixin — in-training deployment-quality monitor.

Runs a fixed 800-frame canary set through the model every N steps and logs
deployment-relevant scalars (per-identity FPR, FPR-calibrated lockbox recall,
score-distribution shape, Wilcoxon vs P8A reference) to W&B.

Why this exists: train AUC, value_composite, and class_separation are all
empirically decoupled from deployment-grade quality (memories
project_train_auc_not_valid_promotion_signal.md, project_class_sep_not_predictive.md).
The canary is the deployment-shaped signal:
  - FPR-calibrated recall on a fixed canary mirrors what we'd compute on the
    promotion contract scorecard, just on a small fixed cohort.
  - Per-chronic-identity FPR catches the F1 ∩ F5 overlap-empty failure mode
    that surfaced in P1 (analysis/p1_pe_eval_2026-05-07/joint_tau_sweep_2026-05-07/).
  - Wilcoxon-vs-P8A-reference catches "drift from a known-good behavior on
    held-out identities" — the Roy_D regression class.

Discipline: this mixin must NEVER crash training. Every operation is wrapped
in try/except; failures degrade to logging a warning + skipping the probe.
The canary tensor is loaded lazily on first call and cached on the instance.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class CanaryProbeMixin:
    """Mixin providing in-training deployment-quality monitoring.

    Assumes the base class has:
      - self.config: dict with optional 'canary_probe' block
      - self.model: PyTorch model (eval'd via setEval)
      - self.logger: logger
      - self.wandb_run: optional W&B run

    Config schema (`canary_probe:` block in yaml):
        enabled: bool                # default false; opt-in
        parquet_path: str            # path to canary parquet file (in image)
        frequency_steps: int         # log every N global steps; default 1000
        device: str                  # 'cuda' or 'cpu'; defaults to model device
        max_frames: int              # safety cap; defaults to 1000
        batch_size: int              # forward batch size; default 64

    Hook this from `_run_validation` (or wherever you want; idempotent — it
    no-ops if `step_cnt % frequency_steps != 0`).
    """

    # ---------------------------- init ---------------------------------------

    def init_canary_probe(self) -> None:
        """Initialize canary probe state. Call from Trainer.__init__."""
        cfg = (self.config.get('canary_probe') or {})
        # Tolerate wandb.Config sub-objects (memory project_wandb_flattens_nested_dicts.md)
        if not isinstance(cfg, dict):
            try:
                cfg = dict(cfg)
            except Exception:
                cfg = {}

        self.canary_enabled: bool = bool(cfg.get('enabled', False))
        self.canary_parquet_path: Optional[str] = cfg.get('parquet_path')
        self.canary_frequency_steps: int = int(cfg.get('frequency_steps', 1000) or 1000)
        self.canary_max_frames: int = int(cfg.get('max_frames', 1000) or 1000)
        self.canary_batch_size: int = int(cfg.get('batch_size', 64) or 64)

        # Lazy state — populated on first successful probe.
        self._canary_tensor: Optional[torch.Tensor] = None  # [N, 3, H, W]
        self._canary_meta: Optional[Dict[str, np.ndarray]] = None  # parquet columns
        self._canary_load_attempted: bool = False
        self._canary_load_succeeded: bool = False

        if self.canary_enabled:
            self.logger.info(
                "✅ Canary probe ENABLED: parquet=%s, frequency_steps=%d, max_frames=%d",
                self.canary_parquet_path, self.canary_frequency_steps, self.canary_max_frames,
            )
        else:
            self.logger.info("Canary probe disabled (config.canary_probe.enabled=false)")

    # ---------------------------- public API ---------------------------------

    def _run_canary_probe(self, step_cnt: int) -> None:
        """Run the canary forward pass + log scalars at the configured cadence.

        Bulletproof: any exception is caught and logged; training continues.
        """
        try:
            if not self.canary_enabled:
                return
            if self.config.get('local_rank', 0) != 0:
                return
            if step_cnt <= 0:
                return
            # Fire on first probe step (step == frequency) and every multiple after.
            if step_cnt % self.canary_frequency_steps != 0:
                return

            t_start = time.time()

            # Lazy load on first invocation. If load fails, set
            # canary_enabled=False to stop retrying.
            if not self._canary_load_attempted:
                self._canary_load_attempted = True
                ok = self._lazy_load_canary()
                if not ok:
                    self.logger.warning(
                        "Canary probe lazy-load FAILED — disabling for the rest of this run."
                    )
                    self.canary_enabled = False
                    return
                self._canary_load_succeeded = True

            if not self._canary_load_succeeded:
                return

            metrics = self._compute_canary_metrics()
            self._log_canary_metrics(metrics, step_cnt)

            elapsed = time.time() - t_start
            self.logger.info(
                "Canary probe @ step %d: %d scalars logged in %.2fs",
                step_cnt, len(metrics), elapsed,
            )
        except Exception as e:
            # Never raise. Training is sacred.
            try:
                self.logger.warning(
                    "Canary probe FAILED at step %d (non-fatal): %s",
                    step_cnt, e, exc_info=True,
                )
            except Exception:
                pass

    # ---------------------------- internals ----------------------------------

    def _lazy_load_canary(self) -> bool:
        """Download + decode canary frames into a single tensor.

        Returns True on success; False on any failure (logged).
        """
        if not self.canary_parquet_path:
            self.logger.warning("Canary probe: no parquet_path configured.")
            return False

        try:
            import pandas as pd
        except Exception as e:
            self.logger.warning("Canary probe: pandas import failed: %s", e)
            return False

        try:
            df = pd.read_parquet(self.canary_parquet_path)
        except Exception as e:
            self.logger.warning(
                "Canary probe: failed to read parquet at %s: %s",
                self.canary_parquet_path, e,
            )
            return False

        # Validate schema
        required_cols = {'frame_path', 'label', 'cohort', 'base_identity', 'p8a_reference_score'}
        missing = required_cols - set(df.columns)
        if missing:
            self.logger.warning(
                "Canary probe: parquet missing columns %s; have %s",
                missing, list(df.columns),
            )
            return False

        # Cap frames for safety
        if len(df) > self.canary_max_frames:
            self.logger.warning(
                "Canary probe: parquet has %d frames, capping at max_frames=%d",
                len(df), self.canary_max_frames,
            )
            df = df.head(self.canary_max_frames).reset_index(drop=True)

        # Download + decode all frames into one tensor
        tensor = self._download_and_stack(df)
        if tensor is None:
            return False

        # Drop frames that failed to decode (rows where tensor is all-zero)
        # and update metadata accordingly.
        non_zero_mask = tensor.abs().sum(dim=(1, 2, 3)) > 1e-6
        n_valid = int(non_zero_mask.sum().item())
        n_total = len(df)
        if n_valid < n_total:
            self.logger.warning(
                "Canary probe: %d of %d frames failed to decode; using %d valid frames.",
                n_total - n_valid, n_total, n_valid,
            )
            tensor = tensor[non_zero_mask]
            df = df[non_zero_mask.cpu().numpy()].reset_index(drop=True)

        if n_valid < 50:
            self.logger.warning(
                "Canary probe: only %d valid frames after decode — too few; aborting.",
                n_valid,
            )
            return False

        # Cache
        self._canary_tensor = tensor  # CPU, float32
        self._canary_meta = {
            'frame_path': df['frame_path'].to_numpy(),
            'label': df['label'].to_numpy(),
            'cohort': df['cohort'].to_numpy(),
            'base_identity': df['base_identity'].to_numpy(),
            'p8a_reference_score': df['p8a_reference_score'].to_numpy(),
        }

        self.logger.info(
            "Canary probe: loaded %d valid frames into [%s] tensor.",
            tensor.shape[0], 'x'.join(str(s) for s in tensor.shape),
        )
        return True

    def _download_and_stack(self, df) -> Optional[torch.Tensor]:
        """Download + preprocess + stack canary frames.

        Mirrors the inference preprocessing in batch_inference_gcs.py:
          - cv2.imdecode → cv2.resize INTER_LINEAR → cv2.cvtColor BGR→RGB
          - torchvision ToTensor + Normalize(CLIP_MEAN, CLIP_STD)
        """
        try:
            import cv2
            import numpy as np
            from torchvision import transforms as T
            from google.cloud import storage
        except Exception as e:
            self.logger.warning("Canary probe: import failed: %s", e)
            return None

        CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
        CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
        RESOLUTION = 224

        normalize = T.Compose([T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])

        client = None
        bucket_cache: Dict[str, Any] = {}

        try:
            client = storage.Client()
        except Exception as e:
            self.logger.warning("Canary probe: GCS client init failed: %s", e)
            return None

        n = len(df)
        out = torch.zeros(n, 3, RESOLUTION, RESOLUTION, dtype=torch.float32)
        fail_count = 0

        for i in range(n):
            try:
                fp = str(df['frame_path'].iloc[i])
                if not fp.startswith('gs://'):
                    fail_count += 1
                    continue
                bucket_name, blob_path = fp[5:].split('/', 1)
                if bucket_name not in bucket_cache:
                    bucket_cache[bucket_name] = client.bucket(bucket_name)
                blob = bucket_cache[bucket_name].blob(blob_path)
                img_bytes = blob.download_as_bytes()
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    fail_count += 1
                    continue
                img_bgr = cv2.resize(img_bgr, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LINEAR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                out[i] = normalize(img_rgb)
            except Exception as e:
                fail_count += 1
                if fail_count <= 3:
                    self.logger.warning("Canary probe: frame %d (%s) load failed: %s", i, fp, e)

        if fail_count > 0:
            self.logger.info("Canary probe: %d/%d frames failed to load", fail_count, n)
        return out

    @torch.no_grad()
    def _compute_canary_metrics(self) -> Dict[str, float]:
        """Forward-pass the canary tensor and compute deployment-grade scalars."""
        was_training = self.model.training
        try:
            self.model.eval()
        except Exception:
            pass

        device = next(self.model.parameters()).device

        scores: List[float] = []
        bs = self.canary_batch_size
        n = self._canary_tensor.shape[0]

        try:
            for start in range(0, n, bs):
                end = min(start + bs, n)
                batch = self._canary_tensor[start:end].to(device, non_blocking=True)
                # Construct a minimal data_dict matching the model's forward signature.
                # The model's forward takes data_dict['image']; we pass a per-frame batch
                # and read the softmax fake-class probability.
                data_dict = {'image': batch}
                # inference=True bypasses (a) the ArcFace 2-tuple unpack at
                # detectors/effort_detector.py:1813 (use_arcface_head + label=None)
                # and (b) the use_quality_head + use_multi_axis_grl forward
                # branches at lines 1836+1843 — those are training-side heads
                # that aren't needed for canary scoring. Without inference=True,
                # any run with use_multi_axis_grl=true silently throws here and
                # the canary disables itself for the rest of training (the
                # 2026-05-20 T5C_TRIPLE batch lost 4h+1h of canary visibility
                # on Slots 1+3 to this bug). See open loop
                # `canary-silence-when-multi-axis-grl-active` in
                # docs/packet_retrospectives/threads/in_training_canary_signal.md.
                pred = self.model(data_dict, inference=True)
                # Pred can be a dict {'cls': logits, ...} or a tensor; cover both.
                # Prefer 'prob' (the model emits softmax fake-class directly
                # when inference=True). Falls back to cls/raw_logits otherwise.
                if isinstance(pred, dict):
                    if 'prob' in pred and pred['prob'] is not None:
                        probs = pred['prob']
                        if probs.dim() == 2:
                            probs = probs[:, 1] if probs.shape[1] >= 2 else probs[:, 0]
                        scores.extend(probs.float().cpu().tolist())
                        continue
                    logits = pred.get('cls')
                    if logits is None:
                        # try common alternative keys
                        for k in ('raw_logits', 'logits', 'classifier_logits', 'pred_logits'):
                            if k in pred:
                                logits = pred[k]
                                break
                else:
                    logits = pred
                if logits is None:
                    raise RuntimeError(
                        "Canary probe: could not extract logits from model output."
                    )
                # Reduce video tensor dimension if present (model emits [B*T, 2] or [B, 2])
                if logits.dim() == 3:
                    # [B, T, 2] — average over T
                    logits = logits.mean(dim=1)
                probs = torch.softmax(logits, dim=-1)[:, 1]  # P(fake)
                scores.extend(probs.float().cpu().tolist())

            scores_arr = np.asarray(scores, dtype=np.float64)
        finally:
            try:
                if was_training:
                    self.model.train()
            except Exception:
                pass

        return self._aggregate_metrics(scores_arr)

    def _aggregate_metrics(self, scores: np.ndarray) -> Dict[str, float]:
        """Reduce per-frame scores to logged scalars. All sections defensive."""
        meta = self._canary_meta
        labels = meta['label'].astype(np.int64)
        cohorts = meta['cohort']
        identities = meta['base_identity']
        p8a_ref = meta['p8a_reference_score'].astype(np.float64)

        is_real = labels == 0
        is_fake = labels == 1
        out: Dict[str, float] = {}

        # 1) Score distribution shape on reals — the pair_rank tail-collapse signal
        try:
            real_scores = scores[is_real]
            if real_scores.size > 0:
                out['canary/score_p50_on_reals'] = float(np.median(real_scores))
                out['canary/score_p95_on_reals'] = float(np.percentile(real_scores, 95))
                out['canary/score_mean_on_reals'] = float(real_scores.mean())
                out['canary/score_std_on_reals'] = float(real_scores.std(ddof=0))
        except Exception as e:
            self.logger.warning("Canary metric (real distribution) failed: %s", e)

        # 2) Score distribution shape on fakes
        try:
            fake_scores = scores[is_fake]
            if fake_scores.size > 0:
                out['canary/score_p50_on_fakes'] = float(np.median(fake_scores))
                out['canary/score_p05_on_fakes'] = float(np.percentile(fake_scores, 5))
                out['canary/score_mean_on_fakes'] = float(fake_scores.mean())
        except Exception as e:
            self.logger.warning("Canary metric (fake distribution) failed: %s", e)

        # 3) Per-chronic-identity mean score (the F5 binding-constraint signal)
        try:
            chronic_means: List[float] = []
            chronic_cohorts = np.asarray([
                str(c).startswith('chronic_') for c in cohorts
            ])
            chronic_idents = np.unique(identities[chronic_cohorts])
            for ident in chronic_idents:
                mask = (identities == ident) & is_real
                if mask.sum() == 0:
                    continue
                m = float(scores[mask].mean())
                chronic_means.append(m)
                # Sanitize identity string for W&B key
                key_safe = ''.join(c if (c.isalnum() or c == '_') else '_' for c in str(ident))
                out[f'canary/chronic_mean/{key_safe}'] = m
            if chronic_means:
                out['canary/max_per_identity_mean_score'] = float(max(chronic_means))
                out['canary/mean_per_identity_mean_score'] = float(np.mean(chronic_means))
        except Exception as e:
            self.logger.warning("Canary metric (chronic per-identity) failed: %s", e)

        # 4) FPR-calibrated lockbox recall (the F1 deployment metric, monotone-invariant)
        try:
            lockbox_mask = np.asarray([str(c) == 'lockbox_fake' for c in cohorts])
            lockbox_scores = scores[lockbox_mask]
            real_scores_all = scores[is_real]
            if lockbox_scores.size > 0 and real_scores_all.size > 0:
                # Sweep tau on the reals' score distribution; pick tau that gives ~10% FPR.
                tau_grid = np.percentile(real_scores_all, np.linspace(0, 100, 401))
                best_recall_at_10 = 0.0
                best_tau_at_10 = float('nan')
                best_recall_at_5 = 0.0
                for tau in tau_grid:
                    fpr = float((real_scores_all > tau).mean())
                    recall = float((lockbox_scores > tau).mean())
                    if fpr <= 0.10 and recall > best_recall_at_10:
                        best_recall_at_10 = recall
                        best_tau_at_10 = float(tau)
                    if fpr <= 0.05 and recall > best_recall_at_5:
                        best_recall_at_5 = recall
                out['canary/lockbox_recall_at_FPR_10pct'] = best_recall_at_10
                out['canary/lockbox_recall_at_FPR_5pct'] = best_recall_at_5
                out['canary/lockbox_tau_at_FPR_10pct'] = best_tau_at_10
        except Exception as e:
            self.logger.warning("Canary metric (lockbox FPR-calibrated) failed: %s", e)

        # 5) Per-method fake recall at tau=0.5 (calibration-invariant baseline)
        try:
            for method_cohort in ('lockbox_fake', 'viso_fake', 'deeplive_fake', 'teams_fake'):
                m_mask = np.asarray([str(c) == method_cohort for c in cohorts])
                if m_mask.sum() == 0:
                    continue
                recall = float((scores[m_mask] > 0.5).mean())
                out[f'canary/recall_at_tau05/{method_cohort}'] = recall
        except Exception as e:
            self.logger.warning("Canary metric (per-method recall@0.5) failed: %s", e)

        # 6) Wilcoxon vs P8A reference (drift signal from a known-good baseline)
        try:
            from scipy.stats import wilcoxon
            # Limit to reals where p8a_reference_score is finite — this catches
            # the "drift on identities P8A handled well" signal.
            ref_mask = is_real & np.isfinite(p8a_ref)
            if int(ref_mask.sum()) >= 10:
                cur = scores[ref_mask]
                ref = p8a_ref[ref_mask]
                # Use signed-rank — large stat magnitude ≡ large drift
                stat, pval = wilcoxon(cur, ref, zero_method='wilcox', alternative='two-sided')
                out['canary/wilcoxon_stat_vs_p8a_reals'] = float(stat)
                out['canary/wilcoxon_pval_vs_p8a_reals'] = float(pval)
                # Also report absolute mean drift (signed)
                out['canary/mean_score_drift_vs_p8a_reals'] = float((cur - ref).mean())
                out['canary/abs_mean_score_drift_vs_p8a_reals'] = float(np.abs(cur - ref).mean())
        except Exception as e:
            self.logger.warning("Canary metric (Wilcoxon vs P8A) failed: %s", e)

        # 7) Sanity counters
        try:
            out['canary/n_frames_evaluated'] = float(scores.size)
            out['canary/n_reals'] = float(int(is_real.sum()))
            out['canary/n_fakes'] = float(int(is_fake.sum()))
        except Exception:
            pass

        return out

    def _log_canary_metrics(self, metrics: Dict[str, float], step_cnt: int) -> None:
        """Push the dict to W&B at the current step."""
        if not metrics:
            return
        try:
            if self.wandb_run is not None:
                # Filter NaN/Inf so W&B doesn't choke
                clean = {
                    k: v for k, v in metrics.items()
                    if isinstance(v, (int, float)) and np.isfinite(v)
                }
                clean['canary/probe_step'] = int(step_cnt)
                self.wandb_run.log(clean, step=step_cnt)
        except Exception as e:
            self.logger.warning("Canary probe: W&B log failed: %s", e)
