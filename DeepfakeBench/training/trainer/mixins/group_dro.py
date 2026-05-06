"""
Group-DRO (Distributionally Robust Optimization) mixin for Trainer.

Implements Group-DRO loss that upweights groups with higher loss, encouraging
the model to perform well across all generation methods / data subgroups.

Two grouping schemes are supported:

1. **method_mapping** (legacy): groups = generation methods. Pre-2026-05-07
   default. Batch must include `method_id` (int tensor).
2. **group_id_mapping** (PE_PAIR_RANK_DRO, 2026-05-07): groups = asymmetric
   string keys constructed via
   `analysis/group_id_design_audit_2026-05-06/outputs/group_id_python_snippet.py`
   with R-D real-side keys (`source × transport × quality_band × chronic_flag`)
   and F-B fake-side keys (`method_family × enhancer_family × transport ×
   quality_band`). Batch must include `group_id` (List[str] or int tensor) and
   the config must carry `data_params.group_id_mapping: Dict[str, int]`.

When both are configured, group_id_mapping takes precedence.
"""
from typing import Any, Dict, List, Optional, Union

import torch


class GroupDROMixin:
    """
    Mixin that provides Group-DRO functionality for the Trainer.

    Group-DRO upweights samples from groups (methods or arbitrary subgroups)
    that have higher loss, which helps the model generalize across distinct
    generation methods or substrate cohorts.

    Assumes the base class has:
    - self.config: Training configuration dict
    - self.model: The PyTorch model
    - self.logger: Logger instance
    """

    def init_group_dro(self) -> None:
        """Initialize Group-DRO state. Call from __init__ if use_group_dro is True."""
        self.logger.info("Initializing Group-DRO loss strategy.")

        dro_params = self.config.get('group_dro_params', {})
        self.group_dro_beta = dro_params.get('beta', 3.0)
        self.group_dro_clip_min = dro_params.get('clip_min', 1.0)
        self.group_dro_clip_max = dro_params.get('clip_max', 4.0)
        self.ema_alpha = dro_params.get('ema_alpha', 0.1)
        # Steps before EMA-based reweighting kicks in. During warmup all
        # samples get uniform weight (per group_id_design_audit recommendation
        # — default 100, mirrors ema_alpha=0.1 saturation timescale).
        self.group_dro_warmup_steps = int(dro_params.get('warmup_steps', 100))
        self._group_dro_step_counter = 0

        data_params = self.config.get('data_params', {})
        group_id_mapping = data_params.get('group_id_mapping')
        method_mapping = data_params.get('method_mapping')

        if group_id_mapping:
            if not isinstance(group_id_mapping, dict):
                raise ValueError(
                    "data_params.group_id_mapping must be a dict[str, int]; "
                    f"got {type(group_id_mapping).__name__}."
                )
            self.group_id_mapping: Optional[Dict[str, int]] = dict(group_id_mapping)
            self.method_mapping: Optional[Dict[str, int]] = None
            self.num_groups = len(self.group_id_mapping)
            self.logger.info(
                f"Group-DRO using group_id_mapping ({self.num_groups} groups; R-D + F-B keying)"
            )
        elif method_mapping:
            self.method_mapping = dict(method_mapping)
            self.group_id_mapping = None
            self.num_groups = len(self.method_mapping)
            self.logger.info(
                f"Group-DRO using method_mapping ({self.num_groups} methods)"
            )
        else:
            raise ValueError(
                "Group-DRO requires either 'data_params.group_id_mapping' (preferred) "
                "or 'data_params.method_mapping' in the config."
            )

        # Initialize EMA of group losses
        device = next(self.model.parameters()).device
        self.group_losses_ema = torch.zeros(self.num_groups, device=device)

        self.logger.info(
            f"Group-DRO initialized with {self.num_groups} groups, "
            f"beta={self.group_dro_beta}, "
            f"clip=[{self.group_dro_clip_min}, {self.group_dro_clip_max}], "
            f"ema_alpha={self.ema_alpha}, warmup_steps={self.group_dro_warmup_steps}"
        )

    def _resolve_group_ids(
        self,
        data_dict: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """Pull per-video group ids out of the batch and convert to int tensor.

        Uses self.group_id_mapping when configured, otherwise falls back to
        the legacy method_id integer field.
        """
        if self.group_id_mapping is not None:
            raw = data_dict.get('group_id')
            if raw is None:
                raise KeyError(
                    "Group-DRO is configured with group_id_mapping but the "
                    "batch is missing 'group_id'. Update the loader/collate "
                    "to emit per-video group_id strings."
                )
            if isinstance(raw, torch.Tensor):
                return raw.to(device=device, dtype=torch.long)
            # Expect list[str] (or list with possible None entries from
            # loaders that don't populate group_id). Map unknowns to 0 with a
            # one-shot warning so the run does not silently skew weighting.
            ids: List[int] = []
            unknown_count = 0
            for g in raw:
                if g is None:
                    unknown_count += 1
                    ids.append(0)
                    continue
                idx = self.group_id_mapping.get(g)
                if idx is None:
                    unknown_count += 1
                    ids.append(0)
                else:
                    ids.append(idx)
            if unknown_count and not getattr(self, '_group_dro_unknown_warned', False):
                self.logger.warning(
                    f"Group-DRO: {unknown_count}/{len(raw)} samples in batch had "
                    "missing or unmapped group_id; bucketing them into group 0. "
                    "Verify the loader emits a group_id present in "
                    "data_params.group_id_mapping. (Warning emitted once.)"
                )
                self._group_dro_unknown_warned = True
            return torch.as_tensor(ids, device=device, dtype=torch.long)

        # Legacy path: integer method_ids already in the batch.
        method_ids = data_dict['method_id']
        if not isinstance(method_ids, torch.Tensor):
            method_ids = torch.as_tensor(method_ids, dtype=torch.long)
        return method_ids.to(device=device, dtype=torch.long)

    def calculate_group_dro_loss(
        self,
        data_dict: Dict[str, Any],
        per_sample_loss: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Computes the Group-DRO loss for a given batch.

        Args:
            data_dict: Data dictionary containing group_id (preferred) or
                method_id (legacy) per-video.
            per_sample_loss: Per-sample loss tensor (before reduction).

        Returns:
            Dictionary with 'overall' (weighted loss) and 'group_weights' tensors.
        """
        device = per_sample_loss.device

        group_ids = self._resolve_group_ids(data_dict, device)

        # Handle video-style batches where loss is per-frame but group_id is per-video.
        # The model reshapes [B, T, C, H, W] → [B*T, C, H, W] for per-frame predictions,
        # so per_sample_loss has B*T elements while group_ids has only B elements.
        # We expand group_ids to match by repeating each id T times.
        if per_sample_loss.shape[0] > group_ids.shape[0]:
            B = group_ids.shape[0]
            T = per_sample_loss.shape[0] // B
            group_ids = group_ids.repeat_interleave(T)

        # Step 1: Compute per-group average loss for the current batch
        batch_group_losses = torch.zeros(self.num_groups, device=device)
        batch_group_counts = torch.zeros(self.num_groups, device=device)

        # Efficient scatter_add operation for calculating group means
        batch_group_counts.index_add_(0, group_ids, torch.ones_like(per_sample_loss))
        batch_group_losses.index_add_(0, group_ids, per_sample_loss.detach())

        valid_groups_mask = batch_group_counts > 0
        batch_group_losses[valid_groups_mask] /= batch_group_counts[valid_groups_mask]

        # Step 2: Update the EMA of group losses
        current_ema = self.group_losses_ema[valid_groups_mask]
        current_batch_loss = batch_group_losses[valid_groups_mask]
        updated_ema = (1 - self.ema_alpha) * current_ema + self.ema_alpha * current_batch_loss
        self.group_losses_ema[valid_groups_mask] = updated_ema

        # Step 3: During warmup, return uniform weights (no reweighting yet).
        # Otherwise compute exponential weights from the EMA.
        in_warmup = self._group_dro_step_counter < self.group_dro_warmup_steps
        self._group_dro_step_counter += 1

        if in_warmup:
            clipped_weights = torch.ones(self.num_groups, device=device)
        else:
            # Use only the EMA of groups present in the current batch
            valid_ema_losses = self.group_losses_ema[self.group_losses_ema > 0]
            avg_ema_loss = valid_ema_losses.mean() if len(valid_ema_losses) > 0 else 0

            relative_losses = self.group_losses_ema - avg_ema_loss

            # Exponential weighting based on relative loss
            weights = torch.exp(self.group_dro_beta * relative_losses)
            weights = weights * (self.num_groups / torch.sum(weights))  # Normalize
            clipped_weights = torch.clip(weights, self.group_dro_clip_min, self.group_dro_clip_max)

        # Step 4: Calculate the final weighted loss for the batch
        sample_weights = clipped_weights[group_ids].detach()
        weighted_loss = (per_sample_loss * sample_weights).mean()

        return {
            'overall': weighted_loss,
            'group_weights': clipped_weights,
            'group_losses_ema': self.group_losses_ema.clone(),
            'group_dro_in_warmup': torch.tensor(1.0 if in_warmup else 0.0, device=device),
        }

    def get_group_dro_stats(self) -> Dict[str, Any]:
        """Get current Group-DRO statistics for logging."""
        if not hasattr(self, 'group_losses_ema'):
            return {}

        # Get group names from the active mapping (reverse lookup)
        if self.group_id_mapping is not None:
            reverse_mapping = {v: k for k, v in self.group_id_mapping.items()}
        else:
            reverse_mapping = {v: k for k, v in self.method_mapping.items()}

        stats = {}
        for idx in range(self.num_groups):
            group_name = reverse_mapping.get(idx, f"group_{idx}")
            # Prefix kept as `group_dro/ema_loss/...` for dashboard continuity.
            stats[f"group_dro/ema_loss/{group_name}"] = self.group_losses_ema[idx].item()

        return stats
