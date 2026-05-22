"""Asymmetric substrate-pair hinge loss for BACKBONE-T5C (2026-05-22).

Penalty target. Pull `prob_fake(raw_clean)` UP to match `prob_fake(raw_teams)`
when the clean side is lower (only). The reverse direction (pulling teams
down to clean) was refuted on 2026-05-04 + 2026-05-23 — pulling teams down
collapses fake recall.

Motivation (FACTS). CPU-2 on 2026-05-23 measured 275 paired viso fakes
on Slot A v2 step3500:
  - mean(raw_clean) = 0.5029, mean(raw_teams) = 0.5419, Δ = +0.0390
  - Wilcoxon p = 0.0020
  - Cohort partition at τ=0.20: 62 frames in (teams>τ AND raw≤τ) vs
    only 1 in (raw>τ AND teams≤τ) — strong asymmetry.

The 62:1 asymmetry indicates the model already favors teams over clean
in the right tail. The lever is to raise clean scores up to match teams.

Loss form (one-sided hinge):

    L = lambda_pair * mean( ReLU(prob_teams - prob_clean - margin) )

Computed only over matched-pair rows in the batch (same `substrate_pair_id`,
one row with `substrate_transport=0` and one with `=1`). Rows without a
match are skipped — the loss returns 0.

The trainer (or any caller) invokes `compute_from_batch(prob, data_dict)`
where:
  - `prob`: tensor [B, 2] of softmaxed class probabilities or [B] of
    prob_fake (the loss prefers `prob[:, 1]` when 2-D, else flat).
  - `data_dict`: collate output containing `substrate_pair_id` and
    `substrate_transport` (1-D tensors, B-long).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from loss.abstract_loss_func import AbstractLossClass
from utils.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="substrate_pair_asymmetric")
class SubstratePairAsymmetricLoss(AbstractLossClass):
    """One-sided hinge: penalize (prob_teams - prob_clean) when positive.

    Trainable knobs:
      - lambda_pair: scalar multiplier on the mean hinge.
      - margin: epsilon under which the hinge is silent.

    Defaults match the task spec for BACKBONE-T5C (lambda_pair=0.3,
    margin=0.0).
    """

    TRANSPORT_CLEAN = 0
    TRANSPORT_TEAMS = 1

    def __init__(
        self,
        lambda_pair: float = 0.3,
        margin: float = 0.0,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_pair = float(lambda_pair)
        self.margin = float(margin)
        self.enabled = bool(enabled)

    def forward(
        self,
        prob_clean: torch.Tensor,
        prob_teams: torch.Tensor,
    ) -> torch.Tensor:
        """Direct one-sided hinge given already-paired arrays.

        prob_clean and prob_teams must be the same length; element i is the
        clean / teams pair partner.

        IMPORTANT: prob_teams is detached internally so the gradient ONLY
        flows through prob_clean. This implements the task spec: "Pulls raw
        probabilities UP to match teams probabilities; does NOT pull teams
        down to raw." (The teams-down direction was refuted on 2026-05-04
        + 2026-05-23.)
        """
        if not self.enabled or self.lambda_pair <= 0.0:
            device = prob_clean.device if isinstance(prob_clean, torch.Tensor) else None
            return torch.zeros((), device=device) if device is not None else torch.tensor(0.0)
        # Detach teams side so gradient flows only into prob_clean.
        diff = prob_teams.detach() - prob_clean - self.margin
        return self.lambda_pair * F.relu(diff).mean()

    def compute_from_batch(
        self,
        prob: torch.Tensor,
        data_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Peel matched (clean, teams) pairs from a batch and apply the hinge.

        prob: model output. If 2-D, prob[:, 1] is the fake probability. If
        1-D, used as-is.

        data_dict must contain:
          - 'substrate_pair_id': LongTensor [B] (or list[int]).
          - 'substrate_transport': LongTensor [B] (or list[int]) with values
            0=clean, 1=teams, -1=skip.

        Behavior:
          - Filters to rows with substrate_transport in {0, 1}.
          - Groups by substrate_pair_id; for each pair_id with at least
            one clean AND one teams row, pairs the FIRST clean with the
            FIRST teams (deterministic within a batch).
          - Returns the mean hinge across paired diffs, scaled by lambda_pair.
          - Returns 0 (no grad path) when no matched pair exists.
        """
        if not self.enabled or self.lambda_pair <= 0.0:
            device = prob.device
            return torch.zeros((), device=device)

        device = prob.device
        # Reduce prob to per-sample fake probability.
        if prob.dim() == 2:
            prob_fake = prob[:, 1]
        else:
            prob_fake = prob.view(-1)

        pair_ids = data_dict.get("substrate_pair_id")
        transports = data_dict.get("substrate_transport")
        if pair_ids is None or transports is None:
            return torch.zeros((), device=device)

        if not isinstance(pair_ids, torch.Tensor):
            pair_ids = torch.as_tensor(pair_ids, dtype=torch.long, device=device)
        else:
            pair_ids = pair_ids.to(device=device, dtype=torch.long)
        if not isinstance(transports, torch.Tensor):
            transports = torch.as_tensor(transports, dtype=torch.long, device=device)
        else:
            transports = transports.to(device=device, dtype=torch.long)

        # Per-video collate can produce B that doesn't match prob_fake's length
        # when the model splits [B, T, C, H, W] → [B*T, ...]. We expand pair_ids
        # by replicate-T to match the per-frame output the model emits.
        if prob_fake.shape[0] != pair_ids.shape[0]:
            B = pair_ids.shape[0]
            if B > 0 and prob_fake.shape[0] % B == 0:
                T = prob_fake.shape[0] // B
                pair_ids = pair_ids.repeat_interleave(T)
                transports = transports.repeat_interleave(T)
            else:
                # Shape mismatch we can't reconcile — bail safely.
                return torch.zeros((), device=device)

        # Filter to valid transports.
        valid_mask = (transports == self.TRANSPORT_CLEAN) | (transports == self.TRANSPORT_TEAMS)
        if not valid_mask.any():
            return torch.zeros((), device=device)

        valid_prob = prob_fake[valid_mask]
        valid_pair_ids = pair_ids[valid_mask]
        valid_transports = transports[valid_mask]

        # Walk unique pair_ids, peel matched (clean, teams) pairs.
        # Python-side loop is fine — N is small (≤ frames per batch).
        clean_diffs = []
        teams_diffs = []
        unique_ids = torch.unique(valid_pair_ids)
        for pid in unique_ids.tolist():
            if pid < 0:
                continue
            id_mask = (valid_pair_ids == pid)
            id_prob = valid_prob[id_mask]
            id_trans = valid_transports[id_mask]
            clean_idx = (id_trans == self.TRANSPORT_CLEAN).nonzero(as_tuple=True)[0]
            teams_idx = (id_trans == self.TRANSPORT_TEAMS).nonzero(as_tuple=True)[0]
            if clean_idx.numel() == 0 or teams_idx.numel() == 0:
                continue
            # Take the first row from each side. The collate orders frames by
            # frame_idx so this is deterministic within a batch.
            clean_diffs.append(id_prob[clean_idx[0]])
            teams_diffs.append(id_prob[teams_idx[0]])

        if not clean_diffs:
            return torch.zeros((), device=device)

        prob_c = torch.stack(clean_diffs)
        prob_t = torch.stack(teams_diffs)
        return self.forward(prob_c, prob_t)


__all__ = ["SubstratePairAsymmetricLoss"]
