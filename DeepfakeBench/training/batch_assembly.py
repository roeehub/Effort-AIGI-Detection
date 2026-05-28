"""Pure helpers for assembling the `/check_frame_batch` response.

Extracted from app3.py so the t5c index-alignment contract can be unit-tested
without importing the model / torch / GCS stack. Intentionally has NO heavy
imports — keep it that way so its tests stay fast and env-independent.
"""
from __future__ import annotations

from typing import Any, Dict, List

# Sentinel probability for frames that were NOT scored (gated / decode-failed /
# processing-failed) in the t5c "align to input" profile. Outside the valid
# model-output range [0, 1] so downstream consumers filter it with `p >= 0.0`.
# Must stay equal to app3.GATE_SENTINEL_PROB (app3 passes its own constant in).
GATE_SENTINEL_PROB = -1.0


def assemble_probs_list(
    per_frame_status: List[Dict[str, Any]],
    align_to_input: bool,
    sentinel: float = GATE_SENTINEL_PROB,
) -> List[float]:
    """Build the response `probs` list from the per-frame status entries.

    `per_frame_status` is 1:1 with the input files, in input order (every
    handler path appends exactly one entry per file). Two profiles, matching
    app3.GATE_PROFILES:

    * ``align_to_input=True`` (t5c): EVERY entry yields exactly one slot, in
      order. Scored frames carry the real model prob; any entry lacking a
      ``"prob"`` key (gated / decode-failed / processing-failed) carries
      ``sentinel``. This guarantees ``len(probs) == len(per_frame_status)`` so a
      positional per-participant split on the client can never shift across
      frame/participant boundaries — and a future non-scoring code path that
      forgets to set a prob still can't silently drop a slot.

    * ``align_to_input=False`` (legacy): only entries that carry a ``"prob"``
      appear (gated frames vote with their default prob; decode/processing
      failures are dropped). Pre-existing contract, preserved byte-for-byte.
    """
    if align_to_input:
        return [s.get("prob", sentinel) for s in per_frame_status]
    return [s["prob"] for s in per_frame_status if "prob" in s]
