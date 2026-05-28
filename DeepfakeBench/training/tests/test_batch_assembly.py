"""TDD tests for `batch_assembly.assemble_probs_list`.

This is the `/check_frame_batch` response-`probs` contract, extracted from
app3.py so the t5c index-alignment guarantee can be unit-tested WITHOUT importing
the model / torch / GCS stack.

The load-bearing case is B1: in the t5c ("align to input") profile every input
frame MUST get exactly one slot in `probs` so the client's positional
per-participant split can never shift across frame/participant boundaries. A
frame that raises AFTER decode (recrop/resize/transform) lands as
`{"kind": "failed"}` with no `"prob"` key — before this fix that slot was
dropped, shifting every later prob one position left (the same misattribution
class the 2026-05-28 cross-repo fix addressed for decode-failures, but on an
unpatched sibling path).
"""
from __future__ import annotations

import sys
from pathlib import Path

# batch_assembly.py lives in the training/ dir (parent of tests/).
_TRAIN = Path(__file__).parent.parent
if str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))

from batch_assembly import GATE_SENTINEL_PROB, assemble_probs_list  # noqa: E402


class TestAssembleProbsListT5C:
    """align_to_input=True: one slot per input frame, in order."""

    def test_scored_and_gated_frames_keep_order(self):
        status = [
            {"kind": "tensor", "prob": 0.9},
            {"kind": "gated", "prob": GATE_SENTINEL_PROB, "reason": "min_dim"},
            {"kind": "tensor", "prob": 0.1},
        ]
        assert assemble_probs_list(status, align_to_input=True) == [0.9, GATE_SENTINEL_PROB, 0.1]

    def test_processing_failed_entry_without_prob_becomes_sentinel(self):
        # B1 REGRESSION: a post-decode processing exception appends
        # {"kind": "failed"} with NO "prob" key. In t5c that frame must still
        # occupy a (sentinel) slot so len(probs) == number of input frames.
        status = [
            {"kind": "tensor", "prob": 0.8},
            {"kind": "failed"},                 # processing exception, no "prob"
            {"kind": "tensor", "prob": 0.2},
        ]
        probs = assemble_probs_list(status, align_to_input=True)
        assert probs == [0.8, GATE_SENTINEL_PROB, 0.2]
        assert len(probs) == len(status)        # alignment with input preserved

    def test_len_always_matches_input_even_if_nothing_scored(self):
        status = [{"kind": "failed"}, {"kind": "gated", "prob": GATE_SENTINEL_PROB}]
        probs = assemble_probs_list(status, align_to_input=True)
        assert probs == [GATE_SENTINEL_PROB, GATE_SENTINEL_PROB]
        assert len(probs) == len(status)

    def test_custom_sentinel_value_is_used_for_missing_prob(self):
        status = [{"kind": "failed"}, {"kind": "tensor", "prob": 0.5}]
        assert assemble_probs_list(status, align_to_input=True, sentinel=-9.0) == [-9.0, 0.5]

    def test_empty_status_returns_empty(self):
        assert assemble_probs_list([], align_to_input=True) == []


class TestAssembleProbsListLegacy:
    """align_to_input=False: pre-existing contract, preserved byte-for-byte."""

    def test_drops_entries_without_a_prob_key(self):
        # Legacy: decode/processing failures are dropped; gated frames vote with
        # their default (0.25) and stay.
        status = [
            {"kind": "tensor", "prob": 0.7},
            {"kind": "failed"},                 # dropped (no "prob")
            {"kind": "gated", "prob": 0.25},    # kept
        ]
        assert assemble_probs_list(status, align_to_input=False) == [0.7, 0.25]

    def test_never_injects_sentinels(self):
        status = [{"kind": "failed"}, {"kind": "failed"}]
        assert assemble_probs_list(status, align_to_input=False) == []
