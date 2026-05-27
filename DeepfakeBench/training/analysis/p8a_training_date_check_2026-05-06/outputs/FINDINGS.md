# P8A Training-Date Check — FINDINGS

**Job:** `P8A_TRAINING_DATE_CHECK_2026-05-06`
**Date:** 2026-05-06
**Verdict:** **PRE-FIX**
**Confidence:** **HIGH**

> Note: this file was written by the parent agent because the sub-agent harness policy blocked direct `.md` writes from the dispatched probe. The verdict and evidence log written directly by the sub-agent are at `verdict.json` and `evidence_log.md` in this same directory; this file mirrors the sub-agent's report content.

## Headline

`P8A_REFERENCE_STEP5000` (run `9lmvb5b4`, "smooth-haze-250") trained **2026-04-24 21:41 UTC → 23:48 UTC** (~2h 7min, Vertex job `1205078281779412992` on us-east1). The in-proj-SVD gradient bug fix landed in commit `2feea58` at **2026-04-26 19:13 UTC** — **43h 25m AFTER P8A finished training.**

P8A's `apply_svd_to_in_proj=true` config was therefore active across all ~5,000 training steps but the q/k/v residual SVD parameters received **zero classification gradient** for the entire run. Every reference to "P8A's recipe" in the retros refers to a pre-fix codepath.

## Anchoring evidence

1. **P8A run timestamp** — `analysis/overnight_packet7_packet8_summary_2026-04-25.md:67-69`:
   - run_id: `9lmvb5b4`
   - vertex_job_id: `1205078281779412992`
   - region: us-east1
   - start_iso: 2026-04-24T21:41:00Z
   - end_iso: 2026-04-24T23:48:00Z

2. **Fix commit `2feea58` timestamp** — `git show 2feea58 --format=%aI --no-patch`:
   - 2026-04-26T21:13:21+02:00 = 2026-04-26T19:13:21Z
   - Subject: "Fix silent zero-gradient bug in apply_svd_to_in_proj path"

3. **Three independent in-repo sources explicitly classify P8A as pre-fix:**
   - `docs/packet_retrospectives/packets/P8A.md:29,108` — "Every P8A run had `apply_svd_to_in_proj: true` but the q/k/v residuals received zero classification gradient until the 2026-04-26 fix."
   - `april-26-training-master-plan-v2.LOG.md:214` — "P8A (`9lmvb5b4/step5000`, ~5k broken-bug steps)."
   - Memory `project_in_proj_svd_gradient_bug.md` — affected scope is the entire R12/RLP/P-* lineage including P8A.

4. **Independent corroboration via filename**: 23+ checkpoint maps in `arena/checkpoint_maps/*.yaml` resolve `P8A_REFERENCE_STEP5000` to the same GCS object whose filename embeds `20260424` as the training day. Confirms the date independent of the retros.

## Implication for `PE_PAIR_RANK_DRO` base choice

The external advisor's structural argument holds: **a fresh FT-from-P8A on the post-fix codepath would route real classification gradient through ~36M q/k/v residual params for the first time**. This is the "unmeasured headroom" the advisor flagged in plan §6.5.

**Caveat (load-bearing):** `packets/P8A.md:108` documents that the post-fix C-ablation slot tied within noise vs C.1 on the **P10_SYM-on-P8A recipe** — i.e., "in_proj-SVD does not materially move the canonical P8A configuration." But that null was tested under one specific FT recipe (P10 symmetric router), not under `PE_PAIR_RANK_DRO`'s pair-rank + multi-axis GroupDRO loss. The lever's payoff under a new loss is plausible but unverified.

## Recommendation for `PE_PAIR_RANK_DRO` design

If FT-from-P8A is chosen as the base (pending `CHECKPOINT_COHORT_DIAGNOSIS` verdict from Agent 3):

1. **Include a cheap grad-audit at step 100, 500, 1000** to confirm the in-proj-SVD lever is actually live under the new loss before banking on the headroom claim. Use existing `analysis/probe_battery_2026-04-26/grad_audit.py` if available.
2. **Document the grad audit in the packet retro** so the lever's payoff under pair-rank+DRO is recorded for future ledger entries — closes a small piece of the open loop on "P8A re-train value under post-fix codepath."

## Outputs

- `verdict.json` — machine-readable record (run_id, timestamps, pre_fix bool, justification).
- `evidence_log.md` — append-only log of what was searched and what was found.
- `FINDINGS.md` — this file.

## Cross-references

- Plan: `docs/relaunch_handoffs/NEXT_STEPS_PLAN_2026-05-06.md` §6.5 (advisor's argument), §8.2 P1 (PE_PAIR_RANK_DRO recipe, FT-base TBD).
- P8A retro: `docs/packet_retrospectives/packets/P8A.md`.
- In-proj SVD bug memory: `project_in_proj_svd_gradient_bug.md`.
- P10_SYM-on-P8A C-ablation: `packets/P8A.md:108`.
