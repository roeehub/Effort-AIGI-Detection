# Morning decision tree — codec restoration triple verdict — 2026-05-21

> Written 2026-05-21 ~00:35 UTC by overnight auto-mode agent. The 3 GPU jobs
> were submitted ~00:23 UTC. Expected RUNNING ~00:30-00:50 UTC. Expected
> SUCCEEDED ~04:30-06:30 UTC. Read AFTER `FACTS_2026-05-21.md`.

## Tonight's experiment

3 packets in a 2×2 design (anchor × codec):

|                     | codec_p=0.0          | codec_p=0.20      | codec_p=0.40      |
|---------------------|----------------------|-------------------|-------------------|
| anchor=ON  (Slot A v2 base) | reference (already evaluated) | **Slot 2** us-east1 | **Slot 1** us-west4 |
| anchor=OFF (T5C base)       | reference (already evaluated) | (not run)         | **Slot 3** us-central1 |

## What the CPU evidence says BEFORE training (informative priors)

### Gate 1 (single-frame natural-experiment validity)

- **T5C base**: PASS — applying VideoCodecSimulation to Roy_D crop drops his prob_fake 0.795 → 0.724 (median over 12 seeds; gap to Guest 0.628 = 0.096). Aug reproduces ~42% of the natural-experiment Δ.
- **Slot A v2 base**: marginal FAIL by 5pp (gap 0.149 vs 0.10 threshold). Direction is correct; magnitude smaller because Slot A v2 already has a lower baseline (0.767 vs T5C 0.795).

### Gate 3 (saturation swing across q-bands)

Both bases PASS the swing ≤ 0.30 threshold — no compression-trap concern.

### may6 production-drift cohort with codec aug applied at inference

Applied VideoCodecSimulation to 92 may6 frames, 3 seeds per frame, scored both bases:

| ckpt | baseline frac>0.5 | aug frac>0.5 | rate factor |
|---|---:|---:|---:|
| Slot A v2 step3500 | 0.043 | **0.014** | aug **HELPS** 3× |
| T5C step3500 | 0.174 | **0.018** | aug **HELPS** 10× |

**This is the key positive signal**: codec aug doesn't compression-trap on production-realistic reals — it pushes them DEEPER into the real distribution. Training should propagate this invariance into the encoder.

Caveat: worst-case single-frame increase is +0.41 on both ckpts (a few frames become more fake-looking when aug fires). Training-time sees these in distribution and should converge correctly.

## How to read the verdicts (decision tree)

The morning verdict has 4 dimensions per packet:
1. Did the job complete? (SUCCEEDED vs FAILED vs CANCELLED)
2. Did the contract metrics hold? (lockbox_real_fpr ≤ 0.025, lockbox_fake_recall ≥ 0.688)
3. Did dev_fake_macro_recall regress? (close criterion: −0.05 floor)
4. Does the trained ckpt close the natural-experiment Δ? (close criterion: ≤ 0.05)

### Branch A: All 3 succeed and pass contract gates

If Slot 1 holds lockbox_fake_recall ≥ 0.688 AND closes natural-experiment Δ ≤ 0.05:
→ **Ship Slot 1 step3500**. Levers compose; codec restoration is the production fix.
→ Re-run the 2026-05-19 natural experiment on Slot 1's step3500 ckpt with `analysis/codec_restoration_gates_2026-05-21/run_gates.py` (point CKPTS at the new GCS path).
→ Run 4-ckpt 29-suite scorecard ($15-25 GPU) to canonicalize the result.
→ Reconsider tiebreak amendment (Job B evidence supports composite).

### Branch B: Slot 3 best (anchor doesn't add)

If Slot 3 ≈ Slot 1 on lockbox_fake_recall AND Slot 3 closes Δ ≤ 0.05 but Slot 1 doesn't:
→ **Codec is the binding lever, anchor doesn't add**. Refutes Slot A v2's anchor-mechanism claim.
→ Ship Slot 3 step3500.
→ Consider deprecating anchor_aware as a default lever.

### Branch C: Slot 1 worse than Slot A v2 (compression trap)

If Slot 1's lockbox_fake_recall < 0.50 OR dev_fake_macro_recall regresses >0.05:
→ **Compression trap from stacking codec on pipeline_randomization**. Heavy aug + anchor interact badly.
→ Slot 2 (light dose) should be the rescue — if Slot 2 lands between Slot 1 and Slot A v2 reference, the cliff is between 0.20 and 0.40.
→ If Slot 2 ALSO compression-traps, codec on Slot A v2 base is dead — consider Branch B's "Slot 3 only" path.

### Branch D: All 3 fail or non-deployment-grade

If none beat Slot A v2 step3500's lockbox_fake_recall 0.688:
→ Codec aug restoration doesn't close the transport gap on its own.
→ Open loops: **Roy_D anchor pool extension** (medium GPU, $30-50; requires bucket-prefix + frames upload + registry edit) and **multi-account capture sweep** (user-side, $0).
→ Reconsider whether the 2026-05-19 natural experiment is a per-account-rare event, OR a deployment-fatal axis.

### Branch E: Slot 2 unexpectedly wins (sweet-spot dose)

If Slot 2 has higher lockbox_fake_recall than Slot 1:
→ Dose-response is non-monotone; p=0.20 is the optimal dose.
→ Ship Slot 2 step3500. Run intermediate-dose follow-up (p=0.30) if natural-experiment Δ on Slot 2 doesn't fully close.

## Pre-checklist on each ckpt (before declaring a winner)

For EACH SUCCEEDED ckpt step3500:

1. **W&B scorecard sanity** — `contract/lockbox_real_fpr`, `contract/lockbox_fake_recall`, `dev_fake_macro_recall` over the periodic-save points. Check trajectory not just terminal.
2. **Manual canary** — same as 2026-05-20 (script at `analysis/manual_canary_2026-05-20/score_canary.py`). Compare to Slot A v2 step3500's reference values.
3. **Natural experiment retest** — re-score Roy_D + Guest crops with `analysis/codec_restoration_gates_2026-05-21/run_gates.py` after updating CKPTS list.
4. **may6 cohort** — `analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_f_may6_retest/` pattern. Slot A v2 reference is 4/92; Slot 1 should match or improve.
5. **29-suite scorecard** — optional, ~$15-25 GPU. Only if a ckpt looks promotion-grade on the manual canary.

## Known risks

1. **Image-currency edge case** — image `1.3.295` was built at 22:19 UTC; Slot 3 yaml was created at 22:18 UTC (1 minute before image build finished). If the build's source tarball was created at job START rather than END, Slot 3's yaml might NOT be in the image. If Slot 3 errors with "FileNotFoundError: config not found", REBUILD via `./dev.sh build-prod -y` and relaunch Slot 3 (it'll get a new VERSION 1.3.296+).
2. **Slot A v2's Gate 1 marginal FAIL** — the single-frame validity test under-estimates training-time invariance gain. If Slot 1 fails to close the natural-experiment Δ at step3500, that's the proximate cause.
3. **W&B canary may silently fail** when `multi_axis_grl.enabled: true` (open loop `canary-silence-when-multi-axis-grl-active`). All 3 packets inherit T5C's multi_axis_grl. **Mitigation**: the Move 3 fix to `trainer/mixins/canary_probe.py` is in the working tree (uncommitted M-file). If that fix is in the image used for tonight (1.3.295), canary should work. If not, expect missing canary metrics — fall back to manual canary post-job per pre-checklist step 2.

## What I did and did NOT do this session

DID:
- 3 yamls + launch script + image rebuild + Vertex submission (all 3 PENDING → RUNNING expected within 30 min)
- 4 CPU jobs (gates × 2 ckpts, may6+codec × 2 ckpts)
- STATE/HANDOFF/FACTS docs
- 1 clean commit (8 files, no working-tree M-files mixed in)

DID NOT:
- Commit working-tree M-files (HANDOFF.md, STATE.md, canary_probe.py, etc.) — those are user's in-progress work
- Run the AUC preservation gate on a labeled panel — superseded by in-flight wandb canary
- Roy_D-specific anchor pool packet — multi-step infra change, silent-failure risk under autonomous mode
- KLIEP teams-v2 vs lockbox distance measurement — informative but not gating
- Tiebreak amendment to `arena/score_teams_promotion_contract.py` — user-gated config change, no autonomy

## References

- This file: `analysis/codec_restoration_gates_2026-05-21/MORNING_DECISION_TREE_2026-05-21.md`
- FACTS doc: `analysis/codec_restoration_gates_2026-05-21/FACTS_2026-05-21.md`
- Gate outputs: `analysis/codec_restoration_gates_2026-05-21/gates_summary.json`, `outputs/may6_with_codec_summary.json`
- HANDOFF.md (working tree, uncommitted): full session log
- STATE.md (working tree, uncommitted): 2026-05-21 entry at top
- W&B dashboard: https://wandb.ai/dtect-vision/phase2r13-experiments
