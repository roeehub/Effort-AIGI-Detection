# P18 FINAL VERDICT — diagnostics A-G + D contract scorecard

**Date generated**: 2026-05-02 morning CEST, ~10:10 UTC
**Branch**: `teams-relaunch-root-2026-04-17`
**Predecessors** (read in order):
1. [`HANDOFF_2026-05-02_P18_CORRECTIVE.md`](HANDOFF_2026-05-02_P18_CORRECTIVE.md) — original "inconclusive" handoff with the diagnostics-needed list
2. [`HANDOFF_2026-05-02_P18_DIAGNOSTICS_COMPLETE.md`](HANDOFF_2026-05-02_P18_DIAGNOSTICS_COMPLETE.md) — CPU-only A/B/C/E/F/G writeup
3. [`HANDOFF_2026-05-02_NEXT_PACKET_DECISION_DRAFT.md`](HANDOFF_2026-05-02_NEXT_PACKET_DECISION_DRAFT.md) — pre-D candidate matrix
4. **THIS document** — final verdict combining diagnostics + D's contract numbers

---

## TL;DR

**P8A is the v3-contract winner** (rank 1). **NO arm clears the v3 `target_fake_recall_min=0.70` floor** — all three score 13.6%–17.9% on `dev_fake_macro_recall`, well below 0.70. The promotion contract picks P8A on the lexicographic tiebreaker (lowest lockbox FPR + highest lockbox fake recall combination).

The deeper findings:

1. **GRL works defensively, as predicted**: P18T's lockbox FPR (0.29%) is half of P18C's (0.81%). Without the GRL, FT would have introduced 5× more false positives at the contract τ.
2. **P18 arms ARE substantially better than P8A on `deeplive_enhanced` fake recall** (P8A 2.4% → P18T 7.5% → P18C 19.4%). The FT data does help on this academic-source fake category.
3. **P8A is dramatically better on `teams_real_dor_dev`** (0.000 FPR on 50 frames vs P18T/C both 0.040). Confirms the corrective-probe finding at full-suite scale.
4. **All three arms fail `visomaster_enhanced_macro_dev`** (1.1-1.8% recall). This is the production-target fake category, and all current models are ~unable to detect it. **This is the production-blocking gap, not the dor_shkedi issue.**

The next packet should not be a tweak of P18. The single most decision-relevant finding is that **the entire FT-from-P8A-on-current-data trajectory is plateaued well below the v3 contract floor**, with `visomaster_enhanced` being the gating bucket. Either:
- **Move 4 (paired same-identity contrastive)** to force method-conditional learning, or
- **Rebalance family_weights** to push visomaster substantially harder, or
- **Both** in sequence.

---

## D — full contract scorecard (Vertex job `4550151094264659968`)

Submitted 2026-05-02 08:31Z, RUNNING from 08:35Z, SUCCEEDED 10:04Z (1h33m). Image `1.3.241`, region `us-east1`. Cost ~$8-12.

Output GCS: `gs://training-job-outputs/test_results/teams_promotion_contract/p18-corrective-contract-20260502-103127/`

Local pull: `analysis/p18_probe_2026-05-01/d_results/`

### Promotion winner (`promotion_winner.json`)

```json
{
  "checkpoint_key": "P8A_REFERENCE_STEP5000",
  "promotion_rank": 1,
  "selected_threshold": 0.991,
  "dev_fake_macro_recall": 0.136,
  "dev_primary_real_fpr": 0.020,
  "dev_worst_real_stress_fpr": 0.016,
  "lockbox_fake_recall": 0.237,
  "lockbox_real_fpr": 0.0015
}
```

> **NB**: `target_fake_recall_min = 0.70` is the v3 floor; the winner's `dev_fake_macro_recall = 0.136` falls 56pp short. Contract still ranks within candidates that don't clear, hence "P8A wins" doesn't mean P8A is deployment-ready — it means it's the least-bad of the three.

### Checkpoint summary (all 3 arms, contract-selected τ)

| Metric | **P8A (rank 1)** | P18T (rank 2) | P18C (rank 3) |
|---|---:|---:|---:|
| selected_threshold | 0.991 | 0.994 | 0.995 |
| dev_primary_real_fpr | 0.0200 | 0.0200 | 0.0197 |
| dev_worst_real_stress_fpr | 0.0157 | 0.0300 | 0.0350 |
| **dev_fake_macro_recall** | **0.136** | 0.151 | **0.179** |
| **lockbox_real_fpr** | **0.0015** | 0.0029 | 0.0081 |
| **lockbox_fake_recall** | **0.237** | 0.178 | 0.174 |
| teams_fake_all_dev recall | 0.373 | 0.358 | 0.329 |
| visomaster_enhanced_macro_dev recall | 0.011 | 0.018 | 0.015 |
| deeplive_enhanced_dev recall | 0.024 | 0.075 | **0.194** |

### Per-suite real FPR (lockbox + dev real suites at the contract-selected τ)

| Suite | n_real | **P8A** | P18T | P18C |
|---|---:|---:|---:|---:|
| teams_real_all_dev (calibration target) | 3253 | 0.0200 | 0.0200 | 0.0197 |
| teams_real_all_lockbox | 1361 | **0.0015** | 0.0029 | 0.0081 |
| **teams_real_dor_dev** | 50 | **0.000** | 0.040 | 0.040 |
| teams_real_lighting_extreme_dev | 1401 | **0.016** | 0.030 | 0.035 |
| teams_real_poor_quality_dev | 923 | 0.0033 | 0.0022 | 0.0022 |

### Per-suite fake recall (4 fake suites at the contract-selected τ)

| Suite | n_fake | **P8A** | P18T | P18C |
|---|---:|---:|---:|---:|
| teams_fake_all_dev | 2409 | **0.373** | 0.358 | 0.329 |
| teams_fake_all_lockbox | 253 | **0.237** | 0.178 | 0.174 |
| visomaster_enhanced_macro_dev | 550 | 0.011 | **0.018** | 0.015 |
| deeplive_enhanced_dev | 545 | 0.024 | 0.075 | **0.194** |

---

## Synthesis — what the diagnostics + D mean together

### What's confirmed

1. **P8A wins overall** — contract rank 1, lowest lockbox FPR, highest lockbox fake recall.
2. **GRL has a real defensive effect** — P18T's lockbox FPR (0.29%) is ½ of P18C's (0.81%); P18T's `dor_dev` regression is no worse than P18C's; on every lockbox/stress suite P18T beats P18C.
3. **P8A's dor invariance is unique and full-suite-confirmed** — 0/50 false positives on the dedicated `teams_real_dor_dev` suite. Both P18 arms regress to 2/50.
4. **The "GRL is defensive against FT regression" reframe holds** — without GRL, the FT introduces a 5× higher lockbox FPR; with GRL, only a 2× regression.
5. **No arm clears the v3 `target_fake_recall_min = 0.70` floor** — winners are determined by lexicographic fallback, not by floor compliance.

### What's new from D (not in the diagnostics)

1. **P18 arms substantially improve `deeplive_enhanced_dev` recall** (P8A 2.4% → P18T 7.5% → P18C 19.4%). The FT data composition does teach the model new signal on academic deeplive content. P18C captures more of it than P18T because no GRL pressure.
2. **The visomaster_enhanced gap is the production-blocker, not dor_shkedi**. All three arms hit 1.1-1.8% recall on visomaster fakes. The training data is not delivering visomaster signal regardless of architecture choices.
3. **P18C's lockbox FPR (0.81%) at the contract-selected τ is much smaller than the mini-scorecard suggested (40% at τ=0.92).** The contract picks a tighter τ (0.995) which sweeps most of the FPR concern under the rug — but at the cost of reduced fake recall.

### Where my pre-D framing was right and wrong

| Pre-D claim | D outcome | Verdict |
|---|---|---|
| "P8A wins lockbox aggregate" | P8A rank 1, lockbox FPR 0.0015 vs P18T 0.0029 vs P18C 0.0081 | **Right** |
| "P18T trades 9pp recall for 5pp FPR vs P8A" | At contract τ: 6pp lockbox fake recall down, 1.5pp lockbox FPR up — but operating point is different (contract τ ≈ 0.99 vs my mini-scorecard τ=0.92) | **Right in direction, wrong in magnitudes**: contract picks tighter τ where both sides shrink |
| "GRL is defensive, not additive over P8A" | Confirmed — P18T halves P18C's regression but doesn't beat P8A | **Right** |
| "Lowering `deeplive_teams_fake` weight is the cheapest test" | NOT contradicted by D, but D reveals a more important gap: visomaster_enhanced is at 1-2% across all arms | **Partially right — pivots to "visomaster_enhanced" as the load-bearing missing capability** |
| "Move 4 is the structural fix" | Likely correct — the dor regression is real but is dwarfed by the visomaster gap | **Right but for a different reason than I originally said** |

---

## Next-packet recommendation (final)

The decision-matrix from the pre-D draft, now resolved with D's numbers:

### Recommended: **Move 4 (paired same-identity contrastive) is the right next major work, but it's not urgent — first try a cheap data-rebalance probe.**

**Rationale**: D shows the production-blocking gap is `visomaster_enhanced` (1-2% recall across all arms), not dor_shkedi. The dor finding is a real but secondary signal. The next packet should target visomaster.

#### Step 1 (cheap, ~$60-100): Try a `visomaster_*` family-weight bump

Hypothesis: P18 used `visomaster_fake: 4.0` and `deeplive_teams_fake: 7.0`. Pumping `visomaster_fake` to 10.0+ and slashing `deeplive_teams_fake` to 1.0 might shift learning toward what the contract actually scores.

What we'd expect: `visomaster_enhanced_macro_dev` recall climbs from 1-2% to 5-15% (still below floor but moves the needle); lockbox FPR plausibly stays similar; `deeplive_enhanced` recall drops from 19.4% (P18C) toward 5-10% (less FT data). Trade is tilted toward what the contract values.

Cost: ~$60-100 / ~12-18h Vertex. Just family_weights yaml changes; no new code.

#### Step 2 (medium, ~$80-120 + 2-3 days code): Move 4 paired same-identity contrastive

Hypothesis: Per `project_clean_teams_same_identity`, 705 paired identities are cached. A batch sampler that pulls (real_X, fake_X_swapped_to_Y) pairs forces the encoder to learn "what changed in the swap" rather than identity-cluster shortcuts. Should fix dor regression by construction AND improve method-conditional learning broadly.

This is a bigger swing than Step 1; probably worth running EVEN IF Step 1 hits something promotable, because Step 1 only addresses data weighting, not the structural shortcut problem.

#### Skip: P19 with stronger λ on the same data composition

Diagnostic E showed GRL effect lives at FINAL CLS, not intermediate layers. Stronger λ won't change the locus. The visomaster gap isn't a GRL-discoverable problem; it's a data-coverage/curriculum problem.

#### Skip: Promote P8A as production model

P8A is rank 1 of three options that all fail the floor by ~55pp. Don't ship; iterate.

### Provisional plan if running autonomously

If user authorizes, **Step 1** (the cheap rebalance probe) is the highest-EV next move. ~$80 within budget; one yaml change; takes overnight. Result tells us:
- If visomaster recall climbs materially: continue this trajectory.
- If visomaster recall is stuck at 1-2%: confirms the data simply doesn't have enough visomaster signal; pivot to Move 4 with confidence.

I am **not** launching Step 1 without explicit user OK.

---

## Operational state at end of this session

- **Vertex jobs**: D succeeded; no active jobs.
- **Image**: 1.3.241 (built today; baked the new P18 corrective checkpoint map).
- **Local artifacts**: `/tmp/p18_ckpts/` (~17GB), `/tmp/p17_ckpts/` (~8GB) — both safe to delete (downloadable from GCS).
- **D outputs pulled**: `analysis/p18_probe_2026-05-01/d_results/` (full local copy of contract artifacts).
- **Working tree**: uncommitted, per project pattern. Do NOT commit without explicit user OK.
- **Watcher background job (`bm0589os3`)**: completed when D succeeded.

## File inventory (this session)

```
NEW (uncommitted) — code & analysis:
A  analysis/p18_probe_2026-05-01/verify_l3_hook.py
A  analysis/p18_probe_2026-05-01/extract_all_arms_layers.py
A  analysis/p18_probe_2026-05-01/run_diagnostics_abce.py
A  analysis/p18_probe_2026-05-01/run_minicard.py
A  analysis/p18_probe_2026-05-01/run_roc_curves.py
A  analysis/p18_probe_2026-05-01/decompose_lockbox_auc.py
A  analysis/p18_probe_2026-05-01/analyze_d_results.py
A  analysis/p18_probe_2026-05-01/outputs/{corrective_probes__P8A_step5000_FINAL.json,
   diagnostics_abce_2026-05-02.json, mini_scorecard_2026-05-02.json,
   roc_curves_2026-05-02.json}
A  analysis/p18_probe_2026-05-01/d_results/promotion_contract/{promotion_winner.json,
   promotion_contract.json, checkpoint_summary.csv,
   selected_threshold_scorecard.csv, threshold_grid.csv}
A  analysis/p18_probe_2026-05-01/d_results/diagnostic_scorecard/{scorecard.csv, scorecard.json}
A  analysis/_features_cache_2026-04-30/{final_cls__{P8A,P18T,P18C}__n800.npz,
   intermediate__P18T,P18C__layer{03,06,09,11}__n800.npz}
A  arena/checkpoint_maps/teams_target_domain.p18_corrective_2026-05-02.yaml

NEW (uncommitted) — handoffs:
A  docs/relaunch_handoffs/HANDOFF_2026-05-02_P18_DIAGNOSTICS_COMPLETE.md
A  docs/relaunch_handoffs/HANDOFF_2026-05-02_NEXT_PACKET_DECISION_DRAFT.md
A  docs/relaunch_handoffs/HANDOFF_2026-05-02_P18_FINAL_VERDICT_WITH_D.md   ← THIS

VERSION bumped: 1.3.240 → 1.3.241 (image build to bake in new checkpoint map yaml).

Memory:
A  ~/.claude/.../memory/project_p18_diagnostics_complete_2026-05-02.md
M  ~/.claude/.../memory/project_p18_corrective_probe_2026-05-02.md (marked SUPERSEDED)
A  ~/.claude/.../memory/feedback_no_secrets_in_tmp_files.md
M  ~/.claude/.../memory/MEMORY.md (updated pointers)
```

## Next-agent checklist

1. Read this doc top to bottom.
2. **Decide**: Step 1 (visomaster-bump rebalance, $60-100) vs Step 2 (Move 4, 2-3 days code + $80-120) vs both in sequence.
3. If Step 1: edit `R13_P18_*` yaml to bump `visomaster_fake` family weight + lower `deeplive_teams_fake`; rebuild image; submit.
4. If Step 2: read `project_clean_teams_same_identity` memory and `data/sources/combined_paired.py` companion-bucket plumbing; design the paired-batch sampler; write tests; rebuild; submit.
5. Update `MEMORY.md` and add a new memory entry capturing the chosen direction + outcome.
6. Optionally commit this session's work (with user OK).

## Memory entries to write

After deciding the next packet:
- `project_p18_d_contract_p8a_wins_no_floor_clear.md` — capture "P8A wins; no arm clears 0.70 floor; visomaster gap is the bottleneck".
- Update `project_p18_diagnostics_complete_2026-05-02.md` with the D-confirmed numbers.
