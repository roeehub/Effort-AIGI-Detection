# P1 (PE_PAIR_RANK_DRO) — Structured Analysis, 2026-05-07

> **⚠ DEPRECATED — read `DEEP_DIVE_FACTS_2026-05-07.md` and `AGENT_PROPOSAL_2026-05-07.md` first.**
>
> This was the agent's FIRST analytical pass. Its §6 ("settled understanding") was partly superseded mid-session:
> - F5 chronic-6 read was based on a buggy regex matcher (PC_Generator/Q chronic identities silently dropped). Corrected version: `DEEP_DIVE_FACTS_2026-05-07.md` §6 + `phase_d/per_identity_fpr_FIXED.csv`.
> - F3 audit was based on 3 axes (sharpness, face_area, luma); extended to 4 axes (added min_dim, color_b_dev) later in the session. Corrected version: `DEEP_DIVE_FACTS_2026-05-07.md` §4 + `FOLLOWUPS_FACTS_2026-05-07.md` §1.
>
> The corrected factual record lives in `DEEP_DIVE_FACTS_2026-05-07.md`. The agent's self-correction log (with explicit retraction of the GroupDRO-balloon hypothesis) lives in `AGENT_PROPOSAL_2026-05-07.md` §4.
>
> **This file is preserved as the original interpretation for forensic purposes.** Renamed from `ANALYSIS_2026-05-07.md` on 2026-05-07.

---

**Companion to**: `RESULTS_FACTS_2026-05-07.md` (which has the raw numbers).

This doc walks through the six analytical layers the user asked for: motivation → expectations → results → interpretations → criticism of interpretations → settled understanding. It explicitly surfaces uncertainty and avoids embedding prior-agent framings.

The "settled understanding" section is the load-bearing one; everything before it is showing the work.

---

## 1. Original motivation

P1 was designed to address a specific situation:

- **P8A had broken the camera-signature shortcut** (the RLP6_04 ceiling around 0.84-0.89 anchor FPR — memory `project_p8a_breakthrough.md` and `project_shortcut_is_upstream.md`). It produced a model that was meaningfully more dor-anchor-invariant than its predecessors.
- **But P8A was still IQ-shortcut-coupled** — score correlated negatively with sharpness (r=−0.644 on dor frames; memory `project_image_quality_shortcut.md`), and reach to the production v2 substrate was bounded ("viso ceiling 27% unbroken across 13+ packets" — memory `project_viso_ceiling_unbroken_10_packets.md`).
- **Multiple data-axis levers had been pulled with no lift** — P14_DATA_FIX (xan4dfto) collapsed at value_composite=0.126; P16_DATA_AXIS (rmic6wrc) didn't lift contract-calibrated viso recall above 1.1% (memory `project_data_axis_lever_pulled_twice_no_lift.md`). Packets A + C (codec aug) showed codec aug was net-negative for viso (memory `project_pc_codec_aug_hurts_viso_2026-05-05.md`).
- **The external advisor's primary recommendation** was to combine two new lever classes: a per-pair invariance loss (`pair_rank_loss`, λ=0.2 m=0.5) + multi-axis GroupDRO (R-D real-side keys + F-B fake-side keys, β=3.0, with `chronic_flag` for the chronic-6 FP cohort).

The two slots — BUNDLE (pair_rank + GroupDRO) and PAIRRANK_ONLY (pair_rank only) — were structured as a **single-lever ablation** per the `anti_shortcut_bundle_decomposition` discipline, after the P14 lesson that stacked-lever bundles can be net-negative compared to single-lever variants.

Both arms FT'd from `P8A_REFERENCE_STEP5000` on the post-`2feea58` codepath — the FIRST test of pair-rank/DRO levers with `apply_svd_to_in_proj` actually firing on the qkv classification gradient.

The packet's hypothesis: pair-rank loss buys lift on previously-missed fakes (by pulling fake scores above their paired real scores); GroupDRO buys worst-group recall + chronic-FP suppression by reweighting the worst-performing (real_side × fake_side) cells during training.

## 2. Expectations

The pre-registered close criterion (F1-F5 in the YAML headers) was demanding:

- **F1**: lockbox fake recall ≥ 90% at FPR ≤ 10%. (At time of design, P8A baseline was 38.7%.)
- **F2**: pair-rank metric — fraction `fake_score > real_score` on previously-missed fakes ≥ 30% on ≥ 2 of 6 paired lanes; worst-group recall lift ≥ 20% on chronic / dor-drift cohorts.
- **F3**: no untargeted axis (is_webcam, face_area_fraction, min_dim, color_b_dev) amplifies +50%.
- **F4**: HDTF cross-substrate FPR ≤ 5%.
- **F5** (BUNDLE only): chronic-FP `pc_generator` cluster failure rate (P8A baseline 0.520) drops by ≥ 0.10 absolute via GroupDRO `chronic_flag`.

The packet was a 4-of-4 gate (F5 only counts for BUNDLE). Implicit expectation: at least one BUNDLE ckpt clears 4-of-4 to "pass"; the matched-step ablation BUNDLE-vs-PAIRRANK_ONLY tells us whether GroupDRO was the load-bearing piece.

The slot-1-vs-slot-2 ablation question was specifically framed: **what tradeoff did GroupDRO buy?** — not "did GroupDRO add anything", per the same discipline. A negative delta on one criterion + positive delta on another would be the expected shape if both levers are doing real work but in different directions.

## 3. Results — what happened

(Numbers in §2 of `RESULTS_FACTS_2026-05-07.md`. Brief headline only here.)

a) **Phase A succeeded** (29-suite contract scorecard, 6h 15m). **Phase C failed** at the contract step (per-suite diagnostic completed; promotion_contract step never ran).

b) **No ckpt cleared F1.** Best lockbox recall: BUNDLE_PERIODIC_STEP500 at 82.6% (untrained, 500 steps); PAIRRANK_PERIODIC_STEP500 at 70.8%; P8A 38.7%; E2B 62.9%.

c) **Contract winner: PAIRRANK_PERIODIC_STEP500** (rank 1). dev_fake_macro_recall=0.354 at τ=0.768, lockbox_real_fpr=1.84%. The earliest PAIRRANK_ONLY ckpt — only 500 training steps in — won the contract.

d) **τ pattern is bimodal.** Two ckpts at τ ~0.71-0.77 (E2B, PAIRRANK_step500); the other six at τ ~0.99-0.999. BUNDLE_step3750 + step4000 specifically at τ=0.9994 + 0.9989, with dev_fake_macro_recall **below the 0.30 floor flag** (0.192 and 0.285).

e) **E2B beats every P1 ckpt on dev_fake_macro_recall** (0.508 vs P1 best 0.354). On the contract's primary lexicographic key, no P1 candidate displaces E2B.

f) **HDTF τ=0.5 diagnostic**: BUNDLE_PERIODIC_STEP500 has 25% FPR on `proper_real_clean_dev`. P8A is at 93%+ recall on every HDTF fake suite. The HDTF substrate continues to show the cross-substrate gap documented in memory `project_job_b_findings_universal_vs_trajectory_2026-05-04.md` — same model is much better on HDTF than on production v2.

g) **Trajectory data** (180-frame dor probe + 50-frame teams_real_dor_dev): BUNDLE arm has lower IQ-axis coupling than PAIRRANK arm at all measured steps. PAIRRANK_PERIODIC_STEP500 has dor_dev FPR=16% (8/50) — worst of all 8 ckpts; PAIRRANK_TOP_N_STEP6750 has dor_dev FPR=4% (2/50) — tied for best.

h) **W&B logging gap** found in trainer.py:1727 — BUNDLE's pair-rank-loss scalar magnitude is unverifiable from logs. Real bug; one-line fix; affects every `use_group_dro=true` run.

## 4. Possible interpretations

These are *candidate* readings. Each is internally consistent with the data; not all can be simultaneously load-bearing.

**(a) "P1 failed the gate."** Straightforward read of F1: 90% lockbox is the bar; nobody hit it; the packet doesn't promote.

**(b) "PAIRRANK won because the contract picks rank-1 in a failing field."** The 0.30 dev_fake_macro_recall floor was set, but doesn't appear to gate τ selection for ckpts that never reach 0.30 across any τ (BUNDLE_step3750 at 0.192 still got τ=0.999x). "Rank 1" is a sort, not a promotion decision; calling PAIRRANK_PERIODIC_STEP500 the "winner" undersells how far it is from F1.

**(c) "The advisor's specific bet did not pay out."** Bet was: pair-rank lift on missed fakes + GroupDRO worst-group reweighting. On dev fake suites: every P1 ckpt is below E2B (production baseline) on dev_fake_macro_recall. On lockbox: best P1 is BUNDLE_step500 at 82.6%, but its dev_fake_macro is 0.075 (catastrophic on visomaster_enhanced_macro_dev = 0.55%, deeplive_enhanced_dev = 0%). The lever class produced one ckpt with high lockbox recall and bad calibration, and several with reasonable calibration but no lockbox lift.

**(d) "BUNDLE_PERIODIC_STEP500 is in an early-training high-recall-bad-calibration regime."** 82.6% lockbox recall + 25% HDTF clean FPR + 0% deeplive recall + 0.55% visomaster recall is the signature of a model that fires near-uniformly across many inputs. As BUNDLE training proceeds (step3750, step4000), FPR drops AND lockbox recall drops — calibration tightens around a worse decision boundary. (Caveat: this is interpretive; we don't know the model is "miscalibrated" vs "structurally different at this stage".)

**(e) "BUNDLE's IQ-axis decoupling didn't translate into deployment-grade outcomes."** §5b of the synthesis template shows BUNDLE has lower sharpness raw_r than PAIRRANK at all steps. But BUNDLE's dev_fake_macro is consistently worse. Even if GroupDRO induced feature-decoupling on the dor real frames, the resulting model is worse at fake detection on the dev fake suites.

**(f) "E2B is structurally better at fake recall on dev than any P1 ckpt."** E2B dev_fake_macro = 0.508. Every P1 ckpt is below this. Whatever P1 trained from the P8A FT base, the resulting model is worse on the contract's primary metric than the production deployment baseline. P1 doesn't displace E2B by this reading.

**(g) "The slot-1-vs-slot-2 ablation answers in a way the discipline anticipated."** Matched-step pair (step500): BUNDLE wins lockbox recall (82.6 vs 70.8). PAIRRANK wins dev_fake_macro (0.354 vs 0.075). The two arms traded axes. The contract's lexicographic ordering picks PAIRRANK (dev_fake_macro is a higher-priority key); under different weighting BUNDLE would win.

**(h) "τ=0.999x for BUNDLE step3750/4000 may indicate the recall floor didn't fire as expected."** With dev_fake_macro_recall=0.192 (step3750) the floor of 0.30 should have been violated. Either floor logic differs from naïve interpretation, or there's a bug. Independent of P1's verdict, this matters for any future packet that relies on the floor flag.

**(i) "P1 produced a real but not-targeted lift on dor invariance at low τ."** PAIRRANK_PERIODIC_STEP500 has lockbox recall 70.8% at τ=0.768 (low-τ regime, not in τ-tail) with lockbox FPR 1.84%. That IS a meaningful uplift over P8A's 38.7% lockbox recall. The packet didn't hit F1 but it did move a real metric.

## 5. Possible criticism of those interpretations

Each interpretation in §4 has at least one weak point worth surfacing.

**(a) "Failed the gate" is technically correct but anchors on F1=90%, an aggressive bar set when the baseline was 38.7%.** Asking for +51pp absolute lift on lockbox in one packet may have been unrealistic from the start. A "softer" F1 (say ≥60%) would change the verdict shape — though that's hindsight calibration not a defense of P1.

**(b) "Rank 1 in a failing field undersells the result"** — but it also UNDERSELLS the result if we ignore the actual rank-1 lift. PAIRRANK_PERIODIC_STEP500's 70.8% lockbox vs P8A 38.7% is +32pp. That's not nothing. The argument cuts both ways.

**(c) "Bet didn't pay out"** — but the bet was framed against a high bar (F1 ≥ 90%). Against the operational baseline of "is there a ckpt that beats E2B+P8A on lockbox at acceptable FPR", PAIRRANK_PERIODIC_STEP500 at 70.8% lockbox / 1.84% lockbox FPR / τ=0.768 is the first such ckpt in many packets. Whether that's "paid out" depends on what we're paying for.

**(d) "BUNDLE_step500 calibration unsettled"** — interpretive label without a control. We'd need to compare to other "early-FT" ckpts to call this typical/atypical. Possible alternative: BUNDLE_step500 captures something real about the dataset boundary that BUNDLE_step4000 has lost.

**(e) "IQ-decoupling didn't translate"** — the IQ-decoupling probe was on 180 frames of dor reals. It was specific. BUNDLE_PERIODIC_STEP500 has dor_dev FPR=4% (2/50) — the BEST dor result at calibrated τ, tied with PAIRRANK_step6750. So the decoupling DID translate to a specific outcome; just not to broad fake recall.

**(f) "E2B is structurally better"** — E2B has 5% recall on visomaster_enhanced_macro_dev and 80% on deeplive_enhanced_dev — extreme spread. P1_PAIRRANK_step500 has 22% / 34% — narrower spread. "Better at dev_fake_macro" is true but the underlying distribution is concentrated in a way that may not generalize. E2B's structural advantage on dev may be substrate-bound.

**(g) "Slot-1-vs-slot-2 traded axes"** — the contract DID pick a winner (PAIRRANK_PERIODIC_STEP500). Calling it a "trade" is a deliberate choice to step outside the contract's own ordering. Whether that's appropriate depends on whether we trust the contract's lexicographic keys for THIS specific question.

**(h) "Recall floor didn't fire"** — speculative until I read `arena/score_teams_promotion_contract.py` and `threshold_grid.csv`. Could simply be that the floor logic does what it says but my mental model of it is wrong. Forensic check needed.

**(i) "Real lift at low τ"** — at τ=0.768 PAIRRANK_PERIODIC_STEP500 has dor_dev FPR = 16% (8/50). That's the WORST dor_dev FPR of all 8 ckpts. So the "lift" comes with a cost on the canonical dor-anchor test. Calling it a "real lift" obscures the trade.

## 6. Settled sober understanding

What I'm comfortable signing my name to, ranked by confidence (most confident first):

**(i) Highest confidence — direct readings of the data:**

1. **No P1 ckpt cleared F1 at calibrated τ.** Highest lockbox recall observed is 82.6% (BUNDLE_PERIODIC_STEP500 at τ=0.9919); F1 bar was 90%. Statement is unambiguous.

2. **Phase A's contract winner is P1_PAIRRANK_PERIODIC_STEP500, with dev_fake_macro_recall = 0.354 and selected_threshold = 0.768.** Factual artifact of the contract algorithm.

3. **E2B (production deployment) beats every P1 ckpt on dev_fake_macro_recall** (0.508 vs P1 best 0.354). On the contract's primary metric, P1 does not displace the production baseline.

4. **Phase C failed in the contract step.** The diagnostic-only τ=0.5 readout is in hand; calibrated-τ HDTF FPR (the actual F4 input) is not. F4 verdict is still pending and recoverable via local computation against the HDTF reports.

**(ii) High confidence — structural readings:**

5. **The slot-1-vs-slot-2 matched-step pair (BUNDLE_step500 vs PAIRRANK_step500) does not have a single-axis winner.** BUNDLE wins lockbox recall (82.6 vs 70.8). PAIRRANK wins dev_fake_macro (0.354 vs 0.075). The contract picked PAIRRANK. The ablation answer is a tradeoff, not a domination — consistent with the `anti_shortcut_bundle_decomposition` discipline's expected shape.

6. **PAIRRANK_PERIODIC_STEP500 is a meaningful lockbox-recall uplift over P8A** (70.8% vs 38.7%, +32pp absolute) at acceptable lockbox FPR (1.84%) and a non-tail τ (0.768). Not a F1 pass, but the first low-τ ckpt with this property in many packets.

7. **BUNDLE's IQ-axis decoupling translated to one specific outcome (low calibrated-τ dor_dev FPR) but not to broad fake recall.** BUNDLE_PERIODIC_STEP500 has dor_dev FPR=4% (best, tied) AND dev_fake_macro=0.075 (worst). Decoupling and classification are not the same.

**(iii) Medium confidence — methodological reads:**

8. **The selected τ for BUNDLE_TOP_N_STEP3750/STEP4000 (τ=0.9994/0.9989) sits in the τ-tail regime that the recall-floor flag was supposed to guard against.** Both ckpts have dev_fake_macro_recall below the 0.30 floor. The floor either didn't gate τ selection here or has different semantics than I expect. **Audit on `arena/score_teams_promotion_contract.py` is needed** to confirm. Independent of P1 verdict.

9. **The trainer.py:1727 W&B logging gap is real and affects every `use_group_dro=true` run.** Cannot reconstruct BUNDLE's pair-rank-loss magnitude from history or checkpoint replay alone. Decision: fix-and-rerun on a future packet, OR accept the gap. Worth raising before the next GroupDRO packet.

10. **The HDTF substrate continues to show a substantial cross-substrate gap vs production.** P8A 93%+ HDTF fake recall vs 13.5% on production visomaster_enhanced_macro_dev (calibrated τ, n=550). This isn't new but P1 didn't change it. F4 ≤ 5% may pass on HDTF for most P1 ckpts, but that doesn't mean the model handles production better.

**(iv) What I am NOT comfortable signing my name to (yet):**

11. **"P1 is a dead lever class."** The ablation rotation, BUNDLE_step500's lockbox recall, and BUNDLE's IQ-axis decoupling at low magnitude all suggest the lever class is doing real work the previous packets weren't. We'd need a follow-up that targets the specific failure mode (e.g., visomaster_enhanced recall via a viso-specific data lever stacked with pair-rank, or longer training to see if BUNDLE's calibration tightens with maintained recall) before declaring the class dead.

12. **"GroupDRO was load-bearing in BUNDLE."** Possible, given BUNDLE-vs-PAIRRANK arm separation at every measured step on IQ-axis r AND on lockbox recall AND on dev_fake_macro. But the W&B logging gap means we can't directly verify GroupDRO fired at non-trivial magnitude. Indirectly suggested; not directly confirmed.

13. **"PAIRRANK_PERIODIC_STEP500 should be promoted."** Promotion is the user's call. The ckpt has properties that look promotable (high lockbox recall vs P8A, low τ, low lockbox FPR) and properties that don't (worst dor_dev FPR; below E2B on dev_fake_macro). The verdict shape is "real signal at low τ; not deployment-grade by F1; needs user weighting".

**(v) Questions this evaluation surfaced — open loops worth tracking:**

- **Recall floor τ-selection forensic** — did the 0.30 floor gate τ selection for BUNDLE_step3750/step4000? If not, why?
- **Phase C failure root cause** — what failed in the promotion_contract step? Image/data/quota issue or contract scorer bug specific to HDTF?
- **F2/F3/F5 verdicts** — pull Phase A reports/, run the prepared scripts. The scaffolding is on disk.
- **F4 calibrated-τ HDTF FPR** — apply Phase A's per-ckpt τ to Phase C's per-frame reports. Cheap local computation; recovers F4 without rerunning Phase C.
- **trainer.py:1727 W&B logging fix** — one-line code change; decision on whether to deploy fix + rerun before next GroupDRO packet.
- **Why does BUNDLE_PERIODIC_STEP500 have 25% HDTF clean FPR at τ=0.5?** — at calibrated τ=0.9919 it should drop, but starting from 25% is qualitatively different from other ckpts' starting points. Calibrated-τ FPR is the real read.

---

## Closing note

The verdict-grade reading of this packet does not yet exist. F2, F3, F5 are not computed; F4 is recoverable but not run. F1 is decisive: no ckpt clears it. PAIRRANK_PERIODIC_STEP500 is a non-trivial low-τ result; whether it is promotion-grade is a user judgment about how much weight to put on lockbox recall lift vs dev_fake_macro displacement of E2B.

The scaffolding to compute F2-F5 is on disk. The clean next step is to pull Phase A reports and run them.
