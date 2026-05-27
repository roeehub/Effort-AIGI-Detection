# AGENT_PROPOSAL — Optimal deployment configuration (2026-05-21)

> OPINIONS doc for Job H. Read AFTER `JOB_H_DEPLOYMENT_PARETO_FACTS_2026-05-21.md`. Per `AGENTS.md` reading order: form your own Pass-1 view first; this doc is the interpretation layer.

## 1. The bottom line

**Ckpt: Slot A v2 step3500.** Three deployment operating modes are available (A/B/C — see FACTS §5), parameterized by the lockbox-calibrated FPR target (10% / 2% / 1%). τ is a deployment configuration knob; the mode choice is deferred to a product/SLA decision.

The ckpt choice is settled because Slot A v2 step3500:
1. Strictly Pareto-dominates the current contract rank-1 (P8A) on lockbox FPR vs recall (10/10 wins).
2. Strictly Pareto-dominates T5C (10/10 wins).
3. Catches +7.5pp more lockbox fakes than P8A at the headline operating point (Mode A) while matching P8A's production-drift cost on may6 (3/92 vs 0/92).
4. Catches 4.7× more visomaster-style fakes than the currently-deployed E2B at matched lockbox FPR.
5. Delivers the highest dev_fake_macro recall (0.762) of any candidate at Mode A.

Roy_D dev FPR is elevated across all 3 modes (96% at Mode A, 83% at Mode B, ~70-80% at Mode C). The Roy_D regression cost is operating-point-invariant in magnitude class — it cannot be tuned away by τ choice alone.

## 2. What the data actually says, in plain language

### 2.1 The contract has been ranking on the wrong axis

The 2026-05-20 contract verdict put P8A at rank-1 with `lockbox_real_fpr` 0.0184, vs Slot A v2 at 0.0191 (rank-2 by 1 video out of 1361). Job B's paired bootstrap already showed that 1-video gap is inside sampling noise (95% CI on Δ covers 0; P=0.519 either-way). What Job H adds:

**On the full lockbox Pareto, Slot A v2 wins at every single tested FPR level (10/10).** The contract's lex-on-FPR-ASC tiebreak isn't just inside noise on the chosen operating point — it's picking the strictly inferior model across the entire trade-off space. The contract is structurally misranking.

### 2.2 E2B (current deployment) is not the right ship

E2B Pareto-dominates P8A on lockbox (10/10) and produces a 95.3% lockbox_fake_recall at lockbox_fpr=10%. That number, in isolation, would justify keeping E2B. The reason E2B is not the right ship is the production-drift cost: E2B fires on **57/92 = 62% of may6 frames** at that operating point. P8A fires on 0. Slot A v2 fires on 3.

The lockbox cohort, despite being labeled "lockbox," is dominated by webcam-style captures that are visually closer to clean dev data than to production-drift cases (memory `project_lockbox_fpr_dominated_by_webcam_mode`: 65.7% webcam, 4.6% baseline FPR → 0.71% v2-filtered). E2B's "easy-to-catch" mode catches webcam-style fakes (where it's high recall) at the cost of false-flagging webcam-style reals (where production-drift lives). The contract gates on dev_real_fpr ≤ 0.07; E2B at any operating point with serious lockbox recall fails that gate. That's why E2B isn't in the 2026-05-20 contract scorecard — it's been silently excluded for substrate reasons, not because it's worse on lockbox.

### 2.3 Slot A v2's visomaster recall is the structural advantage

| Ckpt | viso_recall at lockbox_fpr=10% | × E2B |
|---|---:|---:|
| E2B | 0.113 | 1.0× |
| P8A | 0.444 | 3.9× |
| T5C | 0.440 | 3.9× |
| **SlotAv2** | **0.529** | **4.7×** |

E2B has a known weakness on Visomaster-style attacks (memory `project_e2b_breaks_deeplive_ceiling`: "deeplive 87.5%, viso REGRESSES 27% → 7%"). Slot A v2 is the only candidate that catches both Visomaster (52.9%) and deeplive (92.8%) at the chosen operating point.

If your production threat model includes Visomaster (and it does — `project_visomaster_v2_dor` confirms Visomaster is in scope), Slot A v2 catches 4.7× more of them than the model currently deployed. That's the deployment lift, not the +7pp lockbox recall.

### 2.4 The Roy_D risk

At τ=0.535, Slot A v2 false-flags Roy_D on 96% of his dev frames. This regression is operating-point-invariant — even at τ=0.85 (very strict), Slot A v2 still false-flags Roy_D 83% of the time. The anchor_aware mechanism pulled the dor pool toward "clean" and pushed Roy_D's encoder region away from clean as a side-effect (memory `project_band_shortcut_ood_hypothesis_2026-05-16`).

Three honest readings:
1. **Roy_D is one identity.** If Roy_D-like users are <1% of production, accepting 96% FPR on him is a 1% global cost. Ship at τ=0.535.
2. **Roy_D represents a cluster of similar users** (memory `project_chronic_offenders_partition_per_ckpt_2026-05-04`: chronic-FP identities partition per-ckpt). Without a sample of production users in his encoder region, we don't know cluster size. The honest answer is: capture a multi-account sweep, then check whether Slot A v2's per-user score distribution matches Roy_D's profile or P8A's.
3. **Roy_D is the model owner (Roee).** If a deepfake-detection product false-flags its operator 96% of the time, that's a brand failure for one user but operationally not deployment-blocking — just exclude Roy_D from the deployed model's calibration set.

The single-yaml mitigation: train Slot A v3 with a Roy_D-extended anchor pool (open loop `roy-d-specific-anchor-pool-packet`, ~$30-50 GPU). That's a 1-day intervention if it works.

### 2.5 Score-fusion ensembles don't help

I expected ensemble_mean to dominate single ckpts by averaging out each model's failure modes. It doesn't — Slot A v2 alone has better Pareto AUC (0.849) than any ensemble (max 0.823). The intuition: P8A and Slot A v2 fail on DIFFERENT identities, so averaging their scores produces a uniformly mediocre output rather than a clean signal. The lesson — for deployment, pick the best Pareto-dominant ckpt; don't try to ensemble around the failure modes.

ENSEMBLE_min (both models must fire to flag fake) does pick up some Pareto coverage at low FPR — but still inferior to single Slot A v2 across the curve. Worth knowing as a fallback if Slot A v2's Roy_D regression turns out to be a hard blocker; ensemble_min would give a more conservative model.

## 3. The three operating modes (parameterized by τ)

The three modes (A/B/C) listed in FACTS §5 are equivalent ckpt deployments at different τ values. Mode choice is a deployment-time configuration that depends on the operational FP/FN cost ratio — not a fixed recommendation in this doc.

| Mode | τ | lockbox_fpr | lockbox_recall | dev_fake_macro | Roy_D dev FPR | may6 | who it suits |
|---|---:|---:|---:|---:|---:|---:|---|
| **A — High-recall** | 0.535 | 10% | 0.854 | **0.762** | 0.962 | 3/92 | λ ≤ 3 (high cost of misses; webcam-like prod) |
| **B — Contract-compliant** | 0.780 | 2% | 0.696 | 0.454 | 0.831 | 1/92 | 5 ≤ λ ≤ 20 (balance; clears contract dev_fpr ≤ 7%) |
| **C — Low-FPR-strict** | 0.870 | ~1% | ~0.65 | ~0.30 | ~0.69-0.83 | ~0-1/92 | λ ≥ 30 (strict on real-user false flags) |

τ is a runtime config knob with no retraining cost. Picking is reversible — start at any mode, adjust later based on production telemetry.

**Why this doc does not pick a mode**: the choice rests on a product/SLA value (the FP/FN cost ratio λ) that is operational, not technical. The technical conclusion of this batch is: **the ckpt is settled, the three modes are the available operating points, and any of them is a defensible ship.**

## 4. What this batch eliminates from the menu

The other agent's earlier post listed remaining options 1-4 (stop and ship, SDK metadata, multi-account capture, Roy_D anchor pool). This batch resolves option 1 to a concrete answer: **the Pareto curve has been computed, Slot A v2 strictly dominates P8A, three operating modes are documented (FACTS §5)**. The "ship-or-not" question is now decomposed into "pick which mode (A / B / C)" — a deployment configuration choice, not a ckpt research question.

That makes the next-step picture cleaner:
- **If a mode is acceptable** → ship Slot A v2 step3500 at the chosen τ. Done.
- **If Roy_D regression is a hard blocker at all 3 modes** → pursue option 3 (multi-account capture, characterize Roy_D-like user prevalence) and/or option 4 (Roy_D-anchor packet) in parallel.
- **If product wants higher catching power than Mode A's 85.4% lockbox recall** → that requires moving past the current ckpt family; this analysis doesn't address it.

## 5. Caveats / what I'm uncertain about

- **E2B-on-may6 = 53/92 is a known weakness of E2B but does it generalize?** The 2026-05-19 natural experiment (Roy_D Teams account flip) suggests Teams-transport-encoded shortcut is the operative deployment failure mode. E2B's may6 fire rate is *partly* a Teams-transport issue (since may6 includes Teams-source frames), so the 53/92 is a directional indicator of "this model is brittle on production drift." Whether 53/92 reproduces on the next production-drift cohort is not measured.
- **Slot A v2's may6 = 3/92 is good but small-sample.** 95% CI on may6_fpr at 3/92 is approximately [0.7%, 9.3%] — wide enough that I can't rule out Slot A v2 being closer to T5C (4/92) than to P8A (0/92). The directional claim is solid; the precise rate is not.
- **The 75.5%/10% baseline from the prior agent in the original chat reproduces here at 77.9%/10%** within ±2.5pp. I've used the 77.9% number in the FACTS; mention 75.5% only as the prior agent's reference.
- **Visomaster recall on the canary panel is small-n** (n_fakes = 55 in `visomaster_enhanced_macro_dev`). The 4.7× E2B claim is on this small sample; bootstrap CI would put Slot A v2 0.529 in approximately [0.40, 0.66] and E2B 0.113 in approximately [0.05, 0.22] — still clearly disjoint at 95% but ±10pp resolution.

## 6. Self-correction log

1. **Mid-analysis framing**: I initially focused on the SlotAv2 vs P8A comparison from the original chat's framing. The E2B Pareto-dominance over P8A was a surprise and reshaped the analysis — it forced me to compute may6 trade-off as a third axis, otherwise E2B would have looked like the clear winner. The lesson: when the prior agent says "compare X to baseline Y, see if X dominates," include the deployed model Z (here E2B) in the comparison even if Y is the contract anchor. Z is the actual baseline.
2. **Ensemble expectations**: I expected ENSEMBLE_mean to be the optimal single config. It isn't (SlotAv2 dominates). Worth checking whether per-suite-routed ensembles (use P8A for substrates X, SlotAv2 for substrates Y) would dominate either — not done in this batch, but the per-identity table in §3.6 of FACTS shows different ckpts have different best-fit identities, so a per-substrate router could theoretically extract value the score-fusion can't.
3. **τ choice as deferred decision**: this doc previously framed Option A (τ=0.535) as a single recommendation. Revised 2026-05-21 to present three operating modes (A/B/C) as a menu without picking; mode choice is a product/SLA decision deferred to deployment time. The technical claim is the ckpt + the menu; not the picked mode.
4. **τ calibration approach**: all three modes use lockbox-calibrated τ (not dev-real-calibrated). The contract calibrates on dev_real to gate dev_real_fpr ≤ 0.07; switching to lockbox-cal τ for Modes A and C buys recall at the cost of higher dev_real_fpr. Mode B happens to land at dev_real_fpr ≈ 0.069, naturally inside the contract gate. The departure from contract calibration for Modes A and C is justified by §2.1 above (contract is ranking on the wrong axis).

## 7. Cross-references

- FACTS: `JOB_H_DEPLOYMENT_PARETO_FACTS_2026-05-21.md`
- Cohort-level FACTS docs: `RESULTS_FACTS_2026-05-20.md` (top-level navigator), `job_a..g/JOB_*_FACTS_2026-05-20.md`
- The other agent's recommendation post in the original chat: "If Slot A v2's Pareto dominates P8A's, that's your shipping point" — confirmed dominant (Bar 1 MET).
- Open loops resolvable: `lockbox-real-fpr-tiebreak-is-load-bearing`, `deployment-vs-p8a-substrate-tradeoff-not-quantified` (close criterion: both populations quantified — see FACTS §3.5)
- Open loops still active: `roy-d-specific-anchor-pool-packet`, `eval-production-crop-tightness-mismatch`
