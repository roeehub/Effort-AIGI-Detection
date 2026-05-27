# Agent Proposal — T6 / T7 / T5C packet interpretation (2026-05-12)

> **Status: OPINION, not RECORD.** Reviewer is invited to disagree. Each load-bearing claim cites specific numbers from `RESULTS_FACTS_2026-05-12.md` (`§N.M`).
>
> **Authoring**: written 2026-05-12 UTC by the agent who proposed the T6/T7/T5C bundle, drafted the yamls, ran the pre-launch CPU diagnostics (A1/A2/A3 + A2-extension + head_retrain on 2026-05-11), launched the GPU jobs, ran the scorecard, and authored the FACTS doc above. Single-agent session; the opinion in this doc was not authored independently of the FACTS doc.
>
> **Required disclaimer per convention**: past framings in this project have been demonstrably wrong (see e.g. the T4 2026-05-11 OPINION doc's "substrate-overfit" framing — partially retracted in the iq_shortcut thread 2026-05-11 update once A1/A2/A3 landed; also the P15/P18 GRL packets that predicted invariance lift that did not materialize; the cyclic-λ T5-A packet that was hypothesized to capture attractor moments and was refuted same-night). This doc adds another candidate framing to that ledger; treat it skeptically.

---

## 1. Executive summary

The T6 / T7 / T5C bundle was launched 2026-05-11 morning with three single-lever hypotheses (T6: T3 + face_scale_jitter; T7: T4 + face_scale_jitter; T5C: T4 with classifier hidden_dim 256 → 1024). The scorecard finished 2026-05-12 03:13 UTC. Headline: **P8A retains contract rank-1** (FACTS §3.1); **T5C step3500 is the highest-ranking new ckpt at rank 3** (FACTS §3.2), above the strongest prior-week candidate (T3_SLOT1_step1500 at rank 4).

The data is consistent with one new positive finding, one refuted hypothesis, one partially-supported hypothesis, and one persistent failure mode:

1. **NEW POSITIVE**: T5C step3500 is the first non-P8A / non-E2B ckpt to crack the top 3 on the v3-fix contract since the T3 packet two weeks ago. The single-lever delta (classifier hidden_dim 256→1024) appears to give meaningfully better dev/lockbox balance than its T4 base (T4_step10500 ranked 3 of 7 on its own scorecard with `lockbox_real_fpr` 0.0536; T5C step3500 sits at `lockbox_real_fpr` 0.0279 — about half).
2. **REFUTED**: face_scale_jitter@0.50 was previously shown to be a load-bearing single-lever anti-shortcut on the P14 sister-variant ablation 2026-04-30 (memory `project_face_scale_jitter_load_bearing`). On both T6 (T3 base) and T7 (T4 base), adding jitter regresses `dev_fake_macro_recall` below the 0.30 floor on every sampled step (FACTS §5.1). Jitter does NOT compose as a free additive lever on these bases.
3. **PARTIALLY SUPPORTED**: "stronger online classifier forces uniform encoder invariance covering chronic_6" — the original hypothesis behind T5C, motivated by A3's finding that T4 regressed chronic_6 inv_mean by 0.0315 absolute. T5C step3500's `teams_real_dor_dev` FPR (0.06, 3/50) is below P8A's (0.08, 4/50) — directionally consistent with the hypothesis. But T5C step1500's catastrophic `lockbox_real_fpr` (0.3204) shows the lockbox-clean window is narrow in training step; the bigger classifier doesn't *uniformly* fix invariance.
4. **PERSISTENT**: every R13 candidate continues to under-perform P8A on `lockbox_real_fpr` even when it lifts every other metric. The v3-fix policy's tiebreak on `lockbox_real_fpr` is what keeps P8A at rank 1 across 13+ packets — and it remains the only metric where P8A's lead is decisive (FACTS §5.2 — P8A 0.0184 vs E2B 0.0235 vs T5C step3500 0.0279).

The implications I draw — that T5C is the next refinement direction, that T6/T7 don't merit further investment, that the dev→lockbox transfer gap is now the binding constraint — are interpretations of these facts. I assign them MEDIUM confidence overall; §5 names the framings I retract or downgrade from prior-session OPINION docs.

## 2. High-confidence claims

These are direct paraphrases of FACTS, not interpretations:

1. P8A is contract rank-1; no T6 / T7 / T5C ckpt displaces it. **Citation**: FACTS §3.1.
2. 6 of 12 ckpts pass all three contract gates: P8A, E2B, T5C step3500, T3_SLOT1 step1500, T5C step3750, T5C step1500. **Citation**: FACTS §5.1.
3. 6 of 12 ckpts fail the dev_fake_macro_recall floor: all three T6 ckpts and all three T7 ckpts. **Citation**: FACTS §5.1.
4. T5C step3500 ranks 3, above T3_SLOT1 step1500 at rank 4. **Citation**: FACTS §3.2.
5. Among the 6 all-pass ckpts, rank ordering is consistent with ascending `lockbox_real_fpr` (P8A 0.0184 → T5C step1500 0.3204). **Citation**: FACTS §5.2.
6. T5C step1500 has the highest dev_macro_recall (0.6832) AND the worst `lockbox_real_fpr` (0.3204) of any all-pass ckpt. **Citation**: FACTS §7.2 + §3.2.
7. T6 step1500 has the highest `lockbox_fake_recall` (0.8696) of any scored ckpt but fails the dev_macro_recall floor at 0.2598. **Citation**: FACTS §3.2 + observation 9.
8. T3_SLOT1 step1500 has the highest `lockbox_fake_recall` (0.7747) among all-pass ckpts. **Citation**: FACTS §7.1.

## 3. The mechanism I propose (caveated)

I propose three loosely coupled mechanism claims, each tagged with confidence:

### 3.1 **Classifier capacity is a real lever for the T4 GRL recipe** (MEDIUM)

A3 (2026-05-11) found T4_L1_step10500 regressed chronic_6 inv_mean by −0.0315 absolute at L11 vs P8A on the atlas triptych chronic_6 slice. The T5C hypothesis was that this is a "classifier-capacity" issue: with hidden_dim=256, the online classifier can satisfy GRL on the dev side without forcing the encoder to be uniformly invariant across the chronic_6 cohort.

The scorecard data is directionally consistent: T5C step3500 has `teams_real_dor_dev` FPR 0.06 (3/50) vs T4 step10500 0.18 (9/50, per the T4 eval FACTS §4.1) — a 3× reduction on the chronic_6 cohort. The cohort is small (n=50, 95% CI ±0.14) so the magnitude is noise-band-limited, but the sign is clean.

**Caveat**: I have no direct measurement of T5C's L11 chronic_6 inv_mean (no CPU atlas re-run for T5C yet). The behavioral lift on `teams_real_dor_dev` could be (a) the proposed mechanism, (b) coincidence at this specific step, or (c) downstream of some other effect (e.g., the bigger classifier shifts T5C's overall score distribution and we're seeing the τ-calibration consequence, not an encoder change).

### 3.2 **face_scale_jitter@0.50 does NOT generalize as a free additive lever** (HIGH)

T6 (T3 base + jitter) and T7 (T4 base + jitter) both regress `dev_fake_macro_recall` below 0.30 across all 3 sampled steps each (FACTS §5.1). The P14 sister-variant ablation finding (2026-04-30) showed jitter@0.50 alone won value_composite vs the full P14 bundle; this established jitter as load-bearing *for that base*. The T6/T7 results refute jitter as a *composable* lever — when stacked on top of T3 SLOT1's keep-list lever OR on top of T4's multi-axis L11 GRL, jitter cuts recall by 0.04-0.14 absolute (sampled vs base contracts) without proportional FPR benefit.

**What this would explain**: jitter's anti-shortcut effect on P14 may have been training-distribution-specific (P14's bundle was net-negative; jitter alone undid the damage). On T3 / T4 bases that already break the relevant shortcut differently, jitter introduces noise that the model has to absorb without compensating benefit.

**Caveat**: I tested jitter at one magnitude (scale_limit=0.50) on two bases (T3, T4). I don't know what jitter@0.20 or @0.30 would do; lower magnitudes might pass the floor.

### 3.3 **dev → lockbox substrate-transfer is the binding production constraint** (MEDIUM)

This is the framing carried over from the head_retrain CPU diagnostic (2026-05-11 PM, β-outcome). The scorecard data reinforces it: every all-pass ckpt has `lockbox_real_fpr` strictly worse than P8A's 0.0184, and the rank order tracks this metric exclusively (FACTS §5.2). Every architecture / loss / data lever pulled since T3 has lifted `lockbox_fake_recall` (good) and degraded `lockbox_real_fpr` by 0.005-0.30 (mixed-to-bad).

**What this would explain**: this is the same shape as the head_retrain finding — dev-trained heads can't generalize the dor_shkedi-vs-real_dor signature flip. The encoder may be substrate-invariant on average (per A2's frozen-probe AUC=1.000 result) but the trained head's transfer to lockbox stays limited by what dev distribution exposes.

**What this does NOT explain**: why T5C step1500 catastrophically regresses lockbox_real_fpr to 0.32 while T5C step3500 (2000 more training steps with the same recipe) keeps it at 0.028. If the constraint were purely "what dev shows," the trajectory shouldn't have a clean step1500 vs step3500 dichotomy.

## 4. Confidence-tiered claims

| Claim | Confidence | Citation |
|---|---|---|
| P8A is rank-1; no T6/T7/T5C ckpt displaces it | HIGH | FACTS §3.1 |
| 6 of 12 ckpts pass all three contract gates | HIGH | FACTS §5.1 |
| T5C step3500 ranks 3, above T3_SLOT1 step1500 | HIGH | FACTS §3.2 |
| face_scale_jitter@0.50 regresses dev_macro_recall on both T3 and T4 bases | HIGH | FACTS §5.1, §6 |
| Lockbox_real_fpr is the rank tiebreaker among all-pass ckpts | HIGH | FACTS §5.2 |
| Classifier capacity is the lever responsible for T5C step3500's rank-3 placement | MEDIUM | inference from FACTS §6 + §8 + memory `project_t4_substrate_overfit_inv_mean_misleading_2026-05-11` |
| T5C's chronic_6 behavior is improved vs T4 | MEDIUM | inference from FACTS §8 (dor_fpr 0.06 vs T4 0.18 on n=50) |
| dev → lockbox substrate transfer is the binding production constraint | MEDIUM | head_retrain CPU diagnostic (2026-05-11 PM, β-outcome) + FACTS §5.2 + §7.2 |
| T6 / T7 are dead-ends; further jitter investment is unwarranted | MEDIUM | inference from FACTS §5.1 (6/6 jitter ckpts fail floor) |
| Lower jitter magnitudes (0.20-0.30) would also fail | LOW | speculation; not tested |
| T5C is the best base for a follow-up packet | LOW | depends on §3.1 mechanism being correct |
| The lockbox-clean window is narrow in T5C training step | LOW | n=2 step samples (1500 catastrophic, 3500/3750 clean) — not a trajectory |

## 5. Self-correction log

This is the third agent session in 5 days on the post-T3 lineage. Retractions I am making from prior session opinions:

| Retracted framing | What was said | Why it was overconfident | Where the correction lives now |
|---|---|---|---|
| "face_scale_jitter@0.50 is the load-bearing anti-shortcut single lever" | Memory `project_face_scale_jitter_load_bearing.md` (2026-04-30, P14 sister-variant) | The 2026-04-30 evidence was on a single base (P14 bundle vs P14 jitter-only). T6/T7 results extend that to a 2nd and 3rd base and show non-generalization — jitter is load-bearing AS A REPLACEMENT FOR a bad bundle, not AS AN ADDITIVE LEVER to clean bases. The original memory entry should be amended. | This doc §3.2; recommend amending memory entry `project_face_scale_jitter_load_bearing.md` to scope the load-bearing claim to "P14 bundle replacement" and note refutation as "composable lever". |
| "T5C is the cleanest single-experiment test of classifier-capacity-fixes-chronic_6" | Yaml header comments in `R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.yaml` + thread 2026-05-11 update | The scorecard verdict supports the proposition directionally on n=50 dor cohort (0.06 vs T4 0.18) but I claimed it would resolve the question. Resolving it requires CPU atlas re-run on T5C features (not done in this session). The scorecard alone gives evidence; it does not give a verdict on the mechanism. | This doc §3.1 (tagged MEDIUM); recommend CPU follow-up: run the L11 atlas inv_mean recompute on T5C step3500 + step1500 to test whether chronic_6 inv_mean recovered. |
| "T7 (T4 + jitter) is the right Slot-3 use" | Thread entry 2026-05-11, before scorecard landed | At dispatch time the CPU head_retrain β-outcome was already in, refuting the "head retrain at scale" Slot-3 alternative; T7 held by default. The scorecard now shows T7 ckpts ranked 8, 10, 12 (all below the recall floor). T7 was not the right Slot-3 use; a single-step retry of T5C with different hidden_dim values, or a T5C × different attachment layers, would have been higher EV. | This doc §3.2 + Open loops in §10; recommend the next packet bet investment toward T5C variants, not jitter variants. |

**Lesson for future me / next agent**: when extending a memory-recorded finding to a new context, treat the recorded finding as "this lever worked HERE with THIS combination" — not as "this lever works". Single-base ablations don't establish composability. The eval folder pattern of pre-stating the hypothesis + close criterion + falsifying outcome (a la T4 OPINION doc §7) is what prevents this kind of overreach.

## 6. Confidence-tiered claims table — see §4 above

## 7. Proposed next steps (options, not prescriptions)

Per AGENT_GUIDE Rule 3 (CPU-first-then-GPU), I list CPU options first.

### Investigation (CPU-only, $0)

1. **L11 atlas inv_mean recompute on T5C ckpts**: re-run the per-axis triptych inv_mean from A3 on T5C step1500 + step3500 + step3750. Tests §3.1 mechanism — if T5C step3500 has chronic_6 inv_mean ≥ P8A (vs T4 step10500's −0.0315), classifier capacity is the lever; otherwise §3.1 is refuted. ~3 hours CPU/MPS, no GPU.
2. **Per-frame catch-disagreement on lockbox: T5C step3500 vs P8A**: of the lockbox fakes T5C catches that P8A misses, what are their characteristics? Are they the chronic_6 cohort, or a different subset? §9 pattern from `project_job9_disagreement_audit_2026-05-04`. ~1h.
3. **T5C training trajectory: step1500 → step3500 score-distribution audit**: plot T5C scores on lockbox reals (n=1361) at intermediate ckpts step2000, step2500, step3000. Tests §3.3's "the lockbox-clean window is narrow in step" claim with more than 2 samples. ~1h CPU per ckpt × 3 = 3h if extra ckpts exist; else this is gated on saving more T5C ckpts in a follow-up training run.

### Training (GPU spend)

4. **T5C × hidden_dim sweep** (~$70-100 if launched as a small 3-arm sweep): hidden_dim ∈ {512, 1024, 2048} with same other config; rank by lockbox_real_fpr at calibrated τ. Direct test of "is 1024 a local optimum, or is there a larger sweet spot?". Could be reduced to one variant + reuse of the T5C 1024 ckpt for the third arm.
5. **T5C × attachment layer** (~$70): run T5C recipe but attach GRL at L6 instead of L11. Tests whether attaching the GRL deeper-in-the-tower changes the dev_macro / lockbox_real_fpr balance.

### Deployment (no spend)

6. **Ship T5C step3500 as deployment candidate alongside P8A**: T5C step3500 catches 1.7× more lockbox fakes than P8A (0.66 vs 0.39) at only +0.0095 absolute `lockbox_real_fpr` (FACTS §7.2). The v3-fix policy ranks P8A first because of the tiebreak metric, but the policy may not match the user's actual deployment cost function (false-positive cost vs false-negative cost). If FN cost > 15× FP cost, T5C step3500 is the better operational choice.
7. **Park T6 / T7**: both lever variants fail the recall floor on every sampled step. No further investment unless a structurally different jitter recipe (different magnitude, different base, different aug stack interaction) is proposed.

### Status quo

8. **Ship P8A and continue investigating**: P8A retains contract rank-1; this is the conservative move. Investigation budget goes to the L11 atlas recompute (option 1) before authorizing more GPU.

## 8. Counter-experiments that would falsify the mechanism in §3.1

If the "classifier-capacity" mechanism for T5C is wrong, one of these would show it:

1. **L11 atlas recompute on T5C step3500** (option 1 above): if T5C step3500's chronic_6 inv_mean is still ≤ P8A's (i.e., the regression A3 found on T4 persists in T5C), the lockbox improvement is not coming from chronic_6 invariance and §3.1's mechanism is refuted.
2. **T5C step1500 atlas inv_mean**: if step1500 has BETTER chronic_6 inv_mean than step3500 (despite catastrophic lockbox), the link between atlas inv_mean and lockbox behavior is decoupled even within a single training run.
3. **Per-source disagreement on `teams_real_dor_dev` (n=50)**: which 3 of 50 reals does T5C catch that P8A's 4 of 50 doesn't? If the 3 caught by T5C are a different subset (i.e., T5C and P8A make different mistakes on the same cohort), the "T5C is uniformly better on chronic_6" reading is wrong; it's "T5C trades one chronic-6 mistake mode for another".

## 9. Open questions I cannot answer from this data

- Why does T5C step1500 catastrophically regress `lockbox_real_fpr` (0.3204) while step3500 has it at 0.0279? Both are mid-training; the recipe is identical; the only difference is 2000 more optimization steps. Is the lockbox-clean window narrow, or wide and step1500 is an outlier?
- Why does T6 step1500 catch 87% of lockbox fakes (the highest of any scored ckpt) while T6 step3500 / step10250 catch only 23% / 36%? Is this a τ-calibration artifact (T6 step1500 has lower τ = 0.7811 vs T6 step3500's 0.9374), or a real trajectory effect?
- Is there a different jitter magnitude that would compose with T3/T4? The 0.50 magnitude is the only one I tested; the failure may be magnitude-specific not lever-specific.
- The v3-fix policy ranks on lockbox_real_fpr after gate-passing. Is this the user's actual operating preference, or is it an artifact of the policy design? T5C step3500 catching 1.7× more lockbox fakes at +0.0095 FPR may be operationally preferable to P8A's profile.

## 10. Suggested CPU jobs not yet run

In priority order, these would update my belief regardless of which direction they point:

1. **L11 atlas inv_mean recompute on T5C step3500 + T5C step1500 + T5C step3750** (option 1 above): direct test of §3.1 mechanism. ~3h CPU.
2. **Per-source disagreement matrix on lockbox fakes**: which lockbox fakes does each all-pass ckpt catch? n=253 fakes × 6 ckpts. Tests whether each ckpt's lockbox capture is the same set + extra, or genuinely different sets (= ensemble potential). ~1h CPU.
3. **T5C step3500 face_size axis FPR at calibrated τ**: re-runs the §A-style per-axis-bin FPR audit on T5C, like the 2026-05-10 robustness diagnostics did for P8A/E2B/T3_SLOT1. ~1h CPU.

## 11. Reviewer guidance

If you (the reviewer / next agent) are picking this up cold:

1. Read `RESULTS_FACTS_2026-05-12.md` first. Form your own view on what the 12-ckpt ranking + per-suite numbers imply.
2. Only then read §3 of this doc (the proposed mechanisms). Compare against your independent view.
3. If you disagree with §3.1 or §3.3, the counter-experiments in §8 are designed to be cheap CPU/MPS ways to discriminate; pick the one that most cleanly distinguishes our two views.
4. §5 self-correction log is load-bearing — I retracted the face_scale_jitter "composable lever" framing and the "T5C resolves classifier-capacity question" framing. Memory entry `project_face_scale_jitter_load_bearing.md` needs amendment to scope the original finding.
5. Pillar-1 reading (deployment Q): T5C step3500 catches 1.7× more lockbox fakes than P8A at +0.0095 absolute `lockbox_real_fpr`. The v3-fix policy ranks P8A 1st because of the tiebreak metric, not because P8A is operationally better. Whether to ship T5C step3500 in place of (or alongside) P8A is the user's call, predicated on FP/FN cost ratios that the v3-fix policy doesn't make explicit.
