# Prior codec aug context (added 2026-05-05 post-prediction-registration)

**Why this file exists**: while waiting for PA/PC results to land, I (the agent) discovered a prior codec-aug experiment that should inform my reading of PC-codec. This was NOT in my pre-registered predictions; documenting it now to be honest about updating frame after registration.

## Prior experiment: codec_hedge (2026-04-27)

Source: `analysis/policy_reruns_2026-04-27/default/checkpoint_summary.csv` + `arena/checkpoint_maps/teams_target_domain.codec_hedge_2026-04-27.yaml` + `experiments/phase2_round13/R13_P10_SYM_on_P8A_codec_hedge.yaml`.

### What was tested

A side-by-side of FT-from-P8A with vs without codec aug:

- **C1_CANONICAL_VC_STEP2000** (run `6tzvre0k`): no codec aug; canonical "fixed-bug re-train" of the recipe.
- **C3_CODEC_HEDGE_VC_STEP2000** (run `bm7xxwqo`): same recipe + codec aug (`teams_codec_sim_p=0.60, teams_codec_sim_quality=[15, 55]`).
- **C3_CODEC_HEDGE_OOD_STEP2500** (same `bm7xxwqo` run, later step).

### Results at the 2026-04-27 contract scorecard policy

| Ckpt | dev_macro_recall | viso recall | deeplive recall | teams_fake_dev recall | lockbox_real_FPR | rank |
|---|---:|---:|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.300 | 0.136 | 0.239 | 0.526 | 0.018 | 1 |
| C3_CODEC_HEDGE_VC_STEP2000 | 0.426 | **0.215** | 0.461 | 0.604 | 0.031 | 2 |
| C3_CODEC_HEDGE_OOD_STEP2500 | 0.392 | 0.162 | 0.444 | 0.570 | 0.033 | 3 |
| C1_CANONICAL_VC_STEP2000 (no codec aug) | 0.421 | **0.227** | 0.451 | 0.586 | 0.038 | 4 |

(Note: τ-policy here predates v3; viso recall numbers are NOT directly comparable to the v3-era 0.011 baseline. But the **within-scorecard A vs B comparison of codec-aug effect is valid**.)

### Key reading

**Codec aug does NOT lift viso recall over its no-codec sister recipe** in this prior test:
- C1 (no codec): viso = 22.7%
- C3 (codec): viso = 21.5% (Δ = −1.2pp, slight regression)

Codec aug DID lift the lockbox FPR profile slightly, which moved C3 to rank 2 vs C1's rank 4 — but the viso-recall axis specifically was not improved.

### Implications for PC-codec interpretation

The prior codec test was on FT-from-P8A. PC-codec is FT-from-E2B. The FT-base shift is the new variable.

Hypotheses to consider:

1. **Codec aug is base-agnostic**: PC viso ≈ PA viso (±2pp), consistent with the prior C1 vs C3 finding.
2. **Codec aug is base-specific**: PC > PA on viso because E2B's IQ shortcut (inverted) responds differently to codec aug than P8A's IQ shortcut.
3. **The aug parameters matter**: PC's `policy: adaptive_mixture, p=0.5` is different from C3's `teams_codec_sim_p=0.60`. The C3 result might not generalize.

Of these, hypothesis 1 has the strongest prior evidence. Hypothesis 2 is plausible but speculative.

### Pre-prediction adjustment

My pre-registered prediction was "PC F0 viso recall: 8-20% (codec aug helps marginally on substrate IQ)." The codec_hedge prior suggests this was too optimistic — codec aug had a slight NEGATIVE effect on viso in the only prior single-axis test. A more honest pre-registration would have been: "PC F0 viso recall ≈ PA F0 viso recall ± 3pp, with no strong directional prior; codec aug's value-add is unclear given the C1/C3 evidence."

I am NOT amending the pre-registered predictions retroactively. I'm flagging this as "additional context I should have included pre-registration."

### What this means for the verdict

If PC viso F0 < PA viso F0: consistent with the codec_hedge prior; does NOT refute the codec aug as an axis (different base, different params), but is the most likely outcome.

If PC viso F0 ≈ PA viso F0: still consistent; the codec aug's effect on viso is small in either direction.

If PC viso F0 >> PA viso F0 (e.g., +10pp): would be a novel finding that conflicts with the codec_hedge prior; would warrant investigating whether E2B's inverted IQ shortcut is the mechanism.

### What needs to be documented post-results

- The C1 vs C3 prior should be cited as a "prior single-axis codec test" in the verdict doc.
- If PC outperforms PA on viso, the FT-base mechanism hypothesis becomes testable.
- If PC underperforms or matches PA on viso, the codec_hedge prior is reaffirmed and the codec aug as a viso-recall lever is ~empirically dead.

### Per AGENT_GUIDE Rule 1 (validate-before-suggest)

Per `docs/relaunch_handoffs/AGENT_GUIDE_2026-05-02.md` Rule 1, I should have surfaced this prior BEFORE participating in the PC-codec design and pre-registration. The PC-codec packet was launched by the prior agent (2026-05-04 evening); my role is interpreting the results, but this analysis should be informed by ALL prior tests of the codec aug axis. Adding this context retroactively is a partial mitigation, not a substitute for proper pre-registration.

The pre-registered file `PRE_LANDING_PREDICTIONS.md` notes which patterns of past tests I had cited; the codec_hedge experiment was missed and should be added to that list of prior tests for future predictions.
