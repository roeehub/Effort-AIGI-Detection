# Per-checkpoint IQ-shortcut signatures on v2 viso fakes (n=550)

**Date authored**: 2026-05-05 (during PA+PC eval wait)
**Source**: `analysis/p8a_signature_decomposition_2026-05-05/viso_cohort_assignments.csv` (existing cached scores)
**Purpose**: ground-truth the IQ-shortcut-direction-per-ckpt assumption underlying the IQ-valley framing

---

## Pearson r per ckpt on v2 viso fakes (550 samples)

| Ckpt | r(score, laplacian_var) | r(score, luma_mean) | r(score, sobel_edge) | Interpretation |
|---|---:|---:|---:|---|
| **P8A_step5000** | **+0.508** (p=1.8e-37) | -0.174 | +0.514 | **Sharper = more fake** (canonical shortcut, matches `project_image_quality_shortcut`) |
| **E2B_top_n_step3200** | **-0.254** (p=1.6e-9) | +0.119 | -0.364 | **Sharper = LESS fake** (INVERTED, matches `project_iq_gating_viability_2026-05-04`) |
| **E3_top_n_step6600** | **-0.011** (p=0.79) | -0.203 | -0.100 | **Lap-agnostic** (uses different signal — possibly luma) |

Three architecturally-distinct ckpts have THREE DIFFERENT IQ-shortcut directions on the same viso fakes.

## Per-cohort score medians (where each ckpt sits in its score distribution)

| Cohort | n | P8A median | E2B median | E3 median | Lap p50 |
|---|---:|---:|---:|---:|---:|
| A: caught all | 25 | 0.97 | 0.98 | 0.99 | 13 |
| B: P8A+E3, not E2B | 26 | 0.97 | 0.23 | 0.99 | 61 |
| C: P8A only | 93 | 0.88 | 0.03 | 0.20 | 85 |
| **D: missed all** | **364** | 0.03 | 0.02 | 0.03 | 33 |
| W: E2B+E3, not P8A | 4 | 0.14 | 0.87 | 0.99 | 16 |
| X: E2B only | 13 | 0.12 | 0.60 | 0.49 | 15 |
| Y: E3 only | 21 | 0.11 | 0.04 | 0.96 | 18 |
| Z: P8A+E2B, not E3 | 4 | 0.87 | 0.60 | 0.54 | 149 |

**The Cohort A is the most striking**: very low Lap (13) but caught by all 3 ckpts at high score. These frames must have a strong fake signal that overrides each ckpt's IQ-shortcut. Likely a different mechanism (face artifact, identity, etc.) that all 3 ckpts pick up.

**The Cohort D** (uncaught) has midrange Lap (33). At this Lap:
- P8A's +Lap shortcut would predict score ≈ 0.5 (mid). Actual P8A median = 0.03 (very low).
- E2B's -Lap shortcut would predict score ≈ 0.4-0.5. Actual E2B median = 0.02 (very low).
- E3's r=0 means Lap doesn't matter; some other axis. Actual E3 median = 0.03 (very low).

Why are D scores so low if the IQ shortcut alone can't predict them? Hypothesis: the fake-signal in D frames is structurally weak; the ckpts default to a low-fake prior in the absence of strong IQ signal (P8A) or strong inverted-IQ signal (E2B). The 364 are essentially "models can't detect the fake here, regardless of which shortcut they're using."

## Implication: v2 substrate has THREE distinct miss zones, ckpt-specific

A score-fusion ensemble can in principle combine these by using different ckpts for different IQ regions:
- LOW Lap (cohort A, W, X, Y range, ~63 frames): E2B/E3 sweet spot
- MID Lap (cohort D, 364 frames): nobody's sweet spot — this is the unreachable
- MID-HIGH Lap (cohort B, 26 frames): P8A+E3 sweet spot
- HIGH Lap (cohort C, 93 frames): P8A's specialty
- VERY HIGH Lap (cohort Z, 4 frames): P8A+E2B

Per Job 12: oracle combiner gets 77% by knowing labels per frame. The IQ-conditioned routing would in principle achieve ~33.8% (current loose-OR ceiling) without needing labels — that's the same number, so per-frame IQ-routing isn't free of any improvement; it's the "loose OR" already.

## What would lift the ceiling above 33.8%?

The 364 frames need a NEW catch direction that none of P8A/E2B/E3 has. Options:
1. A new ckpt with r(score, lap) tuned for midrange (not necessarily zero — perhaps a non-monotonic relationship).
2. Out-of-stream features (face artifact detector, identity-based filter) that complement the IQ axis.
3. An aggressive IQ-jitter-trained model that becomes Lap-invariant entirely.

P22 step8k went from r=+0.51 to r=-0.11 on viso (per FACTS doc 6.5.11). Its viso recall at FPR=2% was 4.4%. So **flattening the IQ-r partially helps but doesn't break the ceiling**.

## Implications for PA/PC

If PA's r(score, lap) on viso ≈ E2B's -0.25 → PA inherits E2B's miss profile, misses cohort D.
If PA's r ≈ +0.51 (P8A-like) → unlikely given FT-from-E2B; would need substantial drift.
If PA's r ≈ 0 (P22-like) → would suggest data lever broke the shortcut; unlikely given P22 was the only ckpt that achieved this and via aggressive aug.

**Most parsimonious prediction**: PA r ∈ [-0.3, -0.1]. Catches cohort A subset (low Lap), misses cohort D (mid Lap). F0 viso recall ≈ E2B's 8% ± 5pp. Consistent with my pre-registered 5-15% prediction.

For PC: codec aug degrades IQ during training. If PC's r becomes MORE negative (e.g., -0.4), PC's sweet spot is even narrower (only very-low Lap caught). PC would do WORSE than PA on cohorts B/C/D. Expected PC F0 viso ≤ PA F0 viso.

If PC's r becomes LESS negative (e.g., -0.05) → PC less Lap-dependent → catches midrange. WOULD BE A SURPRISE; would warrant memory entry.

## Computing PA/PC r when frame reports land

When PA/PC frame reports for `visomaster_enhanced_macro_dev` are downloaded, run:

```python
import pandas as pd
from scipy.stats import pearsonr
ck = "PA_TOP_N_STEP5600"
viso_df = pd.read_csv(f"analysis/pa_pc_eval_2026-05-05/raw_reports/visomaster_enhanced_macro_dev_{ck.lower()}_frames_report.csv")
cohort_df = pd.read_csv("analysis/p8a_signature_decomposition_2026-05-05/viso_cohort_assignments.csv")
merged = viso_df.merge(cohort_df[["frame_path", "laplacian_var"]], on="frame_path")
r, p = pearsonr(merged["frame_prob"], merged["laplacian_var"])
print(f"{ck}: r(score, lap) = {r:+.3f} (p={p:.2e})")
```

This is the dispositive test of whether PA inherits E2B's IQ profile.

## Cross-references

- `analysis/p8a_signature_decomposition_2026-05-05/per_cohort_iq_profile.csv` — full per-cohort IQ stats
- `analysis/p8a_signature_decomposition_2026-05-05/cohort_vs_A_ks_tests.csv` — KS tests vs Cohort A
- Memory `project_image_quality_shortcut`
- Memory `project_iq_gating_viability_2026-05-04`
- Memory `project_p22_cpu_followups_reframe_2026-05-02` — P22 r-flattening detail
- This file's companion: `IQ_VALLEY_FINDING.md`
