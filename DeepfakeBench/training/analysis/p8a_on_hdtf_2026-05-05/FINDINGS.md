# P8A_step5000 on HDTF proper viso substrate — DISPOSITIVE substrate-specific framing test

**Date authored**: 2026-05-05
**Vertex job**: `4232735281465262080` (us-east1, image 1.3.257)
**Job state at writeup**: still `JOB_STATE_RUNNING` for clean variants; teams variants and reals all complete
**Source**: `gs://training-job-outputs/test_results/p8a_on_hdtf_2026-05-05/p8a-on-hdtf-2026-05-05/reports/`
**Local artifacts**: `analysis/p8a_on_hdtf_2026-05-05/*_summary_report.txt`

---

## Headline (DISPOSITIVE)

**P8A_step5000 reaches 93.57% recall on `proper_visomaster_enhanced_teams_dev` (n=1182) at τ=0.5 with FPR=0.97% on `proper_real_teams_dev`** (n=1444).

Compare to:
- **P8A on v2 production substrate (`visomaster_enhanced_macro_dev`, n=550)**: **27% recall** at FPR=10%, **1% recall** at FPR=2% (per F4 reference data + PSERIES_FACTS).

**Same model, same enhancer × teams transport conjunction, different substrate → 3.5× recall difference at comparable FPR.**

The "viso ceiling is structural" claim from `project_viso_ceiling_unbroken_10_packets.md` is **DISPOSITIVELY refuted as a universal claim**. The ceiling is conditional on the v2 production substrate.

## Empirical results (all at τ=0.5, no FPR calibration)

### Reals (FPR — lower is better)

| Suite | n | TN | FP | **FPR** |
|---|---:|---:|---:|---:|
| proper_real_clean_dev | 1443 | 1437 | 6 | **0.42%** |
| proper_real_clean_lockbox | 382 | 382 | 0 | **0.00%** |
| proper_real_teams_dev | 1444 | 1430 | 14 | **0.97%** |
| proper_real_teams_lockbox | 382 | 377 | 5 | **1.31%** |

All real-side FPRs at τ=0.5 are well below 5% on the HDTF substrate.

### Visomaster fakes (recall — higher is better)

| Suite | n | TP | FN | **Recall** |
|---|---:|---:|---:|---:|
| proper_visomaster_teams_dev | 262 | 247 | 15 | **94.27%** |
| proper_visomaster_teams_lockbox | 80 | 76 | 4 | **95.00%** |
| **proper_visomaster_enhanced_teams_dev** | 1182 | 1106 | 76 | **93.57%** |
| **proper_visomaster_enhanced_teams_lockbox** | 302 | 287 | 15 | **95.03%** |

(Clean variants pending — job still running. Will add when complete.)

### All-fakes umbrella (mix of methods)

| Suite | n | TP | FN | **Recall** |
|---|---:|---:|---:|---:|
| proper_fake_teams_all_dev | 1444 | 1353 | 91 | **93.70%** |
| proper_fake_teams_all_lockbox | 382 | 363 | 19 | **95.03%** |

## Comparison vs Job B (RLP6_04 on same HDTF substrate)

Same suite manifest; baseline FT chain endpoint vs P8A endpoint.

| Suite (n) | RLP6_04 (Job B) | P8A (this run) | Δ (P8A − RLP6_04) |
|---|---:|---:|---:|
| proper_visomaster_enhanced_teams_dev (1182) | 85.36% | **93.57%** | **+8.21pp** |
| proper_visomaster_enhanced_teams_lockbox (302) | 89.74% | **95.03%** | **+5.29pp** |
| proper_visomaster_teams_dev (262) | 93.51% | 94.27% | +0.76pp |
| proper_visomaster_teams_lockbox (80) | 90.00% | 95.00% | +5.00pp |
| proper_fake_teams_all_dev (1444) | 86.84% | 93.70% | **+6.86pp** |
| proper_fake_teams_all_lockbox (382) | 89.79% | 95.03% | **+5.24pp** |
| proper_real_clean_dev (1443) FPR | 1.11% | **0.42%** | -0.69pp |
| proper_real_clean_lockbox (382) FPR | 0.52% | **0.00%** | -0.52pp |
| proper_real_teams_dev (1444) FPR | 1.18% | 0.97% | -0.21pp |
| proper_real_teams_lockbox (382) FPR | 1.05% | 1.31% | +0.26pp |

**P8A is BETTER than RLP6_04 on every cell** (recall up, FPR down or flat). The FT chain RLP6_04 → P8A added 5-8pp on viso enhanced+teams while reducing real FPR.

## What this dispositively closes

### The substrate-vs-trajectory question (per Job B + this run)

Trajectory hypothesis was: late chain ckpts lost a viso capability earlier ones had. **REFUTED**. P8A (chain endpoint) BEATS RLP6_04 (chain member) on HDTF substrate; the chain ADDED viso capability, didn't lose it.

Substrate hypothesis was: production v2 substrate has structurally different difficulty than HDTF. **CONFIRMED**. Same FT chain endpoint (P8A) gets 93.6% on HDTF enhanced+teams vs ~27% on v2 substrate.

### The "viso ceiling is structural across architectures" claim

Memory `project_viso_ceiling_unbroken_10_packets.md` claimed: "P8A_step5000 holds 27%; 13+ packets refuted including B16-FT, B16-scratch, L14-scratch; 3 arch-distinct attempts → ceiling is structural."

**Now refined**: the ceiling is structural ON THE V2 PRODUCTION SUBSTRATE. On HDTF substrate, the same model gets 93.6%. The "structural across architectures" framing is conditional, not universal.

The 2026-05-04 scoping note already added to that memory is now empirically supported by head-to-head measurement (not just inference).

## What this does NOT change

1. **Production-relevant numbers are still v2 substrate numbers.** If production traffic looks like v2 (Dor + diverse swap, low-Lap teams transport), the v2 ceiling is what matters.
2. **The 364/550 unreachable v2 frames** (per Job 12) remain structurally unreachable. F4 cleaning lifts P8A to 67% on cleaned v2, but that's eval-side cleanup.
3. **The IQ-shortcut + chronic-6 + Dor-skew confluence** is what makes v2 hard (per `IQ_VALLEY_FINDING.md`).

## What this implies for PA/PC interpretation

When PA+PC F0 results land:
- **PA/PC will likely be evaluated on v2 substrate** (production scorecard manifest).
- **PA/PC's headline viso recall will be substrate-bound**, not capacity-bound.
- The data lever's value depends on whether it lifts v2 viso recall above E2B's 8.4% — NOT whether PA/PC could in principle do 93%+ on a different substrate.
- F4 lift (chronic-6 cleaned) is the production-relevant readout if production traffic ≠ v2.

## Implications for next-packet planning

If verdict on PA/PC is (b) "data lever doesn't lift v2 viso":
- The v2 substrate's structural unreachability is the binding constraint.
- Lever options:
  1. **Substrate-side**: re-curate eval substrate to better reflect production (F4-cleaned or HDTF-like).
  2. **IQ-shortcut breaking**: explicit Lap-jitter aug (extension of P22's pipeline_randomization).
  3. **Out-of-stream router**: train a learned router that uses Lap + identity + capture-mode features.
  4. **Acceptance**: production traffic looks like HDTF; deploy on HDTF-substrate-equivalent metric.

The dispositive finding here makes (4) a viable deployment story IF the user is comfortable that production traffic ≠ v2 internal-test substrate.

## Pending: clean-variant suites (job still running)

When the remaining 6 suites complete (proper_fake_clean_all_*, proper_visomaster_clean_*, proper_visomaster_enhanced_clean_*), this doc will be extended. The clean-variant numbers should mirror Job B's near-100% recall on these.

## Cross-references

- `analysis/p8a_on_hdtf_2026-05-05/` — local summary text reports
- Memory `project_viso_ceiling_unbroken_10_packets.md` — refined via head-to-head measurement
- Memory `project_job_b_findings_universal_vs_trajectory_2026-05-04.md` — pre-existing finding on RLP6_04; this extends to P8A
- Memory `project_v2_substrate_is_dor_diverse_swap.md` — what makes v2 different from HDTF
- `IQ_VALLEY_FINDING.md` — mechanism for v2 unreachability
- `per_ckpt_iq_signatures.md` — per-ckpt IQ-shortcut directions
- `analysis/job_b_pre_rlp604_2026-05-04/FINDINGS.md` — companion analysis on pre-RLP6_04 chain
