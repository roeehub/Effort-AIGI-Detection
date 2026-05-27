# Stage A CPU diagnostics — 2026-05-12

> **For an independent fresh agent picking this up cold**: read the FACTS docs first (§1 below); form your own view; then optionally read the OPINIONS doc (§2) to compare.
>
> This folder bundles 4 CPU jobs run 2026-05-12 to answer two questions the user posed:
>
> 1. Do we have any checkpoint that is materially better than P8A under the three pillars (`MODEL_GOALS.md`)?
> 2. Does the answer to (1) give us a direction toward a new single model that combines T5C step3500's catch power with P8A's substrate invariance?

---

## 1. Reading order — FACTS docs (form your own view here)

The numbers are the same across this folder and `../cpu_diagnostics_2026-05-11_t67_t5c_probe/`; the layout splits along the CPU job that produced them, not along subject matter. Read all 5 before forming a view.

1. **[`../cpu_diagnostics_2026-05-11_t67_t5c_probe/CKPT_SCORING_FACTS_2026-05-12.md`](../cpu_diagnostics_2026-05-11_t67_t5c_probe/CKPT_SCORING_FACTS_2026-05-12.md)** — T5C/T6 candidate ckpt scoring on the 952-frame may6/may5/dor/chronic_6/Roy_D cohort matrix (the dispositive data for "better than P8A").
2. **[`../cpu_diagnostics_2026-05-11_t67_t5c_probe/INV_MEAN_FACTS_2026-05-12.md`](../cpu_diagnostics_2026-05-11_t67_t5c_probe/INV_MEAN_FACTS_2026-05-12.md)** — L11 atlas inv_mean per substrate slice for 7 ckpts; tabulates the partial chronic_6 invariance recovery (the headline mechanism finding).
3. **[`STAGE_A_FACTS_2026-05-12.md`](STAGE_A_FACTS_2026-05-12.md)** — 3 probes on the 13,636-frame contract suites: per-IQ-bin policy + ensemble policy grid + per-frame disagreement audit.
4. **[`JOB1_L11_DISTANCE_FACTS_2026-05-12.md`](JOB1_L11_DISTANCE_FACTS_2026-05-12.md)** — per-frame L2 + cosine distance from P8A's L11 CLS features on the 800-frame triptych.
5. **[`JOB2_OVERFIRE_FACTS_2026-05-12.md`](JOB2_OVERFIRE_FACTS_2026-05-12.md)** — T5C step3500 vs P8A real-side overfire identity concentration (95.4% on top 5 identities).
6. **[`JOB3_INFRA_AND_RULE1_AUDIT_FACTS_2026-05-12.md`](JOB3_INFRA_AND_RULE1_AUDIT_FACTS_2026-05-12.md)** — codebase infrastructure inventory + `AGENT_GUIDE.md` Rule 1 grep audit confirming the 3 proposed levers are untested.

## 2. Project context (read after forming an independent view, before OPINIONS)

7. **[`../../docs/packet_retrospectives/MODEL_GOALS.md`](../../docs/packet_retrospectives/MODEL_GOALS.md)** — three pillars + B16-only + **NO ENSEMBLE hard rule** + the promotion bar (beat E2B + don't regress P8A's chronic-FP behavior + ≥90% lockbox recall + HDTF cross-substrate within 5pp of E2B).
8. **[`../../docs/packet_retrospectives/AGENT_GUIDE.md`](../../docs/packet_retrospectives/AGENT_GUIDE.md)** — 6-rule contract: validate-before-suggest, read failure modes, CPU-first-then-GPU, viewer integration, FACT-vs-OPINION split, plus bucket-manifest-verification.

## 3. OPINIONS doc (read after forming your own view)

9. **[`STAGE_A_PLUS_PACKET_PLAN_OPINIONS_2026-05-12.md`](STAGE_A_PLUS_PACKET_PLAN_OPINIONS_2026-05-12.md)** — interpretation of the FACTS + 3-slot GPU plan + falsification criteria for each slot + memory citations + self-correction log.

You are explicitly free to disagree with the OPINIONS doc. If you do, the falsification criteria in §2.1-§2.3 of the OPINIONS doc are designed to be cheap experiments to discriminate competing views.

## 4. Output artifacts (reproducible scripts + CSVs)

All CSV outputs under `outputs/`:

```
disagreement_frames.csv              — Stage A Probe 3 frames where |P8A−T5C|>0.5
ensemble_policy_grid.csv             — Stage A Probe 2 raw policy × suite × τ grid
ensemble_summary_tau{0.50,0.70,0.85,0.90}.csv  — aggregates per τ
l11_distance_per_frame.csv           — Job 1 per-frame L2 + cosine distances
per_iq_bin_policy.csv                — Stage A Probe 1 per-IQ-axis × cohort × ckpt × τ table
real_with_overfire_flag.csv          — Job 2 real-cohort frames with overfire flag + IQ profile
unified_frame_matrix.csv             — 13,636-frame contract matrix joined to IQ atlas
```

Reproducible Python scripts:

```
run_stage_a.py            — Stage A Probes 1, 2, 3
synthesize_facts.py       — Stage A aggregations + FPR-cal τ + specialist-routing test
job1_l11_distance_map.py  — Job 1 L11 distance map
job2_overfire_audit.py    — Job 2 overfire population audit
```

Re-run: `cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training && python3 analysis/cpu_diagnostics_2026-05-12_stage_a/<script>.py`

Sibling folder (companion outputs):

```
../cpu_diagnostics_2026-05-11_t67_t5c_probe/outputs/per_ckpt_cohort_scores.csv  — 952-frame may6/dor/chronic_6/Roy_D matrix
../cpu_diagnostics_2026-05-11_t67_t5c_probe/outputs/per_ckpt_inv_mean.csv       — 7-ckpt L11 atlas per-substrate inv_mean
../cpu_diagnostics_2026-05-11_t67_t5c_probe/outputs/L11_inv_mean_summary.csv    — full-triptych inv_mean summary
```

## 5. Authoring note

All FACTS docs in this folder were authored 2026-05-12 by a single agent, in a single session, after running the CPU jobs. The OPINIONS doc was authored by the same agent in the same session. **The pass-1 / pass-2 independence is NOT guaranteed.** A truly independent reading requires a fresh agent (or fresh you in a later session) to read the FACTS docs in §1 + project context in §2 BEFORE reading the OPINIONS doc in §3.
