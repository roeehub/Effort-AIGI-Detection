# Shortcut Learning Audit — 2026-04-26

## Purpose

The Effort Teams deepfake detector achieves AUC ≥ 0.99 / EER ≤ 0.02 on training-time validation but fails on deployment-relevant probes (camera-signature shortcut, enhanced-fake recall collapse, capture-session outliers). After ~10 days of FT-based attempts to fix this, no approach has worked.

This audit was commissioned to surface — comprehensively, chronologically, skeptically — every piece of evidence and context an investigative agent would need to form an independent view of where the actual problem lives.

**The audit's purpose is NOT to propose solutions.** It is to organize information so a fresh agent can reason cleanly about the methodology problem.

## Structure

```
shortcut_learning_audit_2026-04-26/
├── README.md                              ← you are here
├── MASTER_REPORT.md                       ← read this first
└── sub_reports/
    ├── 01_experiment_lineage.md           — the FT chain, cumulative steps from CLIP base
    ├── 02_data_sources_and_bad_data.md    — sources, manifests, policies, leakage audit
    ├── 03_augmentation_architecture.md    — the per-class asymmetric router and the new symmetric branch
    ├── 04_shortcut_evidence_inventory.md  — every probe, contact sheet, anchor rescore, summary json
    ├── 05_measurement_apparatus.md        — GRL, promotion contract, lockbox, what we don't measure
    └── 06_strategic_doc_trail.md          — chronological history of strategic moves with FLAG markers
```

## How to read

1. Start with `MASTER_REPORT.md` sections 0–3 (10 min).
2. Read sub-report 04 to get the empirical foundation (45 min).
3. Read sub-reports 01, 02, 03, 05 in any order to fill in mechanism (~2 hours total).
4. Read sub-report 06 last for strategic context (30 min).
5. Return to `MASTER_REPORT.md` sections 4–10 to read the skeptical synthesis (45 min).

Total: ~4 hours for the full audit. ~90 minutes for the master + sub-report 04 if you only have one sitting.

## Key questions the receiving agent is being asked

(Full versions in `MASTER_REPORT.md` §8.)

1. **Where does the shortcut actually live?** CLIP backbone weights, R12g FT, RLP6_04 FT, the data composition itself, or some combination? What probes would settle this without retraining?

2. **What does a clean baseline look like?** What probe battery would certify a model as shortcut-free? What thresholds count as passing?

3. **Is the FT-from-RLP6_04 paradigm fundamentally broken?** If shortcut features are baked into the FT chain, can any FT-from-FT-from-FT approach fix it?

4. **Are our "wins" actually wins?** P8A "broke the camera-signature ceiling" — did it, or did it shift the failure mode? RLP6_04 "leader" — by what metric, audited how?

5. **What's the minimum probe battery** that would let us certify a model as shortcut-free before deployment?

6. **Should the existing data even be trusted?** The pre-relaunch training data has had 4904+ samples retroactively flagged as bad. R12g was trained on that data. RLP6_04 was FT'd from R12g. Is a clean-data retrain the actual prerequisite for any progress?

7. **The methodology question (the one that drove this audit):** How do we prove the model has no shortcut learning before we trust its metrics?

## Author

Claude (Opus 4.7, 1M context), in collaboration with Roee Dar. The author has participated in the most recent P10 packet planning and therefore has a vested interest in not declaring prior work wrong; the report tries to be explicit about this and should be read with that disclosure in mind.
