# External Advisor Pack — 2026-05-06

> Self-contained pack to give to an external advisor (an LLM session in a different system, or a human reviewer with no repo access). Total size ~5 MB.

## Contents

```
external_advisor_pack_2026-05-06/
├── README.md                         ← this file
├── ARCHITECTURE.md                   ← model + SVD + paper reference + variants
├── FACTS.md                          ← bias-stripped factual snapshot of project state + 7 probes + R13 history
├── PROMPT_FOR_ADVISOR.md             ← the prompt the user will give to the external advisor
├── data/
│   ├── probe1_xinhe/                 ← Xinhe cross-day audit raw outputs (152 frames × 24 IQ axes)
│   ├── probe2_dor_drift/             ← Dor 22× drift mechanism raw regressions
│   ├── probe3_amp_phase/             ← FFT amplitude vs phase fake-signal localization (5000 frames)
│   ├── probe4_paired_features/       ← 275 paired (raw, teams) viso pair feature distances
│   ├── probe6_fourier_bands/         ← 16-band shortcut-vs-signal AUC partitioning
│   ├── probe7_per_layer/             ← 12-resblock × 3-ckpt × 2-substrate cosine + AUC tables
│   └── identity_browser/             ← consolidated dataset (14,626 rows × 15 cols)
```

## How to use

1. **Bundle**: `tar -czf external_advisor_pack_2026-05-06.tar.gz external_advisor_pack_2026-05-06/` from the parent directory. Send the tarball, or just zip the folder.
2. **Tell the advisor**: read `ARCHITECTURE.md`, then `FACTS.md`, then load any of the CSVs in `data/` they want to inspect more carefully. Then follow the question in `PROMPT_FOR_ADVISOR.md`.
3. **Get their answer**, then bring it back to the in-house workflow (see `THIRD_PROMPT_INGEST_EXTERNAL.md` in `docs/relaunch_handoffs/` for what to do with it).

## What this pack does NOT contain

Deliberately omitted to preserve the advisor's independence:
- The team's ranked list of next-step recommendations (`ANTI_SHORTCUT_TECHNIQUES_RANKED_2026-05-06.md`).
- The team's interpretation/synthesis blocks (the 2026-05-06 sub-sections in `processing_signature_shortcut.md`, the FINDINGS.md docs, the MODEL_SCORES_FINDINGS interpretation).
- The team's memory entries (cross-references and prior-agent framings).

The advisor is asked to form an opinion from data + architecture + the open question — not from prior synthesis.

## Verifying the pack is bias-clean

The advisor should challenge anything in `FACTS.md` that:
- Asserts a "should" without a CSV/JSON/memory citation.
- Frames a measurement with an interpretive verdict (e.g., "GREEN", "MIXED", "deployment-grade") rather than the underlying number.
- Implies a recommendation without quantifying alternatives.

The team has tried to keep the pack to data + cited measurements + open questions only. If the advisor finds bias, that's a useful flag.

## After the advisor responds

Their response feeds into the in-house workflow's third prompt — see the user's notes / `THIRD_PROMPT_INGEST_EXTERNAL.md` for the synthesis step.
