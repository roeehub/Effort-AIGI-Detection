# Independent critic review — task spec (2026-05-13)

## Who you are

You are an independent ML vision specialist reviewing a deepfake-detection project. You have no prior context about this codebase or the prior agents who worked on it. You are being asked to form **your own view** of where the project currently stands and what its next moves should be.

The user is concerned that mid-session agent opinions have accumulated and contaminated the next agent's reasoning. The wiki at `docs/packet_retrospectives/` was restructured to separate FACTS from OPINIONS exactly so a fresh reader can reach independent conclusions. **Use that discipline.** You are explicitly free — and encouraged — to disagree with prior agents if the evidence supports it.

## What you must produce

A single review doc at `analysis/cpu_diagnostics_2026-05-13_independent_critic_review/INDEPENDENT_CRITIC_REVIEW_2026-05-13.md` containing:

1. **§1 — Current standing assessment.** Where is the project relative to its goals? Cite file paths for every numerical claim. 200-400 words.
2. **§2 — Proposed GPU experiments.** A ranked list (1-4 experiments, your call). For each: hypothesis + falsifier + cost estimate + a one-line argument for why this is the right next bet. 300-600 words.
3. **§3 — Confidence calibration.** For each proposed experiment, an explicit confidence level (HIGH / MEDIUM / LOW) with one-line justification. Then a SEPARATE confidence assessment for "if all your proposed experiments succeed, does the project hit its stated goals?" (HIGH / MEDIUM / LOW). Be honest if your answer is LOW — that is more useful than performative optimism. 100-200 words.
4. **§4 — Where you disagree with prior agents.** Pull specific framings from `D1_D5_OPINIONS_2026-05-12.md`, `D1_D5_CRITIC_REVIEW_2026-05-12.md`, or `STAGE_A_PLUS_PACKET_PLAN_OPINIONS_2026-05-12.md` that you read in pass 2 and disagree with based on the FACTS evidence. Cite the disagreement specifically (doc:section). If you don't disagree with anything material, say so explicitly. 100-300 words.
5. **§5 — Self-correction log.** If your pass-2 reading changed your pass-1 view in any material way, document the change. If it didn't, say "no view change." 50-200 words.

## Reading protocol (mandatory)

### Pass 0 — Mission context
1. Read `docs/packet_retrospectives/AGENTS.md` (the docs contract).
2. Read `docs/packet_retrospectives/MODEL_GOALS.md` (the three pillars + NO-ENSEMBLE rule).
3. Read `docs/packet_retrospectives/AGENT_GUIDE.md` (6-rule contract; FORBIDDEN-words list).

### Pass 1 — FACTS only (form your own view)
DO NOT read any `*_OPINIONS_*.md`, `AGENT_PROPOSAL_*.md`, or `*_CRITIC_REVIEW_*.md` in this pass.

4. Read `docs/packet_retrospectives/STATE.md` (top section, refreshed 2026-05-12 late evening).
5. Read the last ~15 entries of `docs/packet_retrospectives/TIMELINE.md`.
6. Read `docs/packet_retrospectives/OPEN_LOOPS.md` (mechanically generated).
7. Read all 10 FACTS docs landed 2026-05-12 (paths in STATE.md FACTS table):
   - D1 through D10 at `analysis/cpu_diagnostics_2026-05-12_d{1..10}_*/D{1..10}_FACTS_2026-05-12.md`
   - Plus `analysis/train_overlap_audit_2026-05-12/TRAIN_OVERLAP_FACTS_2026-05-12.md`
   - Plus `analysis/cpu_diagnostics_2026-05-12_gcs_identity_audit/GCS_IDENTITY_AUDIT_FACTS_2026-05-12.md`
8. Read `docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md` — focus on `## Current stance` sections and the dated late-evening 2026-05-12 entry. Skip any "Implications" / "Current stance" passages that read as interpretation — those are the prior agents' synthesis, which you should reach independently first.
9. **Form your own pass-1 view.** Write §1 + §2 + §3 of your deliverable based on FACTS only.

### Pass 2 — OPINIONS (only after pass-1 §1-§3 are drafted)
10. Read `analysis/cpu_diagnostics_2026-05-12_d1_d5_synthesis/D1_D5_OPINIONS_2026-05-12.md` (planning-agent synthesis; §1 framing retracted, §3 4-slot plan).
11. Read `analysis/cpu_diagnostics_2026-05-12_d1_d5_synthesis/D1_D5_CRITIC_REVIEW_2026-05-12.md` (prior independent critic's review).
12. Read `analysis/cpu_diagnostics_2026-05-12_stage_a/STAGE_A_PLUS_PACKET_PLAN_OPINIONS_2026-05-12.md` (older planning doc; the user's first OPINION input for this session, partially superseded).
13. Compare your pass-1 view against these. Update §4 (disagreements) and §5 (self-correction log).

## Constraints

- **Forbidden words in your FACTS-citing claims** (per AGENT_GUIDE Rule 5): succeeds, fails, wins, loses, promotes, deployment-grade, unfortunately, remarkably, shortcut-aligned, lucky, confirmed, refuted. Use mechanical pass/fail against pre-stated bars. Your OWN view (§2-§5) can use stronger framings, but flag them clearly as interpretation.
- **Cite file:section for every numerical claim.** Don't say "P8A has 5% FPR"; say "P8A has 0.92% lockbox FPR at dev-cal 5% τ per `D4_FACTS_2026-05-12.md` §2."
- **No new analytics this pass.** Don't run new CPU jobs. Don't read code. Don't grep manifests. The 10 FACTS docs + STATE + TIMELINE + thread are sufficient.
- **NO-ENSEMBLE constraint is hard.** Per `MODEL_GOALS.md`: deployable target is a single model. Don't propose score-fusion, specialist-routing, or cascade-of-models. CPU-only diagnostic ensembles are fine; deployment-time ensembles are not.
- **CPU-first-then-GPU rule** (AGENT_GUIDE Rule 3): if you can name a CPU diagnostic that would change your mind about a proposed GPU bet, name it and propose it as Stage A before the GPU launch. The user has done 10 CPU jobs already this session; if you say "we need more CPU before GPU," justify it concretely.

## Project goals reminder (from MODEL_GOALS.md, paraphrased — read the doc directly)

Three pillars (all load-bearing, not just the first two):
1. **Fake recall** on target methods (deeplive, viso, teams pipelines).
2. **Low real FPR** (< 5%) on production-relevant real distributions.
3. **Robustness** across lighting / camera / codec / color / capture-substrate axes.

Hard constraint: **single model at deployment** (NO-ENSEMBLE).

Current production-deployed model is E2B. Current promotion-contract rank-1 anchor is P8A. Current strongest new candidate is T5C step3500 (rank-3, 1.7× more lockbox fake recall than P8A at +0.0095 absolute lockbox real FPR).

## What success for this review looks like

A reader of your deliverable in 1 month should be able to (a) understand where the project stood on 2026-05-13, (b) understand what experiments you proposed and why, (c) decide whether to run them, (d) measure whether they worked. Your confidence calibration in §3 is the most important section — it sets the user's expectations honestly.

If your conclusion is "we don't have enough information to confidently propose a GPU bet, and here's the CPU diagnostic that would change that" — that is a valid and welcome conclusion. The user has been bitten by over-confident GPU proposals before; honest LOW-confidence is more useful than performative HIGH-confidence.
