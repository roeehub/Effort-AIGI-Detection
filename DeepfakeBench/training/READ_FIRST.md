# READ FIRST — fresh-agent onboarding — 2026-05-21

> If you are a fresh agent picking up this codebase: stop. Read this first.
> The auto-memory system has accumulated 80+ session-summary entries over 6
> weeks; several quantitative claims in those summaries are now wrong or
> partially wrong. Follow the reading order below and apply
> `docs/MEMORY_CAVEATS_2026-05-21.md` when reading older memory.

## Project in one paragraph

DeepfakeBench training for a Microsoft Teams deployment deepfake-detection
model. R13 is the current packet round (Phase 2, Round 13), running since
~mid-April 2026. The deployment goal is a fine-tuned CLIP-based binary
classifier that distinguishes real Teams call frames from face-swap fakes.
The production anchor checkpoint is **P8A step5000** (W&B `9lmvb5b4`); the
current best ship-candidate is **Slot A v2 step3500** (W&B `hp35c51p`).
Approximately 25 single-lever R13 packets have been tested; one is the
deploy anchor, one is the current ship-candidate, the rest were refuted or
shelved. See `docs/R13_LEVER_LEDGER_2026-05-21.md` for the full inventory.

## Reading order

### Layer 1 — current state (read these in order, ~30 min)

1. **`docs/R13_LEVER_LEDGER_2026-05-21.md`** — what's been tried, when, what
   the literal verdict was. FACTS-only table. Read this BEFORE you read any
   per-packet retro.
2. **`docs/MEMORY_CAVEATS_2026-05-21.md`** — which auto-memory claims have
   been invalidated or revised. Apply these before acting on any memory
   entry that mentions: per-substrate τ, "21pp lift", inference-side
   remediation, or pre-2026-05-21 verdict numbers.
3. **`HANDOFF.md`** — current state of the most recent overnight session
   (codec restoration triple launched + verdict). Contains pointers to all
   2026-05-21 artifacts.
4. **`analysis/iq_substrate_tau_2026-05-21/slot_a_v2_lockbox_pareto.md`** —
   the lockbox-calibrated Pareto comparison. Slot A v2 dominates P8A at
   every measured operating point. This is the most directly actionable
   number for "ship vs train" decisions.

### Layer 2 — key recent FACTS docs (~30-60 min)

Read only the ones relevant to the question you're trying to answer:

5. **`analysis/codec_restoration_gates_2026-05-21/VERDICT_FACTS_2026-05-21.md`**
   — most recent training experiment. 3 packets testing codec aug
   restoration; partial mechanism activity, no deployment upgrade. Important
   for understanding why "more codec aug" is not the answer.
6. **`analysis/iq_substrate_tau_2026-05-21/FACTS_2026-05-21.md`** — most
   recent deployment-side experiment. Per-frame IQ-feature τ refuted; ALSO
   contains the apples-to-apples comparison that revises older per-substrate
   τ claims.
7. **`analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/RESULTS_FACTS_2026-05-20.md`**
   — 7-job CPU evidence batch on Slot A v2. Quantifies the contract tiebreak
   noise (95% CI [-0.008, +0.010]; P=0.519 that P8A is truly better than
   Slot A v2 at the formal contract).
8. **`analysis/teams_account_natural_experiment_2026-05-19/TEAMS_ACCOUNT_NATURAL_EXPERIMENT_FACTS_2026-05-19.md`**
   — the per-Teams-account transport-shortcut finding. Same person, same
   moment, two Teams accounts → Δ prob_fake 0.168. Underlies several
   strategic options.

### Layer 3 — packet-specific retros (as needed)

`docs/packet_retrospectives/packets/*.md` — one file per packet family.
Read a specific one ONLY when the ledger row says "see this retro." Do not
read them all sequentially — they were written across 6 weeks and reflect
the framing of their authoring session.

`docs/packet_retrospectives/threads/*.md` — open-loop threads. Each is a
single issue tracked over time. Useful when an open loop is named in the
ledger or in `OPEN_LOOPS.md` and you need history.

### Layer 4 — auto-memory (with caveats)

`~/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/*.md`
— per-session summaries. These are OPINIONS, not facts. The index in
`MEMORY.md` is one line per memory; the bodies are paragraph-length.

Caveats:
- Several claims are wrong as originally framed; see
  `MEMORY_CAVEATS_2026-05-21.md` first.
- Memory entries are NOT timestamped consistently; some say "ship X" or
  "X is refuted" that reflect at-the-time framings now superseded.
- Treat memory as a hypothesis-generating source, NOT a fact source.

## Anti-loop checklist

Before proposing a packet, check:

| Question | Where to look |
|---|---|
| Has this exact lever been tried? | `R13_LEVER_LEDGER_2026-05-21.md` table |
| What was the literal verdict? | Ledger "Verdict" column + linked FACTS doc |
| Did the verdict use a contract that has since shifted? | `MEMORY_CAVEATS_2026-05-21.md` C7 |
| Does this proposal address the binding failure mode? | The binding failure modes as of 2026-05-21 are listed below |
| Is there a CPU experiment that would answer the question before GPU? | Default answer: yes, find it |

## Current binding failure modes (FACTS, frozen 2026-05-21)

1. **Per-Teams-account transport-shortcut** (2026-05-19): same person × same
   camera × different Teams account → Δ prob_fake 0.17 on T5C/Slot A v2.
   Codec aug partially closes (37-60% Δ reduction) but not deployment-grade
   and at may6 cost. NOT addressed by Slot A v2 vs P8A choice.

2. **Roy_D-cluster chronic-FP regression on Slot A v2 dev** (2026-05-20):
   Roy_D dev FPR 0.30 → 0.84 (+0.54) on Slot A v2 vs P8A. Roy_D is NOT in
   the contract lockbox cohort, so this is invisible to the formal contract
   but real on the deployment population.

3. **Dev/lockbox distribution mismatch within every substrate definition
   tried so far** (2026-05-21 new finding): per-mode τ (whether substrate =
   clip_capture_mode or IQ-feature cluster) does not reliably beat naive
   global τ at matched lockbox FPR. Implication: "per-account τ" via
   observable per-frame signals is structurally hard; the option that might
   still work is discrete per-account SDK metadata if Teams exposes it at
   runtime.

## Strategic options remaining (as of 2026-05-21)

In order of effort-to-information ratio:

A. **Compute Slot A v2's full 6-substrate Pareto curve** (~30-60 min MPS).
   Currently only teams_real/fake_lockbox measured. Adding viso/deeplive/teams_fake_dev
   would let the program reason about cross-domain trade-offs without retraining.

B. **Ask Microsoft Teams SDK whether per-account / per-client metadata is
   exposed at frame-receive time** (user-side, hours). If yes, per-account τ
   becomes a one-day deployment fix. If no, this branch is dead.

C. **Multi-account capture sweep** (user-side data task). Capture ~20-50
   frames per Teams account class (Roy D / Guest / free / edu / enterprise).
   Determines whether (i) lockbox is faithful to production, (ii) the
   per-account axis clusters cleanly across N accounts not just 2.

D. **Roy_D anchor pool extension training packet** (~$30 GPU + 60-90 min
   infra). Multi-step infra change (bucket prefix + frames upload + registry
   edit in `analysis/teams_pool_rescore.py`). Different mechanism from any
   tested lever. Highest-confidence remaining training-side option.

E. **Stop and ship Slot A v2 with a chosen operating point**. The lockbox
   Pareto curve says Slot A v2 dominates P8A. Product/cost considerations
   choose the FPR target. This is the boring, defensible answer; should be
   the default if A-D don't open new ground.

## What this onboarding does NOT cover

- Per-CKPT τ-calibration code (it's in `arena/score_teams_promotion_contract.py`;
  read that file directly when you need to modify it).
- Vertex AI launch operational rules (see `CLAUDE.md` in this directory).
- W&B project layout (project = `phase2-experiments` for older runs,
  `phase2r13-experiments` for 2026-05-19+).
- Auto-mode constraints (see `feedback_*` memory entries for "no cancelling
  Vertex jobs without auth", "no secrets in /tmp files", etc. — operational
  guardrails, not technical).
