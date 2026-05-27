# Handoff — independent agent, propose a 24-hour plan

**Date generated**: 2026-05-02
**Branch**: `teams-relaunch-root-2026-04-17`
**You are**: an independent AI agent picking this work up cold. Your task is to **propose a plan for the next 24 hours of work**, then wait for the user to approve before executing.

You have full context-window access to the data; do not skip the reading list. Form your own framing of the problem before reading any of my prior interpretation.

---

## What you must do

1. **Read the inputs in §A** (in order). Form your own picture of the empirical state.
2. **Read the cautions in §B**. Honor them.
3. **Optionally** read the interpretive docs in §C, but treat them as one perspective among possible ones — not as constraints.
4. **Produce** a written plan for the next 24 hours per §D. Do not begin executing without explicit user approval.

---

## §A — Required reading (do not skip; do not skim)

Order matters. Earlier items contextualize later items.

### A.1 — Operational rules (5 min)
- `CLAUDE.md` (root of `DeepfakeBench/training/`): region preference, capacity playbook, US-bucket-locality cost/throughput.
- `~/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/MEMORY.md`: index of project memory entries. Each entry is one line; pull the entries you need.

### A.2 — The original plan (20 min)
- `PLAN.md` (root of `DeepfakeBench/training/`): the R13 Forward Plan, dated 2026-04-29. ~1000 lines. Sections most relevant for next-step planning:
  - §1 Situation
  - §3 Weaknesses and confounds
  - §4 Whack-a-mole patterns
  - §7 Paths to investigate further
  - §9 Recommended experiment slate (Move 1 → Move 4 → Move 5)
  - §10 Risk register
  - §11 Open questions for the user
  - §12 Decision tree
  - **Read this BEFORE the diagnostics docs.** It's the framing the project was on before this session's work invalidated parts of it.

### A.3 — What was measured this session (35 min)
- `docs/relaunch_handoffs/MEASUREMENTS_2026-05-02_P18.md` — pure data, no interpretation. All numbers from this session's CPU diagnostics + Vertex contract scorecard. **Treat as ground truth for what was measured; do not inherit any framing of what it means.**

### A.4 — Predecessor handoffs you should be aware of (skim 10 min, drill into anything that affects your plan)
- `HANDOFF.md` (root): currently points at the final-verdict doc; the file itself is 2 lines of pointer.
- `docs/relaunch_handoffs/HANDOFF_2026-05-02_P18_CORRECTIVE.md` — the predecessor handoff that called for the diagnostics this session ran. Useful to understand the methodology critique that motivated F/G.
- `docs/relaunch_handoffs/PHASE1_2_COMPLETE_STATUS_2026-05-01.md` — Phase 1+2 status as of 2026-05-01 morning.
- `docs/relaunch_handoffs/P17_FINAL_VERDICT_2026-05-01.md` — P17 closeout (layer-3 readout idea was killed before P18 launched).

### A.5 — Recent memory entries that frame the empirical state
Pull these directly:
- `project_p18_d_contract_p8a_wins_no_floor.md` (2026-05-02) — D contract scorecard outcome
- `project_p18_diagnostics_complete_2026-05-02.md` (2026-05-02) — CPU diagnostics summary (CONTAINS PRIOR-AGENT INTERPRETATION; treat the numerical sections as authoritative, the framing as one perspective)
- `project_p18_corrective_probe_2026-05-02.md` — SUPERSEDED but useful for the methodology lessons
- `project_phase1a_method_cluster_axis_2026-05-01.md` — Phase 1A finding that motivated P18
- `project_signature_shortcut_finding.md` — camera-signature shortcut original finding
- `project_eval_production_crop_tightness_gap.md` — eval-vs-production crop gap
- `project_visomaster_hints_lanes_bad_data.md` — known bad data lanes
- `project_face_size_label_leak.md` — face-size leak finding
- `project_p8a_breakthrough.md` — what P8A originally achieved
- `project_clean_teams_same_identity.md` — paired identities (Move 4 prereq)
- `project_promotion_contract.md` — contract authority
- `project_success_criteria.md` — three pillars (recall + FPR + robustness)

---

## §B — Cautions before you plan

1. **The user's deployment target is 90% fake recall at ~10% FPR.** Current best (per `MEASUREMENTS_2026-05-02_P18.md` §1.5) is **43% dev_fake_macro_recall at 7% dev FPR (P18C)**. The gap is ~2× on recall. This is structural-distance territory, not "tweak a hyperparameter".

2. **PLAN.md is from 2026-04-29.** Since it was written:
   - Move 1 ran (memory `project_move1_bucket_gap_refuted` — refuted bucket-gap-as-AUC-failure hypothesis).
   - P14, P15, P16, P17, P18 all ran. None promote.
   - P14_DATA_FIX (the plan's "Move 2") is drafted but not launched.
   - Move 4 (the plan's Priority 8) plumbing partially exists in `data/sources/visomaster.py` (`companion_bucket` field, `_load_merged_teams_enhanced_paired`, ~5 wired call sites) but no batch sampler in `data/sources/combined_paired.py`, and no yaml exists.
   - The plan's "canonical priority sequence" (Move 1 → P14_DATA_FIX → anti-shortcut → Move 4 → P15 GRL) was partially executed and partially obsoleted.
   - The plan does not yet reflect "visomaster_enhanced_macro_dev recall is 1-2% across all arms" — a per-suite finding from D that wasn't visible at PLAN.md authorship time.

3. **The contract scorecard is authoritative for promotion** (memory `project_promotion_contract.md`); `dev_fake_macro_recall` is the contract's primary recall metric. Trainer-side `value_composite` and individual-bucket recalls are not promotion-grade on their own.

4. **Don't be misled by the corrective probe's small-N (n_dor=9, n_lockbox_dor=25)** — Diagnostic A's bootstrap CIs include 0 for all paired AUC deltas, but this is a power problem, not a "no signal" finding (Diagnostic B with paired Wilcoxon does have power). Read both.

5. **Operational hard rules** (from `CLAUDE.md`, memory):
   - Strongly prefer US Vertex regions (`us-east1`, `us-west4`, `us-central1`); ~10× throughput vs asia. Do not launch in non-US without explicit user authorization.
   - If a Vertex job is `PENDING` >30 min in a US region, switch regions (relaunch in another US region; once new job is `RUNNING`, ask user before cancelling original).
   - Don't cancel Vertex jobs without explicit user authorization (memory `feedback_no_cancelling_vertex_jobs.md`).
   - Don't write secrets (W&B API key, etc.) to files in /tmp (memory `feedback_no_secrets_in_tmp_files.md`).
   - Avoid `n_jobs=-1` in sklearn/joblib on this Mac (memory `feedback_sklearn_njobs.md`).
   - Image rebuild auto-bumps VERSION via `./dev.sh build-prod -y`; new in-image yamls require rebuild (memory `reference_image_rebuild.md`).
   - Don't commit without explicit user OK.

---

## §C — Optional: prior-agent interpretive handoffs

These are *one* prior agent's interpretive framing. They contain analysis, recommendations, and verdicts that you are NOT obligated to inherit. Read them if you want to understand the previous agent's reasoning chain or to argue against it. Do not let them constrain your plan.

- `docs/relaunch_handoffs/HANDOFF_2026-05-02_P18_FINAL_VERDICT_WITH_D.md` — interpretive verdict + 3-option recommendation matrix
- `docs/relaunch_handoffs/HANDOFF_2026-05-02_P18_DIAGNOSTICS_COMPLETE.md` — interpretive write-up of CPU diagnostics
- `docs/relaunch_handoffs/HANDOFF_2026-05-02_NEXT_PACKET_DECISION_DRAFT.md` — pre-D candidate matrix

---

## §D — What to deliver

A **written plan** in `docs/relaunch_handoffs/HANDOFF_<your-date>_24H_PROPOSED_PLAN.md` containing:

### D.1 — Your independent reading of the situation (~½ page)
What you think the empirical state is. What you think the binding constraint(s) on `dev_fake_macro_recall ≥ 0.70` is/are. What you think is unknown. **Cite specific numbers from `MEASUREMENTS_2026-05-02_P18.md`.** State explicitly where you agree or disagree with the predecessor agent's framing if you read §C.

### D.2 — A ranked list of candidate moves (3–5 items)
For each candidate, state:
- **Hypothesis being tested** (≤ 2 sentences).
- **Concrete artifact you'd build** (yaml diff, code change, analysis script, etc.).
- **Cost estimate** (CPU hours / Vertex $).
- **What the result would tell you** — how you'd update beliefs given each plausible outcome (α/β/γ-style).
- **Prerequisites** (e.g., does it depend on a prior step, code that doesn't exist, manifests, etc.).

### D.3 — Your recommended 24-hour sequence
Pick a sequence (could be one big move or 2-3 staged moves) that fits in 24 hours of wall-clock time, accounting for: code time, image build (~5-15 min via Cloud Build), Vertex queue (us-east1/us-west4 capacity per `gcloud ai custom-jobs list`), Vertex run time, results pull + analysis. **Total spend ≤ $50 of remaining ~$65 GPU budget.** Identify go/no-go decision points so the user can intervene mid-sequence.

### D.4 — What you would NOT do (and why)
2-3 sentences each on candidate moves you considered and rejected. Helps the user calibrate your reasoning.

### D.5 — Open questions for the user
Items where your plan branches on a user decision. Examples: "should I include `top_n_step8000` ckpt in any contract scorecard re-run?", "is `visomaster_fake: 4.0 → 12.0` an acceptable distortion of the training distribution?", "is overnight wall-clock acceptable or do you want results within 4 hours?".

### D.6 — Standing operational discipline
Confirm you've read §B and will honor it. Confirm you will not commit, cancel jobs, or launch without explicit user OK.

---

## §E — Available resources

### Budget
- ~$65 of $80 GPU budget remaining (CPU is free).
- Cloud Build (~$0.50 per image rebuild) is in scope.

### Local artifacts
- `/tmp/p18_ckpts/` — both P18 ckpts at periodic_step4000 (~900 MB each) + 8 other steps for treatment + 8 for control.
- `/tmp/p17_ckpts/` — P17 ArcFace + LINEAR ckpts.
- P8A reference ckpt: `analysis/_features_cache_2026-04-30/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth`.
- Cached features (per-arm × 4 layers + CLS): `analysis/_features_cache_2026-04-30/intermediate__*__layer*__n800.npz` and `final_cls__*__n800.npz`.
- D contract artifacts pulled to `analysis/p18_probe_2026-05-01/d_results/`.

### Code already in repo
- All P-series yamls in `experiments/phase2_round13/` (P14 through P18 + P14_DATA_FIX drafted).
- All this session's analysis scripts in `analysis/p18_probe_2026-05-01/`.
- Move 4 plumbing partial: `data/sources/visomaster.py` has `companion_bucket` infrastructure; no `combined_paired.py` paired-batch sampler.
- Contract launcher: `arena/launch_teams_promotion_contract.sh`.
- Image build: `./dev.sh build-prod -y` (auto-bumps VERSION).

### Image
- `1.3.241` in GCR (current; built 2026-05-02 to bake P18 corrective ckpt map). Includes contract policy v3 default `target_fake_recall_min=0.70`.

### Eval substrates / manifests
- 800-frame substrate: `analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/sampled_frames.csv`.
- Production-tight (RFA=0.85) re-tag: `analysis/eval_substrate_v3_retag_2026-05-01/outputs/eval_substrate_v3_retag.csv` (840 frames; pre-scored with P8A only).
- Suite manifests: `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml` (latest).
- Checkpoint maps: `arena/checkpoint_maps/teams_target_domain.*.yaml`.

### Vertex / GCS
- Project: `train-cvit2`.
- Region preference (CLAUDE.md): us-east1 → us-west4 → us-central1.
- W&B project: `phase2-experiments`. W&B entity: `dtect-vision`. API key in `~/.netrc` line under `api.wandb.ai`.
- Output GCS root: `gs://training-job-outputs/`.

### Inactive
- 0 active Vertex jobs as of session end. Verify with `gcloud ai custom-jobs list --region=us-east1 --filter="state:JOB_STATE_RUNNING OR state:JOB_STATE_PENDING"`.

---

## §F — A few sanity-check questions to ask yourself before writing your plan

These are not exhaustive; they're examples of the kind of question that should be answered (yes or no) before you commit to a direction. Don't take them as endorsements of any particular path.

1. Looking at `MEASUREMENTS_2026-05-02_P18.md` §1.4 (per-suite FAKE recall), what's the smallest single-suite recall lift that, if achieved, would move `dev_fake_macro_recall` from 0.18 (P18C) to 0.70? Which suite is the binding constraint?

2. What's the largest delta between dev recall and lockbox recall on any fake suite? Does that gap suggest the data has the signal but the model fails to transfer, or that the data lacks the signal entirely?

3. The contract picks τ to satisfy `target_real_fpr ≤ 0.02` first. If the user is willing to accept 7% or 10% FPR (per the user note above), how does the ranking change? (See `MEASUREMENTS_2026-05-02_P18.md` §1.5.)

4. P14_DATA_FIX (the original plan's Move 2) was drafted but never launched. Does its hypothesis still apply given the post-P18 evidence base, or does Move 1's grouped-probe result kill the rationale for P14_DATA_FIX?

5. Move 4 (paired same-identity contrastive) requires writing a new batch sampler in `data/sources/combined_paired.py`. Is there enough scaffolding to do this in <12 hours of code time, or is this a multi-day effort that doesn't fit a 24h window?

6. What's a measurement you could do entirely on CPU (cost $0) that would substantially update your beliefs before spending any GPU? Anything in §1.5 of MEASUREMENTS suggest a specific cross-section probe?

7. The P18 yaml's `family_weights` (MEASUREMENTS §3) heavily up-weights `deeplive_teams_fake` (7.0). Does the per-suite recall data suggest the training is correctly converging on what these weights ask for?

(If you cannot answer any of these confidently from the data, that itself is a sign that an additional measurement should precede a packet launch.)

---

## §G — How to engage with the user

The user wants:
- Smart, methodical decisions that match the empirical state, not pattern-matched packets.
- Concrete plans, not vague directions.
- Recommendations + tradeoffs, then waits for explicit pick (memory `feedback_decision_points.md`).
- Aggregate signal over per-pool details for small samples (memory `feedback_small_sample_guidance.md`).
- Terse responses; no trailing summaries (general project preference; see prior memory entries for tone).

The user reserves judgment calls. **Do not autonomously launch a packet, even within budget.** Propose, then wait.
