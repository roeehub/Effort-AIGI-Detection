# HANDOFF — IQ-shortcut deconvolution program (2026-05-08)

> **⚠ TEMPORARY DOCUMENT.** Delete this file when the program's first GPU
> packet has either landed a verdict or been explicitly abandoned. The
> canonical knowledge base is the wiki thread + memories + per-stage FACTS
> docs (see §1 below). This handoff exists only because the parent
> conversation paused mid-flight while monitoring an unrelated running
> scorecard.
>
> **Authored**: 2026-05-08, after the cross-pool IQ data atlas surfaced
> previously-not-articulated structural facts about IQ-axis confound in
> both training data and eval substrates.

---

## 1. Where the canonical knowledge lives

Read these in order. **Do not start work before reading them.**

1. **[`docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md`](docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md)** — the authoritative proposal thread. Contains FACTS sections, OPINIONS sections, and the staged program with decision criteria. **This is the source of truth; this handoff is just the operational entry-point.**
2. **[`analysis/iq_data_atlas_2026-05-08/IQ_ATLAS_FACTS_2026-05-08.md`](analysis/iq_data_atlas_2026-05-08/IQ_ATLAS_FACTS_2026-05-08.md)** — the cross-pool measurement that motivates the program. 35 pools × ~500 frames × 13 IQ features.
3. **[`docs/packet_retrospectives/MODEL_GOALS.md`](docs/packet_retrospectives/MODEL_GOALS.md)** — three pillars, deployment ≡ E2B, NO-ENSEMBLE rule.
4. **[`docs/packet_retrospectives/SCORECARD_GUIDE.md`](docs/packet_retrospectives/SCORECARD_GUIDE.md)** — when to use which scorecard mode.
5. **[`docs/packet_retrospectives/AGENT_GUIDE.md`](docs/packet_retrospectives/AGENT_GUIDE.md)** — validate-before-suggest discipline + bundle-decomposition rule.
6. The memory entries listed in the proposal thread §6 (Cross-references / Loadbearing memories).

After reading, you should be able to answer:
- What are the three competing hypotheses (A/B/C)?
- Why is Stage 1 (R² probe) the gate for everything else?
- What's the difference between Stage 2a (IQ GRL) and Stage 2b (IQ-balanced sampler), and what determines which to run?
- Why is Stage 4 (drop smooth training reals) explicitly held until Stage 1+2 confirm IQ is the binding constraint?

---

## 2. State of the running work as of 2026-05-08 ~14:00 UTC

**P2 Phase A scorecard** (Vertex `6226906326623059968`) — running at handoff
authoring; ETA SUCCEEDED ~13:55-14:10 UTC. Reports landing at
`gs://training-job-outputs/test_results/teams_promotion_contract/p2-scratch-scorecard-2026-05-08/`.
When it lands, the P2-D-step3000 verdict is known; this is the third
candidate ckpt for Stage 1 of the program. **Stage 1 is gated on this
verdict** because if P2-D-step3000 produces a meaningful contract result,
its decomposition is informative; if it doesn't, we run Stage 1 on P8A +
E2B only.

**No GPU work in flight related to this program.** No packet has been
launched against the IQ-deconvolution hypothesis yet. The user explicitly
asked to NOT run any GPU spend in the parent conversation.

**Iterative scorecard tooling** is image-baked-pending — `--mode iterative`
exists in `arena/launch_teams_promotion_contract.sh` but the iterative
suite manifest yaml is newer than the current image push time. Before
launching iterative-mode, run `./dev.sh build-prod -y` (~17 min Cloud
Build). Trajectory mode + full mode work without rebuild.

---

## 3. First-action sequence

Execute these in order. **Each stage is gated on the previous stage's
result.**

### 3.1. Wait for the P2 Phase A scorecard to finalize

If `gcloud ai custom-jobs describe 6226906326623059968 --region=us-east1 --project=train-cvit2 --format='value(state)'` returns `JOB_STATE_RUNNING`, wait. ETA finalize ~14:10 UTC.

When it transitions to `JOB_STATE_SUCCEEDED`, pull and read:
- `gs://training-job-outputs/test_results/teams_promotion_contract/p2-scratch-scorecard-2026-05-08/promotion_contract/promotion_winner.json`
- `gs://training-job-outputs/test_results/teams_promotion_contract/p2-scratch-scorecard-2026-05-08/promotion_contract/selected_threshold_scorecard.csv`
- `gs://training-job-outputs/test_results/teams_promotion_contract/p2-scratch-scorecard-2026-05-08/diagnostic_scorecard/scorecard.json`

Document the P2 verdict in a new FACTS doc at
`analysis/p2_eval_2026-05-08/P2_PHASE_A_VERDICT_FACTS_<date>.md` and update
`docs/packet_retrospectives/STATE.md` accordingly. **This is not optional**
— the P2 verdict is part of the iteration record regardless of where the
deconvolution program goes next.

### 3.2. Stage 1 — IQ-shortcut R² probe (CPU, ~3-6 hours)

Per `iq_shortcut_deconvolution_program_2026-05-08.md` §4.1.

**Inputs already cached**:
- `analysis/iq_data_atlas_2026-05-08/outputs/per_frame.parquet` — IQ feature panel for 15,236 frames across 35 pools.
- Per-ckpt scores already exist for P8A, E2B, and the P2 candidates from prior eval runs. Look in `arena/run_target_domain_validation_sequential.py` outputs and the canary parquet `analysis/p2_eval_2026-05-08/d1_d4_cpu/scores/`.

**What to build**:
1. A driver script at `analysis/iq_shortcut_decomp_<date>/decompose.py` that:
   a. Joins per-frame IQ features to per-frame model scores by `frame_path`.
   b. For each (ckpt, pool-group): fits `model_score = β · IQ_features + ε` with sklearn `LinearRegression` (single-process, NOT n_jobs=-1 per memory `feedback_sklearn_njobs.md`).
   c. Reports R², residuals, and residual-AUC for fake-vs-real per (ckpt × pool-group).
2. A FACTS doc at `analysis/iq_shortcut_decomp_<date>/IQ_DECOMP_FACTS_<date>.md` containing tables of R² and residual-AUC, no interpretation.
3. A short interpretation doc at `analysis/iq_shortcut_decomp_<date>/IQ_DECOMP_OPINIONS_<date>.md` separately, marked OPINION, that maps the numbers to the Hypothesis A/B/C frame and the Stage 2 decision criterion.

**Decision criterion for Stage 2** (per the proposal §4.1):
- R² > 0.5 across most ckpt × pool cells → Stage 2a (IQ GRL).
- R² ∈ [0.3, 0.5] → Stage 2b (IQ-balanced sampler).
- R² < 0.2 → pivot away from IQ-debiasing.

**Stop and ask the user before launching Stage 2.** GPU spend authorization is required regardless of Stage 1 verdict. The user reserves judgment calls at decision points (memory `feedback_decision_points.md`).

### 3.3. (Conditional) Stage 2 packet design

If Stage 1 indicates Stage 2a or 2b, draft the packet yaml + canary probe per the proposal §4.2 + `AGENT_GUIDE.md` validate-before-suggest discipline. Surface to the user for authorization. **Single-lever discipline mandatory** — no bundle stacking.

### 3.4. Update the wiki

After each stage's FACTS doc lands:
1. Append a §7 (Status / progress) entry to the proposal thread.
2. Add a memory entry summarizing the load-bearing finding.
3. Update `STATE.md` "Where we stand" if the finding shifts the iteration plan.

---

## 4. Operational gotchas

- **Don't cancel the running Vertex job** without explicit user authorization (memory `feedback_no_cancelling_vertex_jobs.md`). Even hung jobs need user OK first.
- **Don't use n_jobs=-1** in sklearn / joblib (memory `feedback_sklearn_njobs.md`; caused 3 system reboots from swap exhaustion). Use `n_jobs=1` or bounded multiprocessing.
- **Don't write secrets to /tmp files** (memory `feedback_no_secrets_in_tmp_files.md`). Set WANDB_API_KEY inline if needed.
- **Image-currency check** — for any GPU launch, edited yamls must be older than the image push time, else the launcher blocks. Run `./dev.sh build-prod -y` (~17 min) to bake new yamls in.
- **FACTS-vs-OPINIONS separation** — every analysis dir gets a `*_FACTS_*.md` (numbers + tables only) and optionally a `*_OPINIONS_*.md` (interpretation + suggestions). Forbidden words in FACTS docs: succeeds, fails, wins, promotes, deployment-grade. See `AGENT_GUIDE.md`.
- **Bundle-decomposition discipline** — single-lever ablations only (memory `project_face_scale_jitter_load_bearing.md`). The 5.7× P14 bundle vs jitter-isolated gap is the precedent that makes this discipline non-negotiable.
- **GPU region preference** — US regions only (memory `project_gcs_region_locality.md`); 10× throughput multiplier. Default order: `us-west4`, `us-east1`, `us-central1`. Never asia/europe without explicit user authorization.
- **Don't propose ensembles** — memory `project_job12_ensemble_ceiling_2026-05-04.md` and `MODEL_GOALS.md` § NO ENSEMBLE. Hard rule.
- **Don't propose L14 production candidates** — memory `project_l14_does_not_break_viso_ceiling.md` and `MODEL_GOALS.md` § Architecture. L14 is for capacity tests only.

---

## 5. What you'll need

- **Python env** — repo has `requirements.txt` and a Dockerized image. For CPU-only Stage 1 work, a local venv with `numpy`, `pandas`, `scikit-learn`, `pyarrow`, `cv2` is sufficient.
- **GCS access** — `gcloud auth application-default login` if not already; project `train-cvit2`.
- **W&B access** — `WANDB_API_KEY` env var; entity `roee-darshan-anthropic-internal-stuff` or whatever the project uses; project `phase2-experiments`. **Do NOT write keys to /tmp.**
- **Memory system** — auto-loaded from `~/.claude/projects/.../memory/MEMORY.md`. Read it on every session start.

---

## 6. When to delete this file

Delete `HANDOFF_IQ_DECONVOLUTION_2026-05-08.md` (this file) when **either**:

- Stage 1 has produced a FACTS doc + memory entry + a Stage 2 decision (regardless of whether Stage 2 launches), AND the proposal thread §7 has been updated. At that point the wiki + memory carry the state and the handoff is redundant.
- The user explicitly abandons the program. At that point the proposal thread should be marked CLOSED in §1.

**Reminder to the user** (Roee): once the agent finishes this work and the wiki + memory are updated, **delete this handoff file**. We are deliberately moving away from per-session HANDOFF.md files toward a rolling wiki + memory model. This handoff is a one-off exception for cross-conversation continuity.

---

## 7. One-line summary for the picking-up agent

You have a fully-specified research program in `docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md`. Read it, check that the running scorecard has finalized, then run Stage 1 (IQ-shortcut R² probe) per §4.1. **Do not skip ahead to Stage 2 without the user's explicit go-ahead after seeing Stage 1's numbers.**
