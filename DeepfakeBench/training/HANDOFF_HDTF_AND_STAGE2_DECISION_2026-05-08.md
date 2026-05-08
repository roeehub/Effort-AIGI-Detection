# HANDOFF — HDTF Phase C landing + Stage 2 decision (2026-05-08)

> **⚠ TEMPORARY DOCUMENT.** Delete this file once you have:
> - Documented the HDTF Phase C results in a FACTS doc + memory entry,
> - Updated [`docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md`](docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md) §7 with your reading,
> - Refreshed [`docs/packet_retrospectives/STATE.md`](docs/packet_retrospectives/STATE.md) "Where we stand",
> - Surfaced your Stage 2 proposal (or pivot proposal) to the user.
>
> The wiki + memory are the canonical knowledge base. This handoff is a
> one-off operational entry-point that exists only because the prior
> conversation is paused mid-program. After your work lands in the wiki,
> this file is redundant noise.

---

## 1. What you are being asked to do

The user wants a **fresh, unbiased read** on the convergent dataset of an
in-flight research program (the IQ-shortcut deconvolution program). Three
checks (a/b/c) were authorized to gate the program's Stage 2 packet
design. Two of them landed before this handoff was written; (b) is in
flight and will likely complete before you start. Specifically:

1. **Read** the FACTS docs FIRST, **form your own view** of what the data
   shows, **then** optionally compare to the prior agents' OPINIONS docs.
   You are explicitly free to disagree with the prior framings.
2. **Document the (b) HDTF Phase C results** in a new FACTS doc (factual-
   only, forbidden words enforced) when the Vertex job lands.
3. **Propose the Stage 2 next step** based on the convergent dataset —
   could be the program's original Stage 2a (IQ GRL), could be Stage 2b
   (IQ-balanced sampler), could be a different intervention entirely
   (e.g. dual-axis approach if the dor identity-cluster axis from check
   (c) deserves its own intervention), could be a pivot away from the
   IQ-deconvolution program altogether. **The user reserves the GPU spend
   authorization** — your job is to surface the decision with reasoning,
   not to launch packets.

The user's explicit framing: *"We don't want to feed the next agent bias,
but we also don't want to deprive it from the information."*

---

## 2. Read these first (mandatory, in order)

**Layer 1 — wiki onboarding** (~30 minutes):

1. [`docs/packet_retrospectives/AGENTS.md`](docs/packet_retrospectives/AGENTS.md) — agent onboarding overview.
2. [`docs/packet_retrospectives/AGENT_GUIDE.md`](docs/packet_retrospectives/AGENT_GUIDE.md) — validate-before-suggest, CPU-first-then-GPU, single-lever discipline, FACTS-vs-OPINIONS separation. Hard rules.
3. [`docs/packet_retrospectives/MODEL_GOALS.md`](docs/packet_retrospectives/MODEL_GOALS.md) — three pillars, deployment ≡ E2B, P8A is the substrate-invariance anchor, NO ENSEMBLE rule, FPR budget, promotion bar.
4. [`docs/packet_retrospectives/SCORECARD_GUIDE.md`](docs/packet_retrospectives/SCORECARD_GUIDE.md) — when to use iterative / trajectory / full mode.
5. [`docs/packet_retrospectives/STATE.md`](docs/packet_retrospectives/STATE.md) — current rolling state. Note the **FRESH-AGENT GUIDANCE** block at the top — it tells you the unbiased reading order for the FACTS docs vs OPINIONS docs.
6. The `MEMORY.md` index at `~/.claude/projects/.../memory/MEMORY.md` — auto-loaded; ~80 entries; load-bearing for project context.

**Layer 2 — the program proposal**:

- [`docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md`](docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md) — the proposal thread. §7 has the Status / progress log including Stage 1 R² probe + (a) + (c) entries.

**Layer 3 — the FACTS docs (read these BEFORE the OPINIONS docs)**:

- [`analysis/p2_eval_2026-05-08/P2_PHASE_A_VERDICT_FACTS_2026-05-08.md`](analysis/p2_eval_2026-05-08/P2_PHASE_A_VERDICT_FACTS_2026-05-08.md) — P2 promotion contract scorecard.
- [`analysis/p2_eval_2026-05-08/P2_DEEPER_ANALYSIS_FACTS_2026-05-08.md`](analysis/p2_eval_2026-05-08/P2_DEEPER_ANALYSIS_FACTS_2026-05-08.md) — J1-J5 deeper characterization.
- [`analysis/iq_data_atlas_2026-05-08/IQ_ATLAS_FACTS_2026-05-08.md`](analysis/iq_data_atlas_2026-05-08/IQ_ATLAS_FACTS_2026-05-08.md) — cross-pool IQ atlas (35 pools, 15,236 frames).
- [`analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_FACTS_2026-05-08.md`](analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_FACTS_2026-05-08.md) — Stage 1 IQ R² probe.
- [`analysis/iq_perlayer_probe_2026-05-08/IQ_PERLAYER_PROBE_FACTS_2026-05-08.md`](analysis/iq_perlayer_probe_2026-05-08/IQ_PERLAYER_PROBE_FACTS_2026-05-08.md) — check (a).
- [`analysis/dor_encoder_axis_2026-05-08/DOR_ENCODER_AXIS_FACTS_2026-05-08.md`](analysis/dor_encoder_axis_2026-05-08/DOR_ENCODER_AXIS_FACTS_2026-05-08.md) — check (c).
- The HDTF FACTS doc you author (check (b) landing).

**Layer 4 — OPINIONS docs (optional second-pass reading)**:

After you have your own reading from layer 3, *then* compare to:

- [`analysis/p2_eval_2026-05-08/P2_DEEPER_ANALYSIS_OPINIONS_2026-05-08.md`](analysis/p2_eval_2026-05-08/P2_DEEPER_ANALYSIS_OPINIONS_2026-05-08.md) — J1-J5 synthesis + proposed pre-Stage-2a sequence.
- [`analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_OPINIONS_2026-05-08.md`](analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_OPINIONS_2026-05-08.md) — Stage 1 reading + Stage 2a recommendation.
- [`analysis/dor_encoder_axis_2026-05-08/DOR_ENCODER_AXIS_OPINIONS_2026-05-08.md`](analysis/dor_encoder_axis_2026-05-08/DOR_ENCODER_AXIS_OPINIONS_2026-05-08.md) — check (c) reading + 3 candidate Stage 2 paths.

**You may disagree with any or all of the OPINIONS docs.** They are one
agent's reading; the data is what it is.

---

## 3. Operational state at handoff time (2026-05-08 ~17:50 UTC)

### In-flight GPU job

- **Vertex `6632089555598049280`** — display name `p2-d-step3000-hdtf-2026-05-08`, region `us-east1`, image `1.3.273`.
- State at handoff: `JOB_STATE_RUNNING` (started 16:31:59 UTC, ~1h17m elapsed).
- Progress: 24 of 48 (suite × ckpt) pairs complete. Throughput ~24/hr.
- ETA SUCCEEDED: ~18:25-18:45 UTC.
- Outputs land at: `gs://training-job-outputs/test_results/teams_promotion_contract/p2-d-step3000-hdtf-2026-05-08/`.
- Persistent monitor armed: task `bpz0ecwgf`. Will fire on terminal state.

If by the time you read this it's not yet `JOB_STATE_SUCCEEDED`, check
status:
```bash
gcloud ai custom-jobs describe 6632089555598049280 \
  --region=us-east1 --project=train-cvit2 --format='value(state)'
```

### Recently completed

- **Sister-agent (handoff-driven)**: Stage 1 R² probe + P2 verdict FACTS doc + memory entry `project_iq_shortcut_decomp_stage1_2026-05-08.md`.
- **Sub-agent `ad2687929ec623ce5`** (CPU): checks (a) per-layer IQ probe + (c) Dor encoder-axis. FACTS docs + OPINIONS doc + 2 memory entries.
- **This-session agent**: J1-J5 deeper analysis FACTS + synthesis OPINIONS, MODEL_GOALS.md, SCORECARD_GUIDE.md, scorecard tooling (iterative/trajectory/full modes), IQ atlas (sub-agent dispatched earlier), and the (b) GPU launch.

Recent commits:
```
963dff6  (a) + (c) FACTS docs + thread §7 update
32b9464  STATE.md FRESH-AGENT GUIDANCE block + thread §7 + handoff cleanup
a069354  J1-J5 deeper analysis FACTS + synthesis OPINIONS
8af9569  IQ-deconv proposal thread + (now-deleted) program handoff
3504a7b  cross-pool IQ data atlas
```

---

## 4. Specific actions when (b) lands

### 4.1 Pull artifacts

```bash
mkdir -p /tmp/p2_d_hdtf_artifacts
gsutil -m cp 'gs://training-job-outputs/test_results/teams_promotion_contract/p2-d-step3000-hdtf-2026-05-08/promotion_contract/*' /tmp/p2_d_hdtf_artifacts/
gsutil -m cp 'gs://training-job-outputs/test_results/teams_promotion_contract/p2-d-step3000-hdtf-2026-05-08/diagnostic_scorecard/*' /tmp/p2_d_hdtf_artifacts/
```

Per-frame reports (if you want frame-level analysis):
```bash
gsutil -m cp 'gs://training-job-outputs/test_results/teams_promotion_contract/p2-d-step3000-hdtf-2026-05-08/reports/*_frames_report.csv' /tmp/p2_d_hdtf_frames/
```

### 4.2 Document in a FACTS doc

Create `analysis/p2_d_hdtf_2026-05-08/P2_D_HDTF_FACTS_2026-05-08.md`.

**FACTS-only discipline** — forbidden words: succeeds, fails, wins,
promotes, deployment-grade. Tables + cross-references only. No
recommendations.

Cover at minimum:
- Promotion-contract ranking from `checkpoint_summary.csv`.
- Per-suite recall + FPR at the contract τ for all 3 ckpts on the 16
  HDTF suites.
- Cross-comparison vs Phase A teams substrate (data is in
  `analysis/p2_eval_2026-05-08/P2_PHASE_A_VERDICT_FACTS_2026-05-08.md`).
- Cross-reference with `IQ_DECOMP_FACTS_2026-05-08.md` § HDTF rows
  (substrate IQ-clean per atlas; Hypothesis B reading).

### 4.3 Optional OPINIONS doc

If you have a reading worth surfacing, create
`analysis/p2_d_hdtf_2026-05-08/P2_D_HDTF_OPINIONS_2026-05-08.md` —
clearly labeled as OPINIONS, present alternative readings explicitly.

The questions worth answering (or at least addressing) in your reading:

1. **Does D step3000's viso lift hold on HDTF?** Memory `project_pa_does_not_generalize_to_hdtf_2026-05-05.md` records PA's v2-substrate lift collapsing on HDTF. Does D step3000 follow the same pattern, or is it content-real?
2. **What does the IQ R² look like on HDTF for D step3000?** P8A and E2B were measured at low R² on HDTF (0.016-0.090). If D step3000 also has low HDTF R², the encoder change from Fourier is substrate-conditional. If high, the encoder is more IQ-driven across substrates.
3. **Does the dor identity-cluster collapse from check (c) manifest on HDTF too?** HDTF doesn't have the same Dor-cohort substrate, but you can probe nearby — `proper_real_teams_dev` slices may have informative subsets.

You don't need to write up an answer to all three; pick whichever the data supports.

### 4.4 Memory entry

Add at `~/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/project_p2_d_hdtf_<descriptor>_2026-05-08.md` — a project-type memory naming the load-bearing finding (one or two sentences), with **Why** and **How to apply** lines per the auto-memory discipline. Index it in `MEMORY.md`.

### 4.5 Update the wiki

- Append an entry to thread §7 (`docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md`) — date, headline numbers, link to the FACTS doc, very short reading.
- Update `STATE.md` "Where we stand right now" paragraph — replace "(a/b/c) gating Stage 2a" with the post-(b) state. Update the FRESH-AGENT GUIDANCE block to add the new HDTF FACTS doc to layer 3.

### 4.6 Surface your Stage 2 proposal to the user

The user reserves the GPU spend authorization. After you've done your
own analysis, propose the next step:
- A specific Stage 2 packet design (single-lever per AGENT_GUIDE Rule 1).
- A different direction (data curation / eval reframe / pivot).
- Or "more CPU diagnostics first" with the specific diagnostic named.

You can disagree with the prior agents' OPINIONS docs. The user
explicitly authorized this — quote from the user instruction that
created this handoff: *"We don't want to feed the next agent bias, but
we also don't want to deprive it from the information."*

---

## 5. Discipline rules (mandatory)

### Hard rules — DO NOT VIOLATE

- **Don't cancel Vertex training jobs without explicit user authorization** (memory `feedback_no_cancelling_vertex_jobs.md`). Even hung jobs need user OK first.
- **Don't use `n_jobs=-1`** in sklearn / joblib (memory `feedback_sklearn_njobs.md`; caused 3 system reboots from swap exhaustion). Use `n_jobs=1` or bounded multiprocessing.
- **Don't write API keys / secrets to /tmp files** (memory `feedback_no_secrets_in_tmp_files.md`). Set inline on the launcher command or use shell env.
- **Don't propose ensembles** (`MODEL_GOALS.md` § NO ENSEMBLE; memory `project_job12_ensemble_ceiling_2026-05-04.md`). Single-model only. Hard rule.
- **Don't propose L14 production candidates** (`MODEL_GOALS.md` § Architecture; memory `project_l14_does_not_break_viso_ceiling.md`). L14 is for capacity tests only.
- **GPU region preference**: US only (memory `project_gcs_region_locality.md`; 10× throughput multiplier). Default order: `us-west4`, `us-east1`, `us-central1`. Never asia/europe without explicit user authorization.

### Process rules

- **FACTS-vs-OPINIONS separation** — every analysis dir gets a `*_FACTS_*.md` (factual-only, forbidden words enforced) and optionally a `*_OPINIONS_*.md` (interpretation, alternatives marked). See `AGENT_GUIDE.md`.
- **Bundle-decomposition discipline** — single-lever ablations only (memory `project_face_scale_jitter_load_bearing.md` — P14 bundle 5.7× weaker than isolated lever). Don't propose stacked-intervention packets without a sister-variant that isolates each lever.
- **CPU-first-then-GPU** (`AGENT_GUIDE.md` Rule 3) — before any GPU spend, run the cheapest CPU diagnostic that would update your beliefs.
- **Validate-before-suggest** (`AGENT_GUIDE.md` Rule 1) — before proposing any packet, grep for the toggle in existing yamls; check viewer's `model_dashboard_runs.yaml`; check MEMORY.md. If the toggle has been tested before, articulate what's structurally different about your proposal.
- **User reserves judgment calls at decision points** (memory `feedback_decision_points.md`) — present recommendation + tradeoffs, wait for explicit pick. Don't auto-launch GPU.

### Wiki-first knowledge management

- **The wiki is canonical, not session handoffs.** Update STATE.md, threads, and MEMORY.md as you work. After you finish, this file (HANDOFF_HDTF_AND_STAGE2_DECISION_2026-05-08.md) gets deleted.
- **Memory entries for load-bearing findings only** — not ephemeral task state. Project memories carry findings; feedback memories carry user preferences; reference memories point at external systems. See the auto-memory format used in existing entries.
- **Don't commit secrets.** Standard repo `.gitignore` excludes `*.csv`, `*.png`, `*.parquet` — figures and CSVs are regenerable, not committed. FACTS docs + driver scripts + parquets ARE committed (the parquets are gitignored too — driver re-runs the analysis from cached scores).

---

## 6. The strategic question (without bias)

The data is what it is. The strategic question you're being asked to
form a view on:

**Given the convergent dataset (P2 verdict, J1-J5, IQ atlas, IQ R² Stage
1, per-layer IQ probe, Dor encoder-axis, HDTF Phase C), what is the
binding constraint on Pillar 3 (robustness across capture conditions)
and what's the highest-leverage intervention to attack it?**

Some sub-questions you may find useful (not exhaustive, not the only
ones, you may identify others):

- Is D step3000's viso break content-real or v2-substrate-bound? (b) answers this.
- Is the binding constraint a single axis (IQ shortcut on lockbox) or
  multiple axes (IQ + dor identity-cluster collapse, possibly more)?
- Are augmentation-side levers (face_scale_jitter, Fourier band-amp,
  potential successors) saturating, or is there headroom on a different
  augmentation axis?
- Is distribution-level intervention (sampler / GRL) the right structural
  change, or is data-side curation (drop smooth training reals) more
  load-bearing?
- Is the eval substrate itself the binding constraint, and does HDTF
  deserve canonical-eval status per the IQ-deconvolution program §4.3?

The prior agents have proposed answers to some of these. Read their
OPINIONS docs after forming your own view, and challenge them where the
data supports it.

---

## 7. When to delete this handoff

Delete `HANDOFF_HDTF_AND_STAGE2_DECISION_2026-05-08.md` (this file) once:

- The (b) HDTF FACTS doc is committed at `analysis/p2_d_hdtf_2026-05-08/`.
- The thread §7 has your reading.
- STATE.md "Where we stand" reflects the post-(b) state.
- A memory entry summarizes the load-bearing HDTF finding.
- Your Stage 2 proposal is surfaced to the user and they have either
  authorized a packet, requested more CPU diagnostics, or asked for a
  pivot.

At that point, the wiki + memory carry the state. This file is redundant.

**Reminder to the user**: once the next agent finishes their work and
the wiki + memory are updated, **delete this handoff**. We are
deliberately moving away from per-session handoffs — this is a one-off
exception for cross-conversation continuity.

---

## 8. One-line summary for the picking-up agent

You have an in-flight GPU job (Vertex `6632089555598049280`, ETA ~18:30 UTC),
a fully-documented convergent dataset across 6 FACTS docs + 3 OPINIONS docs
+ 1 thread proposal, and a strategic question (where is the binding
constraint on Pillar 3?). Read the FACTS first, form your own view, then
compare to OPINIONS, then surface a Stage 2 proposal. **Don't launch
GPU spend** — the user reserves that. **Don't auto-commit beyond the
FACTS+OPINIONS+memory updates** — let the user review.
