# Handoff for the next agent — three-day training perspective + in-flight work

**Date authored**: 2026-05-04 night (PDT)
**Audience**: any agent picking up this work cold
**Purpose**: give you the empirical picture from the last three training days + what's running right now, with enough context that you can form your OWN conclusions about where to take this next. The user (Roee) explicitly does not want you to inherit this agent's framings; the bias-correction pass logged in TIMELINE 2026-05-04 night was specifically to make framings disposable.

---

## ⚠ DO NOT KILL — currently in flight (us-east1)

| Job ID | What | State | Image | Yaml |
|---|---|---|---|---|
| `3140330896851206144` | **Packet A** — single-lever data-availability test (visomaster_enhanced + visomaster_teams_enhanced at fw=4.0 on E2B baseline; NO bundle) | `JOB_STATE_RUNNING` | `1.3.256` | `experiments/phase2_round13/R13_PA_VISOMASTER_ENHANCED_DATA.yaml` (untracked) |
| `1202657157175050240` | **Packet C-codec** — Packet A recipe + `teams_codec_simulation` aug enabled (`policy: adaptive_mixture`, p=0.5, exclude_families covers all `*_teams_*` to avoid double-codec) | `JOB_STATE_RUNNING` | `1.3.257` | `experiments/phase2_round13/R13_PC_CODEC_PLUS_DATA.yaml` (untracked) |

Both launched 2026-05-04 evening. ETA to completion ~24h from launch. Both will produce checkpoints suitable for scorecard evaluation. Per CLAUDE.md, US-region preference is enforced; both are in us-east1 already.

**Already failed (not in flight)**:
- `9082408392701509632` — Job B (pre-RLP6_04 ckpts on held-out HDTF proper viso, second attempt). FAILED 2026-05-04 18:44 UTC, ~4h after submit. Root cause not yet investigated by this agent. Logs: `gcloud logging read 'resource.type="ml_job" AND resource.labels.job_id="9082408392701509632"'`. Memory `project_job_b_status_2026-05-05.md` is the most recent status entry but predates this failure; the original failure was abs paths in the YAML — fixed for this attempt; the second failure cause is open.
- `318825730303590400` — Packet C-codec FIRST attempt. FAILED 2026-05-04 19:45 PDT in 30s on `WANDB_ENTITY=roeehub` 404. Relaunched as `1202657157175050240` after `unset WANDB_ENTITY`. Memory `feedback_promotion_contract_launch.md` extended with the gotcha; thread `wandb_yaml_propagation_bugs` extended (4th wandb-side surface bug).

---

## Read protocol (do this first)

Per `docs/packet_retrospectives/AGENTS.md`:

1. Read `docs/packet_retrospectives/AGENTS.md` (the protocol; it may have evolved)
2. Read `docs/packet_retrospectives/OPEN_LOOPS.md` (28 open, 4 in-progress, 7 resolved, 1 superseded as of 2026-05-04 night). The two new in-progress / open loops from today are at the top of the open list:
   - `data-axis-clean-single-lever-retest-in-progress` (high) — Packet A + PC-codec close criterion
   - `pair-loss-asymmetric-variant-untested` (low) — pair-loss disconfirmation gate
3. Read the last ~15 entries of `docs/packet_retrospectives/TIMELINE.md` (covers 2026-05-04 morning CPU pass + evening Packets + tonight's wiki maintenance)
4. Read this file
5. **Specifically read the critical-reading banners** I added 2026-05-04 night to: `clean_teams_identity_pairing.md`, `viso_bucket_gap.md`, `webcam_fpr_dominance.md`, `calibration_vs_training_aug.md`, and `README.md`. They list framings to read with skepticism.
6. Read `PSERIES_FACTS_2026-05-02.md` and `PSERIES_OPINIONS_2026-05-02.md` in this directory — the FACT/OPINION split established 2026-05-02 is the cleanest separation in the wiki of "what's measured" vs "what an agent inferred."

---

## Three-day training arc (2026-05-02 → 2026-05-04)

Times are local PDT/PST. All packets ran on Vertex (us-east1, us-west4, us-central1 per CLAUDE.md region rules).

### 2026-05-02 — P22 + score-distribution forensics

**P22** (run `vp7zwlt7` per memory): pipeline_randomization aug curriculum on FT-from-P8A. **Mixed verdict, two memory entries**:
- Original (memory `project_p22_succeeded_2026-05-02.md`, AGENT FRAMING) — "P22 step8000 succeeds 2/3 pre-registered falsifiers, new FT-base candidate"
- SUPERSEDED (memory `project_p22_cpu_followups_reframe_2026-05-02.md`) — "P22 step8k score variance collapsed 140× (degraded, not 'wider'); step1k dominates under joint dev+lockbox τ-calibration; P8A+step1k ensemble (min rule) gives 18× viso, 51× deeplive at $0; P22 step8k regresses on P8A's signature dor invariance (0/1170 → 36/1170)"

**Read the second framing as the current understanding**. The P22 succeeded entry is preserved as historical record per the FACTS/OPINIONS pattern.

**Score-distribution forensics 2026-05-02 PM** (memory `project_score_distribution_audit_2026_05_02.md`): 11 CPU diagnostics, 70 figures, 33 CSVs, 739-line FINDINGS doc. Established:
- Image-quality shortcut: model uses sharpness/brightness as fake predictor (P8A); E2B is INVERTED (memory `project_iq_gating_viability_2026-05-04.md` updates this for E2B)
- Train-eval sharpness gap (4-25× on Laplacian)
- Per-identity FPR concentration (chronic-6 axis surfaces here)
- Deployment-τ unlock at 5-10% FPR (4-77% recall depending on suite)

### 2026-05-02 → 03 overnight — S1 / S2 / S3

**S1 / S2 / S3** (memory `project_s1_s2_s3_2026-05-03.md`): three single-lever P22 variants. Verdicts:
- S1 (training-cap): 0/3 falsifiers, worse than P8A
- S2 (FT from P8A_step2500, less-saturated base): 1/3 falsifiers; **strong reproducible win on lockbox transfer**: teams_fake_lockbox 54% → 91.5% at FPR=10% joint, 1.7× lift; chain champion on teams_fake_lockbox
- S3 (S2 + viso fw=8.0): 1/3 falsifiers, regression vs S2 on lockbox transfer

**Conclusion drawn at the time** (OPINION, see `PSERIES_OPINIONS_2026-05-02.md`): aug curriculum + base ckpt + family weight axis is "structurally exhausted." 10+ packets, no lever in this axis improved viso recall at joint dev+lockbox FPR=10%.

**Conclusion contested**: the contrarian read (in same OPINIONS doc) says: data-axis hypothesis was tested only at fw values that were either bundle-stacked (P14_DATA_FIX), sub-baseline (P16=2.0), or saturating (S3=8.0+earlier base). The clean fw=4.0 single-lever test on a non-FT-from-P8A base was NOT done.

### 2026-05-04 — E2B / E3 verdicts + 11 CPU diagnostic jobs + Packet A/PC-codec launch

**E2B** (B16 scratch + CE + heavy aug, memory `project_e2b_breaks_deeplive_ceiling.md`): step3200 breaks deeplive ceiling — 87.5% recall at FPR=10% (vs P8A 42.9%). But viso REGRESSES 27% → 7%. New anchor candidate.

**E3** (L14 scratch + CE, memory `project_l14_does_not_break_viso_ceiling.md`): step6600 viso 11.6% at FPR=10% vs P8A 27.1%. Three arch-distinct packets (P8A B16-FT, E2B B16-scratch, E3 L14-scratch) all fail viso ≥ 30%. Ceiling is structural across architectures. **OPINION (your call to inherit or not)**: this is what motivated the 2026-05-04 morning CPU diagnostic pass.

**11 CPU diagnostic jobs ran 2026-05-04 morning + afternoon** (TIMELINE entries 2026-05-04 morning/afternoon, 11 separate entries). Findings worth surfacing for your perspective:

| Job | Finding | Memory |
|---|---|---|
| Job 1 (suite composition audit) | clean variants of viso/deeplive NOT in production scorecard substrate; teams_fake_all_dev is 40.5% non-teams-captured | (uncodified) |
| Job 2 (per-cell catch patterns) | viso_enhanced 550 frames → P8A 0.269 / E2B 0.084 / E3 0.138 (P8A_only Venn slice = 16.9%); 3-way union recall caps at 33.8% on viso enhanced | `project_eval_substrate_reframe_2026-05-04.md` |
| Job 3 (FP-tail characterization) | Top-5 of 24 dev identities own 80-95% of FPs per ckpt; **chronic-6** identities (`bla_bla_chow`, `bla_bla_chow__s2`, `pc_generator__s22`, `pc_generator__s45`, `roy_d`, `q__s6`) dominate every ckpt; 90.1% of P8A FPs come from frames with `min(W,H)<200` | `project_eval_substrate_reframe_2026-05-04.md` |
| Job 4 (viso subtype) | viso → teams shifts IQ in webcam-failure direction; sharpness coefficient flips +1.00 (raw) → -0.11 (teams); HIGH-confidence reframe of viso→teams as IQ-compounded shortcut, NOT transport-specific | `project_viso_subtype_iq_compounded_2026-05-04.md` |
| Job 6 (frozen-feature region analysis) | P8A frozen probe AUC = 0.969 ± 0.011 on caught-vs-uncaught viso; uncaught region purity 0.924; **encoder DOES separate; head doesn't pick it up**; refutes "representation gap requires encoder change" | `project_viso_head_boundary_finding_2026-05-04.md` |
| Job 7 (head-only retrain) | All 6 retrained heads OVER-FIRE 80-92% on lockbox reals; P8A real_lockbox p50=0.10 (invariant) vs retrained p50=0.79 (substrate-collapsed); **head-only retrain REFUTED**; substrate-invariance is the load-bearing P8A property | `project_job7_head_retrain_REFUTED_2026-05-04.md` |
| Job 9 (per-frame disagreement on viso) | P8A 122 unique viso catches vs E3 56 vs E2B 1 in DISAGREE bands; "scratch has more viso headroom" hypothesis REFUTED | `project_job9_disagreement_audit_2026-05-04.md` |
| Job 11 (chronic-6 partition per ckpt) | Chronic-6 do NOT fail uniformly across ckpts; PC_Generator__s22 P8A 0.91 vs E3 0.06 (Δ=0.85); each chronic identity has at least one ckpt that handles it cleanly | `project_chronic_offenders_partition_per_ckpt_2026-05-04.md` |
| Job 12 (label-free ensemble) | Best non-oracle ensemble on viso = 16.2% < P8A 27%; oracle ceiling 77% on viso → information IS there but no label-free combiner extracts it; out-of-stream router is the structural lever | `project_job12_ensemble_ceiling_2026-05-04.md` |
| Job 13 (per-video viso) | 8 source sequences × {raw, teams} = 16 videos; no video catch >0.90; teams transport uniformly hard with E2B↔E3 r=0.94; raw subtype is where ckpt-disagreement lives | `project_job13_per_video_viso_2026-05-04.md` |
| **Job 14 (substrate cleaning)** | F4 (drop chronic-6 + lowres + no-face): **P8A viso 27→67%, E3 viso 14→78%, E2B viso 8→31%**; FPR collapses to 0.5-3.2%; chronic-6 axis alone explains 96-108% of FPR drop | `project_job14_substrate_clean_2026-05-04.md` |
| Job F1 (recall simulation) | Drop chronic-6 only at tau_F1=0.053: P8A viso 27→63%, deeplive 42→84%, lockbox 54→92%; F1 tau gives 22-28% FPR on full F0 reals | `project_f1_recall_results_2026-05-04.md` |
| IQ gating viability | P8A recall increases monotonically with sharpness (Q1 36% → Q4 98%); E2B INVERTED (Q1 52% → Q3 5%); IQ gating is a P8A lever, non-starter for E2B | `project_iq_gating_viability_2026-05-04.md` |

**Reusable tools authored 2026-05-04 evening** (so the next checkpoints can be evaluated immediately):
- `analysis/substrate_cleaning_eval_2026-05-05/run_clean_eval.py` (F4 substrate-cleaning eval pipeline; reproduces Job 14 within 0.5pp)
- `analysis/per_substrate_tau_calibration_2026-05-05/run_calibration.py` (per-substrate τ tool; reproduces Job 7's 21pp lift, refines to 24.5pp on P8A)

### 2026-05-04 evening — pair-loss verification + Packet A + Packet C-codec

**Pair-loss verification** (`analysis/pair_loss_effect_verification_2026-05-05/FINDINGS.md`): 275 paired (raw, teams) viso fakes with seq_id matching. Verdict LOW. Three findings (computed directly from E2B scores, not via proxy):
- Sign-of-effect REVERSED on E2B: mean raw_score 0.086 < mean teams_score 0.172 (Wilcoxon p=0.0019). Teams transport HELPS E2B catch viso fakes.
- Wrong-way cohort dominates: at τ=0.5, target cohort 3/275 (1.1%) vs wrong-way cohort 29/275 (10.5%); 86% missed in BOTH versions.
- Symmetric pair loss is structurally net-negative across all τ tested (-9 to -14 pp).
- E2B per-pair score correlation r=0.62 → model already exhibits implicit pair-coherence.

**Decision**: pair-loss packet was NOT drafted. Memory `project_pair_loss_premise_refuted_2026-05-04.md` captures the verdict; DEBATE block added to thread `clean_teams_identity_pairing.md`.

**Codec aug verification** (`analysis/codec_aug_verification_2026-05-05/FINDINGS.md`): on 30 paired (raw, teams) viso frames, `TeamsCodecSimulation` cosine similarity to actual transport delta = 0.87-0.99 across 8 IQ axes; magnitude ratio 97-106%. Aug is faithful simulation. This evidence is the load-bearing input to the codec aug lever in Packet C-codec.

**Packets launched** (see top of this doc for IDs / state).

---

## Empirical state (FACTS — cite the source if you re-use these)

What's measurable, no framing:

- **P8A_step5000 viso recall ceiling = 27% on F0 substrate** (P8A baseline, 13+ R13 packets unbroken). On F4 (cleaned) substrate: 67%.
- **E2B_3200 viso recall = 8.4% F0 / 30.9% F4**.
- **E3_6600 viso recall = 13.8% F0 / 77.6% F4**.
- **E2B_3200 deeplive recall at FPR=10% = 87.5%** (vs P8A 42.9%).
- **L14 (E3) does NOT break viso ceiling**: viso 11.6% at FPR=10%.
- **3-way oracle ceiling on viso = 77%**; best non-oracle ensemble = 16.2%.
- **Chronic-6 identities own 80-95% of FPs per ckpt; 90.1% of P8A FPs from `min(W,H)<200`**.
- **P8A frozen-feature probe AUC = 0.969 on caught-vs-uncaught viso**. Encoder separates; head doesn't.
- **Pair-loss premise refuted on E2B for viso** (sign reversed; opposite cohort dominates).
- **Per-mode τ offline lift on P8A = 24.5pp** (lockbox teams_fake recall 54.1% → 78.6% at 4× FPR). NOT deployable (Teams does not surface mode at inference).
- **F4 substrate cleaning is reproducible** (within 0.5pp on every checkpoint tested). Reusable script ready.
- **TeamsCodecSimulation aug calibration verified** on a different sample (cosine 0.87-0.99 vs actual transport).

---

## Framing disputes (areas where bias has accumulated — read with skepticism)

These are framings prior agents (including me) elevated based on partial evidence. Read the dated DEBATE blocks and `PSERIES_OPINIONS_2026-05-02.md`'s superseded entries:

1. **"Data axis is exhausted"** (memory `project_data_axis_lever_pulled_twice_no_lift.md` — see critical-reading note added 2026-05-04 night). Both prior tests had recipe-side confounds (bundle stack on P14_DATA_FIX; sub-baseline fw=2.0 on P16). Packet A is the first clean fw=4.0 single-lever test on a non-P8A base. Verdict ETA ~24h.

2. **"Pair loss is structurally well-targeted"** (thread `clean_teams_identity_pairing.md` — see DEBATE block 2026-05-04). Symmetric KL pair loss is empirically refuted on E2B for viso. Asymmetric variants are untested. The structural same-identity-two-transport finding remains valid; the inference layer to pair-loss design implications is contested.

3. **"P8A is the right anchor because it has the best viso ceiling"** vs **"P8A is a 5x FT chain with confounds, derive a clean version from scratch"**. User has explicitly stated skepticism toward P8A as the anchor (this session). E2B is the new candidate per `project_e2b_breaks_deeplive_ceiling.md`. Packet A and PC-codec FT-from-E2B, not P8A.

4. **"face_scale_jitter@0.50 is load-bearing"** (README "Confirmed good" line 37 — see critical-reading note 2026-05-04 night). Refuted on its own design-intent mechanism the same afternoon (43.9% face-size flip rate vs P8A 39.4%); leader did not promote on contract scorecard. Operating recommendation should be read as "weak strictly less bad," not endorsement.

5. **"Per-mode τ is a viable forward lever"** (threads `webcam_fpr_dominance.md` and `calibration_vs_training_aug.md` — see critical-reading notes 2026-05-04 night). Per-mode τ is OFFLINE-ONLY because Teams does not surface capture mode at inference. The 24.5pp lift is real but undeployable. Substrate-level filters (modern_v2) and per-known-identity readouts ARE deployable.

6. **"Frame-level AUC reframe — model is meaningfully less broken than headline metrics suggested"** (memory `project_p8a_frame_level_auc_2026-04-29.md`). True at the AUC layer; the per-substrate τ tool's deployable single-τ readout shows P8A viso recall is 0.18% at 5% FPR ceiling, which is operationally devastating. Frame-level AUC and deployment-τ recall are not interchangeable readings of model quality.

---

## Where to look first (the canonical surfaces)

Per AGENTS.md update protocol, threads are the canonical record. Cross-cutting surfaces:

- **`docs/packet_retrospectives/threads/viso_bucket_gap.md`** — primary thread for the data-axis lever; contains the new in-progress loop tracking Packet A / PC-codec.
- **`docs/packet_retrospectives/threads/clean_teams_identity_pairing.md`** — DEBATE block (2026-05-04) on pair-loss refutation.
- **`docs/packet_retrospectives/threads/eval_substrate_data_hygiene.md`** — F4 substrate-cleaning pipeline reference.
- **`docs/packet_retrospectives/packets/PA.md`** and **`PC.md`** — in-flight packet retros (results sections to fill in when training lands).
- **Memory MEMORY.md** — index of project memories. The 3 new entries from today are at the bottom.
- **TIMELINE.md** — chronological master index, append-only. Last 18 entries are 2026-05-04.
- **OPEN_LOOPS.md** — mechanically generated; do not edit. To change an entry, edit the source thread block and re-run `python tools/regenerate_open_loops.py` from `docs/packet_retrospectives/`.

---

## Reusable analysis pipelines (CPU, ready to run on Packet A / PC-codec checkpoints)

When Packet A / PC-codec finish (~24h), score them through:

```bash
# F4 substrate-cleaning re-eval
python3 analysis/substrate_cleaning_eval_2026-05-05/run_clean_eval.py \
  --ckpt-name <pkt> \
  --real-csv <real_dev.csv> \
  --fake-csv visomaster_enhanced_macro_dev=<viso.csv> \
  --fake-csv deeplive_enhanced_dev=<dl.csv> \
  --fake-csv teams_fake_all_dev=<tf.csv> \
  --out-dir analysis/substrate_cleaning_eval_2026-05-05

# Per-substrate τ-calibration (deployable single-τ + oracle per-mode for comparison)
python3 analysis/per_substrate_tau_calibration_2026-05-05/run_calibration.py \
  --real-dev <real_dev.csv> \
  --real-lockbox <real_lockbox.csv> \
  --fake-suite "name=viso_dev path=<viso.csv>" \
  --fake-suite "name=deeplive_dev path=<deeplive.csv>" \
  --fake-suite "name=teams_fake_dev path=<teams_fake_dev.csv>" \
  --fake-suite "name=teams_fake_lockbox path=<teams_fake_lockbox.csv>" \
  --out analysis/per_substrate_tau_calibration_2026-05-05/run_<ckpt>/
```

USAGE.md in each dir has full invocation details.

---

## What the user wants you to do

Direct quote (paraphrased) from this session's authorization:

> Reach his own conclusions.

So: read the FACTS. Read the framing disputes. Form your own read on:

- Whether Packet A and/or PC-codec produce a contract-grade lift when they finish.
- Whether the data-axis lever is dead, alive, or in-between (the verdict closes the in-progress loop either way).
- Whether the user's 24h deeplive ship + 72h overall ship goals are tractable on the current evidence.
- What the next move is — IF a next move is even justified given pending verdicts.

What this agent did NOT do, that you might:

- Investigate why Job B (`9082408392701509632`) failed at 18:44 UTC on 2026-05-04. Logs are accessible. Memory `project_job_b_status_2026-05-05.md` captures the prior attempt's status; this attempt's failure is undiagnosed.
- Decide whether to spend $5 on the GPU disconfirmation probe for pair loss (Q1/Q2 on E2B features instead of P8A proxy). The cohort math is dispositive even if the probe finds strong feature signal; the probe could only inform an asymmetric variant.
- Run the F4 substrate-cleaning + per-substrate τ tools on Packet A / PC-codec checkpoints when they finish.
- Decide whether to commit any of the uncommitted YAMLs (PA / PC-codec, prior P14_DATA_FIX, F4 manifest, etc.) — git status shows many untracked files.

---

## Critical-reading notes pattern (newly established 2026-05-04 night)

If you find a thread / memory with framing that's been refuted or significantly nuanced by later evidence, **add a `> **⚠ Critical-reading note (added YYYY-MM-DD)**: ...` banner at the top** rather than rewriting the body. Preserve numbers, citations, and dated subsections; soften only the inference-layer adjectives ("fundamentally", "dispositive", etc.) when authorized to neutralize bias. The five existing critical-reading banners (added 2026-05-04 night to `clean_teams_identity_pairing`, `viso_bucket_gap`, `webcam_fpr_dominance`, `calibration_vs_training_aug`, `README.md`) are the template.

This is the user's preferred pattern for dealing with accumulated bias: never delete factual content; flag the framing as conditional via a top-of-file banner + dated DEBATE block.

---

## What this agent worked on tonight (so you can audit if needed)

- 11 CPU diagnostic jobs ran morning + afternoon (per TIMELINE).
- 4 new CPU verification jobs ran evening (codec aug verification, pair-loss scoping, pair-loss verification sub-agent, substrate cleaning + per-substrate τ tools).
- Packet A drafted + launched.
- Packet C-codec drafted, first launch failed on wandb-entity gotcha, relaunched.
- Wiki maintenance pass per AGENTS.md update protocol: 7 thread extensions, 2 new packet retro stubs, TIMELINE 8 entries, OPEN_LOOPS regenerated, 3 new memories, MEMORY.md index updated.
- Bias-correction pass per user authorization: 5 critical-reading banners on threads/README, 4 phrasings softened in threads, 1 critical-reading banner + 2 framing softens on a memory entry. No factual content deleted.

Session log preserved at `docs/relaunch_handoffs/SESSION_LOG_2026-05-04.md`. That's the contemporaneous narrative; this handoff is the cross-agent baton.

Good luck.
