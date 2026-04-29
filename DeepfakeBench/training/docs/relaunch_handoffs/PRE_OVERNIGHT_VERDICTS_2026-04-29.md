# Pre-overnight verdicts — 2026-04-29 (successor to BLOCK_A_LANDED)

**Audience:** the next agent drafting and launching the overnight YAML slate. The five non-GPU moves from `BLOCK_A_LANDED_PRE_OVERNIGHT_2026-04-29.md` are done. This handoff carries the verdicts and the load-bearing data so the slate can be designed against ground truth.

**Branch:** `teams-relaunch-root-2026-04-17`. Two new commits on top of the predecessor handoff. One VERSION bump pending (see "Pending commits").

---

## 1. What landed in this session

| Commit | Subject | Notes |
|---|---|---|
| `974e033` | Promotion contract: default recall floor to 0.70 (Block A) | 5 files, +563/-16. Tests 6/6 green locally. |
| `2660856` | W&B Block B: per-dataset-bucket recall/FPR mid-training observability | trainer/trainer.py +93 + new test file +193. 6 new TDD tests; helper `_compute_per_bucket_recall_fpr`; key shape `{log_prefix}/per_bucket/<dataset>/{recall_fake,recall_real,fpr}`. Out of scope (deferred): clip_capture_mode webcam-vs-studio split, face_size_corr panel — both require new metadata flow that was explicitly excluded. |

**Image rebuilds:**

- Cloud Build #1: VERSION `1.3.226 → 1.3.230`, build `17223cdd-4b9a-4be3-b6ed-1cc412d43749` SUCCESS, ~4m39s. Image `effort-detector:1.3.230` = Block A only (snapshotted before Block B committed). **Superseded by #2 — do not launch overnight runs against 1.3.230.**
- Cloud Build #2: VERSION `1.3.230 → 1.3.231` IN FLIGHT at write time. Image `effort-detector:1.3.231` = Block A + Block B. **This is the image to launch overnight runs against.** Verify completion via `gcloud builds list --filter="tags=effort-detector" --limit=3` before launch.

---

## 2. Verdicts from the five Moves

### Move 1 — floor=0.70 rescore: **no current checkpoint is deployment-grade**

Outputs at `analysis/policy_reruns_2026-04-29_floor_0p70/`. Source: `gs://training-job-outputs/test_results/teams_promotion_contract/p14-ft-from-p8a-scorecard-20260429-uswest4/reports/`. Note: invoked with `--readout_only_suites ""` because `teams_real_dor_dev` is not present in this scorecard root.

| checkpoint | τ | macro recall | primary FPR | stress FPR | lockbox real FPR | tier (sel/prom) | rank |
|---|---|---|---|---|---|---|---|
| P8A_REFERENCE_STEP5000 | 0.9156 | 0.300 | 0.0695 | 0.0685 | 0.0184 | 1/1 | 5 |
| P14_FT_FROM_P8A_STEP500 | 0.8773 | 0.178 | 0.0689 | 0.0999 | 0.0051 | 1/1 | **1 (winner — degenerate)** |
| P14_FT_FROM_P8A_STEP1000 | 0.9279 | 0.164 | 0.0698 | 0.0992 | 0.0154 | 1/1 | 3 |
| P14_FT_FROM_P8A_STEP1500 | 0.7621 | 0.277 | 0.0692 | 0.0985 | 0.0169 | 1/1 | 4 |
| P14_FT_FROM_P8A_STEP2000 | 0.8013 | 0.279 | 0.0695 | 0.0999 | 0.0198 | 1/1 | 6 |
| P14_FT_FROM_P8A_STEP3000 | 0.7238 | **0.449** | 0.0689 | 0.0999 | 0.0411 | 1/1 | 10 |
| P14_FT_FROM_P8A_STEP3500 | 0.6208 | 0.333 | 0.0649 | 0.0999 | 0.0345 | 1/1 | 8 |
| P14_FT_FROM_P8A_STEP4000 | 0.5877 | 0.329 | 0.0667 | 0.0999 | 0.0331 | 1/1 | 7 |
| P13_FROM_SCRATCH_STEP18000 | 0.9930 | 0.110 | 0.0563 | 0.0999 | 0.0147 | 1/1 | 2 |
| RLP6_04_BASELINE_STEP23500 | 0.9706 | 0.378 | 0.0698 | 0.0657 | 0.0375 | 1/1 | 9 |

**Verdict:** every checkpoint fails the 0.70 macro-recall floor. Best macro recall = 0.449 on `P14_FT_FROM_P8A_STEP3000`; P8A baseline = 0.300. FPR budgets are easily met — recall is the binding constraint. The ranker's "winner" is degenerate (`P14_FT_FROM_P8A_STEP500` is just the lowest-FPR floor-failure, only 17.8 % macro recall): exactly the τ-tail-collapse pathology the recall floor was added to surface.

**Reframe for the slate:** the overnight runs are mandatory science, not optional polish. There is no deployable detector to ship today.

**Note on P8A baseline reproducibility:** floor=0.70 P8A τ=0.9156 / lockbox real FPR 1.84 % matches the prior floor=0.30 datapoint in `STATE_2026-04-29.md` line ~29 — confirming the rescore is a faithful re-aggregation of identical predictions. Macro recall 0.300 ↔ per-suite 13.6 % / 52.6 % / 23.9 % (visomaster / teams_fake / deeplive).

### Move 3 — `visomaster_teams_enhanced` inventory: **Hybrid (Path-B + weighted Path-A)**

Authoritative source: resolver manifest `gs://training-job-outputs/cache/visomaster/enhanced_visomaster_resolver_2026-04-06.json`, viewer cache at `.viewer_cache/discovery/visomaster_teams_enhanced.json`.

- **Total sample_ids in resolver:** 999 (≈ identities by clip-level convention).
- **Strict conjunction (`teams_v2_companion`):** 54 sample_ids — Path-A pool. Below the <100 threshold flagged for under-power risk.
- **Clean-companion-only:** 943. Enhancers exist but reals/base-fakes only resolved to the clean bucket.
- **Methods with zero conjunction coverage:** `InStyleSwapper256-C`, `SimSwap512` (139 + 8 enhancer samples but no Teams-v2 companion).

**Method spread on the 54-id conjunction pool:** GhostFace-v1 11, GhostFace-v3 11, InStyleSwapper256-A 9, InStyleSwapper256-B 7, Inswapper128 6, CSCS 5, GhostFace-v2 5. Per-method floors as low as 5 — not enough breadth to claim per-method generalization on the conjunction axis.

**Companion-bucket alignment:** 100 % of fake sample_ids have a paired real (`has_pair: True` for every entry in viewer cache). Paired-transport assumption from `project_clean_teams_same_identity.md` is intact.

**Spot-check (6 frames, 168–219 px range):** all readable, faces visible, no corruption. Resolutions are at the small end of expected (production-tight crops); enhanced PNGs vs base JPGs is a structural feature; Japanese subtitle overlay on Teams-v2 frames confirms the production-pipeline transport is genuine.

**Verdict — Hybrid:**
1. **Path-B as volume backbone** — `visomaster_enhanced` (997 enhanced fakes, no Teams transport) + `visomaster_hints_teams` (~404 Teams transport, no enhancers). Provides ~10× the identity coverage of Path-A and full method spread, at the cost of an empirical assumption that the union approximates the conjunction.
2. **Path-A (54 ids) as a small weighted source** — `visomaster_teams_enhanced` with `companion_domains=["teams_v2"]` enabled but down-weighted so it biases gradient toward the eval-substrate distribution without dominating. Plumbing already exists at `data/sources/combined_paired.py:704` and the resolver-of-record path above.

**Confidence: high** — based on resolver-of-record + viewer cache + live `gsutil ls` cross-check (999 dirs match resolver). Verdict could shift to "Path-A alone is viable" only if the resolver is re-run and `teams_v2_companion` count crosses ~150–200. Worth re-running `tools/audit_enhanced_visomaster_resolver.py` before launch — current snapshot is dated 2026-04-06 (3 weeks old).

### Move 5 — face-size leak scatter: **swap slot 3 to face-scale-jitter**

Outputs at `analysis/face_size_leak_scatter_2026-04-29/` (per-bucket PNGs + grid + per_bucket_verdicts.csv). Inputs: P8A scorecard `*_videos_report.csv` from `gs://training-job-outputs/test_results/teams_promotion_contract/p8a-review-scorecard-20260425/` joined to `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` for face_pixel_area.

| bucket | n_real | n_fake | r_real | r_fake | verdict |
|---|---:|---:|---:|---:|---|
| deeplive_enhanced_dev | 0 | 536 | n/a | -0.214 | weak fake |
| teams_fake_all_dev | 0 | 1825 | n/a | +0.383 | moderate fake |
| **teams_fake_all_lockbox** | 0 | 253 | n/a | **-0.529** | **STRONG fake** |
| teams_real_all_dev | 2610 | 0 | -0.287 | n/a | weak real |
| **teams_real_all_lockbox** | 385 | 0 | **-0.565** | n/a | **STRONG real** |
| teams_real_lighting_extreme_dev | 1183 | 0 | -0.231 | n/a | weak real |
| teams_real_poor_quality_dev | 722 | 0 | -0.279 | n/a | weak real |

Skipped: `visomaster_enhanced_macro_dev`, `teams_real_dor_dev` — parquet tagging cohort mismatch (different filenames), not a data gap.

**Reading the verdict:** the lockbox cohorts (the OOD holdout — i.e. the deployment-grade regime) show **strong** correlation in both classes with the same shortcut: small face ⇒ model says fake.

- Reals lockbox: tiny-face reals (~1–3k px²) are the FPR generators; large-face reals (~25–45k) are correctly low-prob.
- Fakes lockbox: small-face fakes (~25–45k, training-distribution band) caught at prob ~1; out-of-distribution larger-face fakes (~75–110k) widely missed.
- The pooled-cross-class plot (`pooled_lockbox_fake_real.png`) shows the two pools sit in nearly disjoint face-size ranges — exactly the leak pattern memory `face_size_label_leak.md` predicted, now empirically confirmed on P8A's own outputs.

**Structural overlap with the camera-signature shortcut:** the 65.7 % webcam-mode FPR finding (memory `lockbox_fpr_dominated_by_webcam_mode.md`) is consistent — webcam captures produce smaller face crops. The face-size leak and the webcam shortcut are partially the same underlying phenomenon.

**Recommendation:** swap the third overnight slot from "P14_DATA_FIX + GRL compound" to **P8A + face-scale-jitter** — this attacks the leak directly (random face-crop resize at training time prevents absolute face_pixel_area from being a class signal). The strong magnitudes in the lockbox regime make this a load-bearing intervention worth one slot.

### Moves 2 + 4 — landed (commits + image)

See section 1.

---

## 3. Implications for the overnight slate

**Provisional slate from `MASTER_PLAN_2026-04-29.md` was:**
1. P14_DATA_FIX (visomaster_teams_enhanced wired)
2. P8A + GRL on `capture_mode`
3. **P14_DATA_FIX + GRL compound** OR **face-scale-jittered P8A** — TBD on Move 5

**Updated recommended slate:**

| Slot | YAML | Rationale | Source verdict |
|---|---|---|---|
| 1 | **P14_DATA_FIX (Hybrid)** — Path-B union + Path-A weighted | 54 strict-conjunction ids alone is under-powered; union gives volume + correct distribution coverage | Move 3 |
| 2 | **P8A + GRL on `capture_mode`** | Camera-signature shortcut is the named upstream (project_shortcut_is_upstream.md, project_p8a_breakthrough.md) | Master plan, unchanged |
| 3 | **P8A + face-scale-jitter** | Strong leak on lockbox in BOTH classes (r=-0.565 / -0.529); attacks shortcut directly | Move 5 (replaces compound) |

**Cost:** ~3 × $70 ≈ $210, US regions per `CLAUDE.md` (us-west4 / us-east1 / us-central1, queue across if needed).

**Image to launch against:** `effort-detector:1.3.231` (after Cloud Build #2 completes; verify before launch).

**Pre-launch checklist (each YAML, from predecessor handoff):**

- [ ] `WANDB_API_KEY` / `WANDB_ENTITY=dtect-vision` / `WANDB_PROJECT=phase2-experiments` exported (`feedback_promotion_contract_launch.md`)
- [ ] Identity-disjoint group splits (groups by `identity, capture_session, source_bucket`)
- [ ] `train_sweep.py` re-apply allowlist updated if yaml block is new (`project_wandb_flattens_nested_dicts.md`, ~lines 176–282)
- [ ] Smoke-test the YAML with `--dry-run` or config-only mode
- [ ] User explicit authorization to launch (no Vertex jobs without OK — `feedback_no_cancelling_vertex_jobs.md`)
- [ ] Image tag in YAML matches `1.3.231` (or whatever Cloud Build #2 produces)

**Mid-flight monitoring (Block B — newly available):** the new W&B keys `{log_prefix}/per_bucket/<dataset>/{recall_fake,recall_real,fpr}` will fire at the existing `evaluate_every_steps` cadence. For each overnight run, set `evaluate_every_steps: 500` (or 1000) in the YAML. Pin the per-bucket panels in a custom W&B workspace so all three runs share one view. Per-bucket recall on `teams_fake`, `deeplive`, `viso` is the load-bearing trajectory check; aggregate `val_acc` is no longer the only signal.

**Recommended pre-launch refresh:** before launching Slot 1, re-run `tools/audit_enhanced_visomaster_resolver.py` to refresh the 2026-04-06 resolver snapshot. If `teams_v2_companion` has grown to ~150+ since 2026-04-06, re-evaluate whether Path-A alone becomes viable (would simplify the YAML).

---

## 4. Pending commits

- VERSION 1.3.231 bump (in working tree at write time, will land when Cloud Build #2 reports SUCCESS). Suggested message: `Bump VERSION to 1.3.231 (Block A + Block B image)`.
- Predecessor handoff `BLOCK_A_LANDED_PRE_OVERNIGHT_2026-04-29.md` was committed as part of `974e033`. This successor handoff (`PRE_OVERNIGHT_VERDICTS_2026-04-29.md`) should be added in the same commit as the VERSION bump, or in a follow-on commit — user's choice.

Other working-tree changes (`CLAUDE.md`, `HANDOFF.md`, `arena/build_visomaster_proper_data_artifacts.py`, etc.) are pre-existing and outside the scope of this session.

---

## 5. Deferred items (for the next agent or future session)

| Item | Why deferred | Effort estimate |
|---|---|---|
| `clip_capture_mode` metadata flow into `data_dict` | Requires dataloader changes; explicitly out-of-scope for Move 4 | ~30–60 min, medium risk |
| `mid_eval/face_size_corr` panel | Requires per-frame face-pixel-area at training time; not in current eval flow | ~45 min, low risk if metadata available |
| Custom W&B workspace pinning the new panels | UI-only; user can do in W&B directly | ~10 min, no code |
| Resolver re-audit of `visomaster_teams_enhanced` | Recommended pre-launch but not blocking | ~15 min, read-only |
| Overnight slate YAML drafting + launch | User explicitly deferred to a clean chat | bulk of the next session |
| Commit VERSION 1.3.231 after Cloud Build #2 succeeds | Build still in flight | ~5 min |

---

## 6. Cross-references

- `docs/relaunch_handoffs/BLOCK_A_LANDED_PRE_OVERNIGHT_2026-04-29.md` — predecessor (the 5 Moves brief)
- `docs/packet_retrospectives/STATE_2026-04-29.md` — state of the world (frame-level AUCs, eval substrate caveats, contract policy verdict)
- `docs/packet_retrospectives/plans/MASTER_PLAN_2026-04-29.md` — master plan (supersede slot-3 with Move 5 verdict)
- `docs/packet_retrospectives/threads/contract_policy_bug.md` — full bug history; append "Resolved 2026-04-29" once VERSION 1.3.231 lands
- `docs/relaunch_handoffs/P15_GRL_READINESS_NOTE_2026-04-29.md` — GRL design notes for overnight slot 2
- `analysis/policy_reruns_2026-04-29_floor_0p70/` — Move 1 outputs
- `analysis/face_size_leak_scatter_2026-04-29/` — Move 5 outputs (PNGs + verdicts CSV)
- `gs://training-job-outputs/cache/visomaster/enhanced_visomaster_resolver_2026-04-06.json` — Move 3 authoritative resolver
- `CLAUDE.md` — Vertex region preference (US-only) and queue policy
- `MEMORY.md` — load-bearing memories: face_size_label_leak, p8a_breakthrough, shortcut_is_upstream, lockbox_fpr_dominated_by_webcam_mode, viso_train_eval_bucket_gap, clean_teams_same_identity, in_proj_svd_gradient_bug, wandb_flattens_nested_dicts, no_cancelling_vertex_jobs, gcs_region_locality

---

## 7. Pattern for the next agent's handoff

When you finish drafting + launching the overnight slate, write a successor handoff (e.g. `OVERNIGHT_LAUNCH_LIVE_2026-04-30.md` or similar) carrying: (a) which slate you launched and against which image tag, (b) per-job Vertex IDs and W&B run URLs, (c) live monitoring instructions, (d) early-warning thresholds for per-bucket panels, (e) cross-references back to this handoff and the master plan. Don't duplicate the verdicts or master plan — point at them.
