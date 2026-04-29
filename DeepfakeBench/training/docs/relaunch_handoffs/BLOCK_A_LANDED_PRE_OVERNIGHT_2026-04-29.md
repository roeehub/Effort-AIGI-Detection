# Block A landed + pre-overnight checklist — 2026-04-29

**Audience:** the next agent picking this up before tomorrow's overnight launches. Read top to bottom; the work is sequenced.

**Branch:** `teams-relaunch-root-2026-04-17` (uncommitted working tree as of write time — see "Pending commits" below).

**Scope of this evening:** five non-GPU moves remain before drafting + launching the overnight slate. Each is sized for ≤2 h. Together they unblock or *inform* the overnight YAMLs — do them first, draft second.

---

## 1. What landed in the prior session (Block A — contract v3 lock)

Working-tree diffs, all uncommitted:

| File | Change |
|---|---|
| `arena/score_teams_promotion_contract.py` | `ContractConfig.target_fake_recall_min` default `0.0 → 0.70`; matching CLI default; refreshed help text and field docstring. |
| `arena/run_target_domain_validation_sequential.py` | `--promotion_target_fake_recall_min` CLI default `0.0 → 0.70`; help text updated. |
| `tests/test_score_teams_promotion_contract.py` | +2 regression guards: `test_default_contract_config_has_active_recall_floor` (dataclass default ≥ 0.70) and `test_wrapper_and_scorer_cli_defaults_match_contract_default` (regex-grep both CLI defaults vs dataclass). |
| `arena/checkpoint_maps/teams_target_domain.p14_ft_from_p8a_2026-04-29.yaml` | Stale "uncommitted, pass `--target_fake_recall_min 0.30`" comment → "default is now 0.70, no flag needed". |

**Why 0.70:** the user fixed this as the deployment floor — a Teams detector with macro fake recall < 70 % is not deployment-grade.

**Why default-flip rather than just adding a CLI flag:** the recall-floor gate in `_threshold_sort_key` and `_promotion_summary_sort_key` only activates when `target_fake_recall_min > 0.0`. A 0.0 default means any wrapper that omits the flag silently re-enters τ-tail-collapse (the bug Codex caught in round 2). Making the safe path the default everywhere closes that latent bug. To opt out for legacy comparison, pass `--target_fake_recall_min 0.0` (or `--promotion_target_fake_recall_min 0.0` via the wrapper).

**Tests:** 6/6 passing locally in 0.08 s (4 prior + 2 new).

**Wrapper audit:** all 5 arena launchers (`launch_arena.sh`, `launch_r13_best_megaval.sh`, `launch_target_domain_scorecard.sh`, `launch_teams_promotion_contract.sh`, `run_teams_promotion_contract.sh`, `run_teams_target_domain_scorecard.sh`) inherit the wrapper default — none hardcoded the legacy 0.0. No Cloud Build / cron / CI references. Patch surface complete.

---

## 2. Non-GPU work remaining tonight (~4 h)

Order is recommended. Move 2 should kick off in parallel with Move 1 (image rebuild runs unattended).

### Move 1 — Rescore P8A + P14_FT under floor=0.70 (~20 min)

**Why:** floor=0.30 readout is documented (P8A τ=0.916, viso 13.6 %, deeplive 23.9 %, teams_fake 52.6 %, lockbox FPR 1.8 %; see `docs/packet_retrospectives/STATE_2026-04-29.md:29`). Floor=0.70 readout is **not documented anywhere**. Computing it produces:

- The missing data point (and the question "is any current checkpoint deployment-grade *now*?").
- A baseline against which tomorrow's overnight checkpoints are scored.
- Verdict on tomorrow's plan: if any checkpoint cleanly clears 70 % macro recall within FPR budget (≤7 % primary, ≤10 % stress), we have a deployable candidate already. If not, the overnight experiments are mandatory science, not optional polish.

**Why this is honest, not bar-lowering:** the eval data does not change between floor=0.30 and floor=0.70 — same `*_videos_report.csv` artifacts, same model predictions. Only τ-selection and aggregation change. So a "doesn't clear 70 %" verdict is a real verdict on the existing model, not a scoring artifact.

**Inputs:** existing GCS scorecard reports under `gs://training-job-outputs/test_results/teams_promotion_contract/p14-ft-from-p8a-scorecard-20260429-uswest4/` (and equivalent P8A-only paths if separate). Identify the report root by looking at `arena/checkpoint_maps/teams_target_domain.p14_ft_from_p8a_2026-04-29.yaml` for checkpoint aliases.

**Output:** local CSV + JSON in `analysis/policy_reruns_2026-04-29_floor_0p70/`. Document headline numbers in this handoff (append to "Open data points" section below).

**Run command (illustrative):**

```bash
WANDB_API_KEY=... WANDB_ENTITY=dtect-vision WANDB_PROJECT=phase2-experiments \
  python arena/score_teams_promotion_contract.py \
  --report_root gs://training-job-outputs/test_results/teams_promotion_contract/p14-ft-from-p8a-scorecard-20260429-uswest4/ \
  --checkpoint_map arena/checkpoint_maps/teams_target_domain.p14_ft_from_p8a_2026-04-29.yaml \
  --checkpoints ALL \
  --output_dir analysis/policy_reruns_2026-04-29_floor_0p70/ \
  --target_fake_recall_min 0.70
```

(With Block A landed, `--target_fake_recall_min 0.70` is already the default; passing it explicitly here is documentation for the rerun artifact.)

**Watch for:** the contract will fall back to tier 1 or tier 2 if no τ clears both budget AND floor. Inspect the `selected_threshold_scorecard.csv` and threshold-grid outputs to confirm which tier was used and at what (recall, FPR) operating point.

**Definition of done:** an entry in this handoff under "Open data points" naming the τ, macro recall, primary/stress FPR, lockbox real FPR, and tier for at least P8A_REFERENCE_STEP5000.

---

### Move 2 — Commit Block A + image rebuild (5 min + ~20 min background)

**Why required:** Vertex scorecards run inside the production image. Without an image rebuild, any Vertex-side rerun reads the old code. The rebuild is CPU-only (Cloud Build), doesn't burn GPU budget.

**Sequence:**

1. With user explicit OK: `git add -A` (only the four Block A files — no other working-tree changes), then `git commit` with a clear message.
2. `./dev.sh build-prod -y` — auto-bumps VERSION patch, kicks Cloud Build. Run in background (`Bash run_in_background=true`); user is notified on completion. **Do not** sleep-poll.
3. After rebuild: `git status` to confirm VERSION bumped + a fresh commit landed for the version bump.

**Definition of done:** Cloud Build succeeded, `gcloud artifacts docker tags list` (or local `cat VERSION`) shows the new tag, and the new image's promotion-contract scorer has the 0.70 default (verifiable via a one-line `gcloud ai custom-jobs ...` smoke test if needed, but the local test suite is sufficient signal).

**Authorization gate:** user has not yet given explicit OK to commit. **Do not commit without it.** (User preference — `feedback_decision_points.md`.)

---

### Move 3 — Inventory `visomaster_teams_enhanced` bucket (~30 min)

**Critical context:** training has **never used `visomaster_teams_enhanced`** data. Training viso has no enhancers and no Teams pipeline; eval viso has both. This is the load-bearing bucket gap (`project_viso_train_eval_bucket_gap.md`, `STATE_2026-04-29.md`). `R13_P14_DATA_FIX.yaml` was drafted to fix it and never launched. P14_DATA_FIX is the overnight slot that closes this loop. Inventory now decides whether Path-A (direct conjunction source) is feasible or whether the YAML must fall back to Path-B (union of single-axis sources).

**Why this matters tonight:** if the bucket has fewer identities than expected (e.g. <100), Path-A produces an under-powered training signal and a P14_DATA_FIX run risks failing for data-volume reasons that look identical to the model failing — wasting a 12-hour overnight slot.

**What to compute:**

- GCS list of the `visomaster_teams_enhanced` bucket (or the equivalent labelled source — confirm exact path from existing `arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml` and the `arena/build_visomaster_proper_data_artifacts.py` plumbing).
- Identity count (deduplicate by `identity_id` or equivalent).
- Sub-method distribution (which face-swap engines, how many frames each).
- Companion bucket / paired transport check: do these identities have matching reals in `visomaster_real` and `*_clean` buckets? (`project_clean_teams_same_identity.md` says yes — companion_bucket plumbing exists in code.)
- Spot-check 4–6 random frames: not corrupted, faces present, expected resolution range.

**Decision output:**

- Path-A feasible? Yes / No / Constrained — with identity count.
- Recommendation: which path the P14_DATA_FIX YAML should use.

**Definition of done:** entry in this handoff under "Open data points" with bucket statistics + Path-A/B verdict.

---

### Move 4 — W&B Block B: mid-training observability (~2 h, scout first)

**Why:** the user's stated pain point is "we are always surprised when the model is actually not that good." Aggregate `val_acc` averages across buckets where shortcuts inflate them. Per-bucket panels make trajectory legible *while* overnight runs are live, so misbehaviour is caught at hour 2 not hour 12.

**Target panels (logged at fixed step intervals — user picked 1000):**

- `mid_eval/deeplive_recall_clean`, `mid_eval/deeplive_recall_teams`
- `mid_eval/viso_recall_clean`, `mid_eval/viso_recall_teams`
- `mid_eval/lockbox_FPR_webcam`, `mid_eval/lockbox_FPR_studio` (split per `project_lockbox_fpr_dominated_by_webcam_mode.md`: webcam ≈ 65.7 % FPR drives the headline)
- `mid_eval/face_size_corr` (per-batch face-pixel-area ↔ score correlation; drift up = label leak active per `project_face_size_label_leak.md`)

**Pin all to a custom W&B workspace** so all three overnights show in one view.

**Bug-prevention rules (user-required):**

- If touching any yaml structure: update the `train_sweep.py` re-apply allowlist (~lines 176–282) per `project_wandb_flattens_nested_dicts.md`. Failure mode: trainer reads None, run looks fine, results corrupt.
- Add a focused unit test before image rebuild for any trainer-side change.
- Verify the in_proj-SVD-style silent-zero-gradient pattern doesn't recur (`project_in_proj_svd_gradient_bug.md`): if registering a new logging tap on a tensor, confirm autograd path with a deliberate shape-check test.

**Scope-control gate (first 15 min):** scout whether the trainer already exposes a mid-training eval hook. If yes, this is a 30 min wiring job. If no — i.e. would require new hook plumbing — drop to a downgraded scope: log per-bucket recall/FPR inside the existing eval cycle (slower cadence but no new hook). User pre-authorized the downgrade.

**Definition of done:** a new W&B custom workspace exists; at least one local dry-run shows the panels populating; tests added for any trainer-side change.

---

### Move 5 — Face-size leak scatter (~30 min)

**Why:** `project_face_size_label_leak.md` records that each fake method clusters at a tight face-pixel-area band; reals span wider; the model uses face size as a fake predictor. Empirically validating this on the existing P8A scorecard reports tells us *how active* the leak is in current models. Strong correlation → swap one overnight slot for "P8A + face-scale-jitter" instead of compound. Weak correlation → keep current slate.

**Inputs:** any existing `*_videos_report.csv` that has both `avg_video_prob` and frame-level face-size info. The per-frame metadata likely lives upstream of the video-report aggregation; check `analysis/check_frame_4people_2026-04-24.py` and the cohort manifests for paths to per-frame face-pixel area.

**Output:** local matplotlib scatter — face_pixel_area vs avg_video_prob, coloured by label (real/fake), one panel per bucket. Save to `analysis/face_size_leak_scatter_2026-04-29/`.

**Definition of done:** scatter plots saved locally + a one-line verdict ("strong / moderate / weak / no correlation") logged in this handoff under "Open data points".

---

## 3. Then — and only then — draft the overnight slate

User explicitly deferred the overnight slate decision to a clean chat. The pre-work above informs that decision; do not pre-commit to YAMLs before Moves 1, 3, and 5 produce their verdicts.

**Provisional slate (`MASTER_PLAN_2026-04-29.md` consensus):**

1. **P14_DATA_FIX** — visomaster_teams_enhanced wired in (Path A or B per Move 3 verdict).
2. **P8A + GRL on `capture_mode`** — camera-signature shortcut (`project_p8a_breakthrough.md`, `project_shortcut_is_upstream.md`).
3. **P14_DATA_FIX + GRL compound** OR **face-scale-jittered P8A** — choice contingent on Move 5 verdict.

**Cost envelope:** 3 × ~$70 ≈ $210. Well inside the user's $1000 budget. All in US regions per `CLAUDE.md` region preference (us-west4 / us-east1 / us-central1, queue across if needed).

**Pre-launch checklist (each YAML):**

- [ ] WANDB env vars exported (`feedback_promotion_contract_launch.md`)
- [ ] Identity-disjoint group splits (groups by `identity, capture_session, source_bucket` per memory)
- [ ] `train_sweep.py` allowlist updated if yaml block is new
- [ ] Smoke-test the YAML with `--dry-run` or config-only mode
- [ ] User explicit authorization to launch (no Vertex jobs without OK)

---

## 4. Pending commits

Working-tree changes from Block A still uncommitted (user prefers explicit commits):

- `arena/score_teams_promotion_contract.py`
- `arena/run_target_domain_validation_sequential.py`
- `tests/test_score_teams_promotion_contract.py`
- `arena/checkpoint_maps/teams_target_domain.p14_ft_from_p8a_2026-04-29.yaml`

Plus this handoff file (`docs/relaunch_handoffs/BLOCK_A_LANDED_PRE_OVERNIGHT_2026-04-29.md`) once written.

---

## 5. Open data points (fill in as you go)

Append entries here as Moves complete. This is the durable record the next-next-agent reads.

### Move 1 — floor=0.70 rescore (TODO)

| checkpoint | τ | macro recall | primary FPR | stress FPR | lockbox real FPR | tier |
|---|---|---|---|---|---|---|
| P8A_REFERENCE_STEP5000 | _ | _ | _ | _ | _ | _ |
| P14_FT_step3000 | _ | _ | _ | _ | _ | _ |

### Move 3 — visomaster_teams_enhanced inventory (TODO)

- Bucket path: _
- Identity count: _
- Methods: _
- Companion-bucket alignment: _
- Path-A / Path-B verdict: _

### Move 5 — face-size leak scatter verdict (TODO)

- Per-bucket correlation strength: _
- Implication for overnight slate: _

---

## 6. Cross-references

- `docs/packet_retrospectives/STATE_2026-04-29.md` — current state of the world (frame-level AUCs, eval substrate caveats, contract policy verdict).
- `docs/packet_retrospectives/plans/MASTER_PLAN_2026-04-29.md` — master plan; lines 107, 370, 666, 897 cite the load-bearing facts above.
- `docs/packet_retrospectives/threads/contract_policy_bug.md` — full bug history and Slice 7 fix design. Append a "Resolved 2026-04-29" section once Block A is committed.
- `docs/relaunch_handoffs/P13_DAY4_VERDICT_2026-04-29.md` — P13 verdict (modal α — don't launch P13 again).
- `docs/relaunch_handoffs/P15_GRL_READINESS_NOTE_2026-04-29.md` — GRL design notes for overnight slot 2.
- `PLAN.md` (top-level, untracked) — earlier planning context.
- `CLAUDE.md` — Vertex region preference (US-only) and queue policy.

---

## 7. Pattern for the next agent's handoff

When you finish your share of these moves, write a successor handoff in this same `docs/relaunch_handoffs/` directory using the `<TOPIC>_<YYYY-MM-DD>.md` naming. Carry forward (a) what landed in your session, (b) what's still open, (c) the "Open data points" table, and (d) cross-references. Don't duplicate the master plan or thread files — point at them.
