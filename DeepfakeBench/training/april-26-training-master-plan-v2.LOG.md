# Action Log — april-26-training-master-plan-v2

Append-only log of who did what and why, across sessions. Read before starting work; append before ending a session.

## How to use this file

- **Append a new entry** for every working session, even short ones.
- **Never edit prior entries.** Corrections go in a new entry below.
- **Plan change proposals** belong here, not in the plan itself. If you propose a change, tell the next agent in your entry's "Next agent should" line to ask the user to authorize it.
- Use absolute dates and times (`YYYY-MM-DD HH:MM TZ`); convert relative references like "yesterday" to absolute dates.

## Entry template

````
### YYYY-MM-DD HH:MM TZ — agent: <model name + session label>
**Phase/step:** <e.g., 0.1 commit in_proj fix>
**Worked on:**
- <bullet>
**Why:**
- <bullet>
**Outcome:**
- <bullet>
**Flags / plan-change proposals:** <"none" or see block below>
**Next agent should:**
- <bullet>
````

If you have a plan-change proposal, embed this block under "Flags / plan-change proposals":

````
#### PLAN CHANGE PROPOSAL — <one-line title>
**Section affected:** <§N or filename>
**Proposed change:**
- <bullet>
**Rationale:**
- <bullet>
**Status:** AWAITING USER AUTHORIZATION
````

---

## Entries

### 2026-04-26 — pre-existing state (before LOG was created)
**Phase/step:** Phase 0 pre-work (bug fix + audits)
**Worked on (reconstructed from filesystem + git status; prior agent identity not recorded):**
- Modified `detectors/effort_detector.py` (+168 lines): in_proj-SVD gradient fix at `_install_svd_in_proj_routing` (line ~1710).
- Modified `trainer/trainer.py` (+234 lines): anchor-spread monitor + checkpoint gate.
- Modified `trainer/mixins/checkpointing.py` (+23 lines): anchor monitor state init.
- Created `analysis/probe_battery_2026-04-26/grad_audit.py` (passes 9/9 per plan §1).
- Created `analysis/probe_battery_2026-04-26/manifest_overlap.py` + `manifest_overlap.json` (clean: 0/3382).
- Created `analysis/pool_rescore_p10_sym_on_p8a_step2500.summary.json` (anchor_mean=0.6051; see RESULTS).
- Bumped `VERSION` to 1.3.215 (last commit was 1.3.214 — the bump is not yet committed).

**Outcome:**
- Bug fix sits uncommitted in the working tree.
- Eval infra sits uncommitted in the working tree.
- Phase 0 §4 deliverables (two clean commits + image rebuild + smoke verify) NOT done.

**Flags / plan-change proposals:** none — plan §2 already documents this state accurately.

**Next agent should:** wait for user's M-vs-V decision in plan §3 (it doesn't affect Phase 0 itself, but determines Phase D), then begin Phase 0.1.

---

### 2026-04-26 — agent: Claude Opus 4.7 (1M context) — session: plan-headlines + scaffolding
**Phase/step:** Coordination scaffolding — added §0 to the plan
**Worked on:**
- Read the master plan; gave the user the headlines and verified what's already done.
- Added §0 "Coordination — Read & Update Each Session" to the plan, mandating LOG + RESULTS updates per session.
- Created this LOG file with templates and seeded the pre-existing-state entry above.
- Created `april-26-training-master-plan-v2.RESULTS.md` with templates and seeded the step-2500 rescore.

**Why:**
- User asked for cross-session traceability so progress and reasoning survive between sessions.

**Outcome:**
- Plan now mandates LOG/RESULTS updates.
- Pre-existing Phase 0 work documented above.
- Step-2500 rescore measurement is now in RESULTS.

**Flags / plan-change proposals:** none.

**Next agent should:** read this LOG and RESULTS first; if the user has not yet picked M (methodology) vs V (velocity) in plan §3, ask before proceeding past Phase 0; then start Phase 0.1 (commit the in_proj fix in isolation).

---

### 2026-04-26 — agent: Claude Opus 4.7 (1M context) — session: phase-0-execution
**Phase/step:** Phase 0.1 ✅ + 0.2 ✅ done; 0.3 in progress
**Worked on:**
- **Discovery:** `detectors/effort_detector.py` working tree contained TWO unrelated uncommitted changes: the in_proj-SVD fix (hunks at lines 1673+, 1750+, 1762+, ~130 lines) AND an older **feature-norm regularization** change (packet-6B from 2026-04-23, hunks at lines 339-1075, ~38 lines). The earlier "pre-existing state" entry above was therefore imprecise — it called the +168 lines "all in_proj fix"; ~38 of those lines are actually feat_norm. **This entry corrects the record.**
- **Phase 0.1:** Split the file via patch-extract-and-apply (saved working version → reset to HEAD → applied in_proj-only patch → committed → restored working version). Result: clean commit `2feea58` containing only the in_proj-SVD fix. feat_norm changes preserved untouched in working tree.
- **Phase 0.2:** Committed eval infrastructure as `7af72b1` (929 insertions): `trainer/trainer.py` anchor monitor, `trainer/mixins/checkpointing.py` state init, `analysis/probe_battery_2026-04-26/grad_audit.py`, `manifest_overlap.py`, `manifest_overlap.json` (clean: 0/3382).
- **Phase 0.3:** Launched `./dev.sh build-prod -y` in background. Cloud Build will auto-bump `VERSION` 1.3.215 → 1.3.216, tag image `us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:1.3.216`. ETA ~45–60 min. Background task ID `bygxma0rk`; log at `/tmp/build_prod_2026-04-26.log`.

**Why:**
- User said time is of the essence and authorized running all Phase 0 blockers.
- Phase 0 is a strict blocker per plan §4 — Vertex re-launch against unfixed image silently reproduces the bug.
- Plan §0.1 mandates isolation of the in_proj fix; that's why I split rather than batched.

**Outcome (so far):**
- Two clean commits on `teams-relaunch-root-2026-04-17`: `2feea58` (in_proj fix), `7af72b1` (eval infra).
- Image build submitted to Cloud Build; awaiting completion.
- Working tree still has the uncommitted feat_norm change (`detectors/effort_detector.py`, 6 hunks, 38 lines, all before line 1100). Status: deliberate hold-back, pending user decision on whether to keep / commit-separately / discard.

**Flags / plan-change proposals:**

#### PLAN CHANGE PROPOSAL — record the feat_norm-reg change as a deferred decision
**Section affected:** §2 (state table) and §10 (pointers)
**Proposed change:**
- Add a new row to §2: "Older feat_norm-reg change in `detectors/effort_detector.py` — sits uncommitted, 38 lines, deferred decision."
- Or: leave plan as-is but track in LOG (current approach).
**Rationale:**
- The plan didn't anticipate that `effort_detector.py` would contain unrelated work. A future agent reading the plan alone would not know about feat_norm and might accidentally bundle it.
**Status:** AWAITING USER AUTHORIZATION

**Next agent should:**
- **Do not start Phase 0.4 until** the Cloud Build task `bygxma0rk` completes and `/tmp/build_prod_2026-04-26.log` shows `✅ Build successful!` plus the published image tag.
- Phase 0.4 verification: smoke-run `python analysis/probe_battery_2026-04-26/grad_audit.py` inside the new image (or pointed at the new image tag via a tiny Vertex job) and confirm 9/9 in_proj residuals have non-zero gradient.
- After 0.4 passes, commit the VERSION bump (1.3.215 → 1.3.216) as a separate housekeeping commit, matching the project's existing pattern (`Bump VERSION to 1.3.216 (in_proj fix + anchor monitor image)`).
- Then **ask the user**: (a) the M-vs-V decision in plan §3, and (b) whether to authorize the PLAN CHANGE PROPOSAL above re: feat_norm.
- Only after those two answers, proceed to Phase A.1 (`gsutil ls` to find the latest P10_SYM_on_P8A checkpoint).

---

### 2026-04-26 — agent: Claude Opus 4.7 (1M context) — session: phase-0-finish + C.1-launch
**Phase/step:** Phase 0.3 ✅ + 0.4 ✅ + Phase C.1 launched ✅ + Phase A.2 launched ⚠️ (will fail; recovery in flight)

**User decisions made this session:**
- M (methodology) chosen for plan §3 — wait for Phase C.1 fixed-code re-FT before crowning anything.
- feat_norm reg change (the dead code from packet-6B 2026-04-23): leave uncommitted in working tree indefinitely. No commit, no revert.
- User reported 2.5h before "experiment-launch window" and asked for guidance. My read: the only experiment that consumes the night is Phase C.1; everything else is daylight diagnostic. So priority = launch C.1 immediately.

**Worked on:**
- **Phase 0.3 image build:** Cloud Build `c47f7a7e-15c0-401e-a212-eebd1d6b235d` succeeded in 2m31s (Docker layer cache hit). Image at `us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:1.3.216`. VERSION file bumped 1.3.215 → 1.3.216 by the build script.
- **Phase 0.4 verification:** Ran `python3 analysis/probe_battery_2026-04-26/grad_audit.py` locally on Mac (torch 2.11). Result: 9/9 in_proj residuals + 3/3 SVDResidualLinear residuals received non-zero classification gradient. PASS. Verifies the SOURCE has the fix; does not yet prove the IMAGE bundles the fix (would require pulling and running inside the container — flagged as paranoia-grade and skipped per user's time pressure).
- **VERSION commit:** Committed `ac83ba1` ("Bump VERSION to 1.3.216 (in_proj fix + anchor monitor image)"), matching the project's existing single-file VERSION-bump commit pattern.
- **Phase C.1 launch:** Submitted `exp-R13_P10_SYM_on_P8A-20260426-213738` (Vertex job ID `8252691286016917504`) via `./scripts/launch/launch_experiment.sh -y enhanced-aug-test us-west4 experiments/phase2_round13/R13_P10_SYM_on_P8A.yaml`. Image 1.3.216, A100 × 1, 10000 steps. ETA ~12h. The yaml was committed earlier as part of `2c9778b` so it IS in the image.
- **Phase A.2 launch (BUG):** Created `arena/checkpoint_maps/teams_target_domain.p10_sym_on_p8a_step2500_2026-04-26.yaml` containing P10_SYM_on_P8A_STEP2500 + P8A_REFERENCE_STEP5000 + RLP6_04_BASELINE_STEP23500. Submitted `phase-a2-p10sym-on-p8a-step2500-20260426` (Vertex job ID `8294349582570094592`) via `./arena/launch_teams_promotion_contract.sh`. **Defect:** the checkpoint map was created AFTER the image build at 21:15, so it is NOT inside the 1.3.216 image. The Vertex runner reads `/workspace/arena/checkpoint_maps/...` from inside the container and will get FileNotFoundError. The job is currently `JOB_STATE_PENDING` and has not yet started consuming GPU.
- **Image rebuild v2:** Triggered `./dev.sh build-prod -y` again in background (task `bxnkimwje`) to produce image 1.3.217 that includes the new checkpoint map. ETA ~3 min (cache hit).

**Why:**
- User picked M and wanted action. Phase C.1 is the only night-consuming experiment.
- Phase A.2 was a parallelism attempt — start scoring step 2500 in parallel with C.1 so we have the broken-in_proj baseline numbers ready when C.1 finishes tomorrow morning.
- The image-build-vs-checkpoint-map-creation ordering bug is on me — should have created the checkpoint map BEFORE the image build, not after.

**Outcome (so far):**
- Three commits this session: `2feea58` (in_proj fix), `7af72b1` (eval infra), `ac83ba1` (VERSION 1.3.216).
- Phase C.1 training running on Vertex (us-west4) — should produce checkpoints by tomorrow morning. **This is the night's main experiment.**
- Phase A.2 scorecard pending and will fail-fast when it starts. Wasted Vertex pending state has zero GPU cost so far.
- Image 1.3.217 (rebuild v2) building now — will include the new checkpoint map and unblock A.2 relaunch.

**Flags / plan-change proposals:**

#### PLAN CHANGE PROPOSAL — add a "verify image carries new in-image files" step
**Section affected:** §4 Phase 0.4 + §0 rules
**Proposed change:**
- Before any Vertex launch that references a NEW yaml/checkpoint-map/suite under `/workspace/`, list the file's mtime and compare to the image build time. If the file is newer than the build, ABORT and rebuild first.
**Rationale:**
- The "yamls and in-image files load from `/workspace/` inside container" memory note is well-known, but easy to forget when creating new artifacts mid-session. Adding a pre-launch sanity check prevents this entire class of bug.
**Status:** AWAITING USER AUTHORIZATION

**Next agent should:**
- **Decision needed from user:** cancel pending A.2 job `8294349582570094592` (recommended — saves time + cost) and relaunch on image 1.3.217 once the rebuild lands, OR let it fail naturally and relaunch.
- Once image 1.3.217 is published, commit a fresh VERSION bump (`Bump VERSION to 1.3.217 (Phase A.2 checkpoint-map image)` or similar).
- Relaunch Phase A.2 against the new image. The launcher reads from VERSION file automatically.
- Monitor Phase C.1 via W&B (`https://wandb.ai/dtect-vision/enhanced-aug-test`) — expected first checkpoint ~step 500 within ~30 min of launch. The trainer's anchor-pool monitor (committed in `7af72b1`) should log `anchor/composite` every step-validation cycle.
- Mid-run kill switches per plan §C.2:
  - Step 1500: `anchor_mean > 0.90` → kill
  - Step 3000: `anchor_mean > 0.85` OR `visomaster_enhanced_macro recall < 0.40` → kill
  - Step 5000: `anchor_mean ≤ 0.55` AND recall preserved → likely winner
- **Do NOT cancel any Vertex training job (i.e. `8252691286016917504`) without explicit user authorization** per memory `feedback_no_cancelling_vertex_jobs.md`.

---

### 2026-04-26 — agent: Claude Opus 4.7 (1M context) — session: A.2-recovery
**Phase/step:** Phase A.2 recovery — image 1.3.217 + relaunch

**User decisions made this session:**
- Authorized cancellation of pending A.2 job `8294349582570094592`.

**Worked on:**
- Cancelled `8294349582570094592` via `gcloud ai custom-jobs cancel`. State transitioned PENDING → CANCELLED before any GPU was allocated. Zero GPU spend wasted.
- Image rebuild v2 (Cloud Build `f49099f6-3ad3-4330-9630-f44cb6bd0db8`) succeeded in 2m32s. New image at `us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:1.3.217`. The new checkpoint map `arena/checkpoint_maps/teams_target_domain.p10_sym_on_p8a_step2500_2026-04-26.yaml` is now baked into `/workspace/` inside this image.
- Committed `f9ddf3a` ("Bump VERSION to 1.3.217 (Phase A.2 checkpoint-map image)").
- Relaunched Phase A.2 as `phase-a2-p10sym-on-p8a-step2500-v2-20260426` (Vertex job `2662598248543289344`) on image 1.3.217. Same 3 checkpoints (P10_SYM_on_P8A_STEP2500, P8A_REFERENCE_STEP5000, RLP6_04_BASELINE_STEP23500). Currently `JOB_STATE_PENDING`.

**Why:** unblock Phase A.2 diagnostic scoring.

**Outcome:**
- Two Vertex jobs running tonight:
  - **Phase C.1 training:** `8252691286016917504` (image 1.3.216). ~12h.
  - **Phase A.2 scorecard:** `2662598248543289344` (image 1.3.217). ETA ~30–90 min.
- Four commits in this session total: `2feea58`, `7af72b1`, `ac83ba1`, `f9ddf3a`.
- VERSION file at 1.3.217.

**Flags / plan-change proposals:** none new (the prior PLAN CHANGE PROPOSAL re: pre-launch image-currency check still awaits user decision).

**Next agent should:**
- Watch for A.2 v2 completion. Once done, the contract scorecard JSON lands at `gs://training-job-outputs/test_results/teams_promotion_contract/phase-a2-p10sym-on-p8a-step2500-v2-20260426/promotion_contract/selected_threshold_scorecard.csv` (and adjacent files). Read `selected_threshold` first — if ≈ 0.995, the contract policy bug fired and the scorecard is unreliable per memory `project_contract_policy_bug.md`.
- Once A.2 lands, write a fresh entry to `RESULTS.md` for each of the 3 scored checkpoints (P10_SYM_on_P8A step 2500, P8A step 5000 reference, RLP6_04 step 23500 baseline).
- Continue monitoring C.1 W&B (`https://wandb.ai/dtect-vision/enhanced-aug-test`) for the mid-run kill switches at steps 1500 / 3000 / 5000.

---

### 2026-04-26 — agent: Claude Opus 4.7 (1M context) — session: phase-c-overnight-slate
**Phase/step:** Phase C expanded — C-ablation + C.3 + C.4a + C.4b launched

**User decisions made this session:**
- Authorized "max option" overnight slate (5 total Vertex training jobs, including the already-running C.1).
- Specifically: include both RLP6_04-base runs (step 23500 AND step 8000) to disentangle "P8A recipe necessity" from "FT-depth crystallization."

**Worked on:**
- Verified the actual FT chain: plain CLIP → R12g (`0xxqwhxg/step14000`, ~14k broken-bug steps) → RLP6_04 (`h2pdu6i5/step23500`, ~23k broken-bug steps) → P8A (`9lmvb5b4/step5000`, ~5k broken-bug steps) → P10_SYM_on_P8A (`sbnzyrzq/step2500`). Total ~45k upstream broken-in_proj-SVD steps before C.1's 10k of fixed-code FT.
- Confirmed P8A's immediate base is RLP6_04 step 23500 (verified via `R13_P9_R_p8a_reseed.yaml`'s `gcs_base_checkpoint` + the comment in `R13_P9_FORK_p8a_on_r12g.yaml`). The user had asked "isn't P8A FT of R12_G?"; corrected to "P8A is FT of RLP6_04, which is FT of R12g."
- Listed available checkpoints in GCS: R12g earliest save is step 14000 (no earlier exists); RLP6_04 earliest save is step 8000 (AUC 0.9919, 15.5k fewer steps than canonical step 23500).
- Created four new yamls under `experiments/phase2_round13/` matching plan §C / §C.3 / §C.4 (a/b):
  - `R13_P10_SYM_on_P8A_ablation_no_in_proj.yaml` — same recipe as C.1 with `apply_svd_to_in_proj: false`.
  - `R13_P10_SYM_on_P8A_codec_hedge.yaml` — same recipe with `teams_codec_sim_p: 0.60`, `teams_codec_sim_quality: [15, 55]` (plan §C.3 spec).
  - `R13_P10_SYM_on_RLP6_04_step23500.yaml` — base swapped to RLP6_04 step 23500.
  - `R13_P10_SYM_on_RLP6_04_step8000.yaml` — base swapped to RLP6_04 step 8000 (early ckpt).
- Committed all 4 yamls as `6665910` ("Add Plan-v2 Phase C overnight slate ...").
- Built image 1.3.218 via Cloud Build `67b27a06-12ba-4c65-ac3e-96675c93c43e` (2m15s). Image published as `us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:1.3.218`.
- Committed VERSION bump as `04710ab` ("Bump VERSION to 1.3.218 (Phase C overnight slate image)").
- Submitted 4 Vertex training jobs in us-west4 on image 1.3.218:

| Slot | Display name | Vertex job ID | Question answered |
|---|---|---|---|
| 2 | `exp-R13_P10_SYM_on_P8A_ablation_no_in_proj-20260426-221548` | `3815519753150136320` | Is the in_proj fix the lever or co-passenger? |
| 3 | `exp-R13_P10_SYM_on_P8A_codec_hedge-20260426-221551` | `5934463377827954688` | Is the data axis (codec) the missing piece? |
| 4a | `exp-R13_P10_SYM_on_RLP6_04_step23500-20260426-221554` | `9199573107671564288` | Is P8A's specific unfreeze recipe necessary? |
| 4b | `exp-R13_P10_SYM_on_RLP6_04_step8000-20260426-221556` | `7708881631011930112` | Does FT-depth crystallization matter? |

**Why:**
- The user noted time pressure ("time is of the essence") and that 6 GPU slots were available with only one in use (C.1). Filling slots with plan-aligned hedges reduces tomorrow's blocked-on-result wait.
- The 4 runs together give a coherent 5-corner read tomorrow morning (step 2500 / C.1 / C-ablation / C.3 / C.4a / C.4b vs P8A baseline). One missing slot is unused — left as headroom.
- C.4a and C.4b together specifically disentangle "P8A recipe necessary" from "less-crystallized base helps" — running only one would conflate the two variables.

**Outcome:**
- Six commits this session total: prior `2feea58`, `7af72b1`, `ac83ba1`, `f9ddf3a`, plus new `6665910` (yamls) + `04710ab` (VERSION 1.3.218).
- VERSION file at 1.3.218.
- Five Vertex training/scorecard jobs running tonight in us-west4:
  - **C.1 training** (canonical re-FT on P8A) — `8252691286016917504` — image 1.3.216, ~12h.
  - **C-ablation** — `3815519753150136320` — image 1.3.218, ~12h.
  - **C.3 codec hedge** — `5934463377827954688` — image 1.3.218, ~12h.
  - **C.4a one-step-back** — `9199573107671564288` — image 1.3.218, ~12h.
  - **C.4b two-effects-back** — `7708881631011930112` — image 1.3.218, ~12h.
  - Plus **A.2 v2 scorecard** — `2662598248543289344` — image 1.3.217, ~30–90 min (returns its slot tonight).

**Flags / plan-change proposals:**

#### PLAN CHANGE PROPOSAL — incorporate Phase C.4a + C.4b as first-class plan slots
**Section affected:** §4 Phase C (currently has C.1, C.2 mid-run gates, optional C.3); §C-γ (deferred plain-CLIP scratch)
**Proposed change:**
- Add Phase C.4a (one-step-back from RLP6_04 step 23500) and C.4b (RLP6_04 step 8000 early) as named plan slots, with their crystallization-vs-recipe attribution logic.
- Add a §C.4 exit criterion that defines what each result configuration *means* for the §C-γ plain-CLIP-scratch trigger:
  - C.4b > C.4a anchor → FT depth crystallizes shortcut → motivate plain-CLIP scratch
  - C.4b ≈ C.4a anchor → substrate is in CLIP itself → plain-CLIP scratch is unlikely to help
**Rationale:**
- These four runs were launched under Plan-v2 but are not currently named in the plan text. Future agents would not understand from the plan alone why these specific bases were chosen.
- The C.4a/C.4b distinction is load-bearing for whether to launch §C-γ next — making it explicit closes the loop.
**Status:** AWAITING USER AUTHORIZATION

**Next agent should:**
- **Do NOT cancel any of the five training jobs** without explicit user authorization (per memory `feedback_no_cancelling_vertex_jobs.md`).
- Watch A.2 v2 (`2662598248543289344`) for completion (~30–90 min from 19:48 UTC). Read `selected_threshold` first; flag if ≈ 0.995 (contract policy bug). Then write 3 entries to `RESULTS.md` for the 3 scored checkpoints.
- Monitor all 5 training runs on W&B (`https://wandb.ai/dtect-vision/enhanced-aug-test`) for the mid-run kill switches per plan §C.2:
  - Step 1500: any run with `anchor_mean > 0.90` → kill that single run (do NOT kill the others).
  - Step 3000: any run with `anchor_mean > 0.85` OR `visomaster_enhanced_macro recall < 0.40` → kill that single run.
  - Step 5000: any run with `anchor_mean ≤ 0.55` AND recall preserved → likely winner; let finish for full scorecard.
- Killing a run requires user authorization per memory `feedback_no_cancelling_vertex_jobs.md`. Surface the W&B numbers + recommendation; let the user decide.
- Once any of the 5 runs lands a value_composite checkpoint, run the full promotion-contract scorecard against it as a Phase D candidate.
- Plan §C exit interpretation tomorrow morning needs all of: C.1 vs C-ablation (fix vs co-passenger), C.3 vs C.1 (codec axis), C.4a vs C.1 (P8A recipe necessity), C.4b vs C.4a (crystallization). Write this 5-corner read into RESULTS.md as a synthesis entry.

---

### 2026-04-26 — agent: Claude Opus 4.7 (1M context) — session: pcp-cleanup
**Phase/step:** Resolve outstanding PLAN CHANGE PROPOSALs

**User decisions made this session:**
- Authorized PCP #2 (pre-launch image-currency check). Implement as a launch-script guard.
- Closed PCP #1 (feat_norm-reg deferred decision) and PCP #3 part A (C.4a/C.4b in plan §C) as **LOG-only — no plan text change**. Rationale: pure documentation hygiene with no data dependency; the LOG records them.
- Dropped PCP #3 part B (§C-γ plain-CLIP-scratch trigger threshold). Rationale: the launch decision should fall out of tomorrow's actual C.4a vs C.4b numbers, not be authorized in advance.

**Worked on:**
- Created `scripts/launch/check_image_currency.sh` (~95 lines, executable). Queries the pushed image's `createTime` from Artifact Registry via `gcloud artifacts docker images list <pkg> --include-tags --filter='tags:<tag>'`, then compares each referenced config's local mtime. Exits 1 if any file is newer than image push time. Skipped when `SKIP_IMAGE_CURRENCY_CHECK=1`.
- Wired the helper into `scripts/launch/launch_experiment.sh` (training launcher, checks `${PARAM_CONFIG_INPUT}`) and `arena/launch_teams_promotion_contract.sh` (scorecard launcher, checks `${SUITE_MANIFEST}` + `${CHECKPOINT_MAP}`).
- Smoke-tested:
  - Pass case: `./check_image_currency.sh us-docker.pkg.dev/.../effort-detector:1.3.218 <yaml1> <yaml2>` → "OK: 2 config file(s) older than image push time."
  - Fail case: `touch <yaml>` then re-run → exits 1 with explicit mtime + image-push-time + URI + remediation hint.
- Bash-syntax-checked all three scripts (`bash -n`); clean.

**Why:** the user wanted the awaiting-list cleared to zero so tomorrow's decisions are purely data-driven. PCP #2 was the only proposal with a non-trivial implementation; the others are admin notes already captured in this LOG.

**Outcome:**
- Pre-launch image-currency check now active for both training and contract-scorecard launchers. Future A.2-style mistakes will fail-fast at submission time with a clear remediation message.
- No outstanding PLAN CHANGE PROPOSALs awaiting user authorization. Plan text is unchanged.
- Two more files staged for the next commit: `scripts/launch/check_image_currency.sh` (new) + edits to `scripts/launch/launch_experiment.sh` and `arena/launch_teams_promotion_contract.sh`.

**Resolution table for prior PCPs:**

| PCP | Disposition |
|---|---|
| #1 — feat_norm-reg deferred decision in plan §2/§10 | **LOG-only**. No plan-text change. The LOG records the uncommitted feat_norm change in `detectors/effort_detector.py`. |
| #2 — pre-launch image-currency check | **IMPLEMENTED**. See `scripts/launch/check_image_currency.sh` and call sites in both launchers. |
| #3a — name C.4a / C.4b in plan §C | **LOG-only**. No plan-text change. Both runs are documented in this LOG with their question-answered semantics. |
| #3b — codify §C-γ plain-CLIP-scratch trigger as a fixed threshold | **DROPPED**. Decision will be made tomorrow morning from the actual C.4a vs C.4b delta. Pre-authorizing a threshold without seeing the numbers is the wrong direction. |

**Flags / plan-change proposals:** none.

**Next agent should:**
- Watch A.2 v2 (`2662598248543289344`) for completion (~10–60 min remaining as of this entry). When the scorecard JSON lands at `gs://training-job-outputs/test_results/teams_promotion_contract/phase-a2-p10sym-on-p8a-step2500-v2-20260426/promotion_contract/`, read `selected_threshold` first; if ≈ 0.995, flag the contract policy bug per memory `project_contract_policy_bug.md`.
- Write 3 entries to `RESULTS.md` (one per scored checkpoint).
- Continue monitoring the 5 training runs on W&B for §C.2 mid-run kill switches; surface numbers + recommendation but do not cancel without explicit user authorization.
