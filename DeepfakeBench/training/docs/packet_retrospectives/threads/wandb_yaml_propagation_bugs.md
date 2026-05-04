# Thread: W&B YAML propagation bugs (`train_sweep.py` allowlist drift)

## The question

`wandb.init` flattens nested-dict yaml blocks into dotted keys (e.g., `value_composite.target_mean_fpr`), so `wandb.config.get('value_composite')` returns `None` even when the yaml clearly contains the block. `train_sweep.py` has a per-block allowlist (lines ~170–290) that explicitly copies nested config blocks from `single_cfg` into the runtime `config` dict before launching the trainer, working around the flattening. Whenever a new nested-dict block is added to a yaml, **the allowlist must also be extended** or the trainer reads `None` for the block and silently falls through to legacy defaults.

The question this thread answers: how does this bug class manifest, what's the minimum sufficient fix per occurrence, and why does the pattern recur even after each individual fix?

## Initial belief

Through [RLP1](../packets/RLP1.md), [RLP2](../packets/RLP2.md), and [RLP3](../packets/RLP3.md), the per-block re-application was treated as an idiosyncrasy of `train_sweep.py` — a "weird thing the launcher does" — not as a structural class of bug. The allowlist already handled `dataset_methods`, `combined_paired`, `backbone`, `checkpointing`, `lesson_data_control`, `lesson_gate`, `augmentation`, `deeplive`, `visomaster`, `stability_lambda`, `label_smoothing`, `use_group_dro`, `group_dro_params`. Adding a new nested block was understood as "yaml change + maybe a quick allowlist line if the block is nested," but the recurrence pattern wasn't yet legible.

## What changed our mind

- **2026-04-22 RLP3.5 trainer init logs reveal silent-fallback (commit `872502c`).** Packet 3.5 slots 01 and 02 launched under image `1.3.192` (built from commit `477b00b`, which introduced the `value_composite` yaml block). User-side inspection of the trainer init logs flagged that the legacy values `(0.02 / 0.04 / max)` were printed even though the yamls clearly declared the new `(0.03 / 0.05 / p95)` block (see [RLP3_5.md:37-42](../packets/RLP3_5.md) for the new-gate definition; `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_5_EXPERIMENT_PLAN_2026-04-22.md:37-42` for the yaml schema). Root cause from session `02873fd0`: *"it's a nested dict that falls through"* — i.e., `wandb.init` flattened `value_composite` into dotted keys, the trainer's `wandb.config.get('value_composite')` returned `None`, and the legacy fallback fired silently.
- **2026-04-22 fix lands as commit `872502c`.** The diff is 8 lines added to `train_sweep.py:276-283`:
  ```
  # Apply value_composite config directly (nested dict — W&B flattens; must copy).
  # Keys: target_mean_fpr, max_pool_fpr, stability_jitter_stat. Trainer falls
  # back to legacy (0.02 / 0.04 / "max") when absent, so this is legacy-safe.
  if 'value_composite' in single_cfg:
      config['value_composite'] = single_cfg['value_composite']
      print(f"  ✅ Applied value_composite: {single_cfg['value_composite']}")
      logger.info(f"  Applied value_composite: {single_cfg['value_composite']}")
  ```
  VERSION bumped `1.3.192 → 1.3.193`. The 6 RLP3.5 slots that had launched under `1.3.192` were cancelled and relaunched under `1.3.193` distributed across `us-east1 / asia-southeast1 / us-east4 / us-west1 / europe-west4` to spread quota.
- **The retro-runner has the same class of bug.** `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_RETRO_SCORE_RESULTS_2026-04-22.md:137` records the same pattern in `retro_score_value_composite.py`: *"explicitly copies the `value_composite` nested block from `single_cfg` into `config` **before** `apply_all_wandb_overrides` — missing this copy silently drops the new-gate block (W&B flattens nested dicts). This is the same class of bug that `train_sweep.py` commit `872502c` fixed for live training."* Two distinct code paths (live training launcher + retro runner) had to be patched separately for the same logical change. The bug class is structural, not a one-off.
- **Counterfactual: without `872502c`, every RLP3.5 slot would have silently used legacy gates.** [RLP3.md:55](../packets/RLP3.md) records this: *"`train_sweep.py` originally dropped the nested `value_composite` block (fix in commit `872502c`; session `02873fd0`: 'it's a nested dict that falls through'). Without this every RLP3.5 new-gate run would have silently used legacy gates."* The new-vs-legacy gate distinction was the headline observation of RLP3.5 (the "5× smaller deltas" reading), so a silent fallback would have invalidated the entire packet.
- **The `1.3.194` retro-image had a related but distinct image-snapshot issue.** `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_RETRO_SCORE_RESULTS_2026-04-22.md:138`: *"First build of `1.3.194` captured the repo state before the slot-02 yaml edits landed (source tarball snapshotted on `gcloud builds submit`, not at image push). Sanity happened to be benign because the trainer's legacy fallback matched the sanity gates. Image `1.3.195` was rebuilt to catch the slot-02 edits before firing the new-gate runs."* This is the same flattening-class failure mode (legacy fallback fires silently, results look plausible) caught only because the agent inspected the run config. The pattern: **silent fallback masks the bug; only side-by-side comparison of expected-vs-actual behavior reveals it**.

### Slice 5 evidence — artifact-name length bug (different bug class, same hygiene gap)

- **2026-04-26 commit `f366368` ("Fix wandb artifact name >128 chars on long log_prefix scorecards")** — wandb-side hygiene fix, image bumped 1.3.213 → 1.3.214 via `d609e39`. Symptom: long `log_prefix` values produced wandb artifact names that exceeded the W&B 128-char artifact name limit and tripped the validation error. Fix: truncate + hash long names. **Different bug class** from the train_sweep allowlist drift (this is a string-length validation issue at artifact-write time, not a config-propagation issue at trainer-init time), but **same hygiene gap**: there is no schema-level mechanism that prevents wandb-side surface bugs from silently invalidating runs. Each new wandb-touching surface (config init, artifact name, summary key, log step monotonicity) can independently fail; each gets fixed point-wise after observation. The recurrence pattern that motivates this thread is now structural: at least three distinct wandb-side fixes have landed (`872502c` allowlist re-apply, `f366368` artifact name length, and `c366026` from Slice 6 will add a third — re-applying `anchor_aware`/`face_scale_jitter`/`periodic_saves` blocks to the allowlist). The "wandb-side surface" is large enough that point-wise fixes are not closing the bug class.

### Slice 6 evidence — major recurrence event (the same allowlist drift surfaces 6 days later)

The Slice-3 prediction in this thread's Slice-3 close paragraph (*"Slice 6 will see this pattern recur ... When the Slice 6 agent sees that recurrence, it should reopen this loop or supersede it with a new wandb-flattening pattern not generalized entry"*) was correct.

- **2026-04-28 ~17:50 CEST — P13_FROM_SCRATCH launches with three new top-level nested-dict yaml blocks (`anchor_aware`, `face_scale_jitter`, `periodic_saves`).** Same allowlist drift as Slice 3, three new keys. Two cancelled Vertex jobs (`99039952980934656`, `8345130870696312832`, both us-east1, image 1.3.225), ~$3.50 burned. Trainer init logs at job `8345130870696312832`:
  ```
  PipelineRandomization ENABLED p_real=0.55 p_fake=0.45 …
  AnchorAwarePenalty DISABLED (enabled=False weight=5.0)
  FaceScaleJitter DISABLED
  ```
  PipelineRandomization works because it lives **nested under** `augmentation:` (which IS in the allowlist). AnchorAwarePenalty + FaceScaleJitter are top-level blocks → silent fallback to `enabled=False`. periodic_saves was the same class of failure (added in `c7dc828`, silently no-checkpoint on P12_HEAVY_LONG 2026-04-28 12:25). Source: `docs/relaunch_handoffs/WANDB_FLATTENING_BUG_HANDOFF_2026-04-28.md`. Full Slice-6 narrative in [`wandb_flattening`](wandb_flattening.md).

- **The first diagnosis was wrong-layered.** The agent who saw the DISABLED log first hypothesized "wandb wraps nested dicts as `wandb.sdk.wandb_config.Config` sub-objects" and applied a defensive `_to_plain_dict()` helper at the trainer layer. Wrote 7 tests in `tests/test_to_plain_dict.py`, all passing — premise-false. The trainer-layer fix had no effect on the next run. Cost of misdiagnosis: ~$1.40 + a rebuild + a relaunch. **The trainer-layer changes are still in the working tree as harmless defensive code** (the fix lives in `train_sweep.py`).

- **Commit `c366026` lands the structural fix** (image bumped to 1.3.226 via `6c9a320`). Three new blocks added to `train_sweep.py:283-313` mirroring the Slice-3 `value_composite` pattern. **Crucially**, `c366026` adds `tests/test_train_sweep_reapply_allowlist.py` — the regression test that fails if any key in `TRAINER_NESTED_KEYS` is not re-applied in `train_sweep.py`. **This is the first generalized mechanism for this bug class** — Slice 3's `872502c` only added the missing key without test coverage; Slice 6's `c366026` adds both the missing keys AND a structural test guard. Memory `project_wandb_flattens_nested_dicts.md` is updated to enumerate the 14-key allowlist post-fix and references the test guard as the "regression test guard" for future additions.

- **Why the Slice-3 loop stays resolved instead of reopening.** The Slice-3 loop `train-sweep-allowlist-not-closed-under-schema-changes` was prematurely marked `resolved` at end of Slice 3 — the close criterion in that block was *"every new nested dict in yaml requires explicit re-application in train_sweep.py allowlist (~lines 176-282) — recurrence indicates pattern not generalized"*, which Slice 6's recurrence proves was not yet met. **However**, the Slice-6 fix `c366026` adds the test-guard mechanism that *does* close the structural problem (any future addition without an allowlist entry fails CI). The argument for reopening: the same root cause re-fired with new keys. The argument against: the closure mechanism (test guard) now exists, so further recurrence is structurally unlikely (any new addition that omits the allowlist line would fail the test). **Judgment call: keep the Slice-3 loop resolved**, because the closure mechanism the loop's close criterion was implicitly hoping for now exists. Slice 6 is where the *actual* structural fix lands, even though the symptom recurred between Slice 3 (mechanical fix) and Slice 6 (mechanical fix + test guard). The intermediate recurrence is preserved as evidence in this Slice-6 evidence section.

- **The broader `wandb-side-surface-hygiene-not-systematic` loop accumulates evidence.** Three distinct wandb-side fixes have now landed: `872502c` (Slice 3, allowlist re-apply for `value_composite`), `f366368` (Slice 5, artifact-name length), `c366026` (Slice 6, allowlist re-apply for 3 new keys + test guard). The Slice-6 fix has a generalized mechanism (the test guard) for *its* sub-bug-class but not for the broader wandb-side surface (artifact-name length is not covered; future surfaces are not). The open loop stays `open`, severity `medium`, with this evidence appended.

## Current stance (2026-04-29)

The `train_sweep.py` allowlist is **not closed under future yaml schema changes**. Every new nested-dict block introduced into the yaml schema requires an explicit re-apply line in the allowlist (currently lines ~170–290 of `train_sweep.py`). The fix is mechanical (8 lines per block, all in the same shape as `872502c`), but the failure mode is silent and the bug is invisible until either:

1. An agent inspects the trainer init log and notices a config field's value differs from the yaml's declaration, or
2. A downstream metric behaves implausibly (e.g., RLP3.5 slots 01/02 producing legacy-gate `value_composite` numbers that should have been new-gate numbers).

The detection lag is dangerous because a silently-fallback'd run produces plausible-looking metrics that may pass casual review. RLP3.5 caught it within ~24 hours; the consequence in less-instrumented packets could be worse.

The retro runner (`retro_score_value_composite.py`) replicates the same flattening workaround for the same nested block (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_RETRO_SCORE_RESULTS_2026-04-22.md:137`) — meaning the bug class lives wherever yamls are loaded through `wandb.init`. Each new tool that does this independently must either re-implement the per-block copy-over or import a shared helper.

## Packet timeline

- [RLP3](../packets/RLP3.md) — `value_composite` yaml block introduced via commit `477b00b` (RLP3.5 plumbing, but the trainer code reads it for both packets). Image `1.3.192`.
- [RLP3.5](../packets/RLP3_5.md) — first packet that launches with the new block. Slots 01/02 reveal the silent-fallback bug. Fix `872502c` lands, image `1.3.193`. Slots are relaunched. The "5× smaller deltas" reading is preserved.
- [RLP4](../packets/RLP4.md) — does not introduce new nested blocks; allowlist not exercised.
- Slice 4+ — every packet that adds a new nested config block must update the allowlist. (Slice 6 will see this pattern recur; see open loop below for the named handoff to that slice.)

## Evidence locations

- `train_sweep.py:170-290` — the allowlist; current entries cover `dataset_methods`, `lesson_data_control`, `lesson_gate`, `augmentation`, `combined_paired`, `deeplive`, `visomaster`, `backbone`, `checkpointing`, `stability_lambda`, `label_smoothing`, `use_group_dro`, `group_dro_params`, `value_composite`, `anchor_aware`
- `train_sweep.py:276-283` — the `value_composite` re-apply block (commit `872502c`)
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_RETRO_SCORE_RESULTS_2026-04-22.md:137-138` — retro runner has the same bug class; image-snapshot timing also matters (`1.3.194` → `1.3.195`)
- [RLP3.md:55](../packets/RLP3.md) — counterfactual ("Without this every RLP3.5 new-gate run would have silently used legacy gates")
- [RLP3_5.md:74](../packets/RLP3_5.md) — preprocessing-parity / pre-fix caveat documented same packet
- Commits: `477b00b` (introduces `value_composite` yaml block, builds `1.3.192`), `872502c` (the fix; builds `1.3.193`)
- Memory: `project_wandb_flattens_nested_dicts.md` — out-of-scope for Slice 3 (created after Slice 6's wandb-flattening incident); see open loop below for the cross-slice link

## Open loops

### Open loop: train-sweep-allowlist-not-closed-under-schema-changes
status: resolved
severity: medium
first_seen: 2026-04-22
last_verified: 2026-04-29
close_criterion: every new nested dict in yaml requires explicit re-application in train_sweep.py allowlist (~lines 176-282) — recurrence indicates pattern not generalized.

The `value_composite` nested block was the first instance of this bug class to actually surface in production. Commit `872502c` fixed the specific instance (8 lines added to `train_sweep.py:276-283`) and unblocked RLP3.5. The fix is mechanical and idempotent for any single block, but the **pattern is not generalized** — there is no schema-driven mechanism that automatically re-applies every nested block, no test that catches the silent fallback, and no shared helper between `train_sweep.py` and `retro_score_value_composite.py`. Each new nested-dict yaml block introduced after this point must be re-applied by hand. The Slice-3 evidence is sufficient to close the loop with `status: resolved` for the `value_composite` instance, but Slice 6 will see this pattern recur (per memory `project_wandb_flattens_nested_dicts.md` — the wandb-flattening bug surfaces again ~6 days later when nested blocks for `anchor_aware`, `face_scale_jitter`, and `periodic_saves` are added). When the Slice 6 agent sees that recurrence, it should reopen this loop or supersede it with a new "wandb-flattening pattern not generalized" entry. The mechanical fix per occurrence is fine; the structural absence of a schema-aware mechanism is the un-resolved part.

**Slice 5 verification (2026-04-29).** Slice 5 did not surface a new train_sweep allowlist instance — the allowlist remained intact across the P8A / P9 / P10 packets. However, a **different** wandb-side hygiene bug landed: the artifact-name length fix (commit `f366368`, image 1.3.214). Because it's a different bug class (string-length validation at artifact-write, not config-propagation at trainer-init), it does not reopen this specific loop. It does add evidence that the broader "wandb-side surface" hygiene pattern is structural, not point-wise. Loop stays `resolved` for the allowlist instance; cross-link to a new Slice-5 open loop tracking the broader pattern.

**Slice 6 verification (2026-04-29).** Major recurrence event. Three new nested-dict blocks (`anchor_aware`, `face_scale_jitter`, `periodic_saves`) hit the same silent-fallback path as `value_composite` did 6 days earlier; ~$3.50 + 2 cancelled Vertex jobs burned before diagnosis; one wrong-layered fix attempt (trainer-side `_to_plain_dict()` helper) lost ~$1.40. **Loop stays `resolved`** rather than reopens because Slice 6's `c366026` lands both the missing allowlist entries AND `tests/test_train_sweep_reapply_allowlist.py` — a structural test guard that fails if any future trainer-consumed nested-dict block omits the allowlist line. The pre-Slice-6 close-criterion framing (*"every new nested dict in yaml requires explicit re-application in train_sweep.py allowlist — recurrence indicates pattern not generalized"*) is now met by construction: the test guard makes recurrence detectable at CI time. The intermediate Slice-3-to-Slice-6 recurrence is preserved as evidence in the section above; the loop's status doesn't reopen because the closure mechanism is now in place. Full Slice-6 narrative in [`wandb_flattening`](wandb_flattening.md).

### Open loop: wandb-side-surface-hygiene-not-systematic
status: open
severity: medium
first_seen: 2026-04-26
last_verified: 2026-04-29
close_criterion: a single mechanism (test, lint, schema, or shared helper) catches both (a) silent-fallback config flattening AND (b) artifact-name validation issues at submission time before they fire in production — i.e., the wandb-side surface is hardened wholesale rather than fix-by-fix

Slice 5 added the artifact-name length bug (`f366368`) to the wandb-side surface fixes inventory. With Slice 3's allowlist re-apply (`872502c`) and the Slice-6 anchor_aware re-apply (`c366026` per memory `project_wandb_flattens_nested_dicts.md`), the count of distinct wandb-side hygiene fixes is at three across ~6 weeks. Each fix is mechanical and small; the bug class is structurally invisible until it fires; the symptoms range from "silent fallback to legacy defaults" (allowlist) to "artifact write fails with a hard error" (length). A single mechanism that catches the whole class would need to (1) inspect every wandb call surface in the codebase, (2) validate that nested config blocks are explicitly re-applied or use a flat-key idiom, (3) validate that string fields passed to wandb's artifact/run/summary APIs respect their length and character constraints, and (4) ideally run as a pre-launch check (similar to the image-currency guard `ad76cd8`). Until such a mechanism lands, every new wandb-touching code path inherits the same hygiene gap.

**Slice 6 update (2026-04-29).** The recurrence event landed in Slice 6 (`c366026`, 3 new keys + ~$3.50 burned + one wrong-layered fix attempt). Slice 6's contribution to *closing* this loop is partial: `tests/test_train_sweep_reapply_allowlist.py` (added in `c366026`) is a structural test guard for **the allowlist sub-class only**. It does not cover artifact-name length, summary-key validation, or any future wandb-side surface. The loop stays `open` because the close criterion explicitly requires "a single mechanism that catches both (a) and (b)"; the Slice-6 mechanism is sub-class only. Severity stays `medium` — the consequences are still cost-bounded ($3.50 per recurrence + dev time), but the broader pattern remains undriven by a single hygiene mechanism. Cross-link: see [`wandb_flattening`](wandb_flattening.md) for the Slice-6 narrative; see [`silent-feature-failures-pattern`](wandb_flattening.md#open-loop-silent-feature-failures-pattern) for the meta-loop that pairs with this one (same closure mechanism likely; different framing).

**2026-05-04 evening update — fourth wandb-side surface bug: launcher-default vs override entity mismatch.** Different sub-class from allowlist drift, same hygiene gap.

- **Symptom**: Vertex job submitted, runs for ~30 seconds inside the container, crashes at `wandb.init` with HTTP 404: `entity roeehub not found during upsertBucket`. Job state goes `JOB_STATE_FAILED`. Slot is lost (need to relaunch + re-queue).
- **Root cause**: `scripts/launch/launch_experiment.sh:95-96` has `WANDB_API_KEY` and `WANDB_ENTITY` defaults (`bb5a8ea4a27ebe45917587df8c46674d26e43966` and `dtect-vision`). The launcher uses these as fallbacks. If the calling shell exports `WANDB_ENTITY=roeehub` (a wandb username, NOT an entity that exists in this project's wandb workspace), the launcher silently passes that through to the container's wandb credentials, and the container's wandb.init crashes 404 because `roeehub` does not exist as an entity. Memory `feedback_promotion_contract_launch.md` captures the symmetric issue for `arena/launch_*.sh` launchers (those have NO defaults; require explicit export).
- **Verified failure**: 2026-05-04 19:45 PDT, Vertex job `318825730303590400` (Packet C-codec first attempt, image 1.3.257), failed in 30s. Logs at `gcloud logging read 'resource.type="ml_job" AND resource.labels.job_id="318825730303590400"'`.
- **Workaround**: relaunched as `1202657157175050240` after `unset WANDB_ENTITY` (defaulting to `dtect-vision`). Memory entry updated with new section flagging this for `launch_experiment.sh` specifically.
- **Bug-class fit**: this is the same class as `wandb-side-surface-hygiene-not-systematic` — a wandb input that is silently invalid, fires only at runtime, costs a Vertex slot per occurrence (~$0.05 + relaunch latency). Cleaner-mechanism candidates: a pre-launch check in `launch_experiment.sh` that validates `${WANDB_ENTITY}` resolves (lightweight HTTP HEAD against the wandb API) before submitting the Vertex job. Estimated effort: ~10 lines + 1 test.

This adds a fourth concrete instance to the wandb-side surface fixes inventory (after `872502c` allowlist, `f366368` artifact-name length, `c366026` allowlist + test). The `wandb-side-surface-hygiene-not-systematic` open loop's evidence base is now four bugs across three sub-classes (config flattening, string validation, default/override mismatch). The single-mechanism close criterion is materially harder than originally framed because the surfaces are heterogeneous (config dict, artifact name string, env-var passthrough); a unified check is no longer the obvious shape. A more tractable framing: a per-surface checklist enforced at launcher boundaries.
