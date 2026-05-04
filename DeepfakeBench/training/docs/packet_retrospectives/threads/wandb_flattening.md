# Thread: wandb-flattening of nested yaml blocks (the 2026-04-28 recurrence)

> **Slice 6 recurrence of the Slice 3 `train_sweep.py` allowlist pattern.** Same root cause (`wandb.init` flattens nested dicts to dotted keys; `train_sweep.py:172-282` re-applies an allowlist), different keys (`anchor_aware`, `face_scale_jitter`, `periodic_saves`), separate Vertex burn (~$3.50). See [`wandb_yaml_propagation_bugs`](wandb_yaml_propagation_bugs.md) for the bug-class lifecycle; this thread captures the 2026-04-28 incident specifically.

## The question

When commit `cab2909` (P13 anti-shortcut interventions) added three new top-level nested-dict yaml blocks (`anchor_aware`, `face_scale_jitter`, `periodic_saves`), the trainer silently logged them as **DISABLED** because `train_sweep.py`'s manual re-apply allowlist did not include them. **Why did the same bug class as Slice 3's `872502c` recur six days later, given that the pattern was named in memory `project_wandb_flattens_nested_dicts.md`?** And: should the structural fix (a pre-launch test that enforces the allowlist invariant) be promoted, or is per-occurrence mechanical patching sufficient?

## Initial belief

Through Slice 3 the team treated the `train_sweep.py` allowlist as a known one-time idiosyncrasy of the launcher — `872502c` added 8 lines for `value_composite`, the pattern was understood, and memory `project_wandb_flattens_nested_dicts.md` codified the rule "every new nested dict needs a re-apply line." Through Slice 5 the broader wandb-side surface hygiene gap was acknowledged via the artifact-name length fix (`f366368`), surfaced as the open loop `wandb-side-surface-hygiene-not-systematic` (medium). The implicit assumption: the memory rule + the existing allowlist would prevent recurrence at the train_sweep layer specifically, even if the broader wandb-side hygiene was not yet systematized.

That assumption was wrong.

## What changed our mind

- **2026-04-28 17:50 CEST — P13_FROM_SCRATCH launches under image 1.3.225 with three new nested blocks (`anchor_aware`, `face_scale_jitter`, `periodic_saves`).** Trainer init logs at job `8345130870696312832` (us-east1) show:
  ```
  PipelineRandomization ENABLED p_real=0.55 p_fake=0.45 …
  AnchorAwarePenalty DISABLED (enabled=False weight=5.0)
  FaceScaleJitter DISABLED
  ```
  PipelineRandomization is correctly **ENABLED** because it lives nested *under* `augmentation:` (which IS in the allowlist at `train_sweep.py:207-214`); AnchorAwarePenalty + FaceScaleJitter are **DISABLED** despite the yaml asserting `enabled: true`. Same symptom on the previous launch attempt (`99039952980934656`). Two cancelled jobs, ~$3.50 burned. Source: `docs/relaunch_handoffs/WANDB_FLATTENING_BUG_HANDOFF_2026-04-28.md:1-45`.

- **The first diagnosis was wrong-layered.** The handoff (`:128-153`) is explicit that the agent who saw the DISABLED log first hypothesized "wandb wraps nested dicts as `wandb.sdk.wandb_config.Config` sub-objects" and applied a defensive `_to_plain_dict()` helper at the trainer layer (`trainer/trainer.py:56-73`, plus the call sites at `:495-501`, `:503-510`, `:2374-2375`). Wrote 7 tests in `tests/test_to_plain_dict.py`, all passing — **but the tests were premise-false**. The trainer-layer fix had no effect on the next run; the DISABLED log lines reappeared verbatim. **Cost of misdiagnosis: ~$1.40 + a rebuild + a relaunch.** The trainer-layer changes are still in the working tree as harmless defensive code.

- **The correct diagnosis lives upstream in `train_sweep.py`.** wandb.init flattens nested dicts into dot-notation top-level keys; `wandb.config.get('anchor_aware')` returns `None` because the actual keys are `anchor_aware.enabled`, `anchor_aware.weight`, etc. The trainer can't recover what wandb has already flattened. The fix layer is the existing per-block re-apply allowlist (handoff `:46-122`).

- **Commit `c366026` (2026-04-28) lands the fix.** Three new blocks added to `train_sweep.py:283-313` mirroring the Slice-3 `value_composite` pattern (8 lines per block, structural copy of `if 'X' in single_cfg: config['X'] = single_cfg['X']` + log line). Memory `project_wandb_flattens_nested_dicts.md` is updated to enumerate the 14-key allowlist post-fix, plus a new test guard:
  ```
  Regression test guard: tests/test_train_sweep_reapply_allowlist.py (added in commit c366026) fails if any key in TRAINER_NESTED_KEYS is not re-applied in train_sweep.py.
  ```
  VERSION bumped 1.3.225 → 1.3.226 via commit `6c9a320`. The structural test guard is the **first generalized mechanism** for this bug class — Slice 3's `872502c` only added the missing key without test coverage. The Slice-6 fix adds both the missing keys AND a test that fails if a future key is added without an allowlist entry. This is what the open loop `train-sweep-allowlist-not-closed-under-schema-changes` (Slice 3, marked `resolved` at end of Slice 3) was implicitly hoping for; Slice 6 is the slice that delivered it.

- **PipelineRandomization works because it lives under `augmentation:`** (handoff `:117-124`). The augmentation block IS in the allowlist (`train_sweep.py:207-214`), so the entire nested sub-tree (including `pipeline_random_*` keys) is copied as-is. This is structural luck: any nested feature added under an existing top-level allowlisted block silently works; any new top-level block silently fails. The pattern is: **the wandb-flattening mechanism is sensitive to the depth of the new key relative to existing allowlist coverage**.

- **The "value_composite" precedent in Slice 3 was the structural template.** Both `872502c` (2026-04-22) and `c366026` (2026-04-28) follow the identical 8-lines-per-block pattern. Memory `project_wandb_flattens_nested_dicts.md` cites both as reference implementations. The recurrence is mechanical, not conceptual — the team knew the pattern; the fix didn't get pre-applied because the new yaml blocks landed in commit `cab2909` without a paired allowlist update. **Both `cab2909` (anchor_aware + face_scale_jitter additions) and `c7dc828` (periodic_saves addition) shipped without their `train_sweep.py` companion update**; the trainer-side feature lands in commit N, the launcher-side allowlist lands in commit N+1 only after symptom observation.

## Current stance (2026-04-29)

The fix is **structural this time**: `c366026` adds both the three missing keys AND `tests/test_train_sweep_reapply_allowlist.py` enforcing the allowlist invariant for any future nested config block. Memory `project_wandb_flattens_nested_dicts.md` is the canonical operator-discipline rule:

> Whenever you add a top-level nested-dict block to any experiment yaml AND read it from the trainer via `self.config.get('<block_name>')`, you MUST also add a block to `train_sweep.py` … plus add the key to `TRAINER_NESTED_KEYS` in `tests/test_train_sweep_reapply_allowlist.py`.

But the **broader wandb-side surface** (artifact-name length, summary key validation, log-step monotonicity, future flattening surfaces) remains piecemeal. The `wandb-side-surface-hygiene-not-systematic` open loop in [`wandb_yaml_propagation_bugs`](wandb_yaml_propagation_bugs.md) tracks the broader pattern; it stays `open` because c366026 closes the train_sweep allowlist instance but does not address the broader category. The Slice-3 loop `train-sweep-allowlist-not-closed-under-schema-changes` was marked `resolved` at end of Slice 3 — Slice 6's recurrence demonstrates that "marking resolved because mechanical fix exists" was premature; the Slice-6 fix (with the test guard) is what actually closes the structural loop. The Slice-3 loop is **kept resolved** but with a Slice-6 update note acknowledging the intermediate recurrence — it is not reopened because the *closure mechanism* (the test guard `tests/test_train_sweep_reapply_allowlist.py`) now exists; pre-Slice-6 the loop was prematurely marked resolved without the closure mechanism.

## Packet timeline

- [P11](../packets/P11.md) — adds `webcam_harden` aug primitive (2026-04-27 23:55), but lives nested under `augmentation:` so does not trigger the allowlist gap. Not a contributor; mentioned because the augmentation block was the known-good case that misled the wrong-layer hypothesis.
- [P12](../packets/P12.md) — `c7dc828` adds `periodic_saves` to `trainer.py` + the yaml schema; the launcher allowlist is **not** updated in this commit. The `periodic_saves` patch silently failed on P12_HEAVY_LONG (2026-04-28 12:25) — same bug class, separate symptom (silent no-checkpoint vs silent DISABLED). The P12 packet retro carries the load-bearing periodic_saves silent-failure narrative; this thread carries the wandb-flattening narrative they share.
- [P13](../packets/P13.md) — `cab2909` adds `anchor_aware` + `face_scale_jitter` + `pipeline_randomization` to the trainer; allowlist update **missing**. P13_FROM_SCRATCH launches twice with broken-bug behavior. Commit `c366026` lands the fix mid-Slice-6.

## Evidence locations

- `train_sweep.py:172-282` — pre-`c366026` allowlist (11 nested-dict re-apply blocks; 3 new blocks added at `:283-313` post-`c366026`, total 14 keys)
- `train_sweep.py:144-147` — the `wandb.init(config=single_cfg)` call (the flattening source)
- `train_sweep.py:172-174` — the explicit `# CRITICAL FIX: Apply single_cfg directly to data_config BEFORE wandb overrides — W&B flattens nested dicts` comment from Slice 3
- `experiments/phase2_round13/R13_P13_FROM_SCRATCH.yaml:78-91, 269-278` — the three new nested blocks (anchor_aware, face_scale_jitter at the first range; periodic_saves at the second)
- `trainer/trainer.py:495-501, 503-510, 2374-2375` — the trainer-side load points; uncommitted `_to_plain_dict()` defensive helper at `:56-73` (harmless, can stay)
- `tests/test_train_sweep_reapply_allowlist.py` — new test in `c366026` that enforces the allowlist invariant for any future trainer-consumed nested-dict block
- `tests/test_to_plain_dict.py` — wrong-premise tests added during the misdiagnosis (7 tests pass; do not reproduce the actual bug); the handoff (`:293-340`) suggests replacing this with a test that exercises the actual train_sweep.py allowlist contract — `tests/test_train_sweep_reapply_allowlist.py` is what landed
- Master plan LOG: `april-26-training-master-plan-v2.LOG.md:1213-1336` — the 04-28 sequence (P12 dud + periodic_saves silent failure leading to the diagnosis path)
- Handoff: `docs/relaunch_handoffs/WANDB_FLATTENING_BUG_HANDOFF_2026-04-28.md` (full incident writeup, root cause, fix patch, verification plan)
- Memory: `project_wandb_flattens_nested_dicts.md` (originSession `feb6daa1-6d67-463d-9aff-a3b8d19b0981`)
- Commits: `cab2909` (P13 anti-shortcut yaml + trainer code, no allowlist update), `c7dc828` (periodic_saves trainer code + yaml schema, no allowlist update), `c366026` (the fix; 3 new keys + test guard), `6c9a320` (image 1.3.226 bump)

## Open loops

### Open loop: silent-feature-failures-pattern
status: open
severity: medium
first_seen: 2026-04-28
last_verified: 2026-04-29
close_criterion: a single pre-launch CI / smoke / lint step verifies that every yaml-declared trainer feature actually emits an "ENABLED" log line at trainer init — i.e., for every nested-dict block in the launched yaml that the trainer reads via `self.config.get('<block>')`, a runtime check fails fast (and a CI test fails offline) if the block is silently fall-through-disabled

This is a **meta-loop** consolidating the pattern that surfaced 3+ times across Slices 5/6:

1. **Slice 5 — `apply_svd_to_in_proj` zero-gradient** (memory `project_in_proj_svd_gradient_bug.md`): added 2026-03 ish; broken since first landing because `forward_pre_hook` did `.data.copy_` (no autograd graph). Discovered + fixed 2026-04-26 (`2feea58`). Implication for every prior R12/RLP/P-* run with `apply_svd_to_in_proj: true`: the q/k/v residuals trained on regularizer gradient only.
2. **Slice 6 — `anchor_aware` / `face_scale_jitter` / `periodic_saves` silent DISABLED**: yaml-declared, trainer reads `None`, falls back to defaults that say DISABLED. ~$3.50 burned across two cancelled Vertex jobs before symptom diagnosis.
3. **Slice 6 — `periodic_saves` silent no-checkpoint** (this thread's sibling open loop in [P12.md](../packets/P12.md)): the `periodic_saves` patch in `c7dc828` was supposed to fire `save_ckpt(prefix='periodic')` at every step in `step_list`; it never fired on P12_HEAVY_LONG (2026-04-28 12:25). Same root cause class (config flattening + no isinstance(dict) defense) as the wandb-flattening symptom; same symptom class as the in_proj-SVD bug (added in good faith; turned out not to be firing). 

The closing criterion above is the observable that lets the next agent treat all three (and any future instance) as one solved class. Pre-launch CI is the canonical answer; smoke + a YAML-declared-feature-vs-runtime-log audit at `Trainer.__init__` is a cheaper variant that catches the same class.

The `wandb-side-surface-hygiene-not-systematic` loop (medium, [`wandb_yaml_propagation_bugs`](wandb_yaml_propagation_bugs.md)) is **paired with this loop** — that one tracks the wandb-side specifically (string-length validation + flattening + future surfaces); this one tracks the broader "added-in-good-faith-but-not-firing" pattern. They will likely close together via the same pre-launch verification mechanism.

**Slice 7 verification (2026-04-29) — stays `open`; no new instance, but the post-fix P13 run validated the allowlist closure.** The post-`c366026` P13_FROM_SCRATCH relaunch fired ALL three new yaml blocks correctly: trainer init logged `AnchorAwarePenalty ENABLED weight=5.0 target_mean_prob=0.10`, `FaceScaleJitter ENABLED scale_limit=0.25`, `PeriodicSaves ENABLED step_list=[2000,4000,...]`. The Day-4 verdict γ (`docs/relaunch_handoffs/P13_DAY4_VERDICT_2026-04-29.md`) reflects the recipe's true behavior on the second-generation infrastructure. **The train_sweep allowlist closure pattern works in production**; the broader meta-pattern (any future config block in a different code path) stays open until the canonical pre-launch CI lands. Loop status unchanged; `last_verified=2026-04-29`.

### Cross-thread refs

- [`wandb_yaml_propagation_bugs`](wandb_yaml_propagation_bugs.md) — the parent thread for the bug class (Slice 3 origin); the `wandb-side-surface-hygiene-not-systematic` open loop is the broader-pattern surface; this thread is the Slice-6-specific synthesis.
- [P12 retro](../packets/P12.md) — the periodic_saves silent failure is the sibling symptom of the same root cause class.
- [P13 retro](../packets/P13.md) — the P13 launch where the wandb-flattening symptom fired live.
- [`in_proj_svd_gradient_bug`](in_proj_svd_gradient_bug.md) — Slice 5's instance of the "silent feature failure" pattern; cited above in the meta-loop.
- [`contract_policy_bug`](contract_policy_bug.md) — different root cause but the same meta-pattern of "design lands but doesn't ship to deployment artifact".
