# Thread: Image-rebuild discipline (yamls baked into the container)

## The question

The `scripts/launch/launch_experiment.sh` launcher bakes yamls into the container image at `/workspace/experiments/`, not at runtime. Any new yaml or modified yaml that is not in the **image** when a job runs will fail the job at startup with `FileNotFoundError`. The question this thread answers: under what conditions does an agent need to rebuild the image vs. reuse the current one, and what is the minimal cost-correct procedure?

The packet that forced this thread into existence was [RLP4](../packets/RLP4.md): all 8 jobs failed because the 8 yamls were committed locally but image `1.3.195` had been built before the commits landed. Every agent after RLP4 inherits the rebuild rule.

## Initial belief

Before [RLP4](../packets/RLP4.md), the operating assumption was that yamls under `experiments/phase2_round13/` were authoritative at launch time — i.e., the launcher would read whichever yaml was on disk at submission. Through [RLP1](../packets/RLP1.md), [RLP2](../packets/RLP2.md), [RLP3](../packets/RLP3.md), and [RLP3.5](../packets/RLP3_5.md), this assumption was incidentally satisfied because new yamls were always added together with code changes and a VERSION bump, so the image was always rebuilt for the same launch wave that introduced them.

[RLP3](../packets/RLP3.md) and [RLP3.5](../packets/RLP3_5.md) had no incident because their commits (`c6f1034`, `477b00b`, `872502c`) landed code + yamls + VERSION bump as a single atomic operation, with `./dev.sh build-prod -y` triggering an automatic image build. RLP4 was the first packet that tried to launch yaml-only changes against an image that had been built before the yamls existed.

## What changed our mind

- **2026-04-22 RLP4 8-of-8 launch failure** ([RLP4.md:40-42](../packets/RLP4.md), session `cd19ddfb`). Image `1.3.195` was built immediately after the RLP3.5 commit `872502c` (the train_sweep.py value_composite fix). The 8 RLP4 yamls were authored later, on the same branch, but **never committed and never rebuilt into a new image** before launch. All 8 containers died ~30s after start with `FileNotFoundError: [Errno 2] … /workspace/experiments/phase2_round13/R13_RLP4_<slot>_*.yaml`. No W&B init, no OOD composite, no checkpoint — zero trained `value_composite` for RLP4. The packet's only contribution was the postmortem.
- **The launcher contract.** `scripts/launch/launch_experiment.sh` packages the yaml path as a `/workspace/...`-prefixed string and passes it to the Vertex container. The container has only the files baked into the image (the source-tarball snapshot taken at `gcloud builds submit` time). Files committed after the build are invisible to the running job. `arena/launch_*.sh` siblings have the same property — see memory `reference_image_rebuild.md` for the canonical statement of the rule.
- **2026-04-22 RLP4 → RLP5 forcing-function pivot** ([RLP4.md:59](../packets/RLP4.md), session `cd19ddfb`). The pivot to RLP5 explicitly added: (1) `./dev.sh build-prod -y` bumps VERSION patch automatically (`1.3.195 → 1.3.196`); (2) a single canary launched in `us-west4` before fan-out; (3) a yaml-tracked-in-git invariant. These three together close the failure mode.
- **Memory `reference_image_rebuild.md` codifies the rule.** The memory entry was created in direct response to RLP4 and lists the four "DO need a rebuild" cases (new checkpoint map yaml; new suite manifest yaml; new training experiment yaml; any code change in the runtime path the new job will hit). The "do NOT need a rebuild" cases are the inverse: code-only changes that no current job will touch, or one-off validations against a checkpoint with existing eval suites.

## Current stance (2026-04-29)

The image-rebuild-before-launch rule is permanent. The minimal correct procedure when an agent prepares a new packet of training jobs:

1. **Land all yamls and code changes on the branch.** Commit them. The branch state at this moment is what will be in the image.
2. **Run `./dev.sh build-prod -y` from `DeepfakeBench/training/`** (Mac Darwin/arm64). This auto-increments the VERSION patch (e.g., `1.3.195 → 1.3.196`), tarballs the working tree at this commit, submits to Google Cloud Build via `cloudbuild.yaml`, pushes the new image to `us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:<NEW_VERSION>` tagged `latest`. The VERSION file is written but **not auto-committed** — the user prefers explicit commits for VERSION bumps.
3. **Confirm the image exists** before launching:
   ```
   gcloud artifacts docker images list us-docker.pkg.dev/train-cvit2/effort-detector \
     --include-tags --filter='tags~"<VERSION>"'
   ```
4. **Canary first.** Launch one yaml in `us-west4` (or whichever US region is known-good today) before fanning out the rest of the slate. A canary that goes `RUNNING` in <2 minutes confirms the yaml is in the image and the launcher is working. RLP4's 8/8 failure was visible within ~30 seconds of launch, but a canary makes the failure cheap to recover from.
5. **Then fan out.** Launch the remaining slots, ideally spread across multiple US regions for quota hedging (per CLAUDE.md region-preference rules).

The rule does **not** apply when:
- Code-only changes that no in-flight job will hit (e.g., a fix in a code path used only by the next packet).
- One-off retro evaluations against an existing checkpoint with existing eval suites and existing checkpoint-maps in the current image.
- Inspection/diagnostic scripts that run locally, not on Vertex.

## Packet timeline

- [RLP1](../packets/RLP1.md), [RLP2](../packets/RLP2.md), [RLP3](../packets/RLP3.md) — incidentally satisfied the rule because every yaml was added alongside a code/VERSION bump that triggered a rebuild.
- [RLP3.5](../packets/RLP3_5.md) — same; commits `477b00b` (yamls + code) and `872502c` (`train_sweep.py` fix) each triggered a build (`1.3.192`, `1.3.193`).
- [RLP4](../packets/RLP4.md) — the founding incident. 8 yamls authored without a commit + rebuild. All 8 jobs failed.
- [RLP5](../packets/RLP5.md) onward — adopt the canary-first pattern; `./dev.sh build-prod -y` bumps `1.3.195 → 1.3.196` for the RLP5 launch.

## Evidence locations

- [RLP4.md:40-42, 49, 59](../packets/RLP4.md) — the failure postmortem and the procedural fix
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md:113-118` — gotchas section (yaml-staleness, anneal_steps, header comment lineage)
- `scripts/launch/launch_experiment.sh` — launcher (yaml-baking path)
- Memory: `reference_image_rebuild.md` — canonical statement of the rule (created in direct response to RLP4)
- Commits: `c6f1034` (RLP3 yamls + code, builds `1.3.190`), `477b00b` (RLP3.5 yamls + plumbing, builds `1.3.192`), `872502c` (train_sweep.py fix, builds `1.3.193`); RLP4 yamls were drafted but never committed before `1.3.195` was built — that gap is the bug.

## Open loops

### Open loop: rlp4-image-rebuild-discipline-forged
status: resolved
severity: medium
first_seen: 2026-04-22
last_verified: 2026-04-29
close_criterion: a packet successfully launches yaml-only or yaml+code changes after a deliberate `./dev.sh build-prod -y` rebuild + canary-first pattern (no untracked-yaml FileNotFoundError class repeats)

RLP4's 8-of-8 launch failure ([RLP4.md:40-42](../packets/RLP4.md), session `cd19ddfb`) is the founding incident. The procedural fix is captured in (a) memory `reference_image_rebuild.md` (codifies the four "DO rebuild" cases), (b) the RLP5 launch protocol (single canary in `us-west4` before fan-out, `./dev.sh build-prod -y` bumps `1.3.195 → 1.3.196`), and (c) the pattern reinforced through every later packet with no recurrence. The loop is preserved as the historical anchor — slice agents on later packets should expect to see the rule referenced when a wave of yamls lands. If a future slice agent observes a recurrence (any "yamls untracked at image build time → FileNotFoundError" pattern), reopen the loop.
