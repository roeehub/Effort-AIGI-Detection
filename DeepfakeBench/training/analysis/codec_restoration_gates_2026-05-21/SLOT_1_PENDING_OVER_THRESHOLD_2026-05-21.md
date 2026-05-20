# Slot 1 PENDING > 30 min in us-west4 — user decision pending

> Auto-mode flag raised 2026-05-20 23:02 UTC. Slot 1 has been in
> JOB_STATE_PENDING for ~39 minutes in us-west4 (submitted 22:23:50 UTC).
> CLAUDE.md says switch regions; user's "up to 3 GPU jobs" cap + memory
> `feedback_no_cancelling_vertex_jobs` (no cancel without user OK) put me
> in a tension I did not have explicit authorization to resolve.

## Facts

- Slot 1 Vertex job: `5839421592722997248` in us-west4, submitted 2026-05-20 22:23:50 UTC.
- State at 23:02:43 UTC: `JOB_STATE_PENDING`. Elapsed: 38m 53s.
- CLAUDE.md rule (verbatim): "If a Vertex job is `PENDING` in a US region for **more than 30 minutes**, switch regions ... Do not cancel the original until the replacement has reached `RUNNING` state — avoids losing the slot if the new region also queues."
- User instruction tonight: "Run up to 3 GPU jobs and monitor."
- Memory `feedback_no_cancelling_vertex_jobs.md`: "Don't cancel Vertex training jobs without explicit authorization — cancellation is destructive class; even hung jobs need user OK first."

## Tension

The CLAUDE.md region-switch protocol creates 2 instances of the same packet running in parallel until the original is cancelled. Cancelling the original requires user OK per the memory. Without cancelling, the parallel instance would consume a 4th GPU seat (over budget). Two readings:

1. **Conservative (chosen)**: respect the explicit "up to 3 GPU jobs" cap and the no-cancel-without-OK rule. Leave Slot 1 in us-west4 to continue queuing. Document the situation for user review.
2. **CLAUDE.md-first**: treat region-switch as a same-job operational action (not a 4th experiment); launch in us-east1 or us-central1, wait for the new one to RUN, then ask for user authorization to cancel the us-west4 original.

I chose reading #1 because:
- (a) The "up to 3" cap is the more explicit and quantitative constraint.
- (b) Vertex PENDING doesn't auto-expire; the slot will eventually start when capacity opens. 38 min is over CLAUDE.md threshold but well within Vertex's normal scheduling latency for contested A100 quota.
- (c) Slot 2 and Slot 3-relaunched together cover both anchor=ON arms of the design; Slot 1's specific contribution is the higher dose (codec=0.40 + anchor=ON). The dose-response between Slot 1 and Slot 2 is informative but not load-bearing for the morning verdict — Slot 1 result can be inferred from Slot 2 + Slot 3 if Slot 1 doesn't land by morning.

## User options (review in the morning)

If Slot 1 is still PENDING:

A. **Cancel Slot 1 us-west4 and relaunch in us-east1**: per CLAUDE.md operational protocol. Cost: ~50 min of additional wait time; ~$0 (no GPU usage during pending). Use the launch script pattern:
   ```bash
   ./launch_experiment.sh -y phase2r13-experiments us-east1 experiments/phase2_round13/R13_T5C_ANCHOR_AWARE_PLUS_CODEC_2026-05-21.yaml
   gcloud ai custom-jobs cancel projects/700371397073/locations/us-west4/customJobs/5839421592722997248
   ```

B. **Wait — let us-west4 queue clear**: Vertex will eventually grant an A100. Cost: 0-2 more hours of wait time.

C. **Drop Slot 1 and proceed with Slot 2 + Slot 3 only**:
   ```bash
   gcloud ai custom-jobs cancel projects/700371397073/locations/us-west4/customJobs/5839421592722997248
   ```
   The 2×2 design degrades to anchor=ON × codec=0.20 vs anchor=OFF × codec=0.40 — still informative but loses the p=0.40 + anchor=ON cell (the primary recommendation).

If Slot 1 has transitioned to RUNNING by morning: no action needed.

## What I did NOT do (and why)

- **Did NOT launch Slot 1 in another region**: would exceed "up to 3 GPU jobs" cap.
- **Did NOT cancel Slot 1 us-west4**: per `feedback_no_cancelling_vertex_jobs`, cancellation requires explicit user OK.

## Open loops

- This incident is the second auto-mode tension this session (the first was Slot 3 failure → image rebuild, which I COULD handle autonomously because no rule prohibited rebuild). It surfaces the need for a clearer policy in CLAUDE.md or an explicit auto-mode flag: "in auto mode, allow region switching with same-job cancellation".

## References

- HANDOFF.md (working tree): full session log including this incident.
- STATE.md (working tree): 2026-05-21 entry, Vertex job table.
- Monitor task `blbxkk5u3` continues tracking all 3 jobs.
