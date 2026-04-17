# Next-Round Research Plan

This plan assumes the next agent should continue from Round 2, not restart from first principles.

## Track 1: Calibrated promotion measurement

### Questions

- Which frozen checkpoint is actually best under a low-FP Teams deployment contract?
- Does the winner change once thresholds are calibrated per checkpoint?
- Are `FT8` and `FT10` really duplicates of `FT7` and `FT9` in the actual remote runtime?

### Required evidence

- calibrated dev-threshold sweep on the reduced shortlist
- lockbox readout at frozen threshold
- explicit proof of whether external unpaired reals loaded in the finalist runs

### Likely payoff

Very high. This can stop the project from training in the wrong direction.

## Track 2: Augmentation plumbing and nuisance ablations

### Questions

- Does a real GammaUp path reduce real Teams false positives?
- Is direct Teams spatial sensitivity a live issue once tested on the actual Teams branch?
- Is lighting pain mainly upward brightness, colour temperature, or directional shadow?

### Required evidence

- tiny allowlist/plumbing fix
- plumbing-control run
- one gamma-up run
- one Teams-spatial run

### Likely payoff

High. This is the cleanest way to turn the current nuisance story into falsifiable evidence.

## Track 3: Sampler and curriculum redesign

### Questions

- Can a true-Teams-companion family split materially increase useful exposure?
- Does a late-stage target-domain curriculum beat more weight tuning?
- Does per-identity-per-family sampling help without causing obvious forgetting?

### Required evidence

- exact post-filter/post-split family counts
- expected per-epoch exposure logs
- one short pilot run with explicit source split or sampler redesign

### Likely payoff

High. Current sampler leverage is too weak for the current naming story.

## Track 4: Missing-condition data and evaluation

### Questions

- Can the repo produce or recover a real enhanced-through-Teams fake lane?
- Can fake lockbox coverage be expanded beyond the current narrow session set?
- Can enhanced fake lockbox be created?

### Required evidence

- manifest or bucket truth showing actual enhanced-through-Teams samples
- loader truth confirming those samples are emitted
- updated target-domain suite definitions

### Likely payoff

Very high, but possibly slower. This attacks the current structural blind spot directly.

## Track 5: Capacity/backbone only after Tracks 1-4

### Questions

- Once evaluation, nuisance plumbing, and curriculum are cleaned up, is the small model still the bottleneck?
- If yes, which compact architecture change is justified by the remaining error pattern?

### Required evidence

- calibrated shortlist outcome
- cleaned-up ablation results
- target-condition data truth that is no longer structurally missing

### Likely payoff

Unknown until the first four tracks are done. This should stay behind them.

## Priority order

1. Track 1
2. Track 2
3. Track 3
4. Track 4
5. Track 5

## Hand-off rule for the next agent

The next round should try to close at least one of these with new evidence:

- the calibrated checkpoint winner
- a trustworthy augmentation ablation result
- a real sampler/curriculum redesign result
- proof that the missing condition has been created or found

If none of those can be advanced, the next agent should say so plainly rather than padding the package.
