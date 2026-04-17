# Internal Summary: Near-Identical-Frame Inference Instability Mitigation

**Purpose:** skeptical handoff note for another agent.  
**Intent:** separate what was observed, what was proposed, what was actually tested, what helped, what hurt, and what is still not closed.

This note is about the specific problem of **unstable frame-level inference on visually very similar frames**, especially under live Microsoft Teams conditions. It is not a generic project summary.

## 1. Established failure mode

The instability problem is real and has been observed in more than one place:

- `R8` analysis explicitly documented nearly identical frames producing sharply different scores, with an example like `0.03` vs `0.48`.
- The post-launch Teams memo says production still showed **false positives on real faces** and roughly **plus/minus 10 percentage point frame-to-frame swings** on the same participant.
- The repo also documented a strong calibration/domain-gap signal:
  - training/validation EER threshold around `0.45`
  - production/OOD threshold closer to `0.77`
  - training reals much darker than production Teams reals

That matters because the problem is not only "the model sometimes flickers." It is also "the raw score scale is not portable across domains."

## 2. Important distinction: proposals versus validated fixes

Early internal writing can look more confident than the actual experiment record.

The strongest example is `phase2_round8/SCORE_INSTABILITY_ANALYSIS.md`:

- it correctly diagnosed plausible causes:
  - ArcFace scale amplification near the boundary
  - codec jitter
  - ViT patch / compression-block misalignment
  - SVD residual sensitivity
  - crop instability
- but most of its fixes were still **a fix plan**, not proven wins

Do not let another agent read that document as if it were already a successful ablation report.

## 3. What we tried on the training/model side

### 3.1 Generic stability regularization

This is the most important negative result.

The intended intervention was:

- generate a perturbed version of the input
- add a consistency / KL-style stability loss
- optionally combine with label smoothing and a gentler ArcFace scale

But the historical execution matters:

- `R9` intended to test stability regularization
- due to a config propagation bug, `stability_lambda` and `label_smoothing` never reached the trainer
- so all `R9` runs were effectively `lambda = 0`

That means any claim that "`R9` validated stability regularization" is false.

### 3.2 What happened after the bug was fixed in `R9.5`

`R9.5` is the cleanest internal evidence we have for the generic stability-loss idea.

Key runs:

| Setting | Run | YouTube jitter | OOD AUC | VCD real |
|---|---|---:|---:|---:|
| `lambda = 0.0` | `R9_A` | **0.0389** | **0.9768** | **80.3%** |
| `lambda = 0.1` | `R95_B` | 0.0407 | 0.9729 | 76.9% |
| `lambda = 0.3` | `R95_A` | 0.0438 | 0.9661 | 75.4% |
| `lambda = 0.5` | `R95_C` | 0.0453 | 0.9556 | 74.7% |

What that means:

- the stability machinery was confirmed active because `stability_loss` became nonzero
- but the actual deployment-relevant outcomes got worse
- the degradation was monotonic:
  - more stability loss
  - lower OOD AUC
  - worse VCD real
  - worse YouTube jitter
- zero jitter on some monitored lanes in `R9.5` was **not** a success signal:
  - `WMA` and `VCD` jitter hit `0`
  - the report explicitly attributes that to score saturation / floor-ceiling effects, not to genuine stability
- the fine-tuned `R9.5` variants also did not improve Teams holdout relative to `R9_A`

Internal conclusion from `R95_FINAL_REPORT` was direct: **generic stability regularization was counterproductive**, and the accidental `lambda = 0` configuration in `R9` was actually better.

### 3.3 Label smoothing

This also came out negative once the bug was fixed.

- `R95_B` used no label smoothing and was the best `R9.5` fine-tune on OOD
- smoothed variants underperformed it
- later summaries explicitly say label smoothing did not help

So label smoothing should not be described as a validated instability fix in this repo.

### 3.4 ArcFace scale changes

The early instability analysis argued that the ArcFace scale was amplifying tiny feature movements into large probability swings. That theory was reasonable, but the actual result needs to be stated carefully.

What was actually shown:

- a **higher** ArcFace scale in fine-tuning hurt:
  - `R95_E` used the more aggressive scale path
  - OOD AUC dropped from `0.9661` to `0.9585` relative to `R95_A`
- so the gentler schedule was preferable to the aggressive one in that FT regime

What was **not** shown:

- we do **not** have clean evidence that ArcFace scale tuning alone solved the near-identical-frame instability
- the repo evidence is only strong enough to say aggressive scale was harmful in that tested fine-tuning context

### 3.5 Residual feature normalization and multi-crop training

These appeared in the early fix plan, but I did **not** find a decisive later experiment package showing they were run and validated as winners.

So the safe reading is:

- proposed
- scientifically plausible
- not established as completed successful mitigations in this repo

### 3.6 Scratch-versus-fine-tune nuance

One subtlety that another agent should keep in mind:

- `R95_D` was a scratch run with stability enabled
- it was still improving when the run ended
- it preserved hard-method performance better than the FT runs and had stronger Teams EC holdout

But that does **not** rescue the generic stability-loss story:

- `R95_D` was undertrained
- it does not isolate stability as the cause of any benefit
- the main clean comparison remains the FT lambda sweep, and that result was negative

## 4. What we tried on the data / augmentation / domain side

### 4.1 Teams codec simulation and domain adaptation

`R9` is important here.

- `PHASE2_SUMMARY_R8_R11.md` treats Teams codec simulation as a validated part of Teams adaptation
- this likely helped the model become more Teams-aware than the `R8` family

But this should not be overstated:

- it did **not** close the live instability problem
- production still later reported false positives and unstable scores
- the effect is better described as partial domain adaptation, not a demonstrated cure for frame-to-frame instability

### 4.2 Lighting / gamma / nuisance augmentation

This area remains more uncertain than some round narratives imply.

`R12_POST_LAUNCH_PLAN.md` correctly focused on brightness, white balance, contrast, and other Teams-specific nuisance factors. That led to augmentation-side work such as:

- independent context-variation firing instead of `OneOf`
- color temperature shift
- asymmetric brightness
- more lighting-oriented augmentation thinking

But Round 2 later tightened the repo-truth critique:

- current `R13` YAML appears to request GammaUp
- runtime does **not** actually activate it under the current key mismatch
- direct Teams rows still bypass the stronger nuisance pipeline and only get light passthrough augmentation

So the correct status is:

- the lighting/gamma idea is plausible
- augmentation-side work was a reasonable direction
- but the actual plumbing truth is still shaky enough that more lighting conclusions would be premature

Another agent should assume this area is **not closed** until active transforms are verified end to end.

## 5. What we tried on the inference / decision side

This is where the latest research is most skeptical of the old process.

## 5.1 Calibration

The repo already has real calibration tooling:

- `scripts/run/run_r8_calibration_fit.py`
- `scripts/run/run_r8_apply_calibrator.py`
- `app4.py` can load a calibrator bundle and use its recommended threshold

`app4.py` also already applies calibrated probabilities if a calibrator is configured, instead of only using the raw score.

This matters because the repo's own post-launch work identified a serious threshold portability problem. A fixed threshold of `0.5` is not a trustworthy final operating point.

### 5.2 Aggregation and voting

`app4.py` already supports video-level decisions via:

- mean
- median
- majority vote

That is relevant because the live object is a stream, not an isolated frame.

### 5.3 TTA

`app4.py` also contains a simple TTA path based on original plus horizontal flip.

But the critical caveat is:

- I did **not** find decisive internal evidence that this TTA path was the thing that materially solved the near-identical-frame instability in production

So another agent should treat TTA as:

- implemented in inference tooling
- plausible as a variance reducer
- not yet established as a closed win on the current checkpoint shortlist

### 5.4 Older decision-artifact results

The older arena artifacts are important because they show that decision policy can move false positives without retraining.

Examples already in repo artifacts:

- `teams_only` binary strategy:
  - threshold `0.56`
  - `K = 16`
  - Teams `TPR = 0.9691`
  - Teams `FPR = 0.0126`
- `teams_only` uncertain strategy:
  - threshold `0.48`
  - `K = 16`
  - margin `2`
  - Teams `TPR = 0.9537`
  - Teams `FPR = 0.0063`
  - uncertain rate `0.0365`
- broader `all_three` binary configuration:
  - threshold `0.47`
  - `K = 18`
  - Teams `TPR = 0.9537`
  - Teams `FPR = 0.0063`
- `all_three` uncertain strategy:
  - threshold `0.32`
  - `K = 20`
  - margin `4`
  - Teams `TPR = 0.9073`
  - Teams `FPR = 0.0032`
  - uncertain rate `0.0642`

Extremity-gate artifact:

- baseline ungated operating point:
  - `T = 0.30`
  - `K = 18`
  - `TPR = 0.9378`
  - `FP = 18`
  - `BalAcc = 0.9464`
- mild gate:
  - `FP = 15`
  - `TPR = 0.9341`
  - `BalAcc = 0.9483`
- stronger gate:
  - `FP = 12`
  - `TPR = 0.9206`
  - `BalAcc = 0.9453`

These numbers matter because they show that thresholding, abstain logic, and gating are not cosmetic.

But the critical limitation is just as important:

- these are **older artifacts**
- they are **not** the authoritative current `R12_G` versus `R13_*` promotion result
- they show leverage, not closure

## 6. Latest research position

The latest April 15 research synthesis is stricter than some older round narratives.

The current internal position should be summarized like this:

- the instability problem is **partly a model problem and partly a decision-system problem**
- generic training-time stability regularization is **not** the leading answer anymore
- the highest-leverage near-term work is:
  - target-domain calibration
  - short-window temporal aggregation
  - explicit uncertainty / disagreement handling
  - threshold selection on the actual Teams low-FP contract

That is a meaningful shift in emphasis.

The repo no longer supports the story that "we just need one more stability loss or another fixed-threshold table."

## 7. What is still not proven or not done well enough

Another agent should keep these gaps open:

1. We have **not** proved a current shortlist winner under the calibrated low-FP promotion contract.
   - Round 2 and Round 3 both say this remains open.

2. We have **not** validated current decision-layer ideas on the frozen reduced shortlist:
   - `R12_G_FP32`
   - `R13_A_STEP15500`
   - `R13_E_BESTSOFAR`
   - `R13_FT7_FP32`
   - `R13_FT9_FP32`

3. We have **not** shown that threshold `0.5` is promotion-safe.
   - Latest research treats that as explicitly unacceptable.

4. We have **not** closed augmentation-plumbing truth.
   - GammaUp activation is still suspect.
   - direct Teams rows still appear under-augmented

5. We have **not** demonstrated that TTA, median/majority aggregation, hysteresis, or disagreement gates solve the current problem on the current finalists.
   - some are implemented
   - some are recommended
   - older artifacts suggest leverage
   - current frozen proof is still missing

6. We have **not** closed the missing-condition problem around enhanced-through-Teams fake data / eval.
   - decision logic cannot manufacture missing supervision

7. We have **not** run a clearly verified narrow spatial-consistency retrain after fixing augmentation plumbing.
   - if training-side stability is revisited, the latest research says it should be narrow and deployment-aligned, not a broad generic consistency loss

## 8. Bottom-line reading for another agent

If another agent wants the shortest defensible conclusion, it is this:

- We correctly identified that the system was unstable on very similar frames.
- We explored both model-side and inference-side mitigations.
- The strongest completed model-side ablation says **generic stability regularization and label smoothing hurt**.
- Higher ArcFace scale in fine-tuning also hurt.
- Some domain-adaptation and augmentation work was directionally reasonable, but the augmentation-runtime truth is still messy enough that those conclusions are not clean.
- The most promising underexploited lane is now **calibrated low-FP decision policy plus temporal / uncertainty handling**, not another generic stability-loss round.
- We should not pretend the problem is solved, and we should not pretend the current shortlist has already been judged on the right contract.

## 9. Primary repo evidence behind this note

- `experiments/phase2_round8/SCORE_INSTABILITY_ANALYSIS.md`
- `experiments/phase2_round9_5/R95_FINAL_REPORT.md`
- `experiments/PHASE2_SUMMARY_R8_R11.md`
- `docs/R12_POST_LAUNCH_PLAN.md`
- `docs/research_2026-04-15/03_experiment_memory_and_run_forensics.md`
- `docs/research_2026-04-15/07_literature_review_stability_calibration_and_low_fp.md`
- `docs/research_2026-04-15_round2/05_decision_system_low_fp_analysis.md`
- `arena/strategy_results/best_strategies_summary.csv`
- `arena/strategy_results/extremity_gate_full.csv`
- `app4.py`
