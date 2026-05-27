# AGENT GUIDE — read this BEFORE proposing any next-step packet

> **Status**: rolling document (formerly `docs/relaunch_handoffs/AGENT_GUIDE_2026-05-02.md`).
> Moved into the wiki on 2026-05-07. The original dated file is preserved at the old path as a historical anchor for memory entry `reference_agent_guide_2026_05_02.md`.
> Edit this file when you discover a new class of agent mistake; do not edit the dated copy.

**Date authored**: 2026-05-02 (after the agent who wrote this proposed P19 = a relaunched lever that had already been pulled twice and failed both ways)
**Last revised**: 2026-05-07 (moved into wiki, no rule changes)
**Audience**: any agent picking up this work cold
**Purpose**: stop wasting the user's GPU budget and time re-proposing experiments that already ran

---

## Why this guide exists

On 2026-05-02 the previous agent (me) read the handoff packet, the measurements doc, the canonical entry points, and three memory entries — then proposed `P19_VISO_TEAMS_ENHANCED_FT`: FT-from-P8A with `combined_paired.visomaster_teams_enhanced.enabled: true` and family weight 8.0, plus a 12-bucket GRL block.

**That exact recipe had already been launched twice**:
- `xan4dfto` (P14_DATA_FIX, fw=8.0) — collapsed, value_composite=0.126.
- `rmic6wrc` (P16_DATA_AXIS, fw=2.0, no GRL, no anti-shortcut bundle) — didn't promote, viso recall stuck at ~1% at calibrated τ.

The information was in:
- `viewer/model_dashboard_runs.yaml` (modified in the working tree at session start, the `xan4dfto` entry tagged `collapse`)
- `experiments/phase2_round13/R13_P16_DATA_AXIS.yaml:247-258, 282` (the toggle and the `# CONSERVATIVE (DATA_FIX used 8.0 → collapse)` comment)
- Memory `project_p16_data_axis_does_not_promote_2026-04-30.md` (in the MEMORY.md index)

**I did not check.** The user pushed back ("ARE YOU SURE?") and the miss surfaced. This guide is the procedural fix for that class of mistake.

---

## The hard rules

### Rule 0 — Bucket-manifest verification before touching strategy taxonomies

Added 2026-05-05 after the `quality_enhancement` misrouting bug was found (commit 38558ee5, March 2026 — sat undetected through ~150 R13 yamls and contaminated every R13 training run). Before adding/changing/relying on entries in `DEFAULT_ENHANCED_STRATEGIES` (in `utils/grouping.py`) or `enhanced_strategy_names` (in any yaml's `augmentation.routing` block), **you MUST verify the strategy via its bucket manifest**:

```bash
gsutil cat gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/<sample_id>/manifest.json
```

A genuinely-enhanced (GFPGAN-applied) sample carries:
- `"enhancement": "GFPGAN_sample"` (or equivalent enhancer tag)
- `"model": "GFPGANv1.4"` (or equivalent model tag)
- `"original_sample_id": "<base_sample_id>"` (paired to a base strategy)

A base strategy (NOT enhanced) lacks all three fields. Folder name is NOT sufficient evidence — `quality_enhancement_*` folders sit in the same bucket as the actually-enhanced `*_enhanced_*` folders but carry no enhancer tag.

**Same rule applies to any new family-routing logic in `utils/grouping.py`** — the `infer_family_key` mapping must match the manifest evidence, not the strategy name pattern.

The full incident write-up: `docs/packet_retrospectives/threads/quality_enhancement_strategy_misrouting.md`.

### Rule 1 — Validate-before-suggest

Before proposing any packet that includes a hypothesis like "X has never been tested", "Y has not been tried with Z", or "we should enable W for the first time", **you MUST run all of the following**:

1. `grep -l "<key_toggle>" experiments/phase2_round13/*.yaml` — does any prior yaml have this toggle on?
2. For every yaml that matches: read the full yaml, check it actually launched (W&B run id present), check the result (memory entry, scorecard dir, viewer entry).
3. `cat viewer/model_dashboard_runs.yaml` — does the registered-runs config have any entry that already matches your hypothesis? Look at `recipe`, `tags`, `notes`.
4. Search MEMORY.md for the toggle name or hypothesis word. Read every entry whose description mentions it.

If any of (1)-(4) returns a match, **the hypothesis has been tested**. Do NOT propose the packet without first reading the result and articulating what's different about your proposal.

### Rule 2 — Read failure modes, not headlines

If a memory entry says "X failed", read the failure mode in detail before assuming X is closed.

Example: `project_p16_data_axis_does_not_promote_2026-04-30.md` says viso recall was 0.9-1.1% at calibrated τ. The naive read is "data axis lever is dead". The accurate read is "data axis lever moved viso recall to 51.6% at default τ=0.5 but τ-tail collapse pushed it to 1% at deployment τ" — and the recommendation in that entry is **specifically** "future packets need to either (a) directly target the τ-tail or (b) pivot to a different intervention". An agent that proposes P14_DATA_FIX-with-GRL is ignoring the recommendation.

**Rule**: when reading a memory entry, also read the "How to apply" section. It exists because the past agent saw exactly the failure mode you're about to repeat.

### Rule 3 — CPU-first-then-GPU

The user's GPU budget is finite (~$65 remaining of $80 as of 2026-05-02). Each $25-50 GPU spend buys one Vertex job. The CPU budget is **infinite**.

Before proposing any GPU spend, **you MUST first**:
1. Identify the cheapest CPU diagnostic that would update your beliefs about the proposal.
2. Run it (or propose it as Stage A) BEFORE the GPU launch.
3. Justify why the GPU spend is required given what the CPU diagnostic could rule out.

If you can't articulate a CPU-first measurement that would change your mind about the proposal, **the proposal is not ready**.

Example: I proposed P19 ($30-45 Vertex spend) without first checking `analysis/p18_probe_2026-05-01/d_results/promotion_contract/threshold_grid.csv` to see whether prob_fake distributions on viso_enhanced uniquely have a left tail vs all suites being uniformly soft. That CPU analysis ($0) would have pointed me at loss/calibration intervention rather than data-axis re-launch.

### Rule 4 — Viewer-integration

The user has a training data viewer at `viewer/server.py` + `viewer/templates/index.html` with a **Model Diagnostics** tab (the second tab, after Overview). Registered runs are configured in `viewer/model_dashboard_runs.yaml`.

The Model Diagnostics tab supports per-run:
- Feature manifold (t-SNE, color by label / method / capture_mode / face_area_bucket / score_bin / correctness)
- Face-size invariance probe (crop-tightness sweep)
- Domain confusion probe
- Scorecard suites table (suite, n, metric, mean prob, p90)
- Frame galleries (crop-sensitive, manifold sample, false positives, false negatives, near threshold)

**Rule**: every CPU analysis script you write that produces per-frame or per-checkpoint results SHOULD output its artifacts in viewer-compatible format (CSV/JSON under `analysis/<probe-name>/outputs/`) and SHOULD be registered as a run entry or attached artifact in `model_dashboard_runs.yaml`.

This is so the user can monitor the analysis without having to read JSON. The viewer is the human-oriented view.

When you write an analysis script, ask: "where does this show up in the viewer?" If the answer is "it doesn't", consider whether you should add a viewer panel for it. Default is yes for any analysis the user might want to inspect frame-by-frame.

### Rule 6 — Aug-lever-rediscovery check (added 2026-05-19)

Added after an agent ran an 8-packet validation expansion (E1-E9), measured a CPU-simulated "B3 bundle" augmentation closing 79% of dev↔lockbox W1 distance on prob_fake distribution, and proposed it as a new training-side intervention recommendation in [`analysis/teams_account_validation_2026-05-19/EXPANDED_FACTS_2026-05-19.md`](../../analysis/teams_account_validation_2026-05-19/EXPANDED_FACTS_2026-05-19.md). The "B3 bundle" duplicated `data/augmentations/pipeline_randomization.py` + `data/augmentations/resolution_chain_aug.py` — both already in-tree, the latter already empirically run as Slot α (`lsx4n0t7`) 2026-05-15 overnight with observed 25% real score_range cut.

The aug-lever rediscovery class is structurally distinct from Rule 1 (toggle / yaml validation) because the existing code is in `data/augmentations/`, not in an experiment yaml. Rule 1's `grep experiments/phase2_round13/*.yaml` does not catch this. A dedicated check is needed.

**Rule**: Before proposing any data augmentation (blur, jpeg, color jitter, downsample, resolution chain, contrast, brightness, saturation, gamma, hue, channel scaling, noise, JPEG compression, ANY pixel-level perturbation), **you MUST run all of the following**:

1. `ls data/augmentations/` — enumerate all augmentation modules.
2. For each existing module, read its docstring + class signature. Check whether your proposed sub-aug is already implemented (under any name).
3. `grep -l "<sub_aug_name>" experiments/phase2_round13/*.yaml | head -20` — which prior packets enabled this aug?
4. For each match: read the recipe block (typically the `augmentation:` or `pipeline_randomization:` block in the yaml), check whether the magnitude / probability range you're proposing is within the already-tested envelope.
5. Search MEMORY.md and `threads/anti_shortcut_bundle_decomposition.md` for the aug name and its prior outcomes.

If any sub-aug in your proposal matches existing infra:
- Default disposition: do NOT propose a new aug. The empirical question is "did the existing aug recipe close the loop you're targeting?" — that's a scorecard-readout question, not an aug-design question.
- Acceptable exception: you can propose a magnitude-range extension OR a probability-rebalance OR a NEW sub-aug not in the existing modules. In that case, explicitly state "this is a delta to <module>:<sub_aug>" and quote the existing magnitude/probability you're proposing to change.

Existing aug modules as of 2026-05-19 (verify against `ls data/augmentations/` at session start, this list may have grown):

| module | sub-augs implemented |
|---|---|
| `data/augmentations/pipeline_randomization.py` | jpeg roundtrip, downscale-upscale (mild), chroma blur, RGB↔YUV roundtrip, gamma jitter, Gaussian luma blur, brightness shift |
| `data/augmentations/resolution_chain_aug.py` | downsample-upsample chain (more aggressive than pipeline_randomization's downscale) |
| `data/augmentations/teams_simulation.py` | helper kernels for pipeline_randomization (chroma blur, jpeg roundtrip primitives) |

Empirically-tested yaml entry points as of 2026-05-19 (status column reflects most recent contract verdict — verify against `analysis/<latest>_eval_<date>/RESULTS_FACTS_*.md` before citing):
- `R13_P22_AUG_CURRICULUM.yaml` — pipeline_randomization, P22-era; status: **MIXED** (step8k degraded; step1k robust per memory `project_p22_cpu_followups_reframe_2026-05-02.md`)
- `R13_T5C_RESCHAIN_2026-05-15.yaml` — resolution_chain_aug on T5C-base, Slot α (`lsx4n0t7`); status: **REFUTED at 2026-05-16 contract scorecard** (rank 5, `dev_fake_macro_recall=0.226` below 0.30 floor; the 25% score_range CPU cut was score-distribution compression, not encoder invariance — memory `project_overnight_resolution_chain_2026-05-16.md`, eval folder `analysis/reschain_grl6_eval_2026-05-16/`)
- `R13_T5C_ANCHOR_AWARE_2026-05-16.yaml` — Slot A v2 (different lever class, included for context); status: rank-2 in band_shortcut_ood scorecard per memory `project_band_shortcut_ood_hypothesis_2026-05-16.md`

**Critical metric trap (added 2026-05-19 from Slot α post-mortem)**: any CPU probe for an aug-targeting-IQ-axis lever MUST report fake-vs-real AUC, not just W1 / score_range / KS / KL. AUC distinguishes encoder-level substrate invariance (AUC preserved; the success case) from score-distribution compression (AUC collapses; the Slot α failure mode). A 25-79% W1 closure with AUC drop > 0.02 IS the compression trap, not a win. Open loop `cpu-probe-mechanism-discrimination` in `threads/iq_shortcut_deconvolution_program_2026-05-08.md` (2026-05-16) tracks this requirement.

If your proposal stacks two augs that have not been stacked, that IS a novel proposal — but specify both as deltas to their respective modules; do not re-implement either.

### Rule 5 — Separate FACT from OPINION when writing handoffs

The pattern of past handoffs has been: a single doc mixes "we ran X with config Y and got result Z" (FACT) with "this means we should pivot to W" (OPINION). When the next agent reads the doc, they inherit both. They should only inherit the facts.

**Rule**: when you write a handoff, separate facts from interpretations into different files (or different sections, clearly demarcated with headers). See `PSERIES_FACTS_2026-05-02.md` and `PSERIES_OPINIONS_2026-05-02.md` for the pattern.

The OPINIONS file MUST start with a disclaimer that past framings have been demonstrably wrong, and MUST list at least one concrete example of a framing that was superseded.

**Update 2026-05-07**: The handoff-doc pattern is now superseded by the wiki itself (`docs/packet_retrospectives/`). New per-session HANDOFF_*.md files should not be authored. Findings flow into `STATE.md` + threads + packet retros instead. The FACTS-vs-OPINIONS split now lives at the eval-folder level (see `eval_folder_template.md`) and the packet-retro level (see `packet_template.md`'s `## Conclusions drawn in-session` section).

---

## One-page validate-before-suggest checklist

Copy this into your scratchpad. Tick each box before proposing a packet.

```
PROPOSAL: <one-sentence description of the packet>

PRIOR-WORK CHECK:

[ ] Identified the key toggle / configuration delta vs P8A baseline.
    Toggle name(s): _______________________________________________
    
[ ] Grep'd `experiments/phase2_round13/*.yaml` for each toggle.
    Matches found: _______________________________________________

[ ] If proposal involves ANY data augmentation (Rule 6):
    Listed `data/augmentations/` modules: _________________________
    Cross-check each proposed sub-aug against existing modules: ___
    Stated explicitly whether proposal is a NEW aug or a delta: ___

[ ] For each match: read yaml, found W&B run id, found scorecard or
    "drafted-not-launched" status.
    Status of each match: _________________________________________

[ ] Grep'd `viewer/model_dashboard_runs.yaml` for relevant tags,
    recipes, or notes.
    Matches found: _______________________________________________

[ ] Searched MEMORY.md descriptions for the hypothesis. Read every
    entry whose description matches.
    Memories read: _______________________________________________

[ ] If any prior run matches: explicitly stated what's different
    about my proposal and why the difference is load-bearing.
    Differences: _________________________________________________

CPU-FIRST CHECK:

[ ] Identified the cheapest CPU diagnostic that would update my
    beliefs about this proposal's outcome.
    CPU diagnostic: _______________________________________________

[ ] Ran the CPU diagnostic OR proposed it as Stage A before any GPU.

[ ] Articulated what each plausible CPU outcome (α/β/γ) would mean
    for the proposal.

VIEWER-INTEGRATION CHECK:

[ ] If proposing analytics: planned where outputs land in the viewer
    (artifact path + run yaml entry).

[ ] If proposing a new run: planned what the viewer would show for it
    after it lands (which probes / artifacts).

OPINION-VS-FACT CHECK:

[ ] In the proposal, every "we know X" claim cites a file or scorecard
    CSV path.

[ ] In the proposal, every "we believe Y / suggest Z" claim is marked
    explicitly as opinion.

[ ] If the proposal supersedes a past framing, named the framing
    superseded and the file it lives in.
```

If any box is unchecked, the proposal is not ready to surface to the user.

---

## What "the right next step" looks like (template)

A well-formed proposal has three parts:

### Part 1 — Empirical state (cite-everything)

What's measured, with file:line citations. No framings. Pull straight from the FACTS ledger.

### Part 2 — Hypothesis being tested

One sentence. Plus a "what would falsify this hypothesis" note. If you can't say what would falsify it, the hypothesis is not a hypothesis.

### Part 3 — Concrete artifact + cost + decision points

- Yaml diff or CPU script.
- Cost estimate (CPU $0 or Vertex $).
- Pre-launch CPU-first stage (Rule 3).
- α/β/γ outcomes and what each means for next decision.
- Go/no-go gates the user can use to intervene mid-sequence.

---

## Robustness measurement protocol (added 2026-05-10)

When the user asks "which model is the most robust" or "should we ship X," macro recall metrics on a single substrate are not sufficient. Recommended diagnostic protocol before recommending any deployment switch:

1. **Catastrophic-tail count on broad real cohort**: count of reals scoring above {0.5, 0.7, 0.9} per ckpt at fixed thresholds, regardless of calibrated τ. Plus unique catastrophic-FP frames (where one ckpt > 0.9 AND others < 0.5). A model with many unique catastrophic-FP frames has a characteristic failure mode the others avoid.

2. **Per-axis-bin FPR variance**: at each ckpt's calibrated τ (calibrated to 5% overall FPR on `teams_real_all_dev`), bin reals by IQ axis quartile and measure per-bin FPR. The ckpt with lowest FPR variance across bins is the most decoupled per-axis. The worst single bin per ckpt names the ckpt's fragility axis.

3. **Production-frame retest if available**: score all candidate ckpts on production-fragility cohorts (e.g., the may6/may5 frames at `analysis/xinhe_cross_camera_audit_2026-05-06/`). Day-to-day drift (mean(today) − mean(reference_day)) is the production-fragility signature.

4. **Post-IQ-gate FPR per ckpt**: at multiple gate thresholds, measure FPR on the gate-PASSING subset. A gate that filters by axis X may not shield a ckpt whose fragility is on axis Y. Note: gate filtering can work "by accident" on the eval substrate when the failure axis correlates with the gate axis there but not in production.

5. **Cross-ckpt disagreement structure**: pairwise score correlations + outlier-high/low distribution on disputed real frames. Ckpts that are consistently outlier-high are aggressive on reals; outlier-low are under-confident on fakes.

Reference implementation: `analysis/cpu_diagnostics_2026-05-10/scripts/job_{a,b,c,c2,d}_*.py`.

## Cross-substrate τ comparison rule (added 2026-05-10)

When comparing recall numbers across ckpts on a substrate, use **each ckpt's FPR-calibrated τ on that substrate**, not a fixed τ=0.5. Different ckpts have different score distributions; a fixed-τ comparison can show one ckpt at 30% recall when its FPR-cal recall is 77%. The macro metric "fake recall at FPR=5%" is the apples-to-apples comparison; "fake recall at τ=0.5" is not.

This rule was the source of an in-session error 2026-05-10: an HDTF analysis at τ=0.5 made T3_SLOT1_step1500 look broken (28.8%) when its FPR-cal recall was 77.4%. Always use FPR-cal τ for cross-substrate / cross-ckpt fake recall comparisons.

The corollary: **cross-cohort FPR comparisons should use the same FPR target on each cohort**, not the same τ. A ckpt calibrated to 5% FPR on cohort A may have very different FPR on cohort B at the same τ.

## "Deployed model" vs "production anchor" — attribution discipline (added 2026-05-10)

The deployed model and the production anchor are different ckpts:
- **Deployed model** (currently `E2B_TOP_N_STEP3200`): the binary in production. What the user's hands-on experience refers to when they describe "model behavior in the field."
- **Production anchor** (currently `P8A_REFERENCE_STEP5000`): the substrate-invariance reference. The ckpt new candidates are compared against for promotion decisions.

When a memory entry says "[deploy] does X" or "the production model behavior is Y," disambiguate:
- Read the entry against `project_deployment_is_e2b_2026-05-06` (or successor entries).
- The Pearson r=+1.000 between deploy and E2B local is the load-bearing fact.
- "Deploy" in any 2026-05-06+ entry = E2B unless an explicit successor memory says otherwise.

When a user says "model X feels Y" — verify:
- Are they referring to the deployed model (their hands-on production experience) or to a specific ckpt they're locally testing?
- The wording "I use X" can mean either. Ask if unclear before acting on the anecdote.

This rule was the source of an in-session error 2026-05-10: the may6 false-flag memory was assumed to be P8A's behavior because P8A is the production anchor we discuss most often. The 5-ckpt may6 retest showed the false-flagging was E2B's, and P8A handles may6 perfectly.

## How to keep this guide alive

- Whenever a past framing is superseded, add it to the most recent eval folder's `AGENT_PROPOSAL_<date>.md` §"Self-correction log".
- Whenever a new analysis script lands without a viewer integration, log it as tech debt and revisit when the user asks for monitoring.
- Whenever an agent makes a mistake of the class this guide is designed to prevent, add a section here with: what mistake, what the right move would have been, what tooling/checklist could have caught it.

This guide is supposed to grow as we learn more failure modes.
