# Handoff: plan our next steps — 2026-05-04 morning

You're a fresh agent. Roee will use this handoff to discuss what to do next. The previous agent (me) just spent 12+ hours running 3 architectural-axis packets (E1, E2b, E3) and reached conclusions that you should **critically evaluate before agreeing to**.

Your job is to:
1. Get the broader context.
2. Read what was just done.
3. **Be skeptical of the conclusions, especially mine.** Look for what I missed, what I'm overweighting, what I might have misframed.
4. Propose next steps that respect both the broader context and the new data.

A parallel CPU analysis run is happening right now (see §B at end). Use those results when they land.

---

## 0. Mandatory reading order (do all before proposing anything)

1. **`CLAUDE.md`** in repo root — operational rules (US region preference, capacity rules, etc.)
2. **`docs/relaunch_handoffs/AGENT_GUIDE_2026-05-02.md`** — was written specifically because a previous agent proposed a packet that had already been refuted twice. Validate-before-suggest, single-lever discipline, CPU-first-then-GPU rule.
3. **`docs/relaunch_handoffs/PSERIES_FACTS_2026-05-02.md`** — pure data ledger from the broader R13 chain (P14-P22, S1/S2/S3). No interpretation.
4. **`docs/relaunch_handoffs/PSERIES_OPINIONS_2026-05-02.md`** — interpretations and framings (read AFTER FACTS; the disclaimer at top is load-bearing).
5. **Memory** at `/Users/roeedar/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/MEMORY.md` — accumulated context across many sessions. Of particular relevance:
   - `project_viso_ceiling_unbroken_10_packets.md` — the 27% viso ceiling story (now updated to 13+ packets)
   - `project_e2b_breaks_deeplive_ceiling.md` — new today
   - `project_l14_does_not_break_viso_ceiling.md` — new today
   - `project_eval_production_crop_tightness_gap.md` — likely upstream cause
   - `project_image_quality_shortcut.md` — likely upstream cause
   - `project_face_size_label_leak.md` — likely upstream cause
   - `project_p8a_breakthrough.md` and `project_p8a_frame_level_auc_2026-04-29.md` — what makes P8A unique
   - `project_data_axis_lever_pulled_twice_no_lift.md` — data-axis hypothesis already refuted twice
   - `project_train_auc_not_valid_promotion_signal.md` — important methodological note
6. **Recent verdict docs** for the 3 E-packets:
   - `docs/relaunch_handoffs/E_PACKET_VERDICT_2026-05-03.md` (E1)
   - `docs/relaunch_handoffs/E2B_FINAL_VERDICT_2026-05-04.md` (E2b)
   - `docs/relaunch_handoffs/E3_L14_FINAL_VERDICT_2026-05-04.md` (E3 — L14)
   - `docs/relaunch_handoffs/MORNING_BRIEFING_2026-05-04.md` (consolidated)

## 1. The deployment context (don't lose sight of this)

Roee is building a deepfake detector for Microsoft Teams deployment. Per `project_success_criteria.md`, three pillars:
1. **Fake recall on target methods** (visomaster, deeplive, teams_capture)
2. **FPR < 5%** in production
3. **Robustness across lighting/camera/codec/color**

All three are load-bearing. Optimizing one at the cost of others is not the goal.

## 2. What just happened (24h sprint)

User authorized $200 today (2026-05-03) for a 3-packet portfolio explicitly attacking 3 different hypotheses about why viso recall has been stuck at 27% for 13+ packets. The packets:

- **E1** (B16 FT-from-P8A + heavier eval-targeted aug): tested whether train→eval aug distribution gap was the bottleneck.
- **E2b** (B16 SCRATCH + CrossEntropy + heavy aug): tested whether P8A's accumulated FT-chain ossified viso bias. Originally E2 with ArcFace, which collapsed; rewrote with CE per the original Effort paper.
- **E3** (L14 SCRATCH + CrossEntropy + heavy aug): tested whether encoder capacity was the bottleneck (Effort paper uses L14, we've been on B16).

### Headline numbers (calibrated FPR=10% on dev real, contract scorecard)

```
ckpt                       viso     deeplive   teams_fake
P8A_REFERENCE_STEP5000     27.1%    42.9%      64.1%       (B16, FT chain — baseline)
E1_TOP_N_STEP400            4.7%    25.9%      54.8%       (B16 FT + aug)
E2B_TOP_N_STEP3200          7.1%    87.5%      72.0%       (B16 scratch + CE + aug)
E3_TOP_N_STEP6600          11.6%    85.5%      72.7%       (L14 scratch + CE + aug)
```

### The previous agent's conclusions (CRITICALLY EVALUATE)

I (the previous agent) wrote in `MORNING_BRIEFING_2026-05-04.md`:
1. "Viso ceiling is STRUCTURAL" (3 arch-distinct packets all fail viso)
2. "L14 is not worth it" (4× cost, ~same operational result)
3. "Deeplive ceiling SHATTERED" (43% → 88% at FPR=10%)
4. Recommended Path A: ship P8A + E2B_3200 max-rule ensemble today

**Roee has already pushed back on the ensemble path** — they want a real model improvement, not a deployment hack. So Path A is **off the table** for this discussion.

## 3. Things I (the previous agent) want you to be skeptical of

### "Viso ceiling is structural"
- **Counter-evidence to consider**: my conclusion is based on 3 packets that all stack the same aug curriculum (heavy blur + brightness + JPEG). Is the binding constraint actually "this aug curriculum hurts viso," not "training architecture is irrelevant for viso"?
- **Alternative interpretation**: maybe NO aug + scratch on B16 (or L14) would lift viso. We never tested that. The aug was added because S1/S2/S3 + earlier packets suggested aug helps; but maybe for viso specifically, aug is the lever in the *wrong* direction.
- **Test that would change my mind**: a packet identical to E2b but with NO aug at all. If viso > 27% there, my "structural" conclusion is wrong. (Caveat: E2 with ArcFace + no aug collapsed; need to be careful what to ablate.)

### "Deeplive ceiling shattered to 88%"
- **At calibrated FPR=10% on dev real.** The lockbox numbers in `/tmp/e2b_scorecard/checkpoint_summary.csv` show E2B_STEP3200 has lockbox_fake_recall 32.4% at calibrated FPR=2% — much lower than the 88% headline. The deeplive lift may not transfer to lockbox.
- **The contract picked E2B_STEP6000 as winner, not STEP3200.** Why? STEP3200 has slightly higher dev_worst_real_stress_fpr (3.0% vs P8A 1.6%) and higher lockbox_real_FPR (0.73% vs P8A 0.15%). Real-side cost may be unacceptable.
- **Test that would change my mind**: per-method recall breakdown — is the deeplive lift broadly across all deeplive methods or concentrated on a few?

### "L14 is not worth it"
- **Counter-evidence**: I tested only 3 L14 ckpts (top_n_step4800/6600/7200). I never tested an L14 trained from scratch with NO aug or with DIFFERENT aug. The "L14 doesn't help" conclusion is conditional on "L14 + same aug as E2b."
- **Alternative interpretation**: L14 has a much bigger capacity budget that may be wasted by the heavy aug forcing it into the same compressed boundary as B16.
- **Test that would change my mind**: an L14 with different optimization (e.g., longer training, no aug curriculum, the original Effort paper's exact training recipe).

### "27% viso is the ceiling"
- **Counter-evidence**: P8A's 27% was the joint dev+lockbox FPR=10% number. The contract winner across packets has seen 32% in some configurations under FPR=10% calibrated on dev only. The "27% ceiling" depends on which calibration you use.
- **Alternative interpretation**: "ceiling" is a rhetorical frame; actual best-attainable viso recall under deployment-realistic constraints might be much lower or higher depending on how you measure.

### "P8A's viso strength is from its FT chain"
- **Memory** (`project_p8a_frame_level_auc_2026-04-29`) says P8A frame-level viso AUC is 0.75. That's NOT spectacular — there's plenty of room for separability gain.
- **The 27% viso recall ceiling** at FPR=10% reflects threshold calibration constraints, not maximum-attainable separability. A different threshold strategy or per-suite calibration might unlock more.

## 4. Hypotheses I've NOT tested that could change the picture

These are real candidate next-packet directions. I list them without ranking — your job to think about which is best:

### H1: Aug curriculum is hurting viso, not helping
A B16 SCRATCH + CrossEntropy + NO aug packet would isolate this. If viso > 27% there, my "structural" conclusion is wrong.

### H2: Eval substrate / production crops are the real binding constraint
Per `project_eval_production_crop_tightness_gap`, eval frames have looser face-tightness than production. We've never re-cropped eval frames to match production, then re-evaluated. Cheap diagnostic, possibly upstream of everything.

### H3: P8A's FT chain itself is the lever — replicate its exact recipe with new data
If P8A's viso strength is from its trajectory (R12g→RLP6_04→RLP7_02→P8A), then replicating that trajectory with current code (post-bug-fixes) might give us a P8A_2 with different operational characteristics.

### H4: Per-method specialization
Train viso-only model + deeplive-only model + teams-only model, route at inference. Avoids the per-suite trade-off limit.

### H5: Look at WHY deeplive lifted from E2b/E3 (mechanism analysis)
Whatever caused the 43→88% jump on deeplive may also be the lever for viso if applied differently. We don't yet know what mechanism breaks deeplive — could be:
- The aug curriculum specifically helping deeplive (sharpness range matches deeplive artifacts)
- The CE loss vs ArcFace's identity-encoding pressure
- The scratch initialization vs FT-from-P8A
- Some interaction we haven't isolated

H5 is purely diagnostic, no GPU spend, and might inform H1-H4.

### H6: Re-FT P8A with anti-shortcut aug that PRESERVES the viso trajectory
Don't go scratch — start from P8A and add aug that targets the *shortcut* features (sharpness, face size) without overwriting the viso-specific features. This was attempted in P14 etc., but only with a stacked anti-shortcut bundle. A single-lever anti-shortcut aug from P8A has not been tested.

### H7: Fix the wandb-flatten bug confounding (`project_wandb_flattens_nested_dicts`)
We have a known bug where wandb.init flattens nested-dict yaml blocks. Some prior packets may have run with degraded configs — but the recent packets (P22, S1-S3, E1-E3) should be fine since the workaround is in place. Worth verifying for your own packets.

## 5. Things to verify before proposing

Per `AGENT_GUIDE_2026-05-02.md` and memory `reference_agent_guide_2026_05_02.md`:
- **Validate-before-suggest**: check git log + `analysis/` outputs for whether your proposed lever has already been tried.
- **CPU-first-then-GPU**: a lot of "novel ideas" can be killed in 30 min of CPU work.
- **Single-lever discipline**: don't propose stacked bundles. Per `project_face_scale_jitter_load_bearing`, the bundle was net-negative when the single lever (jitter@0.50) was net-positive at 5.7× the gap.
- **Don't repeat refuted axes**: data-axis has been pulled 3 times, not lifting viso (`project_data_axis_lever_pulled_twice_no_lift`). aug curriculum has been pulled many times.

## 6. The viewer is already loaded with relevant data

`viewer/model_dashboard_runs.yaml` has run entries for P8A and other ckpts. The previous agent should have added entries for E1/E2b/E3 — check that. Roee can browse manifolds / per-frame scores / score distributions there. Use the viewer artifacts as evidence in your discussions.

## 7. What's currently NOT happening

- No active Vertex jobs (all complete or cancelled).
- Budget remaining ~$95 of $200 authorized for today.
- No packets queued. The next packet decision is what Roee wants to discuss with you.

## 8. Cost-aware framing

The R13 chain has spent considerable money. Patterns to avoid (from memory):
- Stacked bundles that net-cancel (`project_face_scale_jitter_load_bearing`)
- Re-running refuted axes (`project_data_axis_lever_pulled_twice_no_lift`)
- Trusting val_AUC over operational scorecard (`project_train_auc_not_valid_promotion_signal`)

Your proposals should articulate: *what's the SPECIFIC structural difference between this packet and what's been tried?*

## 9. Tone for the discussion

Roee wants intellectually honest pushback. They've corrected previous agents who overcommitted to interpretations. If you think:
- "I should run more diagnostics before proposing" — say that.
- "The previous agent missed X" — say that.
- "I'm not sure" — say that.

Don't manufacture certainty.

---

## §B. CPU analyses already done (read these before you discuss)

A comprehensive CPU diagnostic pass landed at `analysis/cpu_followups_2026-05-04/INDEX.html`
(viewable in the dashboard via the E2B or E3 run entries). 11 distinct analyses + 29 figures.
Two findings from the pass invalidate or strongly reframe my prior conclusions:

### Finding 1 (CRITICAL REFRAMING): The "27% viso ceiling" is FPR=10%-specific
Threshold-relaxation curves (`outputs/11_threshold_relaxation_curves.csv`) show:

| FPR target | P8A viso | E2B viso | E3 viso |
|-----------:|---------:|---------:|--------:|
|     0.05   |    5.8%  |    4.9%  |    7.8% |
|     0.10   |   26.9%  |    8.4%  |   13.8% |
|     0.20   |   57.6%  |   21.3%  |   38.5% |
|     0.30   |   68.9%  |   39.8%  |  **87.5%** |
|     0.50   |   85.8%  |   90.0%  |   91.1% |

So:
- The "27% ceiling" was a peculiarity of the FPR=10% target — the data has 86-91% intrinsic viso recall available.
- **At production FPR=5%, ALL ckpts give 5-8% viso recall.** That's the actual deployment blocker.
- **E3 (L14) at FPR=30% reaches 87.5% viso recall** — highest of any single ckpt at any threshold. L14 has the most viso headroom but a wider score distribution.

This reframes the next-steps question. The binding constraint isn't "lift viso ceiling at FPR=10%" — it's "lift viso recall at FPR=5%."

### Finding 2: Frame-coverage shows P8A and scratch ckpts use DIFFERENT signals
At FPR=10% on viso (550 fakes total):
- **66.2% of viso fakes (364/550) are caught by NO ckpt**. The viso recall ceiling is partly DATA-bounded (those frames are below threshold for all training axes).
- Of the 186 caught viso fakes, P8A uniquely catches 93 (50% of the catchable subset). The scratch ckpts contribute differently.

On deeplive: only 1/545 frames is uncaught by all 3 → deeplive ceiling is essentially fully reachable, just calibration-bound.

Per-frame Pearson r between ckpts on viso fakes: P8A↔E2B 0.30, P8A↔E3 0.41, E2B↔E3 0.59. **Scratch ckpts use mostly DIFFERENT signal from P8A on viso, and somewhat different from each other.** The previous claim "P8A's viso strength is unique" is supported.

### Other CPU findings worth reading
- Cross-suite AUCs (`outputs/02b_cross_suite_auc.csv`): P8A viso AUC 0.753 highest of 3; deeplive AUC E2B 0.965 (>P8A 0.861, +10pp).
- Score correlations across all suites (`outputs/04_per_frame_score_correlation.csv`).
- Per-identity FPR concentration (`outputs/07_per_identity_fpr.csv`).
- Method champion table (`outputs/05_method_champion.csv`) — for each (suite, method, FPR floor), which ckpt wins.
- Per-video disagreement table (`outputs/06_per_video_disagreements.csv`) — top 200 most-divergent videos for failure-case browsing.
- Score distribution histograms, ROC curves, recall-vs-FPR plots, per-pair scatter plots.

### Viewer integration
Three new entries in `viewer/model_dashboard_runs.yaml`: `e2b_3200`, `e3_6600`, `viso_uncaught`. All
link to the INDEX.html via the score_distribution_report artifact. The Frame Browser shows their
per-frame scores alongside P8A's.

### Cluster statistical analysis (added later 2026-05-04)
After Roee asked "what explains the score-space clusters?", a statistical pass joined the 550 viso
fake frames with `crop_attributes.csv` (image-level features). Output:
`outputs/14a-14e*.csv` covering per-cluster feature stats, caught-vs-uncaught Welch t-tests +
Cohen's d, L1 logistic regression coefficients, pairwise cluster comparisons, and viso subtype
breakdowns. Full data + methodology in `analysis/cpu_followups_2026-05-04/FINDINGS_FACTS.md` —
read that BEFORE proposing anything that depends on the cluster structure.

---

## TL;DR for the next agent

Roee wants to discuss what to do next. The previous agent's conclusions are: viso ceiling is structural, L14 isn't worth it, deeplive shattered. **Be skeptical of all of these.** Read the broader context, look at the new CPU analysis when it lands, and propose what to do with intellectual honesty about what you don't know.
