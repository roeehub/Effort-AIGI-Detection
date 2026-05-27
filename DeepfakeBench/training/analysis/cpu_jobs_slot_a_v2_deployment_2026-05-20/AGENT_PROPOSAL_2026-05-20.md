# AGENT_PROPOSAL — Slot A v2 deployment decision, post-CPU-batch (2026-05-20)

> Single OPINIONS doc for the 2026-05-20 7-job CPU batch.
>
> Read AFTER `RESULTS_FACTS_2026-05-20.md` and the per-job FACTS docs. Per `AGENTS.md` "Reading order for forming an independent view": form your own Pass-1 view from the FACTS first; this doc is the Pass-2 opinion layer.

## 1. The mechanism claim (revised)

The 2026-05-19 user-prompt framed the deployment decision as three Moves (1-3) plus an Image question. The 7-job CPU batch covers Move 1 + Move 3 + the Image question and produces a more nuanced picture than the original chat could see:

> **Slot A v2 step3500's anchor_aware mechanism does two things at once:**
> 1. It *succeeds at suppressing chronic-identity over-fires* on 4 of 8 chronic identities — Chikara_Takahashi, PC_Generator (dev + lockbox), Q, dor_shkedi all see their anchor pool collapse toward clean (Job D shows the per-identity FPR drops). This is the mechanism working as designed.
> 2. It *does NOT reduce per-frame transport-shortcut sensitivity* — Slot A v2 inherits T5C's Teams-account-encoded transport-shortcut at the per-frame level (Job A: Δ = −0.168, identical to T5C; Job G: G_channel swing 0.698 ≈ T5C 0.678; saturation amplified 4.93×). The anchor mechanism only shifts the score distribution vertically by ~3pp, not horizontally on the transport axis.
>
> These are two orthogonal failure modes. The chronic-identity mechanism is one; the transport-encoded pipeline-signature mechanism is another. **Anchor_aware addresses (1) and not (2).**

## 2. The contract tiebreak is dispositively inside sampling noise

This is the highest-confidence claim from this batch:

- Job B (paired per-video bootstrap, n_resamples=10,000): 95% CI on Δ(SlotAv2 − P8A) on `lockbox_real_fpr` = **[−0.00808, +0.00955]**. The CI is ~25× wider than the observed Δ of 0.000735.
- P(Slot A v2 truly has higher FPR) = **0.519** — a coin flip.
- The observed Δ = exactly 1 video out of 1361 lockbox-real videos.
- Job C confirms: at any composite policy (λ ≥ 5) or Pareto-dominance ranking, Slot A v2 step3500 ranks 1. Only lex-on-FPR-ASC (the contract's current tiebreak) puts P8A at rank 1, and only by virtue of that 1-video gap.

**This was the open loop** `lockbox-real-fpr-tiebreak-is-load-bearing` from the RESCHAIN_GRL6 retro (2026-05-16). It has been open for 4 days. The CPU batch closes it: **the tiebreak is NOT load-bearing**. It is comparing values inside sampling noise. The current contract's verdict that "P8A wins by 0.07pp" is not statistically distinguishable from "P8A and Slot A v2 are tied on lockbox FPR."

## 3. The Roy_D regression is real, large, and dev-only

Job D reproduces the 2026-05-16 finding mechanically on the 29-suite contract substrate: Roy_D dev FPR moves from P8A 0.30 → Slot A v2 **0.84** (+0.54 absolute). This is the largest single per-identity regression in the table, much larger than any of the per-identity *fixes* (Chikara_Takahashi −0.36, PC_Generator_lockbox −0.29, Q −0.64 — though Q's fix is largest in absolute terms among fixes).

**Critical caveat**: Roy_D appears in `teams_real_all_dev` (130 videos) and `teams_real_lighting_extreme_dev` (113 videos) but **NOT in the contract's lockbox real cohort `teams_real_all_lockbox`**. The +0.54 dev regression therefore does not propagate to the contract's `lockbox_real_fpr` (which stays at 0.019).

This creates a deployment-surface-vs-eval-surface mismatch. The contract scorer says "Slot A v2 is rank-2 with +30pp lockbox_fake_recall"; the per-identity audit says "Slot A v2 false-flags Roy_D 84% of the time on dev." The contract's tiebreak metric is `lockbox_real_fpr` — Roy_D's regression is invisible to it. **If Roy_D is a real production identity (or representative of a Roy_D-cluster of production identities), this is a deployment blocker that the contract does not surface.**

## 4. The Teams-account transport-shortcut is unaddressed and is the load-bearing production problem

Job A and Job G together establish:

- Slot A v2 step3500 on the same Roy_D / Guest crops as 2026-05-19: Roy_D 0.767 / Guest 0.599, Δ = −0.168 (T5C: 0.795 / 0.628, Δ = −0.168). The vertical shift is −0.028 / −0.029 — the anchor mechanism shifted the absolute level but not the transport-axis Δ.
- Flip window WIDENED: T5C flips at τ ∈ [0.65, 0.70); Slot A v2 flips at τ ∈ [0.60, 0.70). At a deployment τ around 0.65, Slot A v2 produces opposite verdicts on the same person at the same moment in two different Teams accounts.
- Per-axis perturbation: G_channel scale swing is 0.698 (T5C: 0.678). Saturation swing is 0.148 (T5C: 0.030) — **anchor_aware made the model 5× more sensitive to HSV-saturation perturbations on the Roy_D crop**.

The Teams-account natural experiment is the only piece of evidence in the program that captures a same-person, same-camera, same-moment flip in deployment-realistic conditions. **The 2026-05-19 follow-ups already nailed the mechanism** (G-channel scale alone explains most of the swing; ArcFace cosine 0.92 between accounts shows the identity embedding itself is transport-shifted). Slot A v2 does not address this mechanism; it inherits T5C's failure pattern fully.

The implication: **the most operationally-relevant production failure mode is unaddressed by anchor_aware.** Slot A v2 is a strict improvement on chronic-identity FP suppression at the contract surface, but the natural experiment shows real users flipping verdicts based on Teams account encoding — and that's the kind of failure a deployed model will see most frequently.

## 5. Slot A v2 keeps production-drift robustness on may6 (Job F)

The most directly production-relevant data point in this batch: Slot A v2 fires 4/92 frames on the may6 cohort, vs P8A's 0/92 and E2B's 53/92. The harness sanity (P8A re-score 0/92, max |Δ| = 7.25e-06 vs prior) confirms the comparison is apples-to-apples.

4/92 vs 0/92 is a small absolute regression but well inside the band P8A and Slot A v2 share as "low-FP-on-production-drift" ckpts. Bar 4 (Score-collapse p50 ≤ 0.05) was NOT MET (0.097 > 0.05), but the magnitude is small enough that operationally the difference is unlikely to matter.

This is the cleanest evidence that Slot A v2 can ship without creating a P22-style production false-flag spike (P22 was the 2026-05-02 packet that improved train AUC then exploded operational FPR — Slot A v2 does not show that pattern).

## 6. So what should ship?

The deployment decision rests on three judgment calls that this batch does not answer:

### 6.1 Does Roy_D represent a real production failure mode?

Roy_D is one identity. The Roy_D regression is +0.54 dev FPR. If Roy_D is a synthetic test substrate (he's not — Roy_D is the user himself), or if his FP pattern doesn't generalize to other production users, then the +30pp lockbox_fake_recall lift dominates. If Roy_D is a representative example of "production users whose physical profile sits in the same encoder region" (PC1 ≈ −3.5 in the OpenCLIP embedding), then Slot A v2 systematically breaks for that user class.

**Recommendation**: capture 20-50 frames from each of: Roy_D, 2-3 visually-similar production users (similar gender / age / glasses / lighting), 2-3 visually-different production users. Score with P8A and Slot A v2. If the Roy_D regression generalizes to similar users → don't ship Slot A v2. If it doesn't generalize → ship.

### 6.2 Is the Teams-account flip at τ=0.65 operationally tolerable?

In a Teams call, a user using one account sees themselves as real; the same user joining as a guest sees themselves flagged as fake. This is a brand-bad failure mode visible to actual users, regardless of detection accuracy on aggregate cohorts.

**Recommendation**: define an operational SLA. If the SLA is "no within-session flip on same-physical-source" → don't ship Slot A v2 (or T5C, or Slot B, or any current FT'd ckpt). If the SLA is "aggregate FPR ≤ X" → the natural experiment is one data point among many and the +30pp lockbox lift may dominate.

### 6.3 What's the FP/FN cost ratio in deployment?

Job C shows that under any composite tiebreak with λ ≥ 5 (1 percentage point of FPR cost = 5 percentage points of recall benefit), Slot A v2 ranks 1. For a deepfake-detection-in-Teams use case, λ ≥ 10 is plausible — false flags annoy users; missed fakes are security incidents.

**Recommendation**: write down the operational λ explicitly. If λ ≥ 5, the contract should be amended (composite tiebreak) and Slot A v2 ships. If the policy is "minimize FPR at any cost" (λ → ∞), P8A is the right ship.

## 7. What this means for the chat's Moves 1-3

Mapping back to the original chat's framing:

- **Move 1 (evidence-based deployment for Slot A v2)**: The batch produces evidence both ways. The contract tiebreak is inside noise (ship Slot A v2 if you accept the deployment metric is the +30pp recall lift, not the 1-video FPR delta). The Roy_D regression and Teams-transport shortcut are unaddressed (don't ship without resolving the operational questions in §6). **The honest read: this is not "ship or don't ship" — it's "ship is defensible if and only if §6.1 and §6.2 are resolved in Slot A v2's favor."**

- **Move 2 (data ingestion)**: Job D and Job G both reinforce the data-ingestion case. The Roy_D regression cannot be fixed by re-tuning the anchor pool weight (anchor_aware is content-bounded, see memory `project_band_shortcut_ood_hypothesis_2026-05-16`). The transport-shortcut cannot be fixed by encoder-level loss-based interventions on existing training data (Slot α `resolution_chain_aug` already empirically failed at this on the contract surface). Bringing in direct_teams reals from multiple account configurations is the structurally distinct lever — and the Cheap_Followups §2 finding (blur σ=2.5 + scale 0.80 reproduces Guest's score within 0.02) suggests a synthetic-augmentation variant might also work, but only if AUC is preserved (open loop `cpu-probe-mechanism-discrimination`).

- **Move 3 (canary + LoRA-load fixes)**: All 16 tests pass. Both fixes are ready for commit. The chat's intuition that "this might have been fixed already" was correct. **Recommend**: commit both fixes in a single PR with the 3 test files, close open loops `canary-silence-when-multi-axis-grl-active` and `lora-enabled-not-propagated-by-load-model`.

## 8. Direction proposal for the next packet (if there is one)

Given the picture, the next packet should be **structurally distinct from another GRL flavor or another LoRA layer placement**. Ranked by EV:

1. **The multi-account capture sweep** (real-world capture task, not a packet). User collects 20-50 frames per Teams account configuration. CPU batch then quantifies the per-account `prob_fake` distribution shift. Closes the question "is the 2026-05-19 natural experiment a generalizable failure mode or a 2-frame artifact?" Cost: minutes of user time + ~30 min CPU. If the per-account shift is robust → the Teams-transport augmentation lever (or direct ingestion) is the next packet.

2. **Roy_D anchor pool extension** (open loop `roy-d-specific-anchor-pool-packet`). Add Roy_D + bla_bla_chow real frames to the anchor pool, re-run a Slot A v3 packet. If Roy_D dev FPR drops below 30% → mechanism generalizes by content addition. If not → Roy_D's encoder region needs a different intervention class entirely. Cost: 4-6 hours GPU, $30-50.

3. **Tiebreak amendment in the contract** ($0 yaml change, reversible). Replace lex-on-FPR-ASC with composite `lockbox_fake_recall − 10 × lockbox_real_fpr` (or another defensible composite). Re-run the existing 2026-05-20 scorecard offline (no GPU needed; the scorecard CSV has all the data). If Slot A v2 ranks 1, it's a contract-internal decision — ship.

4. **Teams-transport synthetic augmentation** (medium GPU packet, $40-60). Take the Cheap_Followups §2 finding (blur σ ≈ 2.5 + per-channel scale ≈ 0.80) as a training-time augmentation on real lockbox-style frames. CRITICAL: gate by AUC-preserving CPU probe before launching, per the `cpu-probe-mechanism-discrimination` open loop, to avoid the Slot α score-compression trap.

5. **Per-substrate τ-calibration at deployment** ($0, no training). Memory `project_job7_head_retrain_REFUTED_2026-05-04` notes "21pp lockbox recall lift available on P8A via per-substrate τ-calibration alone, no retraining." Worth trying on Slot A v2 before any GPU packet.

## 9. Self-correction log

This batch's framings that should be flagged:

1. **Initial framing in original chat (this agent, 2026-05-20)**: "Slot A v2 may inherit T5C's transport-shortcut, but the chronic-identity mechanism is decided." → REVISED. The chronic-identity mechanism is *partially* decided (4 of 8 identities fixed); Roy_D regression is large and dev-only; bla_bla_chow regression appeared. The "decided" framing overstated the mechanism's scope.
2. **Initial framing in original chat**: "Slot A v2 is the operative single-checkpoint candidate." → MAINTAINED (no other ckpt ranks higher on the contract surface), but qualified: the contract surface itself is shown to be noise-limited at the tiebreak.
3. **Job C lex-thresholded policy code**: had a Python banker's-rounding bug that made the script under-count Slot A v2 wins. Corrected by hand in the FACTS doc § 4.1. The corrected verdict is that Slot A v2 wins under 6/11 policies (was reported as 5/11 raw).
4. **Job E panel composition**: the initial script's calibrated-τ summary tried to compute `teams_real_all_lockbox` FPR but the canary panel doesn't contain that suite — the 800-frame panel has `proper_real_clean_lockbox` (HDTF-style cleans). The post-processing fixed the join; the chronic-heavy panel composition is now flagged explicitly in Job E's FACTS doc.
5. **Per-job sub-agent OPINIONS docs**: Job B has a sub-agent-authored OPINIONS doc that may differ in framing from this consolidated doc. Where the two differ, this consolidated AGENT_PROPOSAL is the operative read; per-job opinions are inputs.

## 10. Cross-references

- 29-suite scorecard: `analysis/manual_canary_2026-05-20/scorecard_pull/`
- 2026-05-19 natural experiment: `analysis/teams_account_natural_experiment_2026-05-19/`
- 2026-05-16 auto-mode verdict (Slot A v2 origin): `analysis/auto_mode_2026-05-16_eval/`
- Open loops touched: `lockbox-real-fpr-tiebreak-is-load-bearing` (RESOLVABLE per §2), `canary-silence-when-multi-axis-grl-active` (RESOLVABLE per Move 3), `lora-enabled-not-propagated-by-load-model` (RESOLVABLE per Move 3), `roy-d-specific-anchor-pool-packet` (still open, see §8.2)
- Threads: `processing_signature_shortcut`, `iq_shortcut_deconvolution_program_2026-05-08`, `in_training_canary_signal`, `wandb_yaml_propagation_bugs`
