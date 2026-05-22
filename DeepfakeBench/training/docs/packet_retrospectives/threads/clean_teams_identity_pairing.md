# Thread: Clean and Teams buckets share identities (paired transport)

> **⚠ Critical-reading note (added 2026-05-04 night)**: This thread contains a structural data-layout finding (same-identity, two-transport) that is robust and well-cited, AND an inference layer about contrastive / pair-aware losses being "well-targeted" that was a-priori reasoning, not empirically tested before being elevated to the "Current stance." On 2026-05-04 the inference layer was empirically tested and found refuted on E2B for viso. See the DEBATE block (2026-05-04) below the Current stance and the corresponding new low-severity loop. Read the structural finding (data layout) as fact; read any inference about loss-design implications as conditional on disconfirmation work that has not been done for variants other than symmetric KL on E2B+viso.

> **Slice 7 finding (2026-04-29)**: The repository's data layout is *same-identity, two-transport*. Buckets named `*_teams` (or `_teams_v2`, `companion_bucket=teams_v2_companion`) hold frames of the same source identities (and typically the same swap/method) as their paired clean buckets — recaptured through OBS virtual cam → Teams meeting → recording. Joins are explicit at the manifest level. Memory `project_clean_teams_same_identity.md` is the auto-memory anchor.

## The question

When the wiki's bucket-gap finding (Slice 7, [`viso_bucket_gap`](viso_bucket_gap.md)) named the unused `live-...-teams-v2` and `hdtf_visomaster_*_teams` buckets as the substrate for a Layer-2 retrain, what is the structural relationship between those Teams-bucket identities and the identities already in the (cleaner) training buckets? Are these new people we have not collected before, or the same people recaptured through a different pipeline? The answer determines what kind of contrastive / pair-based losses are available for any anti-shortcut design, and why the bucket-fix is so cheap.

## Initial belief

Through Slices 4-6 the team understood that some buckets ended in `_teams` and treated them as "Teams-recaptured deployment-mode data" without explicitly auditing whether the Teams-bucket identities were **the same identities** as their clean counterparts or a fresh capture set. The implicit assumption skewed toward "different captures, possibly different identities, similar pipelines." The proper-data wave (`arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml`, written 2026-04-19) had been authored with paired identity intent but the explicit join was not surfaced in any handoff that anchored a packet's experimental scope.

## What changed our mind

- **2026-04-29 ~10:00 CEST — bucket structural audit (memory `project_clean_teams_same_identity.md`).** The audit was a follow-on to the [`viso_bucket_gap`](viso_bucket_gap.md) Slice-7 finding: once the eval suite was identified as the Teams-recapture bucket, the question "do we have any identities in that pipeline that we could put in training" became load-bearing. Concrete in-repo evidence that the structural relationship is *same-identity, two-transport*:
  - `arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml` — every `base_capture_id` has variants with `transport: clean` AND `transport: teams` under the same `identity_id`.
  - `arena/reports/hdtf_visomaster_clean_vs_teams_join_2026-04-19.json` — explicit join report for the HDTF source.
  - `arena/reports/quickclips_visomaster_clean_vs_teams_join_2026-04-19.json` — explicit join report for the quickclips source.
  - `data/sources/visomaster.py:748,1085,1105,1176,1523` — `companion_bucket` field with `teams_v2_companion` / `clean_companion_only` status values; `_load_merged_teams_enhanced_paired` function exists.
  - `.viewer_cache/discovery/visomaster_teams_enhanced.json` — every sample has `has_teams_counterpart: True`.
  - `combined_paired.py:3893-3941` — `teams_bucket` and `hint_teams_bucket` separation already plumbed at code level.

- **The bucket pairs in current understanding** (memory `project_clean_teams_same_identity.md`):

  | Clean bucket | Teams-transport bucket | Same identities? |
  |---|---|---|
  | `live-deepfake-methods-real-and-fake-frames-cropped` | `live-deepfake-methods-real-and-fake-frames-cropped-teams` (deeplive_teams) | yes |
  | `live-deepfake-methods-real-and-fake-frames-cropped` | `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2` (visomaster_teams_v2_companion) | yes |
  | `hdtf_visomaster_cropped_frames` | `hdtf_visomaster_cropped_frames_teams` | yes (proper-data wave) |
  | `visomaster-enhanced-face-cropped-v2` | `teams-faces-data-test-2914-fake-4420-real-feb-28` (eval bucket) | likely yes (needs join confirmation) |

- **The eval bucket fits the same pattern.** The eval suite `visomaster_enhanced_macro_dev` reads from `gs://teams-faces-data-test-2914-fake-4420-real-feb-28/fake/` (Slice 7 [`viso_bucket_gap`](viso_bucket_gap.md)). The structural reading of the bucket name (`feb-28` recapture context, `2914-fake-4420-real` count signature) is consistent with the same-identity-two-transport pattern: these are identity-paired captures of identities that exist in clean buckets, recorded through Teams. The only outstanding work is to confirm the join via a manifest-side audit — but the structural prior is high enough that the bucket-fix retrain (`R13_P14_DATA_FIX.yaml`) does not block on it.

- **The corollary for anti-shortcut design (a-priori reasoning, partially refuted 2026-05-04 — read with the DEBATE block below).** Same-identity-different-transport pairs *could in principle* serve as a ground-truth signal for "what should look the same vs different to a deepfake detector": a real frame of identity X via clean transport and a real frame of identity X via Teams transport should both score real; the shortcut by definition is when the model uses the transport-pipeline signal (codec, ISP, color space) instead of the manipulation-content signal. The structural argument — that contrastive losses pulling `(clean[id_X], teams[id_X])` toward each other target the shortcut class in a way augmentation cannot reach (because augmentation cannot synthesize a real Teams-transport version of an existing clean frame; only collecting a recapture can) — was the a-priori reasoning that motivated the Move-4 framing in `docs/relaunch_handoffs/TRAINING_EVAL_VISO_BUCKET_GAP_2026-04-29.md` § 6. See `combined_paired.py:3893-3941` for the existing plumbing.

  **2026-05-04 update**: the symmetric-KL pair-loss instantiation of this argument was empirically tested on E2B for viso (the proposal-as-scoped) and the cohort math went the wrong direction (3 target / 29 wrong-way at τ=0.5). The structural pairing remains; the leap from "pairing exists" to "symmetric pair loss is the right intervention" did not survive empirical check. Asymmetric variants and other model bases remain untested. See DEBATE block below.

## Current stance (2026-04-29, with 2026-05-04 nuancing — see DEBATE below)

**Structural finding (robust, not contested)**: the training data is structured as **same-identity, two-transport**, and the wiring to use that structure is largely already in place (`companion_bucket`, `_load_merged_teams_enhanced_paired`, explicit join reports). The unused buckets named in [`identity_audit`](identity_audit.md) (`live-...-teams-v2`, `hdtf_visomaster_*_teams`, the proper-data wave) are not "more people" — they are *more transports of the existing people*. This makes the [`viso_bucket_gap`](viso_bucket_gap.md) Layer-2 closure structurally tractable (we don't need new collection, we have the eval-pipeline transport already).

**Inference layer (a-priori, partially refuted 2026-05-04)**: the original Slice-7 framing extended the structural finding to claim it "sets up an anti-shortcut design lever that augmentation alone cannot reach" (Move 4 in `docs/relaunch_handoffs/TRAINING_EVAL_VISO_BUCKET_GAP_2026-04-29.md` § 6). The 2026-05-04 verification of the symmetric-KL pair-loss instantiation of that lever (DEBATE block below) found the proposal-as-scoped is structurally net-negative on E2B for viso. The lever space "design intervention that uses paired data" remains open; the specific instantiation "symmetric KL pair loss aligning clean toward teams" is empirically refuted on the model + suite tested.

The thread exists to make the structural pairing visible to a future agent who reads "the unused bucket has 997 identities" without realizing those identities are largely already in active training under a different transport. Without this framing, the bucket-fix looks like a data-acquisition story (expensive, slow, requires new captures) when it is actually a wiring story (one yaml stanza + a smoke test). The bucket-fix wiring story remains valid; the additional anti-shortcut-loss-on-top-of-bucket-fix story is the one in active dispute.

## Packet timeline

- *(none — this thread captures a structural property of the data layout, not a packet's results)*. It informs [P14](../packets/P14.md) (the data-fix variant uses the paired buckets directly) and any future P15+ design (the GRL quality-domain head's domain labels could use the transport pairing as a more principled domain partition than the current 4-class mapping).

## Evidence locations

- `arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml` — every `base_capture_id` has clean+teams variants under the same `identity_id`.
- `arena/reports/hdtf_visomaster_clean_vs_teams_join_2026-04-19.json` — explicit join report for HDTF.
- `arena/reports/quickclips_visomaster_clean_vs_teams_join_2026-04-19.json` — explicit join report for quickclips.
- `data/sources/visomaster.py:748,1085,1105,1176,1523` — `companion_bucket` field; `_load_merged_teams_enhanced_paired` function.
- `data/sources/combined_paired.py:3893-3941` — `teams_bucket` and `hint_teams_bucket` plumbing (already shipped, just not exercised by current training yamls).
- `.viewer_cache/discovery/visomaster_teams_enhanced.json` — `has_teams_counterpart: True` field on every sample.
- `docs/relaunch_handoffs/TRAINING_EVAL_VISO_BUCKET_GAP_2026-04-29.md` § 4.1-4.2 — Slice-7 enumeration of the already-built loaders + already-built data not in any training run.
- Memory: `project_clean_teams_same_identity.md` (auto-memory anchor; the structural same-identity-two-transport finding); `project_data_inventory_identity_diversity.md` (the per-bucket identity counts that show the pairing is real at the magnitude that matters); `project_viso_train_eval_bucket_gap.md` (the dispositive bucket-gap finding that this thread structurally enables).

### DEBATE — 2026-05-04 — pair-aware contrastive loss premise empirically refuted on E2B for viso

**Source**: `analysis/pair_loss_effect_verification_2026-05-05/FINDINGS.md` (CPU verification, run 2026-05-04 evening, 275 paired (raw, teams) viso fakes with seq_id matching, P8A frozen features as proxy for E2B features, E2B scores from `analysis/cpu_followups_2026-05-04/raw_reports/visomaster_enhanced_macro_dev_e2b_top_n_step3200_frames_report.csv`).

**Disagreement with Current stance**: this thread's framing (line 34, 38) — *"contrastive losses that pull `(clean[id_X], teams[id_X])` toward each other are fundamentally well-targeted at the shortcut class, in a way that augmentation-only approaches cannot be"* — was a structural / a-priori argument from data layout. The 2026-05-04 verification tested the proposal-as-scoped (symmetric KL pair loss between original and teams-transport on viso fakes, with E2B as the FT base candidate the proposal would target) and found the empirical premise refuted on E2B for viso. The structural same-identity-two-transport finding remains true; the inference that "pair loss is therefore well-targeted" does not hold for E2B+viso in the regime tested.

**Three numerical findings (all directly from E2B scores, no proxy)**:

1. **Sign of effect is REVERSED on E2B**. Mean E2B raw_score = 0.086, mean E2B teams_score = 0.172, Wilcoxon p = 0.0019. Teams transport HELPS E2B catch viso fakes. The thread's framing assumed teams transport hurts E2B (so aligning teams toward raw would close the gap); the data shows the opposite direction.

2. **Failure mode is wholesale, not transport-specific**. Per-pair partition at deployment-style τ=0.5: both caught 7/275 (2.5%); raw caught & teams missed (target cohort) 3/275 (1.1%); teams caught & raw missed (wrong-way cohort) 29/275 (10.5%); both missed 236/275 (85.8%). 86% of viso pairs miss in BOTH versions — pair loss has no fulcrum here (no within-pair asymmetry to align away).

3. **Symmetric pair loss is structurally net-negative across all τ**. The wrong-way cohort outnumbers the target cohort 2-10× at every operating point: τ=0.05 → 26 vs 62 (net −13.1pp), τ=0.10 → 17 vs 55 (net −13.8pp), τ=0.20 → 11 vs 47 (net −13.1pp), τ=0.50 → 3 vs 29 (net −9.5pp).

**Auxiliary measurements (P8A frozen features, proxy)**:
- Q1 paired cosine similarity = 0.878 (paired closer than random within-subtype 0.686, Δ = +0.19, MODERATE band 0.75-0.95).
- Q2 Pearson r(feat distance, |E2B abs score gap|) = −0.089 (p = 0.14, LOW band).
- E2B per-pair score correlation r = 0.62, P8A per-pair r = 0.77 (model already exhibits implicit pair-coherence without explicit pair loss; headroom for explicit alignment to add value is small).

**What this debate changes vs leaves alone**:
- The structural finding (same-identity-two-transport buckets exist, joins are explicit at the manifest level) is **unchanged**.
- The Move-4 pointer to `combined_paired.py:3893-3941` plumbing is **unchanged**.
- What is changed: the inference that contrastive `(clean[id_X], teams[id_X])` losses are "fundamentally well-targeted" for closing the viso recall gap on E2B is empirically **not supported** at the proposal-as-scoped (symmetric KL pair loss). An asymmetric loss (only align teams toward raw when raw scores higher than teams, never the reverse) is a different intervention with a tighter upper bound (~9pp at τ=0.05 perfect-transfer ceiling).

**Caveat on the verification itself**: Q1/Q2 used P8A frozen features as a proxy for E2B features (E2B features not cached on disk). A $5 GPU disconfirmation probe to extract E2B features and re-run Q1/Q2 directly on E2B's geometry would address the proxy assumption — but cannot reverse the sign-of-effect or cohort math (which are computed directly from E2B scores). Recommendation in FINDINGS.md is to skip both the packet and the probe, reallocating budget to substrate cleaning (Job 14 follow-up) and per-substrate τ-calibration (Job 7 follow-up).

**Decision deferred to user**: pair-loss packet was not drafted as a result of this verification.

## Open loops

### Open loop: pair-loss-asymmetric-variant-untested
status: resolved
severity: low
first_seen: 2026-05-04
last_verified: 2026-05-04
close_criterion: either (a) a $5 GPU probe extracts E2B features on the same 550 (raw, teams) viso pairs and re-runs Q1/Q2 directly on E2B geometry, with documented verdict on whether the P8A-as-proxy assumption hides a real signal; OR (b) an asymmetric-pair-loss variant (only aligning teams toward raw when raw scores higher than teams — never the reverse) is scoped, the cohort math is recomputed under that asymmetry, and a go/no-go decision is documented; OR (c) the loop is explicitly closed-as-not-pursued with a one-line note that the symmetric pair loss verdict is dispositive enough to deprioritize the asymmetric variant given the 9pp upper-bound ceiling and the existence of cheaper alternatives (substrate cleaning, per-substrate τ-calibration).

**Resolution (2026-05-04) — closed via close_criterion option (c) per user decision.**

Decision: User explicitly deprioritized pair-loss work on 2026-05-04 ("lay off the asymmetric pair loss for now"). The deferral is "for now" — see reopen triggers below.

Evidence basis (full detail in DEBATE block above; cite `analysis/pair_loss_effect_verification_2026-05-05/FINDINGS.md` if reopening):

1. **Symmetric KL pair loss is empirically refuted on E2B for viso; sign-of-effect REVERSED.** Mean E2B raw_score = 0.086, mean E2B teams_score = 0.172 (Wilcoxon p = 0.0019, n = 275 paired viso fakes). Teams transport HELPS E2B catch viso fakes — the proposal's premise is upside-down on the model base the proposal would target.
2. **Asymmetric variant has a hard ~9pp upper bound at τ=0.05**, set by the size of the target cohort (raw caught & teams missed = 3-26 frames depending on τ). Even perfect transfer cannot exceed that bound. The wrong-way cohort outnumbers the target cohort 2-10× at every operating point measured (τ=0.05: 26 vs 62; τ=0.10: 17 vs 55; τ=0.20: 11 vs 47; τ=0.50: 3 vs 29).
3. **86% of viso pairs (236/275) miss in BOTH versions at τ=0.5** — pair loss has no fulcrum here. The dominant failure is wholesale, not transport-specific.
4. **Cheaper alternatives are demonstrably larger and zero-cost.** F4 substrate cleaning (Job 14, `analysis/job_14_substrate_clean_simulation_2026-05-04/`) lifts P8A viso 27% → 67% on the same eval surface. Per-substrate τ-calibration (Job 7 follow-up, `analysis/per_substrate_tau_calibration_2026-05-05/`) gives 21pp lockbox lift on P8A. Both are deployment-honest under deployment-time constraints.
5. **The $5 GPU probe (option a) was declined** but cannot reverse the sign-of-effect or cohort math — those numbers are computed directly from E2B scores, not from P8A-as-proxy features. The probe could only inform an asymmetric variant's feature-geometry assumptions, and the cohort math already caps that variant at ~9pp.

**What this resolution does NOT claim**: it does NOT claim asymmetric pair loss has been tested and failed. It has not been tested. The claim is that the upper bound (~9pp) is small enough relative to alternatives (~40pp via substrate cleaning, ~21pp via per-substrate τ) that the user deprioritized scoping it. The lever-space "design intervention that uses paired data" remains structurally open — see Current stance, line 44.

**Reopen triggers — revisit if any of these materializes:**

- Substrate cleaning + per-substrate τ on Packets A / C-codec (in flight as of 2026-05-04, runs `3140330896851206144` / `1202657157175050240`) do NOT produce a contract-grade viso lift. I.e., the cheaper alternatives that displaced pair-loss work fail to deliver.
- A different model base than E2B becomes the production candidate AND the sign-of-effect on (raw, teams) viso pairs has not been measured for that base. **Required first step on reopen**: re-run the Wilcoxon + cohort partition on the new base's scores before drafting any pair-loss packet. Do not assume the E2B finding generalizes.
- A theoretical or empirical reason emerges to expect E2B's sign-of-effect on viso pairs to flip — e.g., a packet that unlearns the codec/transport shortcut, after which teams-transport may again hurt rather than help.

Context: this remains a **low-severity** loop because the symmetric-pair-loss premise is empirically refuted (DEBATE block above), the asymmetric variant has a ~9pp upper bound at τ=0.05, and substrate cleaning + per-substrate τ are zero-cost and demonstrably larger. The loop is preserved per AGENTS.md ("do not delete resolved loops") so a future agent reading the DEBATE block does not redo the symmetric-pair-loss scoping work or re-derive the cost/benefit comparison.

### DEBATE — 2026-05-22 — Probe 1 reframe of substrate axis; reopen-trigger #2 fires for Slot A v2

**Source**: `analysis/substrate_pair_geometry_2026-05-22/FALLBACK1_PROBE1_FACTS_2026-05-22.md` §1 (Probe 1 KLIEP re-fit on trained encoders; cached L11 features built by A0.2 `analysis/substrate_pair_geometry_2026-05-22/run_phase0_geometry.py`). Companion FACTS: `analysis/substrate_pair_geometry_2026-05-22/RESULTS_FACTS_2026-05-22.md` §3 (12-cell table) and `INVENTORY_FACTS_2026-05-22.md` (1,880 paired-transport rows).

**Direct numerical findings**:

1. Each of the 3 trained encoders (`P8A_step5000`, `SlotAv2_step3500`, `T5C_step3500`) constructs its own L11 substrate axis with held-out classifier accuracy 0.9804 / 0.9836 / 0.9836 respectively. The per-ckpt axes successfully separate clean-side from teams-side L11 CLS features at ~98% accuracy on the A0.1 inventory pairs (`FALLBACK1_PROBE1_FACTS_2026-05-22.md` §1.2 table).
2. Cosine alignment between each per-ckpt substrate axis and the frozen-CLIP-L11 KLIEP axis (`_kliep_w_hat.npy`, accuracy 0.9909 on dev_real vs lockbox_real per D10) is small: P8A 0.1037, SlotAv2 0.0400, T5C 0.0592. Angular separation `arccos(0.04) ≈ 87.7°`; `arccos(0.10) ≈ 84.0°`.
3. Per-pair direction `diff_i = L2norm(teams[i]) − L2norm(clean[i])` projects 9.0–22.9× more strongly onto the per-ckpt axis than onto the frozen-CLIP axis: P8A ratio 22.9× (0.1187 / 0.0052), SlotAv2 10.1× (0.1363 / 0.0135), T5C 9.0× (0.1337 / 0.0148). σ ≈ 0.04 on per-ckpt axes; ratios are not noise-driven.
4. Probe 2 (face-region pool of L11 patch tokens) on SlotAv2 raises `cos_pair` from 0.8673 to 0.9626 and shrinks |Δ_pair_vs_within| from 0.0774 to 0.0160 on the same 1,825 pairs (`analysis/face_pool_canary_2026-05-22/RESULTS_FACTS_2026-05-22.md` §7).

**What this reframes**:

- The 2026-05-04 verification used E2B scores directly for the sign-of-effect computation, and used P8A frozen features as proxy for E2B's feature geometry on the Q1/Q2 measurements. The 2026-05-22 Probe 1 finding does not retroactively change the E2B sign-of-effect or cohort math (which are computed directly from E2B scores) — but it shows that the *direction* of the substrate axis a future pair-loss packet would target is itself encoder-dependent. Any substrate-pair contrastive loss aligned to the frozen-CLIP KLIEP direction would target an axis ~85–88° from the trained encoder's own substrate axis on Slot A v2 and T5C.
- Slot A v2 step3500 became the deployment-candidate ckpt today (Track B λ=1.0 rerank in `analysis/contract_reframe_2026-05-22/RESULTS_FACTS_2026-05-22.md` §5, and face-pool full scorecard in `analysis/face_pool_scorecard_2026-05-22/RESULTS_FACTS_2026-05-22.md` §3a–§4c). This fires the second reopen-trigger from the 2026-05-04 resolution of `pair-loss-asymmetric-variant-untested`: "A different model base than E2B becomes the production candidate AND the sign-of-effect on (raw, teams) viso pairs has not been measured for that base. Required first step on reopen: re-run the Wilcoxon + cohort partition on the new base's scores before drafting any pair-loss packet."

**What does NOT change**:

- The structural finding that the data is same-identity-two-transport (lines 17–34, this thread) is unchanged.
- The 2026-05-04 sign-of-effect on E2B (mean E2B raw_score = 0.086, mean E2B teams_score = 0.172, Wilcoxon p = 0.0019) is unchanged — it was computed from E2B scores, not from any axis projection.
- The `pair-loss-asymmetric-variant-untested` loop above remains `resolved` for E2B; the new loop below is scoped specifically to Slot A v2 step3500 (the new deployment candidate).

A new structured open loop is opened below to track the required CPU-2 probe on Slot A v2.

### Open loop: per-base-substrate-pair-cohort-math-untested
status: open
severity: high
first_seen: 2026-05-22
last_verified: 2026-05-22
close_criterion: run Wilcoxon on Slot A v2 step3500 on the 275 paired viso fakes from analysis/pair_loss_effect_verification_2026-05-05; CPU-2 in next Phase 1

The reopen-trigger #2 from the 2026-05-04 resolution of `pair-loss-asymmetric-variant-untested` fires today because Slot A v2 step3500 became the deployment candidate (Track B λ=1.0 rerank + face-pool scorecard Pareto improvement). Before any pair-loss packet is drafted against Slot A v2 step3500, the sign-of-effect on the same 275 paired (raw, teams) viso fakes (from `analysis/pair_loss_effect_verification_2026-05-05/FINDINGS.md`) must be re-measured on Slot A v2 scores — the E2B finding cannot be assumed to generalize. CPU-2 in the next Phase 1 (per `/Users/roeedar/.claude/plans/ok-so-we-don-t-valiant-quasar.md` Phase 1 CPU-2) executes this; close criterion is the FACTS doc at `analysis/pair_loss_slot_a_v2_2026-05-23/RESULTS_FACTS_2026-05-23.md` with an explicit α/β/γ outcome on whether the sign reverses, holds, or is indeterminate at the new base. Source thread reopen text: lines 113–118 above.

*(The original "(none direct — closed by evidence)" framing for this thread's open loops is preserved historically below; that framing referred to the structural same-identity-two-transport finding, not the contrastive-loss inference.)*

*(Original Open loops note, preserved for historical fidelity — none direct — the structural finding is closed by evidence. The actionable corollary lives in [`viso_bucket_gap`](viso_bucket_gap.md) `p14-data-fix-not-launched`. A separate open loop for "explicit eval-bucket join confirmation" was considered but is intentionally rolled into the P14_DATA_FIX close criterion: once the retrain produces a working scorecard, the eval-bucket pairing assumption is validated by construction.)*

### Cross-thread refs

- [`viso_bucket_gap`](viso_bucket_gap.md) — the dispositive Layer-2 finding; this thread is the structural argument for why that closure is cheap.
- [`identity_audit`](identity_audit.md) — refutes the "N too small" hypothesis; this thread refines that further to "N is fine, but the *transports per identity* are what matter."
- [`processing_signature_shortcut`](processing_signature_shortcut.md) — anti-shortcut contrastive losses can use `(clean[id_X], teams[id_X])` as positive pairs targeted at the camera/ISP signature; this is the "Move 4" path in `docs/relaunch_handoffs/TRAINING_EVAL_VISO_BUCKET_GAP_2026-04-29.md` § 6 and is structurally orthogonal to the augmentation-only interventions in P11/P13.
- [`webcam_fpr_dominance`](webcam_fpr_dominance.md) — the modern_lockbox_v2 webcam-mode tail represents the failure mode the camera-signature shortcut produces; pair-based contrastive training on `(clean[id_X], teams[id_X])` should weaken that failure mode by construction.
