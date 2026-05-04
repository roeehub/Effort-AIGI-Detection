# Packet RLP6B  ·  Representation-robustness side-branch (quality-λ, ArcFace margin, mixup, feat-norm-reg)

> **Template contract**: fill every section. If a section truly has no content, write `*(none — reason)*` rather than deleting the heading. Keep the status-card table intact.

## Status card

| Field | Value |
|---|---|
| Dates | 2026-04-23 → 2026-04-24 (yamls authored; no evaluation artifacts surfaced as of retro date) |
| Slots | 8 (`R13_RLP6B_01`..`R13_RLP6B_08`) |
| Headline lever | Representation-robustness sweep aimed at the low-norm / enhancement processing-signature shortcut — four single-lever probes (`quality_domain_loss_weight` ×2/×4, ArcFace `m=0.50`, `mixup_alpha=0.5`, `label_smoothing=0.1`, `feat_norm_reg_lambda=0.05`) plus two combo slots |
| Leader slot | *(none — no evaluation artifacts; see §Results)* |
| Leader metric | *(none — no evaluation artifacts)* |
| Verdict | 🟡 in-flight / 🔬 superseded — yamls authored and untracked; no RLP6B-specific scorecards, checkpoint maps, or handoff exist, and the RLP7 refocus onto the camera/ISP-signature shortcut (`HANDOFF.md:1-10`) pulled attention away before results materialized |
| Next-packet decision | *(none — no in-session conclusion to forward; RLP7 proceeded from `RLP6_04` on a different lever axis: camera/ISP signature via Teams spatial + codec aug)* |
| Themes touched | [promotion_contract_evolution](../threads/promotion_contract_evolution.md) (quality-λ and feat-norm-reg both interact with the value_composite readout via feature-norm geometry), [processing_signature_shortcut](../threads/processing_signature_shortcut.md) (every slot description names the shortcut as its target, but no empirical result ties the two) |

## Configuration

RLP6B is a **side-branch of RLP6**, not a successor. It shares RLP6's data recipe and gate configuration (packet-5 `R13_RLP5_07_E3_seedB` base + RLP6 slot-1 gate-alignment: avspeech+VCD dropped from real-gate, `wma_failure_fake` dropped from fake-gate, `path_exclude_contains: ["/visomaster_"]` on `teams_ood_fake`) but tests a different hypothesis: **representation-robustness levers targeting the diagnosed low-norm + enhancement shortcut**, rather than RLP6's main-line **gate-alignment** hypothesis. The "B" suffix marks the branch. Every yaml carries the tag `representation-robustness` alongside `packet-6b` / `rlp6b` (e.g. `experiments/phase2_round13/R13_RLP6B_02_quality_lambda_4x.yaml:13`).

The RLP6 experiment plan (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md`) defines eight gate-alignment slots and does **not mention RLP6B** — the side-branch was authored later, independently, and has no dedicated handoff document.

Control slot: no explicit control is stated in the yamls; the implicit control is either `RLP6_01_gate_align_canary` (same gate, lever knobs at default) or the RLP6 leader itself. The comment block on every yaml fixes the base as *"R13_RLP5_07_E3_seedB + packet-6 slot-1 gate-alignment changes"* (e.g. `experiments/phase2_round13/R13_RLP6B_01_quality_lambda_2x.yaml:4`).

Variants (each slot = one representation-robustness knob delta vs the RLP6 slot-1 gate):

- **`R13_RLP6B_01_quality_lambda_2x`** — `quality_domain_loss_weight: 0.2 → 0.4` (2×). "Stronger gradient-reversal adversarial pressure on the backbone to forget capture-domain texture." `experiments/phase2_round13/R13_RLP6B_01_quality_lambda_2x.yaml:9,322`.
- **`R13_RLP6B_02_quality_lambda_4x`** — `quality_domain_loss_weight: 0.2 → 0.4` *(comment says "×4"; numeric value is the same 0.4 as slot 01)*. "Saturation probe on quality-adversarial." `experiments/phase2_round13/R13_RLP6B_02_quality_lambda_4x.yaml:5,322`. **Slot-naming vs numeric-value mismatch** — flagged in §Retrospective.
- **`R13_RLP6B_03_arcface_high_margin`** — `arcface_m: 0.15 → 0.50`. "Tighter angular margin forces features into class-cluster geometry where feature-norm becomes less decision-relevant." `experiments/phase2_round13/R13_RLP6B_03_arcface_high_margin.yaml:5`.
- **`R13_RLP6B_04_mixup_strong`** — `mixup_alpha: 0.0 → 0.5`. "Heavy embedding interpolation between real/fake; flattens the low-norm 'fake' region." `experiments/phase2_round13/R13_RLP6B_04_mixup_strong.yaml:5,50`.
- **`R13_RLP6B_05_label_smooth_fake`** — `label_smoothing: 0.0 → 0.1`. "Softens the model's certainty in the confident-fake region." `experiments/phase2_round13/R13_RLP6B_05_label_smooth_fake.yaml:5,48`.
- **`R13_RLP6B_06_feat_norm_reg`** — `feat_norm_reg_lambda: 0.0 → 0.05`. "Direct aux loss penalizing (mean-feature-norm-real − mean-feature-norm-fake)² + variance penalty. Targets the diagnosed shortcut head-on. Requires `detectors/effort_detector.py` code change." The required code is present (`detectors/effort_detector.py:346-348,966-1000,1045-1046`). `experiments/phase2_round13/R13_RLP6B_06_feat_norm_reg.yaml:5,50`.
- **`R13_RLP6B_07_combo_lite`** — `quality_domain_loss_weight=0.4` + `arcface_m=0.30` + `mixup_alpha=0.3`. "Mid-strength combined pressure." `experiments/phase2_round13/R13_RLP6B_07_combo_lite.yaml:5,50,316,323`.
- **`R13_RLP6B_08_combo_aggressive`** — `quality_domain_loss_weight=0.4` (×4-labelled) + `arcface_m=0.50` + `mixup_alpha=0.5` + `feat_norm_reg=0.05`. "Maximum representation pressure; may regress baseline recall if over-regularized." `experiments/phase2_round13/R13_RLP6B_08_combo_aggressive.yaml:5,50-51`.

## Results at the time

*(none — no RLP6B-specific artifacts exist as of 2026-04-24: `arena/checkpoint_maps/` has no RLP6B entry, `arena/reports/` has no RLP6B build/score report, and `HANDOFF.md` is a pre-launch gate for Packet-7 that does not reference the branch. The 8 yamls are present but untracked in git. No leader delta, no per-pool numbers, no W&B readouts were recorded in-repo.)*

## Conclusions drawn in-session

*(none — no in-repo handoff or retrospective captures RLP6B-specific conclusions. The lever menu itself is the only artifact: every slot description names the low-norm / enhancement shortcut as its target, so the packet as a whole documents the team's working hypothesis that the shortcut is reachable by representation-shaping levers rather than by data or gate changes. That hypothesis was not tested in-session. `convmem search "RLP6B"` / `"quality lambda 4x"` / `"Packet 6B"` returned no relevant hits — sessions covering this branch are either very recent, still open, or were not transcribed into convmem.)*

- **Session IDs**: *(none indexed — recent/open sessions not yet searchable)*

## Retrospective (as of 2026-04-24)

- **The packet was authored but never landed as a distinct experimental result.** No scorecard, no checkpoint map, no handoff cites RLP6B. The 8 yamls are present under `experiments/phase2_round13/R13_RLP6B_*.yaml` and untracked on the current branch; the downstream decisions (RLP7 launch priorities, the camera-signature shortcut handoff) proceed from the `RLP6_04` leader at `value_composite=0.9006` without reference to RLP6B (`HANDOFF.md:9`).
- **RLP7 refocused the shortcut story onto a different axis.** The Packet-7 handoff (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md:1-19`) and `HANDOFF.md:9` reframe the remaining error surface as a **camera/ISP-signature** shortcut (same person → different webcam flips the score from 0.02 to 0.94), not the **low-norm feature-norm** shortcut that every RLP6B description names. The RLP7 levers (`teams_passthrough_special_*` spatial aug, `teams_codec_sim` codec aug) attack the new framing through training-time augmentation, not through loss-surface pressure. The RLP6B levers therefore target a hypothesis the team partially **superseded** before the runs launched.
- **Slot-01 / slot-02 may be a copy-paste bug rather than an intentional 2x/4x sweep.** Slot-01 is described and named "2x" (`quality_domain_loss_weight=0.4`) and slot-02 is described and named "4x" — but both files set `quality_domain_loss_weight: 0.4` (`experiments/phase2_round13/R13_RLP6B_01_quality_lambda_2x.yaml:322` and `R13_RLP6B_02_quality_lambda_4x.yaml:322`). The RLP5 baseline carried `0.2`, so slot-01 is a genuine 2× against baseline; slot-02 is **identical to slot-01** despite the header. If launched as-is, the "saturation probe" would be a seed-identical duplicate of slot-01. Worth fixing (or verifying intentional) before any launch.
- **Preprocessing-parity status: PRE-FIX.** All RLP6B yamls predate commit `855871e` (INTER_LINEAR inference fix, 2026-04-24; see [preprocessing_parity_bug](../threads/preprocessing_parity_bug.md)). Because no RLP6B run has been evaluated, there is no post-fix re-score to compare against. If RLP6B is ever launched, numbers should be taken from the post-fix inference path only.
- **Open-loop status.** The quality-λ / ArcFace-margin / mixup / feat-norm-reg sweep has **not** been evaluated under the corrected gates (post-avspeech-removal). That is the experimental question RLP6B was built to answer; as of the retro date, it remains unanswered.
- **Recommendation: fold this file into `RLP6.md` in a future revision.** RLP6B stands alone only by name. It has no results, no conclusions, and the hypothesis it targets has been at least partly superseded by RLP7's camera-signature framing. A single section inside `RLP6.md` titled *"Side-branch 6B (representation-robustness, authored / not fired)"* listing the 8 slots, the lever menu, and this retro note would preserve the historical record without implying a packet-level result that does not exist. Reopen as a standalone file only if the branch is later launched.
- **Cross-reference.** Story continues in [RLP6](RLP6.md) (parent packet; gate-alignment main line), [RLP7](RLP7.md) (camera-signature refocus; supersedes the low-norm framing RLP6B targeted), and cross-cuts through [processing_signature_shortcut](../threads/processing_signature_shortcut.md) and [promotion_contract_evolution](../threads/promotion_contract_evolution.md).

## Source files

- **Handoffs**:
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md` — parent packet plan; **does not mention RLP6B**. Establishes the shared base (`R13_RLP5_07_E3_seedB`) and the slot-1 gate-alignment changes this branch inherits.
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md:1-19` — Packet-7 handoff that reframes the shortcut story away from RLP6B's hypothesis.
  - `HANDOFF.md:1-10` — current pre-launch gate; cites `RLP6_04` as leader, no RLP6B reference.
  - *No RLP6B-dedicated handoff exists.*
- **YAMLs**: `experiments/phase2_round13/R13_RLP6B_0{1..8}_*.yaml` — all 8 enumerated in §Configuration; untracked on branch `teams-relaunch-root-2026-04-17`.
- **Scorecards / analysis**: *(none — `arena/checkpoint_maps/` and `arena/reports/` contain no RLP6B entries).*
- **Code touchpoints**: `detectors/effort_detector.py:346-348,966-1000,1045-1046` — `feat_norm_reg_lambda` code path used by slot 06 and slot 08; already shipped, not blocked on a code change.
- **Memory pointers**: `project_signature_shortcut_finding.md` — the shortcut diagnosis RLP6B's slot descriptions reference (feature-norm geometry; low-norm = fake); `project_promotion_contract.md` — dev-calibrated τ + lockbox authority, the correct readout for any RLP6B run if later launched; `feedback_small_sample_guidance.md` — relevant if RLP6B is partially launched and per-slot sample sizes are small.
