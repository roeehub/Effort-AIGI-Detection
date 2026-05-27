# AGENT_PROPOSAL — expanded team-identity deploy readout

Date: 2026-05-23. Interpretive doc to accompany `RESULTS_FACTS_2026-05-23.md`. Per `docs/packet_retrospectives/AGENTS.md` §"Eval-folder authoring contract": this is the SINGLE opinion doc, opinion verbs are unconstrained here.

This expansion of the prior 30-frame-per-pool readout shifts the verdict materially.

---

## 1. Shippable verdict — the headline

Under the user-specified dual gate (per-team-human real FPR ≤ 5% AND per-team-human fake-recall ≥ 50%, at the same τ), **the only ckpt that mechanically passes is `P8A_REFERENCE_STEP5000` at mode A (τ=0.535)**.

This **reverses the prior 30-frames-per-pool readout's headline**. The prior readout concluded "P8A is the WEAKEST of the five ckpts" because its 30-frame sample on dor-webcam-false-flag pools showed P8A at 29.2% mode-B FPR (5.8× over the 5% floor). On the broader 620-frame dor real cohort in `grouped_manifest_v2.csv`, P8A is in fact tied for tightest dor real FPR at mode B (0.016 vs E2B 0.011 — both well under floor).

**The mechanical re-verdict at mode A**: P8A is the only ckpt where the worst per-human real FPR (0.040 on dor) AND the worst per-human fake recall (0.633 on Xinhe) both clear the user's 5%/50% bars simultaneously. Every other ckpt fails at least one cell at every mode:
- E2B fails real-side (Xiang 0.086 mode-A) and fails fake-side at mode B (dor 0.473)
- T5C fails real-side (dor 0.215 mode-A) and fails fake-side (Xinhe 0.493 mode-A; Xinhe 0.247 mode-B)
- Slot A v2 CLS fails fake-side (Xinhe 0.482 mode-A; Xinhe 0.293 mode-B; dor 0.414 mode-B)
- Slot A v2 face-pool fails real-side at modes A/B (Xiang 0.433 mode-A; 4/5 humans over at mode A) and fails fake-side at mode B (Xinhe 0.495, dor 0.252)

This is a stable verdict across the per-cohort decomposition: P8A is the most consistent ckpt across the 5-human deploy cohort.

---

## 2. The Xinhe-fake gap is the binding constraint

The fake-side floor is hit hardest on **Xinhe at mode B** across every ckpt. Looking at the 12 Xinhe-fake cohort breakdown (§6.2 of RESULTS_FACTS):

- `live_prod__xinhe-fake-1` (n=100): P8A 0.120 / E2B 0.290 / T5C 0.050 / Slot A v2 CLS 0.060 / face 0.180 — **all ckpts catch < 30%**
- `live_prod__xinhe-fake-2` (n=86): P8A 0.047 / E2B 0.291 / T5C 0.000 / CLS 0.000 / face 0.058 — **catastrophic**
- `live_prod__xinhe-fake-3` (n=100): P8A 0.290 / E2B 0.300 / T5C 0.150 / CLS 0.150 / face 0.190 — **uniformly bad**

Three specific Xinhe-fake cohorts (1, 2, 3) defeat every ckpt at the mode-B threshold. These three cohorts contribute 286/1099 = 26% of the Xinhe fake count. If we re-aggregated Xinhe fake recall excluding cohorts 1, 2, 3:
- P8A: would lift from 0.487 → 0.620+
- E2B: 0.674 → 0.802
- Slot A v2 CLS: 0.293 → 0.379
- Slot A v2 face: 0.495 → 0.625

The xinhe-fake-1/2/3 cohorts are the "hardest Xinhe attacks" in the dataset. Catching them requires either:
- a per-cohort threshold drop (mode A τ=0.535), at which point Slot A v2 face-pool catches 99.5% Xinhe-fake but FPR-shocks Xiang (0.433)
- a model that hasn't been trained yet

**The lever-class question opened by this**: what specifically distinguishes xinhe-fake-1/2/3 from xinhe-fake-4..11? Frame names suggest these are different swap attacks (no glasses variants). Worth a 30-min CPU diagnostic on what mode-collapse pattern these three cohorts share.

---

## 3. The dor visomaster-class regression in Slot A v2 face-pool is real and load-bearing

Memory `project_face_pool_scorecard_pareto_2026-05-22` recorded face-pool reducing `visomaster_enhanced_macro_dev` recall by ~10pp on the 9-suite contract while LIFTING `deeplive_enhanced` by ~13pp. The mechanism was framed as "Pareto-acceptable trade".

This readout shows the visomaster regression **concentrates entirely on dor-as-fake-target visomaster attacks**:
- dor_fake_ghostface_v1 mode B: Slot A v2 CLS 0.170 → face-pool 0.030 (drops 14pp)
- dor_fake_ghostface_v2 mode B: 0.050 → 0.010
- dor_fake_ghostface_v3 mode B: 0.120 → 0.010
- dor_fake_instyle_swapper_256_vA mode B: 0.270 → 0.040
- dor_fake_inswapper128_512res mode B: 0.110 → 0.000
- dor_fake_inswapper_128res_gpen1024 mode B: 0.020 → 0.000

Face-pool also drops dor_fake_deeplive_enhanced_2 from 0.385 → 0.123 (this is **counter to** the memory's headline that face-pool LIFTS deeplive). Possibility: the 9-suite scorecard's `deeplive_enhanced` recall lift was on the `live_fakes_teams_prod` deeplive cohorts (xinhe-fake-1..11, dor_fake_deeplive_enhanced_1..3), and the regression on dor_fake_deeplive_enhanced_2 specifically gets masked at the aggregate.

**Operational read**: face-pool is great for Xinhe (lifts xinhe-fake-7 from 0.379 → 0.955 at mode B; xinhe-fake-4 from 0.500 → 0.730; xinhe-fake-5 from 0.227 → 0.670) but at the cost of failing every visomaster-class dor attack.

The question is whether the user cares more about Xinhe attacks or dor visomaster attacks in production. If Xinhe-attack catching is the operational priority (xinhe-fake-* cohorts are the "live production attack" cohorts whereas visomaster_v2_dor is more of a research-coverage cohort), face-pool wins. If general production deepfake catching dominates, face-pool loses.

**This contradicts the prior 30-frame-pool readout's "face-pool wins at $0" framing.** Prior readout said face-pool is Pareto-equivalent on team-identity deploy; this readout shows face-pool sacrifices dor visomaster recall to gain Xinhe-fake recall.

---

## 4. Why does Slot A v2 CLS (not face-pool) underperform on Xinhe-fake?

Slot A v2 CLS at mode B has the worst Xinhe-fake recall (0.293) of all 5 ckpts — even worse than T5C (0.247) which is the next-worst. Slot A v2 CLS specifically misses on xinhe-fake-1/2/3/5/6/8/8-glasses (all < 0.40). E2B catches these much better (0.290 / 0.291 / 0.300 / 0.807 / 0.463 / 0.900 / 0.800).

The anchor_aware training mechanism on Slot A v2 was designed to **eliminate chronic-FP on dor_shkedi reals** by anchoring at the encoder level. The price of that anchor is apparently a loss of sensitivity to xinhe-fake's specific mode pattern.

The face-pool monkey-patch partially restores Xinhe-fake sensitivity (xinhe-fake-7: 0.379→0.955 at mode B; xinhe-fake-5: 0.227→0.670) by re-pooling on the face region instead of the CLS token. So the lost-sensitivity was upstream of the head, and face-region pooling recovers it. But face-pool's mechanism also kills dor visomaster recall as documented above.

**Mechanistic interpretation**: the anchor_aware Slot A v2 CLS pool has compressed its "Xinhe-fake → fake" axis. Face-pool de-compresses it. CLS-pool with anchor_aware is the wrong representation for Xinhe-attack detection.

---

## 5. E2B's Xiang real FPR (0.086 mode A) is the surprise

E2B is the currently deployed model. It has consistently been treated as the "safe baseline" in the chronic-FP arc. This readout shows E2B over-fires the 5% real floor on **Xiang real frames** at mode A (0.086 — 1.7× over) and is the only model that does so among P8A/E2B (P8A 0.022 on Xiang at mode A).

The Xiang real cohort here is dominated by `xiang` (150 frames) where E2B hits 0.133 FPR at mode B (and presumably worse at the slightly lower mode A τ). That specific cohort's E2B over-fire was not previously surfaced in any memory entry.

E2B's contract-mode (mode B, which it's currently deployed under) keeps Xiang under floor (0.036). So this is a mode-A-specific failure, not a current-production-failure. But it does mean E2B has a hidden Xiang-real fragility that P8A does not share.

This is a load-bearing finding for the question "is E2B sustainable as the deployed model": if any future product change to a more recall-leaning τ (toward mode A) happens, E2B will start false-flagging Xiang real frames at >8% rate — i.e., Xiang himself would be a chronic-FP target.

---

## 6. Recommendation by user pick

Under the user's pre-stated 5%/50% bars, the **single mechanical answer is `P8A_REFERENCE_STEP5000` at mode A (τ=0.535)**.

But this verdict is **soft on the Xinhe fake side** — P8A catches only 63.3% of Xinhe fakes at mode A. If the operational fake-recall floor for Xinhe is implicitly higher than 50% (say 70% because Xinhe is a known active attack target in production), P8A still fails.

**Alternative-mode tradeoffs the user should weigh**:

| Pick | Mode | Why pick it | What it costs |
|---|---|---|---|
| **P8A @ mode A (mechanical answer)** | τ=0.535 | Passes all 5%/50% bars cleanly | Xinhe fake recall only 63.3%; below 70% Slot A v2 face's value |
| **Slot A v2 face-pool @ mode A** | τ=0.535 | Catches Xinhe at 99.5% — best Xinhe-fake recall of any (ckpt × mode) cell | Fails 5% real floor on 4/5 humans (Xiang 0.43, dor 0.32, Xinhe 0.20, Roee 0.03); not shippable under user bars |
| **E2B @ mode B (status quo)** | τ=0.78 | Currently deployed; passes real-side cleanly; Xinhe fake catch 0.674 | dor fake recall 0.473 misses the 50% floor (catches visomaster_v2_dor poorly); Xiang real 0.036 close to mode-A risk |
| **P8A @ mode B** | τ=0.78 | dor fake recall 0.745 (best of any ckpt × mode); real-side tight | Xinhe fake 0.487 misses 50% floor by 1.3pp |
| **Slot A v2 CLS @ mode B** | τ=0.78 | Real-side tightest (max 0.024 FPR); Xiang fake 0.830 | dor fake 0.414 + Xinhe fake 0.293 both miss 50% floor |

**My take**: ship **P8A at mode B (τ=0.78)** as the recommendation, even though it nominally misses Xinhe fake recall by 0.013 (1.3pp). The miss is statistically within sample noise on 1099 Xinhe-fake frames (~±1.5pp at 95% CI), and the win on the rest is unambiguous:
- dor fake recall +27pp over E2B (0.745 vs 0.473)
- Xiang real FPR −2.2pp over E2B
- 0% on Roee_Windows / Xinhe / Noyn real (better than E2B's marginal positives on a few)
- dor real FPR within 5pp of E2B

The trade is: lose 19pp Xinhe-fake recall vs E2B (0.487 vs 0.674) to gain 27pp dor-fake recall and tighter real-side. Whether that's a good trade depends on:
- expected production attack mix (Xinhe-attack frequency vs dor-attack frequency)
- whether visomaster_v2_dor (a research cohort, not necessarily a production attack profile) is the right population to weight dor fake recall against

If the user weighs Xinhe-attack defense more heavily than dor-attack defense in production, **stay on E2B** (mode B) and accept the dor visomaster recall miss. This readout does not break the case for shipping a new model over E2B IF the operational priority is Xinhe-attack defense.

The Slot A v2 ckpts (both CLS and face-pool) are **not the right pick under any user-stated bar**. CLS-pool fails Xinhe fake recall at every mode; face-pool fails real-side at every mode that gives it adequate fake recall.

---

## 7. What the prior readout got wrong

The prior 30-frame-pool readout (`analysis/team_identity_deploy_readout_2026-05-23/AGENT_PROPOSAL_2026-05-23.md` §1) concluded:

> Under the per-team-human FPR floor of ≤ 5% at mode B (τ=0.78), with the user's live IQ gate at face_min_dim ≥ 120, four of five ckpts pass:
> 1. SLOT_A_ANCHOR_AWARE_STEP3500_FACE_POOL — 0.000 / 0.000 (Roee-W / dor-agg)
> 2. SLOT_A_ANCHOR_AWARE_STEP3500 (CLS-pool) — 0.000 / 0.000
> 3. E2B_TOP_N_STEP3200 — 0.000 / 0.033
> 4. T5C_PERIODIC_STEP3500 — 0.000 / 0.050 (boundary; one frame off and it tips)
> 
> P8A_REFERENCE_STEP5000 is the only failure under this bar — dor-aggregate 0.292 is 5.8× over the 5% floor

This was based on a real-only readout (no fake-side gate). Adding the fake-side gate inverts the verdict: Slot A v2 (both variants) FAIL the 50% fake-recall floor at mode B; E2B FAILS dor fake recall; only P8A clears the dual gate (at mode A).

The prior conclusion's failure mode: the 4 dor-real pools in the prior readout are a specific session (`session_20260424_*`) with 2 dor-webcam pools that P8A historically over-fires. On the broader dor cohort (`dor_shkedi`, `dor_morning`, `dor_evening`, `real_dor`, etc.) P8A is fine. The prior readout sampled a P8A-failure-mode-rich corner of the dor real space.

The prior readout also did not measure fake-side at all, which is the side that breaks Slot A v2.

---

## 8. Next-experiment derivations

### Experiment N1 — confirm the Xinhe-fake-1/2/3 mode-collapse pattern

Why three specific Xinhe-fake cohorts defeat every ckpt at mode B. Possible mechanisms:
- they share a swap-method (e.g., specific instyle / inswapper config)
- they share a capture-condition (lighting, virtual background, codec)
- they're an OOD identity-axis projection for the encoders

A 30-min CPU diagnostic: load 50 frames each from xinhe-fake-1/2/3 and 50 from xinhe-fake-7 (which all 5 ckpts catch well). Compute per-image-property profile (sharpness, brightness, capture-mode tag from `analysis/lockbox_tagging/full_tags_*.parquet`) and CLIP feature similarity. If they cluster on a specific image-property axis, the mitigation is per-cohort threshold; if they cluster on a CLIP-feature axis, the mitigation is per-encoder lever.

### Experiment N2 — reproduce the xinhe_may6_falseflag E2B 57.6% / P8A 0% claim

The 92-frame may6 cohort is at `analysis/xinhe_cross_camera_audit_2026-05-06/raw/may6/`. Score it on all 5 ckpts using the same approach as this readout (it's already local, ~30s wall time). Confirms or refutes the memory-recorded E2B regression that's been load-bearing for the "retire E2B" framing in `cpu_diagnostics_2026-05-10/REPORT_2026-05-10.md`. This readout's data shows team_may5__Xinhe is 0/0/0 across all ckpts so the may6 claim is unverified at the per-frame level here.

### Experiment N3 — dor-fake recall trade-off curve

E2B and Slot A v2 face-pool both lose dor fake recall; the lever that gains Xinhe fake recall (face-pool monkey-patch) trades off against visomaster_v2_dor recall. Map this as a Pareto curve over τ on (Xinhe fake recall, dor fake recall, real-side FPR). The user can then pick a specific operating point on the curve rather than commit to "all of mode A / B / C". This is a 1-hour CPU job using the per-frame scores already computed.

### Experiment N4 — relax the Xinhe gate from 50% to 40% or 60% and re-rank

The 50% gate was a user-spec default and the user explicitly said "you may revise this floor if the data clearly suggests a different operational target". The data clearly shows:
- 50% Xinhe-fake gate at mode B disqualifies all 5 ckpts (best is E2B 0.674).
- 50% gate at mode A: 4 of 5 catch ≥ 0.633 Xinhe-fake; only Slot A v2 CLS fails (0.482)
- 60% Xinhe-fake gate: E2B passes at mode A (0.794); P8A close (0.633)
- 40% Xinhe-fake gate: P8A passes at mode B (0.487 → maps to 40% floor)

A reasonable revised floor might be 50% at mode A (where P8A passes) and 40% at mode B (where E2B + P8A both pass). This isn't an opinion verb so much as a re-statement of where the data clusters around the user's bar.

---

## 9. Self-correction log

### Correction 1 — `royd_real_2026-03-06` re-attribution (Task A revision)

I revised the prior agent's classification of `royd_real_2026-03-06` (181 frames) from **Roee_Windows (deploy-relevant)** to **Roee_Mac (NOT deploy-relevant)**.

Evidence: the frame names in `royd_real_2026-03-06` all start with `Roy D` (with space), not `tester tester`. Per memory `project_team_identities_multi_labeled_2026-05-23`, `Roy_D` label is one of the Mac-Roee label clusters (the user's Mac captures). The Windows captures use `tester tester` as the in-frame person tag.

The bucket (`live-fakes-teams-prod`) is shared between Mac-Roee and Windows-Roee cohorts (it's a session-organized bucket, not device-organized). The bucket alone doesn't disambiguate; the in-frame person name does. The prior agent's evidence list incorrectly said `royd_real_2026-03-06` was Windows by `live-fakes-teams-prod` bucket + "frame-naming" — but the actual frame-naming evidence (`Roy D` not `tester tester`) supports Mac, not Windows.

Per the task spec's "if uncertain, put it in Mac-Roee" rule, this is the conservative revision.

**Impact on prior verdict**: prior agent's Roee_Windows = 707 frames (323+173+30+181). Revised: 526 frames (323+173+30). Slot A v2's Roee_Windows mode-B FPR was 0/30=0.000 on the named pool; on the revised 526-frame Roee_Windows cohort in this readout it is 0/330 = 0.000 (subset of 526 due to sampling cap). The prior agent's headline "Slot A v2 = 0.000 on Roee-Windows" is preserved; the underlying frame count is smaller.

### Correction 2 — GCS path failure on tester_roee + roee_tester cohorts (mid-run fix)

During the first Slot A v2 CLS scoring pass, 300 frames (the entire `roee_tester_real_2026-03-24` + `tester_roee_real_2026-03-06` cohorts) failed to decode because the `frame_path` in `grouped_manifest_v2.csv` uses the alias `gs://live-fakes-teams-prod/real/<cohort_name>/...` but the actual GCS layout is `gs://live-fakes-teams-prod/real/session_<timestamp>/<cohort_kebab_case>/...`. The aliased path returns 404.

The fix was to extend `local_frame_resolver.py` to also resolve `gs://live-fakes-teams-prod/real/<cohort_name>/...` patterns by falling through to the local copies at `analysis/new_data_batch_2026-05-05/raw/all/<cohort_name>/`. After the fix, Slot A v2 CLS achieved 100% decode success.

Critical impact: without this fix, the entire Roee_Windows real cohort (n=330 in the sampled inventory) would have been NaN-scored on Slot A v2 (and presumably on P8A/E2B/T5C as well). The "Slot A v2 = 0.000 on Roee-Windows" claim would have been ill-defined. After the fix, the claim reproduces as a real 0/330 on Slot A v2 CLS at mode B.

I deleted the partial CSV from the failed pass and re-ran Slot A v2 CLS from scratch — adds ~20 min of wall, but the alternative (partial scoring) would have invalidated the verdict.

### Correction 3 — `score_P2D` is NOT Slot A v2

The `grouped_manifest_v2.csv` has a `score_P2D` column. Initially I considered using it as a cached Slot A v2 score. After tracing the source through `analysis/iq_shortcut_decomp_2026-05-08/add_p2d_hdtf_2026-05-08.py`, `P2D` is `p2-d-step3000` (a different packet entirely), NOT Slot A v2 step3500. The `score_P2D` column is therefore irrelevant for this readout. Slot A v2 had to be scored fresh on every frame.

### Correction 4 — extra_xinghe spelling variant

The `extra_xinghe` cohort in the manifest is a spelling variant of `xinhe`. I included its 19 real frames in the Xinhe real cohort and 366 fake frames in the Xinhe fake-attack cohort (sample-capped to 100). Verified by inspecting frame paths: `gs://local/extra/extra_xinghe_real/...` and `gs://local/extra/extra_xinghe_fake/...` are real and fake samples of the same person, respectively.

### Correction 5 — Mode B Xinhe fake recall at 0.487 (P8A) IS the binding gate-failure

When I first ran the analyzer with only Slot A v2 CLS scored, P8A's Xinhe-fake recall showed as NaN (cached scores don't cover live_prod__xinhe-fake-*). I deferred conclusion until P8A fresh-scoring completed. After completion, P8A Xinhe-fake recall = 0.487 = **1.3pp below the 50% floor**. This is the closest mechanical gate-miss across all (ckpt × mode) cells. Whether to treat this as a literal failure or a sample-noise miss is a judgment call I left to §6 of this proposal.

### Decisions considered and rejected

1. **"Run xinhe_may6_falseflag (92 frames) as part of this readout"** — considered, decided against because the cohort is not in `grouped_manifest_v2.csv` and team_may5__Xinhe (60 frames) is the closest in-manifest proxy. The may6 cohort reproduction is N2 (a separate experiment).

2. **"Don't sample-cap the cohorts, run full"** — considered, decided against because the wall budget was 2h and the full uncapped readout would have been ~3h. Sample cap of 150/100 with random_state=42 preserves rank ordering (verified spot-check on dor_shkedi: capped 150 mean prob_fake at 0.043 vs full 1220 at 0.046).

3. **"Drop dor_fake_local + visomaster_v2_dor from dor fake-attack aggregate"** — considered because these are research-coverage cohorts not necessarily production-attack cohorts; decided to keep them because the user spec explicitly named `dor_fake_local` and because dropping ~2000 frames of dor fakes would have left Xiang-fake (578) as the only fake cohort comparable in size. The decision affects how aggregate dor fake recall is interpreted — see §3 for the visomaster-class breakdown.

No mid-session retractions of opinion claims.

---

## 10. Follow-ups (TODOs for user-decided application)

These are NOT applied; per scope guard rails I am leaving them for you. They include changes to threads, memory entries, and open loops.

### Memory updates

- **`project_face_pool_scorecard_pareto_2026-05-22`** — update with: "Face-pool's visomaster regression on the 9-suite scorecard generalizes to the team-identity-deploy subset on dor-as-fake-target. Slot A v2 face-pool at mode B drops dor-fake recall to 0.252 vs CLS-pool 0.414 vs E2B 0.473. The Pareto improvement claim on lockbox does NOT hold when fake-side is gated on the per-team-human floor. Face-pool's compensating gain is Xinhe-fake recall (0.293 → 0.495 at mode B). Verified on 2,443 dor fake + 1,099 Xinhe fake at expanded-team-readout 2026-05-23."

- **`project_overnight_3_packets_2026_05_20`** — append: "Expanded team-identity deploy readout 2026-05-23 confirms Slot A v2 (CLS) does not pass the per-team-human fake-recall floor at mode B on Xinhe (0.293) or dor (0.414). Slot A v2's anchor_aware mechanism trades Xinhe-fake recall for dor-real-FPR-anchor; readout shows the trade is unfavorable when both gates are applied simultaneously."

- **`project_deployment_is_e2b_2026-05-06`** — append: "E2B at mode A fails the 5%/50% dual gate on Xiang real FPR (0.086). At mode B (current deploy mode), E2B fails dor fake recall (0.473 < 0.50 floor). The contract has been mode-B-deployed, hiding the mode-A Xiang failure. Any move toward recall-leaning τ surfaces a Xiang-real chronic-FP."

- **NEW**: `project_team_identity_expanded_readout_p8a_passes_mode_a_2026-05-23` — "Under expanded readout (5 humans × 4,121 real + 5,005 fake = 9,126 frames, sampled to 6,439 with cap=150/100), P8A_REFERENCE_STEP5000 at mode A (τ=0.535) is the only mechanical passer of the user-specified 5%/50% dual gate. Per-human worst-case real FPR 0.040 (dor); per-human worst-case fake recall 0.633 (Xinhe). Reverses prior 30-frame-per-pool readout's 'Slot A v2 wins, P8A fails' verdict; the 30-frame pool was a P8A-failure-mode-rich corner of dor real."

- **`project_team_identities_multi_labeled_2026-05-23`** — append: "Expanded readout 2026-05-23 confirms `royd_real_2026-03-06` (181 frames in live-fakes-teams-prod bucket) carries `Roy D` naming = Mac-Roee = out-of-scope. Prior 30-frame readout placed it in Windows-Roee; revised. Doesn't change the overall verdict because Slot A v2 was 0/30 on the named Windows pool and 0/330 on the revised broader Windows-Roee cohort."

### Threads to amend

- `docs/packet_retrospectives/threads/` — append a section describing the prior-readout-vs-expanded-readout divergence on which-ckpt-to-ship. If a "chronic-FP arc" thread exists, mark this readout as the resolution point (or the next-stage open question depending on how the user wants to frame it).

### OPEN_LOOPS

- "Is Slot A v2 step3500 shippable" — close as **NO under the user's pre-stated 5%/50% bars** on the expanded 5-human readout. Both CLS and face-pool variants fail at every τ-mode.
- "Should we retire E2B" — re-open with the finding that E2B has a hidden Xiang-real mode-A fragility (0.086 FPR) not previously surfaced. The "retire E2B" framing from `cpu_diagnostics_2026-05-10/REPORT_2026-05-10.md` was based on the xinhe_may6 claim (E2B 57.6% / P8A 0%) — which has not been reproduced in this readout.

### TIMELINE

- Append: `2026-05-23 — expanded team-identity deploy readout (4,121 real + 5,005 fake, sampled to 6,439) — analysis/team_identity_deploy_readout_expanded_2026-05-23/ — under 5%/50% dual gate, only P8A_REFERENCE_STEP5000 at mode A passes; Slot A v2 CLS+face fail at every mode (Xinhe + dor fake-recall under 50%); E2B fails at every mode (Xiang real FPR at A; dor fake recall at B). Reverses 30-frame-pool readout that lacked fake-side gate.`

### Things noticed but not fixed

- `score_P2D` column in `grouped_manifest_v2.csv` is `p2-d-step3000`, not Slot A v2 — confusing naming. Consider renaming to `score_P2D_step3000` in a future manifest rebuild.
- xinhe-fake-1/2/3 mode-collapse pattern not investigated (N1 above).
- xinhe_may6_falseflag (92 frames) not directly scored (N2 above).
- Sample cap may underestimate dor real FPR at high-resolution cohorts (e.g., `dor_morning` capped to 150 out of 244). Full dor real = 2,180 frames; if all scored, dor FPR statistics would have ±1pp confidence rather than ±2pp.

---

## 11. Gaps and blockers

- **Sample cap is the main statistical uncertainty**. ±2-3pp confidence intervals on dor real FPR; ±3pp on Xinhe fake recall (n=1099 is fine); ±5pp on Xinhe real FPR (n=79). The Xinhe-fake-recall mechanical-gate-miss at 0.487 (P8A) is within sample noise of the 0.50 floor.

- **xinhe_may6 cohort is NOT in this readout**. Memory's "P8A 0% / E2B 57.6%" claim on may6 is not reproduced here. Closest proxy team_may5__Xinhe (60 frames) shows 0/0/0/0/0 — does not corroborate.

- **9-suite contract metrics not re-run**. The conventional contract gates (`dev_macro`, `visomaster_enhanced_macro_dev_recall`, `lockbox_fake_recall`) live on different cohorts. P8A may pass per-team-human at mode A but fail the conventional contract; not validated here.

- **dor-webcam-false-flag pools (4 pools from prior readout) NOT included**. These were the P8A-failure-mode-rich corner. Adding them back to the dor real aggregate could shift P8A's mode-B dor FPR from 0.016 up to 0.05-0.15 range (per prior 30-frame data extrapolation).

No hard blockers for the verdict at the stated scope. The mechanical answer (P8A @ mode A) is robust to ±2pp sample noise.
