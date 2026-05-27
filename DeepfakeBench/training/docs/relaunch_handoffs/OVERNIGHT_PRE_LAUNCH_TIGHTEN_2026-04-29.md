# Overnight Pre-Launch Tightening — 2026-04-29 (late evening)

> **Type:** Pre-launch handoff. The user wants the next agent to do a **final tightening + rethinking** of tonight's training slate against the canonical plan, and to surface any **high-value additions** worth folding in before launch. Image rebuild and YAML drafting are gated on this pass.
>
> **Predecessor handoffs in chain:**
> 1. `docs/relaunch_handoffs/BLOCK_A_LANDED_PRE_OVERNIGHT_2026-04-29.md` (afternoon)
> 2. `docs/relaunch_handoffs/PRE_OVERNIGHT_VERDICTS_2026-04-29.md` (early evening)
> 3. **This handoff** (late evening) ← you are here

---

## 0. The plan you should align to

**Plan name:** **R13 Forward Plan — Master Plan 2026-04-29.**

| Surface | Path | Lines | Role |
|---|---|---|---|
| Authoritative long-form | `docs/packet_retrospectives/plans/MASTER_PLAN_2026-04-29.md` | 1527 | Source of truth; section-by-section |
| Compressed | `PLAN.md` (project root) | 1016 | Codex-adversarial-review target |
| Canonical wiki | `docs/packet_retrospectives/` | (multi-file) | Citations resolve here |

**Three-layer ship-readiness model** (master plan §0):

- **Layer 1** — Contract scorecard. *(Closed today via Block A — recall floor 0.70.)*
- **Layer 2** — Headline-metric magnitude (the viso bucket gap). `R13_P14_DATA_FIX.yaml`.
- **Layer 3** — Residual shortcut at Axis 3. P14_FT (in flight); P15 GRL is the next structural lever; face-scale-jitter on P8A is an alternative.

All three must close together for ship-readiness. The decision tree (§12) sequences them.

---

## 1. What landed this session (so you don't re-do it)

Five commits on branch `teams-relaunch-root-2026-04-17`:

| SHA | Subject | Notes |
|---|---|---|
| `974e033` | Promotion contract: default recall floor to 0.70 (Block A) | Scorer + wrapper CLI + 2 regression-guard tests |
| `2660856` | W&B Block B: per-dataset-bucket recall/FPR mid-training observability | `_compute_per_bucket_recall_fpr` + 6 TDD tests |
| `5a0dbbb` | Bump VERSION to 1.3.231 (Block A + Block B image) + verdicts handoff | Image `effort-detector:1.3.231` |
| `528dc6e` | Add per-capture-mode mid-eval observability (Approach B, trainer-side) | `_compute_per_capture_mode_recall_fpr` + `_load_capture_mode_lookup` + 9 TDD tests. **NOT in image yet** — disabled by default; activates with `mid_eval_capture_mode_parquet:` config key |
| `85f1fe5` | Add resolver-audit artifacts for enhanced-visomaster (2026-04-29) | `analysis/resolver_audit_2026-04-29/resolver_manifest.json` (1.8 MB; CSV/log were `.gitignore`-matched and remain local-only, regenerable) |

**Tactical analysis runs this session (outputs uncommitted, in `analysis/`):**

- `analysis/policy_reruns_2026-04-29_floor_0p70/` — full P8A + P14_FT rescore under floor=0.70
- `analysis/face_size_leak_scatter_2026-04-29/` — 10 PNGs + per-bucket verdicts CSV showing strong leak in BOTH classes (lockbox r=−0.565 / −0.529)
- `analysis/resolver_audit_2026-04-29/` — refreshed companion-bucket resolver (committed JSON, local-only CSV/log)

**Beware: "Move N" naming collision.** The previous session used "Move 1..5" for the local pre-overnight punch list (rescore, commit, inventory, observability, scatter). The master plan independently uses "Move 1, 1.5, 4" for diagnostic experiments — these are **different**. See §5 below.

---

## 2. Current image and code state

| Item | Value |
|---|---|
| Branch | `teams-relaunch-root-2026-04-17` |
| HEAD | `85f1fe5` (Add resolver-audit artifacts) |
| VERSION | `1.3.231` |
| Production image | `effort-detector:1.3.231` (Cloud Build #2, contains Block A + Block B) |
| Pending image (NOT built) | `1.3.232` would add per-capture-mode panels (commit `528dc6e`) |
| Working-tree dirty? | Yes — pre-existing unrelated edits (CLAUDE.md, HANDOFF.md, `arena/*.yaml`, etc.) — not from this session, not in scope |

**Layer 1 (contract scorecard) is operationally closed:** recall floor 0.70 is the scorer + wrapper default; `1.3.231` carries it. Two regression-guard tests prevent silent-default-drift.

---

## 3. Tonight's slate as currently recommended

From `PRE_OVERNIGHT_VERDICTS_2026-04-29.md` §3, ~$210 total:

| Slot | YAML | State of YAML | Status of recommendation |
|---|---|---|---|
| 1 | **P14_DATA_FIX (Hybrid)** — Path-B union + Path-A weighted | `experiments/phase2_round13/R13_P14_DATA_FIX.yaml` exists; currently Path-B union only (`visomaster_enhanced` + `visomaster_hints_teams` families) | Hybrid wiring pending — Path-A's 54 strict-conjunction sample_ids weighted higher is a small additive change |
| 2 | **P8A + GRL** | `R13_P15_GRL_FROM_P8A.yaml` exists; uses GRL on **`quality_domain` head** (4-class) | **Conflict.** Verdicts handoff recommended GRL on `capture_mode` (webcam/screen/normal_photo). Decide which cut |
| 3 | **P8A + face-scale-jitter** | **No YAML exists.** Fork P14_FT_FROM_P8A and add `face_scale_jitter` block | Needs authoring + `train_sweep.py` allowlist update (~lines 176–282) per `project_wandb_flattens_nested_dicts.md` |

**Image to launch against:** `effort-detector:1.3.231` is sufficient — Block A + Block B baked in. `1.3.232` (per-capture-mode panels) is only load-bearing if Slot 2 changes to GRL-on-`capture_mode` (then per-mode panels become the live monitor for whether GRL is working).

---

## 4. Pre-launch checklist (per master plan §12.4 + handoff §3)

- [ ] Working-tree-diff check before any commit (`AGENTS.md:50-58`)
- [ ] Image-currency check before launch (`scripts/launch/check_image_currency.sh`)
- [ ] WANDB env vars exported: `WANDB_API_KEY` / `WANDB_ENTITY=dtect-vision` / `WANDB_PROJECT=phase2-experiments` (memory `feedback_promotion_contract_launch.md`)
- [ ] `train_sweep.py` re-apply allowlist updated for any new nested yaml block (memory `project_wandb_flattens_nested_dicts.md`) — **face-scale-jitter is the suspect block**
- [ ] Identity-disjoint group splits (groups by `identity, capture_session, source_bucket`) — master plan flagged as launch-blocker round-2 review
- [ ] Image tag in each YAML matches `1.3.231` (or `1.3.232` if rebuilt)
- [ ] Smoke-test each YAML config-only / dry-run before Vertex submission
- [ ] US regions only: `us-west4` ↔ `us-east1` ↔ `us-central1`; 30-min pending threshold for region switch (`CLAUDE.md`)
- [ ] **Explicit user authorization before any Vertex launch** (memory `feedback_no_cancelling_vertex_jobs.md`)

---

## 5. ⚠️ High-value addition candidates for the rethink pass

The user explicitly asked for "a final tightening and rethinking of the experiments, and see if there is potentially something of high value that we want to add for tonight." Below are the candidates worth surfacing — not directives, **decision points** for the user.

### 5.1. Master-plan Move 1 (frozen-feature linear probe) is NOT YET EXECUTED

Per master plan §1 + §11.1:

> **Move 1 — frozen-feature linear probe** of P8A on eval bucket vs training bucket viso. ~$0 / ~1-2 hours. The pre-spend gate that confirms or downgrades the bucket-gap finding before committing to P14_DATA_FIX. **Not yet executed as of 2026-04-29.**

This is **distinct from this session's "Move 1"** (the floor=0.70 rescore). The master plan's Move 1 is a diagnostic linear-probe experiment that gates whether the bucket-gap mechanism is real before spending $70 on P14_DATA_FIX.

The master plan calls it "the cheapest decisive step on the largest pending uncertainty in the program."

**There is an existing probe-battery scaffold at `analysis/probe_battery_2026-04-26/run_linear_probes.py`** that may be reusable. Verify whether it covers the master-plan Move 1 design or if it's a different probe family.

**If the master-plan Move 1 has not been run**, it is the strongest candidate for "high-value addition before tonight" — it's $0 and may either (a) green-light P14_DATA_FIX with confidence, or (b) downgrade the bucket-gap finding and shift cost to P15 GRL.

### 5.2. GRL cut: `quality_domain` vs `capture_mode`

Existing YAML uses `quality_domain` head (4-class, hidden_dim=128). Verdicts handoff recommended GRL on `capture_mode` (webcam/screen/normal_photo/phone_screen).

**Tradeoff:**

- **`quality_domain`** — already wired, zero authoring. Removes a coarser quality-axis signal. Less aligned with the camera-signature shortcut hypothesis (memory `project_signature_shortcut_finding.md`).
- **`capture_mode`** — directly attacks the load-bearing FPR axis (memory `project_lockbox_fpr_dominated_by_webcam_mode.md`: webcam = 65.7% FPR vs <1% elsewhere). Needs head wiring + label flow at training time. Couples cleanly with the new per-capture-mode mid-eval panels (commit `528dc6e`) for live-monitoring.

If Slot 2 stays `quality_domain`, image rebuild is cosmetic. If Slot 2 changes to `capture_mode`, image rebuild becomes load-bearing — per-capture-mode panels become the GRL-effectiveness monitor.

### 5.3. Move 4 (paired-transport contrastive) — sequenced after but cheap

Master plan §12.3:

> Move 4 (paired-transport contrastive) sits below P14_DATA_FIX and P15 in the canonical priority sequence. It is the data-axis lever that becomes the next obvious move if P14_DATA_FIX delivers the substrate-matched ceiling but the residual shortcut at Axis 3 still fires modern_v2 FPR > 5%.

Not for tonight, but worth surfacing if the user is considering a 4th leg.

### 5.4. Resolver pre-flight is fresh

The resolver was re-audited tonight (commit `85f1fe5`). Headline confirmed: **Path-A pool capped at 54** sample_ids; 943/999 are clean-companion-only (manifest mis-claim, not data loss; past R13 training was unaffected because `discover_visomaster_enhanced_samples` always pulls reals from `original_bucket=clean`).

**Implication:** Slot 1 cannot be Path-A standalone. The current YAML (Path-B union) is correct as a base. "Hybrid" = Path-B union + Path-A 54 weighted higher; consider whether the marginal weighting is worth wiring vs simpler "Path-B union only."

### 5.5. P14_FT verdict status check before P15 launch

Master plan §12.3:

> P14_FT verdict drives P15 launch independent of Move 1. P14_FT is in flight at end of Slice 7; its verdict will land within ~24h regardless of Move 1 status. The α/β/γ matrix at `P15.md:60-66` is the canonical decision rule.

`exp-R13_P14_FT_FROM_P8A-20260429-101008` succeeded today on us-west4 (08:10 UTC). The scorecard `p14-ft-from-p8a-scorecard-20260429-uswest4` succeeded at 10:55 UTC. **Verify which of α/β/γ the verdict actually came in at** before approving P15 launch — that's the canonical decision rule, not a recommendation.

### 5.6. Contract policy round-2 finding (master plan changelog)

Master plan §0 changelog flagged that contract v3 must ship with safe defaults *or* a CI guard "as part of the same landing, not as follow-up. Update every launch wrapper." Block A landed safe defaults (recall_floor 0.70, FPR budgets 0.02/0.05). **Verify all launch wrappers actually use the new defaults** — not just the scorer/wrapper CLI tested.

---

## 6. Resume instructions for the next agent

1. **Read this handoff in full.** Then skim `MASTER_PLAN_2026-04-29.md` §11 (Open questions) + §12 (Decision tree) + §0 changelog.
2. **Verify state hasn't drifted** — git log/status, recent Vertex jobs (us-west4, us-east1, us-central1).
3. **Ask the user the rethink questions** in §5 above, in order:
   - 5.1: Run master-plan Move 1 tonight ($0, ~1-2h)? (Highest expected value of the candidates.)
   - 5.2: GRL on `quality_domain` (existing) vs `capture_mode` (recommended, couples with new panels)?
   - 5.5: What did P14_FT come in at — α / β / γ? (Drives whether P15 launches at all.)
   - 5.3: Add Move 4 as a 4th leg, or hold for next round?
4. **Once decisions are made:** draft the actual slate YAMLs (Slot 1 Hybrid wiring if approved; Slot 2 with chosen GRL cut; Slot 3 face-scale-jitter from scratch), update `train_sweep.py` allowlist if any new nested blocks, run smoke-tests.
5. **Decide image rebuild:** `1.3.231` is sufficient for the slate as currently designed. `1.3.232` (Cloud Build #3) only required if Slot 2 → `capture_mode` GRL.
6. **Get explicit Vertex-launch authorization** for each job, in each US region, per `feedback_no_cancelling_vertex_jobs.md`.
7. **Write a successor handoff** post-launch (`OVERNIGHT_LAUNCH_LIVE_2026-04-30.md` or similar) with: launched slate, Vertex job IDs, W&B run URLs, live monitoring instructions, early-warning thresholds for per-bucket panels.

---

## 7. Pitfalls / things not to redo

- **Do not re-run the resolver audit** — fresh tonight (commit `85f1fe5`). Path-A is 54.
- **Do not re-run the floor=0.70 rescore** — done; outputs in `analysis/policy_reruns_2026-04-29_floor_0p70/`.
- **Do not re-run the face-size leak scatter** — done; outputs in `analysis/face_size_leak_scatter_2026-04-29/`.
- **Do not commit-amend** — codebase convention is new commits only (`feedback_no_cancelling_vertex_jobs.md` + general code-cleanliness).
- **Do not bypass the WANDB env vars** for any contract-scorecard launch — see `feedback_promotion_contract_launch.md`.
- **Do not use n_jobs=-1 in sklearn/joblib on the user's Mac** (memory `feedback_sklearn_njobs.md`) — caused 3 reboots from swap exhaustion on 2026-04-27.
- **Do not cancel any in-flight Vertex job** without explicit user OK (memory `feedback_no_cancelling_vertex_jobs.md`).

---

## 8. Cross-references

- Master plan: `docs/packet_retrospectives/plans/MASTER_PLAN_2026-04-29.md`
- Compressed: `PLAN.md`
- Canonical wiki: `docs/packet_retrospectives/`
- Predecessor handoffs (chain order):
  - `docs/relaunch_handoffs/BLOCK_A_LANDED_PRE_OVERNIGHT_2026-04-29.md`
  - `docs/relaunch_handoffs/PRE_OVERNIGHT_VERDICTS_2026-04-29.md`
- Memory anchors most relevant to tonight:
  - `project_contract_policy_bug.md` — Layer 1 mechanism
  - `project_viso_train_eval_bucket_gap.md` — Layer 2 mechanism
  - `project_signature_shortcut_finding.md` + `project_p8a_breakthrough.md` — Layer 3 mechanism
  - `project_lockbox_fpr_dominated_by_webcam_mode.md` — capture-mode FPR
  - `project_face_size_label_leak.md` — face-scale-jitter motivation
  - `project_eval_production_crop_tightness_gap.md` — substrate caveat upstream of all the above
  - `project_wandb_flattens_nested_dicts.md` — yaml-flattening pitfall
  - `feedback_no_cancelling_vertex_jobs.md` — launch-policy discipline
  - `feedback_promotion_contract_launch.md` — env-var setup
  - `feedback_sklearn_njobs.md` — local-Mac discipline
