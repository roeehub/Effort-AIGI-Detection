# Thread: Identity diversity audit (the "N too small" hypothesis is refuted)

> **Slice 7 finding (2026-04-29)**: hard counts from local discovery caches show that active training contains ~430 viso identities + ~1,900 deeplive identities, and another ~2,300 identities sit in unused buckets — including the bucket whose distribution matches the eval suite. The identity-axis hypothesis is structurally refuted; the dominant explanation for cross-domain failure is the bucket gap, not data narrowness. Memory `project_data_inventory_identity_diversity.md` is the auto-memory anchor.

## The question

Does the training data have enough identity diversity to support cross-domain generalization on the viso / deeplive / teams_fake suites? Recipe-tuning agents had intermittently floated "N is too small" — e.g. "we have only N=8 identities per fake method" — as a possible explanation for persistent cross-domain failures. Is that hypothesis empirically supported, or is the substrate adequate at the identity axis?

## Initial belief

Through Slices 4-6 the team had a sense of the data inventory but no quantitative count. The shortcut-finding session (2026-04-23, [`processing_signature_shortcut`](processing_signature_shortcut.md)) had focused on per-pipeline artifacts (camera/ISP, codec, face-pixel-area) rather than on identity diversity at the per-method level. The implicit assumption was that the existing buckets had "tens" of identities per method — a prior narrow enough that "more identity diversity might fix it" was on the table as a fallback for any cross-domain failure. The hypothesis was never tested rigorously because no packet's design specifically isolated the identity axis.

## What changed our mind

- **2026-04-29 ~10:00 CEST — Bucket audit triggered by P13_FROM_SCRATCH γ verdict** (`docs/relaunch_handoffs/TRAINING_EVAL_VISO_BUCKET_GAP_2026-04-29.md` § 1; `april-26-training-master-plan-v2.LOG.md:1395-1406`). User pushed back on the agent's "training data may be too narrow at the identity axis" line of reasoning: *"I should have at least 200 visomaster and 200 deeplivecam identities; check the buckets."* The agent ran hard counts from local discovery caches `.viewer_cache/discovery/*.json` (last refreshed 2026-04-17) plus gsutil ground-truth where available. The numbers (memory `project_data_inventory_identity_diversity.md`):

  | Bucket | Source | Samples | Unique identities | In P13 training? |
  |---|---|---:|---:|:---:|
  | `live-deepfake-methods-real-and-fake-frames-cropped` | viso (live, no enhancers) | 960 | **429** | ✅ |
  | `live-deepfake-methods-real-and-fake-frames-cropped` | deeplive (3 base + 2 enhanced strategies) | ~5,800 cache / 2,040 top-level dirs | **1,916** | ✅ |
  | `live-...-teams` | teams (deeplive_teams_* + viso_hints_teams) | 2,626 | 1,306 | ✅ partial |
  | `live-...-teams-v2` | visomaster_teams_v2_companion (9 swap × 8 enhancers) | 1,994 | **997** | ❌ |
  | `visomaster-enhanced-face-cropped-v2` | viso enhanced (direct) | 2,073 | many (manifest-hashed) | ❌ |
  | `hdtf_visomaster_cropped_frames` + `_teams` | proper-data wave 2026-04-19 (clean+teams pairs) | 1,322 captures | **705** | ❌ |

- **Conclusion is structural.** Active training has **~430 viso + ~1,900 deeplive identities** — not the "N=8 per method" mental model that had been on the table as an alternative explanation. **+2,300 additional identities sit in unused buckets** including the `visomaster_teams_v2_companion` bucket (the closest match to the eval distribution by transport pipeline) and the 2026-04-19 proper-data wave (purpose-built same-identity-clean-and-teams pairs).
- **The identity hypothesis is structurally refuted at the magnitude that was floated**, but the question reframes rather than disappears: the agents who proposed "more identity diversity" were not entirely wrong — they were detecting the right surface (the wrong distribution) but pinning it on the wrong axis (identity rather than pipeline). The *correct* axis is the bucket transport pipeline (clean vs Teams-recapture, no-enhancer vs enhanced), not the count of unique faces per method. Memory `project_clean_teams_same_identity.md` (also Slice 7) makes the structural argument: the same identity recaptured through Teams is a fundamentally different rendering, and the eval suite reads from the Teams-recapture bucket. See [`viso_bucket_gap`](viso_bucket_gap.md).
- **Caveat: counts are from a 2026-04-17 cache.** The deeplive top-level prefix count via gsutil on 2026-04-29 was 870 + 850 + 320 = 2,040 directories for the 3 base strategies; the cache reported 5,800 entries, suggesting the cache double-counts paired (real, fake) records or the bucket has shrunk since. Either way the order of magnitude (1,000s of identities, not 10s) holds — the conclusion is robust to the audit's precision floor.

## Current stance (2026-04-29)

Identity diversity is **adequate**. The hypothesis "N is too small per method" is empirically refuted at the claimed magnitude — active training has 100s-1000s of identities per major method, and another 2,300+ identities sit in unused but already-collected buckets. **The recipe-tuning effort to lift cross-domain recall by adding identity diversity (or by aggressive sampling weights, which P11_HEAVY_DEEPLIVE tried at 6× and saw a regression) is mis-aimed.** The right axis is *pipeline transport* — clean vs Teams-recapture, no-enhancer vs enhanced — not raw identity count. See [`viso_bucket_gap`](viso_bucket_gap.md) for the dispositive distribution-mismatch finding that replaces the identity-axis hypothesis.

This thread is **load-bearing for the wiki** because it explicitly closes off a class of "more data" recommendations that an outside reader (or a future agent) might naturally propose. Without this thread, an agent looking at viso recall ~1.6% (under the buggy contract policy) would plausibly recommend "add more viso fake identities" — and that recommendation would burn cycles re-deriving the bucket-gap finding. The thread exists to short-circuit that cycle.

## Packet timeline

- *(none — this thread captures a meta-finding, not a packet's results)*. The audit was triggered by the P13_FROM_SCRATCH γ verdict (see [P13](../packets/P13.md)) but does not belong to any packet's experimental scope. It informs [P14](../packets/P14.md) (the data-fix variant explicitly uses the unused buckets) and [P15](../packets/P15.md) (GRL is reframed as still-useful-but-secondary because the identity diversity for adversarial training was always there; it was the bucket transport that was missing).

## Evidence locations

- `docs/relaunch_handoffs/TRAINING_EVAL_VISO_BUCKET_GAP_2026-04-29.md` § 1 — the dispositive identity-count table.
- `april-26-training-master-plan-v2.LOG.md:1395-1406` — session entry where the audit was run.
- `.viewer_cache/discovery/*.json` — local discovery caches (last refreshed 2026-04-17). The substrate for the per-bucket identity counts.
- `arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml` — the 2026-04-19 proper-data wave inventory (705 identities, paired clean+teams).
- `arena/manifests/visomaster_enhanced_v2_manifest_2026-04-13.json` — 2,073 frames in the unused enhanced viso bucket (many identities, manifest-hashed).
- Memory: `project_data_inventory_identity_diversity.md` (auto-memory anchor with the per-bucket counts and the "+2,300 unused identities" framing); `project_clean_teams_same_identity.md` (the structural argument that the unused buckets are the *same identities* in a different transport, making them load-bearing for the bucket-gap fix); `project_viso_train_eval_bucket_gap.md` (the dispositive finding that replaces the identity-axis hypothesis with the pipeline-axis hypothesis).

## Open loops

*(none — this thread is a closed-by-evidence meta-finding. The "more identity diversity" hypothesis is refuted; the corollary action item — adding the unused buckets to training — is tracked under [`viso_bucket_gap`](viso_bucket_gap.md) `p14-data-fix-not-launched`. No standalone open loop is needed here.)*

### Cross-thread refs

- [`viso_bucket_gap`](viso_bucket_gap.md) — the *replacement* hypothesis: it's not identity diversity, it's pipeline transport. The unused buckets enumerated here are the substrate for that thread's Layer-2 closure.
- [`clean_teams_identity_pairing`](clean_teams_identity_pairing.md) — the structural argument for why "+2,300 unused identities" is misleading on its own. Many of those identities are the *same people* as in the existing buckets, just recaptured through Teams. So the gain from wiring them in is not "more faces" but "more transports of the same faces" — which is the exact ground truth signal for "what should look the same vs different to a deepfake detector."
- [`processing_signature_shortcut`](processing_signature_shortcut.md) — the shortcut framing was load-bearing through Slice 6; this thread does not refute it but does recontextualize it. The shortcut's Axis 3 Δ stays at ~0.5 even on best P13; the cross-domain headline metric is dominated by the bucket gap; both layered effects exist.
- [`promotion_contract_evolution`](promotion_contract_evolution.md) — once the contract-policy fix and the bucket-fix both land, the contract scorecard will read out the cross-domain recall correctly and identity diversity will not be on the table as a fallback.
