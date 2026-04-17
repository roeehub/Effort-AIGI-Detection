# Current Training Diet

This report uses the local viewer running on `http://127.0.0.1:8511` for `R13_A_trackA_teams_enhanced`, plus the sampler code in `DeepfakeBench/training/viewer/server.py` and `DeepfakeBench/training/data/sources/combined_paired.py`. The important distinction is:

- The viewer's default stats are row-based: one real row and one fake row per paired object.
- The actual train loader is identity-balanced and selects one paired object per identity per epoch.
- For paired objects, selection is driven by the fake-side family, not by the real-side family.

The discovered pool contains `31,064` rows total, perfectly balanced at `15,532` real and `15,532` fake. That corresponds to `15,532` paired objects overall and `13,294` train paired objects. The configured `external_training_reals` block contributes nothing in the current viewer run: `/api/stats` shows no external source, and `DeepfakeBench/training/.viewer_cache/discovery/external_reals.json` is empty.

| Source | Total rows | Total paired objects | Train pairs | Val pairs | Test pairs | Train share of train pairs |
|---|---:|---:|---:|---:|---:|---:|
| `visomaster` | 11,178 | 5,589 | 4,735 | 594 | 260 | 35.6% |
| `df40` | 9,396 | 4,698 | 4,091 | 392 | 215 | 30.8% |
| `deeplive` | 5,800 | 2,900 | 2,496 | 281 | 123 | 18.8% |
| `deeplive_teams` | 2,696 | 1,348 | 1,135 | 130 | 83 | 8.5% |
| `visomaster_teams_enhanced` | 1,994 | 997 | 837 | 111 | 49 | 6.3% |

Train fake-side family counts, as plain paired objects, are:

| Fake family | Train paired objects |
|---|---:|
| `visomaster_fake` | 4,735 |
| `df40_fake` | 4,091 |
| `deeplive_non_enhanced_fake` | 2,221 |
| `deeplive_teams_fake` | 1,135 |
| `visomaster_enhanced_fake` | 837 |
| `deeplive_enhanced_fake` | 275 |

Method concentration is also uneven. Large train methods include `deeplive_edge_cases` (`1,114` train pairs), `deeplive_minimal_processing` (`1,107`), `simswap` (`817`), and the main clean `visomaster_*` methods (`~508-700` each). The deployment-specific tail is tiny: `deeplive_teams_visomaster_*` methods are only `25-32` train pairs each, and `deeplive_teams_unknown` is only `2`.

The viewer's row-weighted "effective sampling distribution" says the diet is:

| Viewer row-weighted effective share | Percent |
|---|---:|
| `realpool_real` | 33.10% |
| `visomaster_fake` | 21.67% |
| `visomaster_enhanced_fake` | 10.73% |
| `deeplive_teams_fake` | 10.39% |
| `deeplive_non_enhanced_fake` | 10.17% |
| `deeplive_teams_real` | 8.31% |
| `df40_real` | 3.00% |
| `deeplive_enhanced_fake` | 1.51% |
| `df40_fake` | 1.12% |

That is not the actual train-time sampler behavior. The real loader path is `_sample_family_for_sampling(...)` plus `_get_identity_balanced_samples(...)` in `DeepfakeBench/training/data/sources/combined_paired.py`. Under that logic, the actual expected paired-object selection per train epoch is:

| Actual paired-object sampler share | Expected selected objects / epoch | Percent |
|---|---:|---:|
| `visomaster_fake` | 2,316.58 | 34.49% |
| `deeplive_non_enhanced_fake` | 1,396.33 | 20.79% |
| `deeplive_teams_fake` | 1,093.00 | 16.27% |
| `visomaster_enhanced_fake` | 837.00 | 12.46% |
| `df40_fake` | 826.00 | 12.30% |
| `deeplive_enhanced_fake` | 247.09 | 3.68% |

That sampler-level table is the one that matters for training. It shows a much more VisoMaster-heavy diet than the viewer dashboard suggests, and it shows that direct Teams fakes are materially important but still not dominant.

# Hidden Imbalances And Dilution

## 1. Family weights mostly do not control the diet you think they control

The current sampler is identity-balanced. In the train split there are `6,716` identities total:

- `4,399` identities have exactly one candidate paired object. Weights cannot change those at all.
- `2,317` identities have multiple candidate paired objects.
- Of those multi-candidate identities, `2,278` are still single-family only, so weights only randomize method choice within the same family.
- Only `39` train identities are true cross-family competitions.

Those `39` cross-family cases are almost entirely:

- `35` identities with `deeplive_enhanced_fake` competing against `deeplive_non_enhanced_fake`
- `4` identities with `deeplive_non_enhanced_fake` competing against `visomaster_fake`

That means the long YAML weight vector is largely cosmetic for global domain mix. It does not really rebalance most of the pool because most identities never present the sampler with a cross-family choice.

## 2. Real-family weights are mostly inert for paired training

The config declares weights for:

- `deeplive_teams_real`
- `df40_real`
- `realpool_real`
- `external_real`

But `_sample_family_for_sampling(...)` routes paired samples by fake family. The selected paired object then emits both real and fake frames. So for the current run:

- `deeplive_teams_real`, `df40_real`, and `realpool_real` do not change paired-object selection.
- `external_real` would matter only for unpaired external reals, but none were discovered in this viewer run.

This is why the viewer shows large row-weighted real-family shares while the actual paired-object sampler never selects a `*_real` family at all.

## 3. `visomaster_teams_enhanced` creates a misleading sense of Teams coverage

This is the clearest structural dilution in the current diet.

Overall merged-source composition:

- `997` total merged pairs
- `54 / 997` are `teams_v2_companion` (`5.4%`)
- `943 / 997` are `clean_companion_only` (`94.6%`)

Train-split merged-source composition:

- `837` train pairs
- `42 / 837` are true `teams_v2_companion` (`5.0%`)
- `795 / 837` are `clean_companion_only` (`95.0%`)

So the merged source is overwhelmingly clean fallback, not true Teams.

Worse, every merged-source paired object is forced into `sampling_family_key: visomaster_enhanced_fake`. That means:

- the `42` true Teams pairs and the `795` clean fallback pairs all inherit the same target-ish family weight `3.5`
- the clean fallback majority is what actually absorbs that weight

This is the main reason the merged source can look strategically valuable in dashboards while contributing far less true Teams signal than the family name suggests.

## 4. The `p_original = 0.5` branch rule dilutes true Teams fake exposure even further

For `visomaster_teams_enhanced`, the loader chooses one fake branch at iteration time:

- `original` with probability `0.5`
- one enhancer branch with probability `0.5` if enhancers exist

All `997` merged rows have at least one available enhancer, with an average of `7.96` enhancers per sample, so this split is active almost everywhere.

Quality-domain consequence from `_iterate_visomaster_teams_enhanced_sample(...)`:

- real branch is Teams-domain only when companion status is `teams_v2_companion`
- fake branch is Teams-domain only when companion status is `teams_v2_companion` **and** branch is `original`
- enhanced branches are routed to the enhanced fake domain, not to Teams fake

Under the current train split:

- merged-source true Teams real exposure is `42` selected objects per epoch at most, or only `0.63%` of all selected train objects
- merged-source true Teams fake exposure is only about `42 * 0.5 = 21` selected objects per epoch, or `0.31%` of all selected train objects
- inside the merged source itself, only `2.5%` of selections are expected to yield true Teams-quality fake frames

Direct `deeplive_teams` contributes `1,093` selected fake-side objects per epoch. The merged source therefore adds only a very small amount of additional true Teams fake exposure on top of the direct Teams lane.

## 5. Some deployment-important subdomains are tiny or missing

- `deeplive_enhanced_fake` is only `275` train pairs and only `3.68%` of actual sampler exposure, despite enhanced deepfakes being one of the stated weaknesses.
- `deeplive_teams_quality_enhancement` is only `258` train pairs.
- The `deeplive_teams_visomaster_*` tail methods are `25-32` train pairs each, which is too small to carry much domain-specific robustness by themselves.
- In the merged source, `visomaster_InStyleSwapper256-C` and `visomaster_SimSwap512` have `0` true Teams train pairs. They exist only as clean fallback there.
- The configured external webcam reals are absent in the discovered pool, so real-side deployment diversity is lower than the YAML suggests.

# Curriculum Options

## 1. Stop treating merged VisoMaster as one target-domain family

The most important reorganization is to split `visomaster_teams_enhanced` into at least two explicit groups:

- `visomaster_teams_true`: only `teams_v2_companion`
- `visomaster_enhanced_cleanfallback`: only `clean_companion_only`

An even better split would be four groups:

- true Teams + original branch
- true Teams + enhanced branch
- clean fallback + original branch
- clean fallback + enhanced branch

That would make the hidden dilution visible and would let you weight actual deployment signal separately from clean fallback signal.

## 2. Use stage-based curricula, not more family-weight tuning

Because only `39` train identities present real cross-family competition, more weight tuning on the current sampler is unlikely to move the global diet much. A staged curriculum is more plausible:

1. Generalist stage: `visomaster` + `deeplive` + a capped amount of `df40`
2. Hard-fake stage: `deeplive_enhanced` + merged enhanced/clean fallback hard cases
3. Deployment-sharpen stage: direct `deeplive_teams` + true `teams_v2_companion` merged rows, with the clean fallback slice either removed or down-weighted hard

That would create a real late bias toward the Teams domain instead of a nominal one.

## 3. Cap or remove DF40 in late stages if deployment realism matters more than academic coverage

The current config gives `df40_fake: 0.15`, which looks tiny. In reality DF40 still lands at about `12.3%` of actual paired-object selections because its identities are mostly unique and therefore unavoidable. If DF40 is less plausible for live Teams deployment, the only believable way to reduce it late is explicit capping, explicit stage removal, or a separate late curriculum without DF40.

## 4. Promote enhanced slices into a dedicated hard lesson

The enhanced slices that matter to the stated weakness profile are still small:

- `deeplive_enhanced_fake`
- `deeplive_teams_quality_enhancement`
- merged-source enhanced branches

A dedicated enhanced lesson is plausible because the current global mix leaves enhanced exposure structurally secondary, especially once clean `visomaster` and DF40 dominate the train object count.

## 5. Recover the missing external real lane or replace it

Low-FP deployment on Teams calls depends heavily on real diversity. Right now the config claims an external VCD real lane, but the discovered pool contains none. If the data exists, fixing that path may be more valuable than further fake-family weight tweaking.

# Best Data-Organization Bets Under Time Pressure

## 1. Best immediate bet: split `visomaster_teams_enhanced` by status and stop calling the whole thing Teams

This is the highest-confidence fix because the current merged family is objectively `~95%` clean fallback in train. If you do only one organizational change, do this one.

## 2. Best short curriculum: broad training first, then a short deployment-sharpen phase

A realistic late-stage sharpen set is:

- all direct `deeplive_teams`
- only true `visomaster_teams_enhanced` companion rows
- enhanced hard slices from DeepLive and VisoMaster
- strongest real slices you trust for low-FP

Do not expect weight tuning inside the current identity sampler to create this effect by itself.

## 3. Best rebalance move: explicitly cap DF40 and probably cap clean `visomaster` in the late phase

The current train object pool is still dominated by:

- `visomaster`: `35.6%` of train paired objects
- `df40`: `30.8%`

That is far larger than their deployment importance for Teams live deepfake detection. If the late phase is meant to be deployment-specialized, those sources need explicit reduction, not just lower nominal weights.

## 4. Best merged-source policy: if you keep `p_original`, apply it asymmetrically

If true Teams fake exposure is the target, raising `p_original` globally is not enough, because `95%` of merged train rows are still clean fallback. The useful version is:

- higher `p_original` for true Teams companion rows
- lower or separate handling for clean fallback rows

Otherwise you mostly generate more clean original VisoMaster, not more Teams-like fake evidence.

## 5. Best underrepresented slices to prioritize next

If you have time to produce or reorganize only a few slices, prioritize:

- true Teams-companion merged VisoMaster rows
- enhanced DeepLive and enhanced Teams slices
- Teams-domain VisoMaster methods with near-zero coverage
- missing external webcam reals

The hard conclusion is that the current training diet is not as target-domain-focused as the config names and family weights make it look. The biggest gains under time pressure will come from making target-domain slices explicit, then staging them late, not from another round of small weight tweaks inside the current sampler.
