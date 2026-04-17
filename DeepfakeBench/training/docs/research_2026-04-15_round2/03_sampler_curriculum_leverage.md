# Sampler And Curriculum Leverage

## 1. Runtime behavior that matters

The sampler in `data/sources/combined_paired.py` does this:

- group paired samples by identity
- draw exactly one paired sample per identity per epoch
- if `identity_sampling_strategy == identity_resample_weighted`, weight the per-identity draw by fake-family weight

That creates an immediate constraint:

> Family weights only matter on identities that have multiple candidate paired samples.

If an identity has only one candidate paired sample, the configured weight is cosmetic.

## 2. Baseline merged-source lane (`R13_FT7`)

### Exact train-pool structure

- train fake candidates: `13294`
- train identities: `6716`
- single-candidate identities: `3618`
- multi-candidate identities: `3098`
- multi-candidate but single-family: `3017`
- true cross-family identities: `81`

Cross-family competition is concentrated in only three patterns:

| Cross-family composition | Identity count |
| --- | ---: |
| `deeplive_enhanced_fake` vs `deeplive_non_enhanced_fake` | 35 |
| `deeplive_non_enhanced_fake` vs `visomaster_fake` | 4 |
| `deeplive_teams_fake` vs `visomaster_enhanced_fake` | 42 |

That last row is the most important one for this project.

### Expected selected objects per epoch

Using the actual configured family weights:

| Fake family | Expected selections / epoch | Share |
| --- | ---: | ---: |
| `visomaster_fake` | `2316.58` | `34.49%` |
| `deeplive_non_enhanced_fake` | `1396.33` | `20.79%` |
| `deeplive_teams_fake` | `1117.71` | `16.64%` |
| `visomaster_enhanced_fake` | `812.29` | `12.09%` |
| `df40_fake` | `826.00` | `12.30%` |
| `deeplive_enhanced_fake` | `247.09` | `3.68%` |

### What the merged VTE family actually contributes

Expected VTE selections per epoch:

- total VTE selections: `812.29`
- of those, true Teams-companion VTE selections: `17.29`
- clean-fallback VTE selections: `795.00`

With `p_original = 0.5` in `FT7`, expected branch exposure is:

- Teams original fake from VTE: `8.65`
- clean original fake from VTE: `397.50`
- clean enhanced fake from VTE: `406.15`

This is the key reality check. The merged source looks large by raw row count, but the actual true Teams-companion contribution is tiny.

## 3. Teamsonly lane (`R13_FT15`) after correcting runtime split behavior

Round 1 understated one detail: teamsonly filtering happens before the split, so it slightly perturbs the entire train pool.

### Exact `FT15` train-pool structure

- train fake candidates: `12490`
- train identities: `5915`
- single-candidate identities: `2816`
- multi-candidate identities: `3099`
- multi-candidate but single-family: `3011`
- true cross-family identities: `88`

Cross-family compositions:

| Cross-family composition | Identity count |
| --- | ---: |
| `deeplive_enhanced_fake` vs `deeplive_non_enhanced_fake` | 35 |
| `deeplive_non_enhanced_fake` vs `visomaster_fake` | 4 |
| `deeplive_teams_fake` vs `visomaster_enhanced_fake` | 49 |

### Expected selected objects per epoch in `FT15`

| Fake family | Expected selections / epoch | Share |
| --- | ---: | ---: |
| `visomaster_fake` | `2309.58` | `39.05%` |
| `deeplive_non_enhanced_fake` | `1392.33` | `23.54%` |
| `deeplive_teams_fake` | `1120.82` | `18.95%` |
| `df40_fake` | `826.00` | `13.96%` |
| `deeplive_enhanced_fake` | `246.09` | `4.16%` |
| `visomaster_enhanced_fake` | `20.18` | `0.34%` |

Expected VTE branch exposure with `p_original = 0.5`:

- Teams original fake from VTE: `10.09`
- clean enhanced fake from VTE: `10.09`

So teamsonly helps less than its name implies. It does not create a strong late lesson on true Teams-companion merged data.

## 4. `p_original` changes branch mix, not candidate competition

`FT9` and `FT10` keep the same candidate competition as `FT7`; they only change branch mix inside VTE.

With `p_original = 0.7`:

- expected VTE total stays `812.29`
- expected Teams original fake from VTE rises to `12.11`
- expected clean original fake from VTE rises to `556.50`
- expected clean enhanced fake from VTE falls to `243.69`

Likewise for `R13_I` teamsonly:

- expected VTE total: `20.18`
- expected Teams original fake: `14.12`
- expected clean enhanced fake: `6.05`

This is still a small intervention.

## 5. Where weights are real and where they are cosmetic

### Real leverage

- the `42`-identity baseline overlap between `deeplive_teams_fake` and `visomaster_enhanced_fake`
- the `49`-identity overlap in the corrected `FT15` teamsonly split
- the `35` enhanced-vs-non-enhanced DeepLive overlap

### Mostly cosmetic

- any family weight applied to identities with exactly one candidate paired sample
- `FT8` relative to `FT7` if no external unpaired reals loaded
- `FT10` relative to `FT9` under the same condition

## 6. Realboost is probably not doing what the label implies

Under local truth:

- `external_reals.json` is empty
- no external unpaired real pool was discovered

Therefore:

- the real-side weight changes in `FT8`, `FT10`, and `FT16` are likely near-cosmetic
- any claim that these runs materially differed because of "realboost" needs proof that external unpaired reals actually loaded in the real runtime

## 7. One more effect teamsonly does have

Even though its merged Teams exposure is weak, teamsonly does shorten the epoch.

- `FT7`: `6716` paired identities -> about `3358` steps/epoch
- `FT15`: `5915` paired identities -> about `2957.5` steps/epoch

With a fixed total step budget, that means the remaining identities repeat more often. That is a real effect. It is just not the same as materially increasing true Teams-companion coverage.

## 8. Concrete curriculum/sampler designs worth trying

### Design A: split VTE into two explicit training families

Create separate families for:

- `visomaster_teams_true_fake`
- `visomaster_clean_fallback_fake`

What this changes in practice:

- makes the true Teams-companion subset measurable and schedulable
- prevents the `795` clean-fallback pairs from hiding inside one merged family name
- allows a late stage that keeps only the true Teams-companion branch

### Design B: late target-domain curriculum, not more weight tuning

Use a broad initializer, then a short late stage that removes:

- `df40_fake`
- `visomaster_fake`
- clean-fallback VTE

and keeps:

- `deeplive_teams_fake`
- `visomaster_teams_true_fake`
- whichever enhanced-clean family is still needed for stress coverage

What this changes in practice:

- changes actual epoch exposure, not just YAML intent
- turns teamsonly from a naming convention into a real late curriculum

### Design C: sampler redesign for overlapping identities

For identities that have both direct Teams and VTE candidates, draw one sample per identity **per family** or allow a capped two-draw rule.

What this changes in practice:

- removes the current `5.0` vs `3.5` winner-take-most behavior on the small overlap set
- can move true Teams-companion VTE exposure from about `17-20` toward the full `42-49` available identities

## 9. What should not be done next

- Do not spend another round on family-weight tweaking inside the current sampler and expect large target-domain change.
- Do not treat `teamsonly` as a strong target-domain curriculum unless the late-stage source composition actually changes.
- Do not treat `realboost` lanes as distinct scientific evidence until the external-real load path is proven.

## 10. Status

- Established:
  - current sampler leverage is concentrated in a small overlap set
  - merged VTE raw size overstates true Teams-companion exposure
  - teamsonly remains weak even after correcting the split story
- Plausible:
  - a short late curriculum or a per-family identity sampler would matter more than more weight tuning
- Still unknown:
  - whether a redesigned sampler can improve Teams safety without sacrificing too much enhanced-clean recall
