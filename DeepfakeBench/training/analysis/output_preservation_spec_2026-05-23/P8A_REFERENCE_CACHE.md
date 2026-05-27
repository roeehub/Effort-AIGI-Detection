# P8A reference features cache for output-preservation aux loss

Prerequisite to the B.I.1 GPU launch (output-preservation aux loss) per
`analysis/output_preservation_spec_2026-05-23/AGENT_PROPOSAL_2026-05-23.md`.

## Contents

- `p8a_substrate_pair_reference_features.npz` (59.5 MB)

Keys:
- `p8a_l11_clean` (5475, 768) — P8A L11 CLS features for clean-substrate frames
- `p8a_l11_teams` (5478, 768) — P8A L11 CLS features for teams-substrate frames
- `p8a_l8_clean` (5475, 768) — P8A L8 CLS features for clean-substrate frames
- `p8a_l8_teams` (5478, 768) — P8A L8 CLS features for teams-substrate frames
- `blob_paths_{clean,teams}` — frame paths (e.g. `samples/HDTF20260416_00000/frames/real/000123.png`)
- `pair_ids_{clean,teams}` — pair_id assignments (use to match clean ↔ teams for the same identity)
- `identity_ids_{clean,teams}` — identity labels
- `source_{clean,teams}` — capture-pipeline source tag

## Usage in trainer

```python
import numpy as np
cache = np.load('analysis/output_preservation_spec_2026-05-23/outputs/p8a_substrate_pair_reference_features.npz')

# Build path → feature lookup
ref_lookup = {}
for i, path in enumerate(cache['blob_paths_clean']):
    ref_lookup[path] = (cache['p8a_l11_clean'][i], cache['p8a_l8_clean'][i])
for i, path in enumerate(cache['blob_paths_teams']):
    ref_lookup[path] = (cache['p8a_l11_teams'][i], cache['p8a_l8_teams'][i])

# In training loop:
# for each frame_path in batch, lookup ref features and compute cosine aux loss
```

## Provenance

Features extracted by `analysis/substrate_pair_geometry_2026-05-22/_probe1.log` from
P8A_REFERENCE_STEP5000 ckpt at
`analysis/manual_canary_2026-05-20/ckpts/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth`.
Inventory at `analysis/substrate_pair_geometry_2026-05-22/inventory_manifest.csv` (1,880 identity pairs).
