# Validation + Checkpoint Audit Tasks

- [ ] Locate checkpoint saving/loading code and confirm whether metadata includes backbone details (B16/L14/etc) and ArcFace head presence.
- [ ] Inspect checkpoint contents to see if backbone/ArcFace can be inferred from state dict or config sidecar.
- [ ] Identify or implement validation data loaders for:
  - [ ] DF40 paired data with method-specific orientation (target_source vs source_target).
  - [ ] Real-only external videos from `effort-collected-data/real/external_youtube_avspeech/`.
- [ ] Validate that trained models can be reloaded with correct backbone + head and run inference on new validation data.
- [ ] Create a validation script (reuse or adapt `rerun_validation`) to compute:
  - [ ] acc / eer / f1
  - [ ] frame-level vs N-frame averaging (e.g., 8 frames)
  - [ ] optimal threshold selection per strategy
