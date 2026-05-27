# Pair-loss scoping — what would it take, and what's the closest existing analog

## TL;DR

The closest existing analog to our intended pair-aware loss is the `StabilityRegMixin` (`trainer/mixins/stability.py`). It already implements `KL(perturbed_logits || clean_logits.detach())` with a configurable `stability_lambda`. The only missing piece is using **actual companion frames** (real teams transport) instead of synthetic perturbations.

Implementation effort: **4-6 hours of focused dev work**. Not blocking the 72h ship, but viable as a v2 packet immediately after.

## What exists today

### Stability mixin (the template)

`trainer/mixins/stability.py` provides:
- `compute_stability_loss()` — computes `stability_lambda * KL(perturbed || clean.detach())`
- `_generate_perturbation()` — currently: Gaussian noise + crop-and-resize
- The mixin is already integrated in `trainer/trainer.py:1735-1738`:
  ```python
  stability_loss = self.compute_stability_loss(self.model, data_dict, predictions)
  losses['overall'] = losses['overall'] + stability_loss
  ```
- Comment at `stability.py:104`: "perturbations mimic the jitter introduced by codec re-encoding and face-detector bounding-box instability" — same conceptual goal

### Companion data infrastructure

`data/sources/visomaster.py` and `combined_paired.py`:
- `VisoMasterTeamsEnhancedSample` carries `companion_bucket` reference (line 748)
- `_iterate_visomaster_teams_enhanced_sample` (combined_paired.py:3067) currently picks ONE branch per iteration:
  ```python
  branch = "original"
  if available_enhancers and rng.random() >= p_original:
      branch = rng.choice(available_enhancers)
  ```
- Same identity is seen sometimes original, sometimes teams-transported, but **NEVER both in the same forward pass**

### Loss-composition pattern

The trainer's loss path is straightforward (trainer/trainer.py:1730-1742):
```python
losses = loss_fn_owner.get_losses(data_dict, predictions)        # base CE/ArcFace
losses['overall'] = losses['overall'] + stability_loss            # stability mixin
losses['overall'] = losses['overall'] + anchor_loss               # anchor-aware mixin
```

Adding pair loss = same pattern: `losses['overall'] = losses['overall'] + pair_loss`.

## What's missing

1. **Dataloader returning both branches in same item.**
   `_iterate_visomaster_teams_enhanced_sample` needs to yield BOTH `original` and `teams_transport` versions of the same fake frame in a single batch entry. Each frame entry would carry an extra key `companion_image` alongside `image`.

2. **A new mixin** `trainer/mixins/pair_consistency.py` modeled on `stability.py`:
   ```python
   class PairConsistencyMixin:
       def compute_pair_loss(self, model, data_dict, predictions):
           if self.pair_lambda <= 0:
               return torch.tensor(0.0, device=...)
           if 'companion_image' not in data_dict:
               return torch.tensor(0.0, device=...)
           # Forward companion through model
           companion_predictions = model({'image': data_dict['companion_image'], ...})
           # KL between original and companion logits
           kl = F.kl_div(
               F.log_softmax(companion_predictions['logits'], dim=-1),
               F.softmax(predictions['logits'].detach(), dim=-1),
               reduction='batchmean',
           )
           return self.pair_lambda * kl
   ```

3. **Mixin registration** in `trainer/__init__.py` and `trainer/trainer.py`:
   ```python
   pair_loss = self.compute_pair_loss(self.model, data_dict, predictions)
   losses['overall'] = losses['overall'] + pair_loss
   losses['pair'] = pair_loss.detach()
   ```

4. **Config knob** `pair_lambda: 0.0` (default OFF) in yaml; only enabled in pair-loss packets.

5. **Wiring test** to verify companion-image plumbing actually delivers both versions.

## Why it would help

The user's intuition: "we have frames from the same exact video one clean and one that passed through teams. We should be able to leverage that."

A KL-pair loss enforces that the model's *features* (or logit distribution) for the original and teams-transported versions of the SAME source frame should be similar. This:
- Forces the model to find features INVARIANT to teams transport
- Drops codec-fragile features (high-frequency content destroyed by transport)
- Keeps codec-robust features (semantic shape, color statistics, identity-stable features)
- The model learns: "this same person, same scene, viewed two ways → both fake; same answer regardless of transport"

This is structurally the right intervention for "teams-transported viso is the binding metric."

## Why it's a v2 packet, not a 72h ship packet

- Code change is non-trivial (4-6h focused work + tests)
- Risk of bugs in companion-image plumbing
- The `p_original` mechanic means the model already SEES both versions during training, just not jointly
- The simpler intervention (Packet A enable + Packet C codec aug) might get us most of the way
- Best to launch Packet A (and possibly C) first, see results, then decide if pair loss is needed

## Cost estimate

If we decide to do it post-72h:
- Dev work: 4-6h (one focused session)
- Vertex packet: ~$87 (24h training)
- Eval: ~$5

Total cost: ~$92 + 6h dev. Same as a regular packet.

## Recommendation

For 72h ship:
- **Don't block on pair loss for the 72h deadline.** Launch Packet A + Packet C-codec instead.
- **Keep pair loss as the v2 packet** if 72h ship doesn't hit teams-transported viso targets.
- The infrastructure scoping above is sufficient to start the dev work whenever it's prioritized.
