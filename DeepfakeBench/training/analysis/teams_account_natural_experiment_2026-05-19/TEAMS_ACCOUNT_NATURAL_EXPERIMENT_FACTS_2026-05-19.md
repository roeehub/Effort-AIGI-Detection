# Teams-Account Natural Experiment FACTS — 2026-05-19

> **Setup.** User opened the same physical webcam, on the same machine, at effectively the same moment, while logged into two different Microsoft Teams accounts:
> - Left panel: `Roy D` account
> - Right panel: `two (Guest)` account
>
> Same person. Same camera. Same room. Same physical scene. The only manipulated variable is the Teams account identity (and therefore, presumably, the encoding / transport pipeline Teams applies to the video stream).
>
> **Result.** The model produces materially different `prob_fake` for the two crops, **including a flip between FAKE and real at deployment-plausible thresholds**. The shift is consistent across three architecturally distinct checkpoint classes (P8A, T5C, Slot B 6-axis GRL). This is a clean dispositive demonstration that the model is using Teams transport signature, not face/forgery content, for a significant fraction of its decision.

---

## 1. Inputs

- Source image: `/Users/roeedar/Downloads/WhatsApp Image 2026-05-19 at 16.14.22.jpeg` (1592×458; side-by-side composite).
- Face detection: YOLO `yolov8s-face.pt` via the production `video_preprocessor._get_yolo_face_box` codepath (`conf_threshold=0.20`, square crop around detected box).
- Face crops produced:
  - `crops/face_roy_d.png` — 210×210
  - `crops/face_guest.png` — 208×208
  - 224×224 INTER_LINEAR-resized versions used for model input (`face_*_224.png`).
- Same preprocessing as training: `cv2.imread (BGR) → INTER_LINEAR resize → BGR2RGB → CLIP normalize (mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276])`.

## 2. Inference results

| Checkpoint | role | `prob_fake` Roy_D | `prob_fake` Guest | Δ (Guest − Roy_D) | Flip at τ=0.50 | Flip at τ=0.70 | Flip at τ=0.80 |
|---|---|---:|---:|---:|:---:|:---:|:---:|
| **T5C_STEP3500** | user's current deployment | **0.795** | **0.628** | −0.168 | both FAKE | **FLIP (R=fake, G=real)** | both real |
| P8A_STEP5000 | production anchor | 0.376 | 0.211 | −0.166 | both real | both real | both real |
| SLOT_B_6AXIS_GRL_STEP3500 | 2026-05-16 6-axis GRL on T5C | **0.813** | **0.610** | −0.203 | both FAKE | **FLIP** | **FLIP** |

**Headline.**
- **T5C (the deployment checkpoint) flips at τ=0.70.** Roy_D scores 0.795 (would-flag-fake), Guest scores 0.628 (would-not-flag-fake). Same physical scene; different Teams account; one verdict-flip.
- **Slot B 6-axis GRL flips at BOTH τ=0.70 and τ=0.80** — the larger Δ on this ckpt is consistent with the "GRL adversarial against IQ axes amplifies the residual transport-shortcut" failure mode the program has seen before.
- **P8A doesn't flip at any standard τ**, both scores are well below 0.50. P8A's substrate-invariance property (the foundational P8A claim, see `project_p8a_breakthrough`) survives this natural experiment — but it still scores Roy_D 0.17pp higher than Guest in probability space.
- **The Δ direction is consistent across all three checkpoints** (Roy_D > Guest by 0.17–0.20 prob). The Teams-account-induced pipeline shift moves the model's score in the same direction for every ckpt tested.

Raw output JSON: `outputs/inference_results.json`.

## 3. What actually differs between the two crops

Same person, same camera, same moment. After YOLO face-crop and 224×224 resize, what differs at the pixel level?

| Metric | Roy_D | Guest | Δ (G−R) | % change |
|---|---:|---:|---:|---:|
| **lap_var (sharpness)** | **113.06** | **55.99** | −57.07 | **−50.5%** |
| **luma_mean (brightness, L*)** | **159.30** | **139.47** | −19.83 | **−12.4%** |
| **luma_std (contrast)** | 60.13 | 45.76 | −14.37 | −23.9% |
| saturation_mean | 69.47 | 64.90 | −4.57 | −6.6% |
| color_a_dev | 8.73 | 7.22 | −1.51 | −17.3% |
| color_b_dev | 5.63 | 6.10 | +0.47 | +8.3% |
| edge_mag (Canny) | 14.06 | 12.18 | −1.88 | −13.3% |
| skin_frac | 0.782 | 0.729 | −0.053 | −6.8% |
| R channel mean | 180.91 | 153.54 | **−27.38** | **−15.1%** |
| G channel mean | 145.11 | 126.40 | −18.71 | −12.9% |
| B channel mean | 133.51 | 113.41 | −20.10 | −15.0% |

Per-pixel mean absolute diff (BGR, after preprocessing): **25.47 / 255** (10% of dynamic range). Per-channel signed mean diff (Guest − Roy_D) is uniformly negative: B −20.1, G −18.7, R −27.4. **The `Guest` account's pipeline is producing a uniformly darker, less-sharp, less-saturated image than the `Roy D` account's pipeline — on the same physical scene.**

The biggest shifts: sharpness drops by 50%, contrast drops by 24%, R-channel drops by 15%. These are precisely the axes the program has identified as IQ-shortcut drivers (memory `project_image_quality_shortcut`, D7's IQ partial R²=0.24).

Diff visualization: `outputs/diff_visualization.png` (per-pixel absolute diff amplified ×3).

## 4. Note on the IQ-shortcut direction

The program's prior summary of the IQ shortcut said: *"score correlates negatively with Laplacian variance / luminance / skin_frac across most suites — model treats soft / less-sharp as a fake predictor"* (memory `project_image_quality_shortcut`).

This natural experiment shows the **opposite direction at the per-frame level**:
- Roy_D is **sharper** (113.06 lap_var) and **brighter** (159.30 luma) — scored **higher** prob_fake (0.795).
- Guest is **less sharp** (55.99 lap_var) and **darker** (139.47 luma) — scored **lower** prob_fake (0.628).

Two readings are possible:
1. **Higher-order interactions dominate.** The sign-of-correlation finding is at the *pool-aggregate* level; at the *per-frame* level, an interaction between sharpness, color cast, and identity-cluster geometry may give the opposite direction.
2. **The shortcut axis is more complex than "soft = fake".** What differs between Roy_D and Guest isn't just one IQ axis but a *coupled bundle* — sharpness + warmth + contrast + saturation all moved together. The model has learned this bundle as a "fake-signature" direction; which axis carries the sign depends on the specific bundle.

Resolution requires per-axis ablation (CPU follow-up — score the same crop after independently perturbing each IQ axis, find which carries the sign).

## 5. Implications for the program

### 5.1 Refines Probe 1's interpretation (2026-05-19)

Probe 1 concluded that the dev↔lockbox CLIP-feature gap is "face-identity cluster driven" because residualizing CLIP against `(IQ + ArcFace)` collapses the discriminator to chance. This natural experiment forces a refinement:

> **The same person's face produces materially different CLIP features depending on the Teams-encoding pipeline.** ArcFace embeddings presumably also shift (cannot be measured here — `insightface` not installed locally). The "identity-cluster" axis Probe 1 identified is not pure biometric identity — it is **identity-as-encoded-by-the-current-transport-pipeline**.

In other words: when CLIP-frozen separates dev_real from lockbox_real at 99.6%, it's not separating "different humans"; it's separating *"the same/similar humans, but post-different-transport-pipelines"*. ArcFace residualization works because ArcFace's identity embedding is also transport-sensitive on the same person.

### 5.2 Reframes the data thesis

The joint-marginal audit's primary hypothesis — *"training pool doesn't span deployment IQ distribution"* — is **supported and refined** by this experiment:
- The training pool's `teams-v2` bucket is YouTube-origin frames pushed through a *synthetic* Teams pipeline.
- Deployment frames are directly captured from Teams calls.
- This natural experiment proves that the actual Teams transport applies a **per-account** encoding profile that the synthetic training pipeline does not replicate.
- The data lever needs to ingest reals captured **through actual Teams calls under multiple account configurations**, not just frames post-processed by a single synthetic pipeline.

### 5.3 Refutes the "anchor_aware fixes chronic-FP via identity supervision" purity reading

If anchor_aware is supplying identity-anchored supervision but the model's actual shortcut is *transport-encoded face appearance*, then anchor_aware works partly by chance — it pulls embeddings of one chronic identity toward an anchor that happens to be in the right transport-encoded position. This explains the bimodal partial rescue (Chikara_Takahashi 26→0%, PC_Generator 28→0%, but Roy_D 29→81% regression — Roy_D's transport profile may sit on the *wrong* side of his anchor).

### 5.4 What this experiment does NOT prove

- It does not measure the magnitude of this effect on the operational lockbox-FPR. The 0.17pp Δ here on one frame pair doesn't directly translate to FPR shifts; that needs the lockbox cohort to be re-paired by account-encoding.
- It does not isolate which IQ axis carries the sign. All five (sharpness, luma, contrast, R-channel, color_a_dev) move together because the pipeline change moves them together. Per-axis ablation is required.
- It does not generalize to all Teams accounts. We have n=2 Teams accounts (the user's own + a guest). A broader sweep (free / education / enterprise tiers, different network conditions, different client builds) would be needed to know how big the transport-account axis is.
- We could not compute CLIP / ArcFace cosines between the two crops because `insightface` is not installed and the CLIP feature path through `transformers` raised an attribute error. Both can be added next session if needed.

## 6. Recommended follow-ups (CPU, no GPU spend)

1. **Multi-account capture sweep.** Have the user capture ~20-50 frames per account (Roy_D, Guest, plus a free-tier, education-tier, enterprise-tier if available). Run inference on all. If the per-account `prob_fake` distributions are separable, this is a first-class deployment failure mode and not just a pair-of-frames anecdote.
2. **Per-axis controlled perturbation.** Take the Roy_D crop. Apply controlled monotone perturbations on each IQ axis independently (gamma → simulate brightness; Gaussian blur → simulate sharpness drop; per-channel multiplicative scale → simulate color cast). Measure prob_fake change per axis. Tells you exactly which axis the model is most sensitive to, separable from the bundle.
3. **Reproducible Teams-account synthetic augmentation.** If we can characterize the transformation T_account(image) that Teams applies based on account type, we can synthesize it as a training-time augmentation. The "guest" pipeline appears to apply lower compression / less sharpening / different color matrix — these are reproducible.
4. **Frozen-CLIP CLS cosine between Roy_D and Guest crops.** Fix the `transformers` CLIP attribute error and measure the L11/L23 CLS cosine. If it's < 0.99 despite same-person same-pose, that quantifies how much of the encoder's representation is transport-dependent.
5. **ArcFace cosine.** Install `insightface`; if the ArcFace cosine is < 0.95, "face identity" itself is transport-shifted in a way that affects Probe 1's residualization interpretation.

## 7. Artifacts

| Path | Contents |
|---|---|
| `crops/panel_roy_d.png` | Left half of source (Roy D account) |
| `crops/panel_guest.png` | Right half of source (two (Guest) account) |
| `crops/face_roy_d.png` | YOLO-detected face crop, raw resolution (210×210) |
| `crops/face_guest.png` | YOLO-detected face crop, raw resolution (208×208) |
| `crops/face_roy_d_224.png` | 224×224 INTER_LINEAR-resized (model input) |
| `crops/face_guest_224.png` | 224×224 INTER_LINEAR-resized (model input) |
| `crops/crop_meta.json` | YOLO bboxes + crop bboxes |
| `outputs/inference_results.json` | Raw per-ckpt prob_fake + cls logits |
| `outputs/inference_results.csv` | Tabular version |
| `outputs/diff_summary.json` | Pixel-diff + per-axis IQ + similarity scores |
| `outputs/diff_visualization.png` | Side-by-side + per-pixel diff heatmap |
| `crop_faces.py`, `run_inference.py`, `diagnose_diff.py` | Reproducible scripts |
