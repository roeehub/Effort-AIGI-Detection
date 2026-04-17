# Literature Review: Augmentations and Domain Generalization for Small Frame-Based Deepfake Detection on Microsoft Teams

## Scope

This note is narrowly scoped to the current repo goal: a **small frame-based detector** that must **generalize well**, be **specifically effective on live deepfakes in Microsoft Teams calls**, and keep **real-Teams false positives low**.

The most important conclusion from the literature is simple: **compression-robust training helps, but real target-domain capture still dominates synthetic-only approaches when the deployment domain is video conferencing**. For this repo, the highest-ROI direction is not a dramatic architecture change. It is better **data construction, pairing, sampling, curriculum, and stability-oriented augmentation/losses**.

## Strong Evidence

### 1. Realistic degradation and target-domain capture matter more than benchmark accuracy

- **Video-conference conditions are a separate problem, not just "normal deepfake detection plus compression."** The recent **VCF benchmark** was built specifically for video conferencing, with multiple resolutions and H.264 compression levels. Its authors report strong degradation under conferencing conditions; one example in the paper is **X-CLIP dropping from AUC 0.804 on 1080p raw to 0.560 on 270p c40**, and they note that many settings remain below a usable real-world threshold. This is the most direct literature support for treating Teams as its own domain, not a small perturbation of FF++/Celeb-DF style benchmarks. [[VCF 2025](https://isprs-archives.copernicus.org/articles/XLVIII-2-W9-2025/169/2025/isprs-archives-XLVIII-2-W9-2025-169-2025.pdf)]

- **DeeperForensics-1.0** reached the same conclusion earlier from a broader "real-world perturbation" angle. Compared with models trained on FaceForensics++, models trained on DeeperForensics generalized much better to a hidden test set designed to be human-deceptive and perturbed; for example, the paper reports hidden-set accuracies around **74.75 to 79.25** for DeeperForensics-trained baselines, versus roughly **52 to 64** for FF++-trained counterparts. The same paper also shows that **adding distorted training variants** improves hidden-test performance, and explicitly points to two future directions: better source data quality and more diverse distortions in training. [[Jiang et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_DeeperForensics-1.0_A_Large-Scale_Dataset_for_Real-World_Face_Forgery_Detection_CVPR_2020_paper.pdf)]

- The **Assessment Framework for Deepfake Detection in Real-world Situations** is less Teams-specific, but it is one of the stronger review-style robustness papers for augmentation policy. Its main practical result is that a **stochastic degradation-based augmentation (SDAug)** built from realistic processing operations improves robustness across detectors. This supports using degradations such as resizing, compression, blur, and photometric shifts, but only when they are treated as a robustness aid rather than a substitute for target-domain data. [[Lu and Ebrahimi, 2024](https://link.springer.com/article/10.1186/s13640-024-00621-8)]

### 2. Compression-aware training works, but modestly; it is not a magic fix

- **FaceForensics++** remains a useful anchor for compression sensitivity. On the combined benchmark, the paper reports **Xception at 99.26% on raw, 95.73% on HQ, and 81.00% on LQ**, while smaller baselines degrade more. Two implications are still relevant:
  1. compression materially changes detector behavior;
  2. larger-capacity backbones usually keep more headroom under heavy degradation.  
  But this is still a benchmark-style result, not video-call reality. [[Rossler et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.pdf)]

- **QAD (Quality-Agnostic Deepfake Detection)** is stronger evidence for a training regime that a small model can actually borrow from. QAD trains one model across multiple quality levels and uses intermediate-representation alignment plus adversarial weight perturbation to improve robustness to compression. The paper reports average improvements of about **0.86 to 1.3 points across seven datasets** and shows the idea transfers to smaller backbones such as **ResNet-18/34 and EfficientNet-B0**. This is good evidence that **quality-consistency objectives** are worth trying in a compact model. [[Le and Woo, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Le_Quality-Agnostic_Deepfake_Detection_with_Intra-model_Collaborative_Learning_ICCV_2023_paper.pdf)]

### 3. Paired real/fake construction and identity control are highly relevant to low-FP generalization

- A recent cross-benchmark paper, **Deepfake Detection that Generalizes Across Benchmarks**, directly tests whether training data should be **paired**. Its result is highly relevant to this repo: models trained on datasets where each fake has its real counterpart from the same source video overfit less and validate better than models trained on unpaired splits. The authors explicitly attribute the gain to reducing shortcut learning and forcing the model to focus on manipulation artifacts instead of source/background identity cues. This is not the same thing as "family-aware sampling," but it strongly supports **source-aware and pair-aware sampling** in your pipeline. [[Yermakov et al., WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Yermakov_Deepfake_Detection_that_Generalizes_Across_Benchmarks_WACV_2026_paper.pdf)]

- **Implicit Identity Leakage** provides the mechanism behind that result. It shows that binary deepfake detectors can accidentally learn an identity-based boundary, so cross-dataset failure is not only about forgery family shift; it is also about learning "who looks fake" instead of "what is fake." For a Teams deployment with low FP requirements, this matters a lot because identity/style shortcuts easily become false positives on unseen real users. [[Dong et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Dong_Implicit_Identity_Leakage_The_Stumbling_Block_to_Improving_Deepfake_Detection_CVPR_2023_paper.pdf)]

### 4. Synthetic fake generation can improve generalization, but only when it forces the detector away from method-specific artifacts

- **Self-Blended Images (SBI)** is one of the strongest synthetic-training papers in practice. It constructs hard fake samples from pristine images and reports substantial cross-dataset gains, including **+4.90 points on DFDC** and **+11.78 points on DFDCP** over baseline in the paper's setting. The core lesson is not "use SBI specifically"; it is that **hard synthetic examples that are close to real data can improve generalization**. [[Shiohara and Yamasaki, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Shiohara_Detecting_Deepfakes_With_Self-Blended_Images_CVPR_2022_paper.pdf)]

- But SBI also states the limitation that matters for your setup: older synthetic artifacts and simple blending cues can fail on **low-quality, heavily compressed, or otherwise disturbed videos** where the artifacts become hard to recognize. That lines up with your "enhanced through Teams" failure mode.

- **Face X-Ray** is similar: strong idea, real generalization gains on unseen manipulation methods, but its strength comes from learning **blending boundary statistics**. That makes it more vulnerable when compression, blur, enhancement, denoising, or cleaner mask integration suppress those boundaries. For live Teams deepfakes, especially enhanced ones, it is better used as inspiration for "artifact-focused training" than as the central strategy. [[Li et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Face_X-Ray_for_More_General_Face_Forgery_Detection_CVPR_2020_paper.pdf)]

### 5. Small-vs-large tradeoff: regime and data matter more than chasing a bigger backbone, but capacity still helps under compression

- The literature does **not** support "small models are fine no matter what." Under compression, larger backbones keep more margin; FF++ already showed Xception holding up better than smaller baselines on LQ videos. [[Rossler et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.pdf)]

- At the same time, papers like **QAD** show that **training regime improvements transfer to compact backbones**, and recent cross-domain papers increasingly report that generalization gains come from **representation constraints, pairing, disentanglement, and augmentation policy**, not only from scaling the encoder. [[Le and Woo, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Le_Quality-Agnostic_Deepfake_Detection_with_Intra-model_Collaborative_Learning_ICCV_2023_paper.pdf)] [[Yermakov et al., WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Yermakov_Deepfake_Detection_that_Generalizes_Across_Benchmarks_WACV_2026_paper.pdf)]

- For this repo, that means: **staying on a CLIP ViT-B/16 class backbone is defensible**, but only if the training recipe becomes more target-aware and more stable. If not, a larger model will often look better simply because it has enough slack to memorize nuisance variation.

## Promising But Less Proven

### 1. Curriculum learning and hard-sample scheduling

- **CDFA** and **DFFC** are the clearest recent papers arguing that deepfake detectors should not see all samples uniformly from the first epoch. CDFA uses a **curricular dynamic forgery augmentation** policy and reports cross-dataset and cross-manipulation gains; DFFC uses **dynamic forensic hardness** based on quality and current loss, then paces training from easier to harder samples. Both are strong conceptual fits for your repo because you already observe:
  - enhanced fakes are harder,
  - lighting strongly changes behavior,
  - visually similar frames can receive unstable scores.  
  The caveat is that these approaches are **not yet deeply replicated across many labs or deployment domains**. They are credible, but not yet "settled science." [[CDFA, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11581.pdf)] [[DFFC, arXiv 2024](https://arxiv.org/abs/2410.11162)]

### 2. Latent-space augmentation instead of pixel-space fake synthesis

- **LSDA** is one of the more interesting papers for your specific failure modes. Its argument is that many RGB-space augmentations teach the model to rely on pixel-level artifacts that are later erased by **compression and blur**; latent-space augmentation instead expands the forgery manifold without depending on a visible blending cue. That is extremely relevant to your enhanced/clean-mask concern.  
  The downside is implementation complexity and the fact that the evidence is still mostly benchmark-centered rather than Teams-centered. [[Yan et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Yan_Transcending_Forgery_Specificity_with_Latent_Space_Augmentation_for_Generalizable_Deepfake_CVPR_2024_paper.pdf)]

### 3. Disentanglement / common-feature learning

- **UCF** and related domain-generalization papers argue that the detector should explicitly separate **common forgery cues** from method-specific cues. UCF is also notable because it explicitly warns that post-processing differences can break methods that over-rely on boundary or frequency signatures. This is promising for Teams, where post-processing is exactly the problem. The limitation is that many of these methods bring non-trivial architectural or training complexity. [[UCF, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Yan_UCF_Uncovering_Common_Features_for_Generalizable_Deepfake_Detection_ICCV_2023_paper.pdf)] [[Controllable Guide-Space, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Guo_Controllable_Guide-Space_for_Generalizable_Face_Forgery_Detection_ICCV_2023_paper.pdf)] [[SDIF, arXiv 2024](https://arxiv.org/abs/2403.12707)]

### 4. Test-time adaptation / test-time augmentation

- There is literature showing cross-dataset gains from **adaptive test-time augmentation** or test-time training, including RL-selected augmentations. I would classify this as promising but operationally awkward for a live Teams detector because:
  - it adds latency and complexity,
  - it can be brittle,
  - it does not directly fix false-positive stability in training.  
  It is better as a last-mile trick than a core research direction for this repo. [[Nadimpalli and Rattani, CVPRW 2022](https://openaccess.thecvf.com/content/CVPR2022W/WMF/papers/Nadimpalli_On_Improving_Cross-Dataset_Generalization_of_Deepfake_Detectors_CVPRW_2022_paper.pdf)]

## Ideas That Sound Good But Probably Waste Time

### 1. Spending a lot of time perfecting a synthetic "Teams simulator" and expecting it to replace real Teams captures

This is the most seductive trap. The literature supports synthetic degradation as a **robustness supplement**, not as a replacement for the deployment domain. VCF and DeeperForensics both show that realistic evaluation domains expose gaps that generic or synthetic perturbation training does not close. For your repo, **real enhanced-through-Teams captures are disproportionately valuable**, even if they are slow to collect. [[VCF 2025](https://isprs-archives.copernicus.org/articles/XLVIII-2-W9-2025/169/2025/isprs-archives-XLVIII-2-W9-2025-169-2025.pdf)] [[Jiang et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_DeeperForensics-1.0_A_Large-Scale_Dataset_for_Real-World_Face_Forgery_Detection_CVPR_2020_paper.pdf)]

### 2. Large architecture churn before fixing pairing, sampling, and quality consistency

The literature does not say backbone choice is irrelevant. It says **generalization failure often comes from shortcut learning, post-processing mismatch, and identity leakage**. If those are not fixed, a larger model often only memorizes them more cleanly. Your current backbone is already in a reasonable regime for trying data/loss improvements first. [[Dong et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Dong_Implicit_Identity_Leakage_The_Stumbling_Block_to_Improving_Deepfake_Detection_CVPR_2023_paper.pdf)] [[Yermakov et al., WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Yermakov_Deepfake_Detection_that_Generalizes_Across_Benchmarks_WACV_2026_paper.pdf)]

### 3. Doubling down on artifact-specific fake synthesis as the primary path for enhanced/live Teams

Face X-Ray, SBI, and related blending-based methods are useful and historically important, but they are exactly the kind of signal that **cleaner masks, enhancement, denoising, blur, and conferencing compression** tend to erase. The more your target data moves toward "enhanced, smooth, live, bandwidth-adapted," the less likely pure boundary-centric training is to carry the load. [[Li et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Face_X-Ray_for_More_General_Face_Forgery_Detection_CVPR_2020_paper.pdf)] [[Yan et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Yan_Transcending_Forgery_Specificity_with_Latent_Space_Augmentation_for_Generalizable_Deepfake_CVPR_2024_paper.pdf)]

### 4. Jumping to heavy temporal or active-defense methods for this iteration

There are good temporal and even active-probing papers in the literature, but they do not match the current constraint set:
  - you are currently frame-based,
  - model size matters,
  - you need a quick next step,
  - deployment is low-latency and low-FP.  
  Temporal models or active probing may be future directions, but they are not the best next move for this repo.

## Top 5 Literature-Informed Experiments For This Repo

### 1. Paired-source, identity-aware, family-balanced sampling

**What to do:** Build batches so that fake samples are preferentially paired with their real counterpart or nearest same-source counterpart, while balancing across:
- live-plausible swap families,
- enhanced vs non-enhanced,
- Teams-passthrough vs clean,
- identity/source buckets.

**Why:** This directly attacks shortcut learning and identity leakage, which the paired-data and identity-leakage papers identify as major causes of poor cross-domain behavior and false positives. [[Yermakov et al., WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Yermakov_Deepfake_Detection_that_Generalizes_Across_Benchmarks_WACV_2026_paper.pdf)] [[Dong et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Dong_Implicit_Identity_Leakage_The_Stumbling_Block_to_Improving_Deepfake_Detection_CVPR_2023_paper.pdf)]

**Why this is quick:** It is mostly a data-loader / sampling-policy change, not an architecture rewrite.

### 2. Teams-first curriculum instead of static sampling

**What to do:** Use a staged curriculum:
1. broad training on all useful families for coverage,
2. mid-stage increase for live-plausible methods,
3. late-stage boost for Teams real, Teams fake, enhanced-through-Teams, and current failure families.

**Why:** Curriculum and hardness-based papers suggest that detectors learn more general and stable cues when hardness is staged rather than mixed uniformly from epoch 1. This also matches your actual objective better than maximizing generic holdout AUC. [[CDFA, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11581.pdf)] [[DFFC, arXiv 2024](https://arxiv.org/abs/2410.11162)]

**Important detail:** DF40 methods that are not plausible for live manipulation should still be used early for breadth, but they should not dominate the last training phase.

### 3. Multi-quality consistency loss for the same frame

**What to do:** For a subset of training samples, create two views of the same crop:
- mild / near-clean view,
- degraded view with rescale, recompression, sharpening-or-blur, brightness/gamma shift, and crop jitter.  
Add a small **logit or feature consistency loss** on top of normal classification.

**Why:** This is the most direct literature-backed way to attack your observed score instability and quality sensitivity without making the model large. It is conceptually close to QAD and the real-world assessment work. [[Le and Woo, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Le_Quality-Agnostic_Deepfake_Detection_with_Intra-model_Collaborative_Learning_ICCV_2023_paper.pdf)] [[Lu and Ebrahimi, 2024](https://link.springer.com/article/10.1186/s13640-024-00621-8)]

**Success metric:** reduced frame-to-frame probability variance on visually similar frames, especially on `teams_real_*` and enhanced fake slices.

### 4. Hard-negative mining focused on real-Teams FPs and enhanced fake FNs

**What to do:** Mine:
- the worst **real Teams false positives**,
- the worst **enhanced deepfake false negatives**, especially enhanced-through-Teams.  
Then oversample them or apply capped extra weight only in the late phase.

**Why:** This is the repo-specific version of DFFC's hardness idea. The literature supports hardness-aware scheduling; your own deployment objective tells you exactly which errors matter most. [[DFFC, arXiv 2024](https://arxiv.org/abs/2410.11162)]

**Guardrail:** cap the weight so the model does not collapse into overfitting a tiny set of pathological frames.

### 5. Real-vs-synthetic target-domain ablation for enhanced-through-Teams

**What to do:** Run a controlled experiment with equal training budget:
- arm A: more synthetic Teams-like degradation on enhanced clean crops,
- arm B: fewer but real enhanced-through-Teams captures.  
Compare not only AUC, but also:
- `teams_real_*` FPR,
- enhanced fake recall,
- prediction stability.

**Why:** The literature strongly suggests that real target-domain capture is more valuable than a large amount of synthetic approximation once the deployment domain is highly specific. This experiment will tell you how much slow capture is worth in your exact setup instead of letting the team guess. [[VCF 2025](https://isprs-archives.copernicus.org/articles/XLVIII-2-W9-2025/169/2025/isprs-archives-XLVIII-2-W9-2025-169-2025.pdf)] [[Jiang et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_DeeperForensics-1.0_A_Large-Scale_Dataset_for_Real-World_Face_Forgery_Detection_CVPR_2020_paper.pdf)] [[Lu and Ebrahimi, 2024](https://link.springer.com/article/10.1186/s13640-024-00621-8)]

## Bottom Line

If I had to rank the next bets for this repo by expected value, I would rank them:

1. **pair-aware and identity-aware data construction**,  
2. **Teams-focused curriculum with late-stage hard mining**,  
3. **quality-consistency training for stability**,  
4. **more real enhanced-through-Teams captures**,  
5. **only then more ambitious synthetic forgery augmentation such as SBI-like or latent-space methods**.

The literature does support synthetic degradations and synthetic forgery generation, but it does **not** support trusting them as a full proxy for live Microsoft Teams behavior. For your exact deployment target, **real domain data and better sampling policy are the center of gravity**.
