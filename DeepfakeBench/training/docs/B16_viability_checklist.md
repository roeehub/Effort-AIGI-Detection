# B16 Viability Investigation Checklist

**Created:** January 14, 2026  
**Goal:** Determine if ViT-B-16 can reach 0.95+ AUC for deepfake detection

---

## Phase 1: Capacity Ceiling Check (BLOCKING)

**Question:** Can B16 achieve 0.95+ AUC with ANY training configuration?

### Launch Commands
```bash
cd DeepfakeBench/training

# Experiment 1: Baseline (uses existing train_sweep.py)
./launch_experiment.sh B16-capacity-ceiling asia-southeast1 experiments/B16_capacity_ceiling/1_baseline_effort.yaml

# Experiments 2-4: Ceiling tests (use new train_capacity_ceiling.py)
./launch_ceiling_experiment.sh 2   # Unfreeze last 2 blocks
./launch_ceiling_experiment.sh 3   # Unfreeze last 4 blocks
./launch_ceiling_experiment.sh 4   # Full finetune (THE CEILING TEST)

# Or run all at once:
./launch_ceiling_experiment.sh all
```

### Experiments

- [ ] **1.1 Baseline: Current Effort (k=1, ~74K params)**
  - Config: `experiments/B16_capacity_ceiling/1_baseline_effort.yaml`
  - Expected: ~0.83 AUC (already have this from cosface_style)
  - Run ID: `_______________`
  - Result: Train AUC=`____` Val AUC=`____` Gap=`____`
  
- [ ] **1.2 Medium Capacity: Unfreeze last 2 transformer blocks**
  - Config: `experiments/B16_capacity_ceiling/2_unfreeze_last_2.yaml`
  - Unfreezes: blocks 10-11 + ln_post + projection + head (~25M params)
  - Run ID: `_______________`
  - Result: Train AUC=`____` Val AUC=`____` Gap=`____`
  
- [ ] **1.3 High Capacity: Unfreeze last 4 transformer blocks**
  - Config: `experiments/B16_capacity_ceiling/3_unfreeze_last_4.yaml`
  - Unfreezes: blocks 8-11 + ln_post + projection + head (~50M params)
  - Run ID: `_______________`
  - Result: Train AUC=`____` Val AUC=`____` Gap=`____`

- [ ] **1.4 Full Finetune: Unfreeze entire B16 (THE CEILING)**
  - Config: `experiments/B16_capacity_ceiling/4_full_finetune.yaml`
  - All ~86M parameters trainable
  - Run ID: `_______________`
  - Result: Train AUC=`____` Val AUC=`____` Gap=`____`

**Decision Matrix:**
| Result | Interpretation | Next Step |
|--------|----------------|-----------|
| Full finetune ≥ 0.95 | B16 CAN learn the task | Find efficient tuning method |
| Full finetune ~0.90-0.94 | B16 is close, may need help | Try capacity expansion tricks |
| Full finetune < 0.88 | B16 likely can't do this | Consider different backbone |

---

## Phase 2: L14 Validation (PARALLEL with Phase 1)

**Question:** Does L14 checkpoint actually generalize as well as metrics suggest?

- [ ] **2.1 Identity Generalization Test**
  - Dataset: 5K real videos with unseen identities
  - Metric: AUC on real-only data (should predict "real" consistently)
  - Pass criteria: >95% predicted as real
  
- [ ] **2.2 Method Generalization Test**
  - Dataset: 8 unseen fake generation methods
  - Metric: Per-method AUC, average AUC
  - Pass criteria: Average AUC >0.90 on unseen methods
  
- [ ] **2.3 Cross-Domain Test (if available)**
  - Dataset: Different compression, resolution, lighting conditions
  - Metric: AUC degradation from in-domain
  - Pass criteria: <10% AUC drop

**L14 Checkpoint Location:** `[INSERT PATH]`

---

## Phase 3: Capacity Expansion (CONDITIONAL)

**Trigger:** Only if Phase 1 shows ceiling ~0.88-0.94

- [ ] **3.1 Lower SVD Rank**
  - Try rank=600 (168 frozen, more trainable)
  - Try rank=500 (268 frozen, even more trainable)
  
- [ ] **3.2 Unfreeze Projection Layer**
  - Keep Effort on attention, but train 768→512 projection
  
- [ ] **3.3 Add MLP Head**
  - Insert 512→256→512 MLP between CLIP output and classifier
  
- [ ] **3.4 Try OpenAI ViT-B/16**
  - Different pretraining, same architecture
  - May have better feature geometry for this task

---

## Phase 4: Knowledge Distillation (CONDITIONAL)

**Trigger:** Only if Phase 1 confirms B16 is viable AND Phase 2 validates L14

- [ ] **4.1 Generate Teacher Logits**
  - Run L14 on training set, save logits/embeddings
  
- [ ] **4.2 Implement Distillation Loss**
  - Logit distillation with temperature
  - Optional: feature distillation
  
- [ ] **4.3 Train B16 with Distillation**
  - Sweep α (distill weight) and T (temperature)

---

## Progress Log

| Date | Task | Result | Notes |
|------|------|--------|-------|
| 2026-01-14 | Document created | - | Starting investigation |
| | | | |

---

## Key Files & Checkpoints

- Best B16 (m=0): `gs://training-job-outputs/best_checkpoints/0ox4mrss/...`
- Best L14: `[TO BE FILLED]`
- Experiment configs: `experiments/B16_capacity_check/`
