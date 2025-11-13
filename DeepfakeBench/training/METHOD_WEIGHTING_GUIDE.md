# Method-Specific Weighting Guide

## Overview
This guide explains how to control the sampling frequency of specific real and fake methods during training using the `lesson_data_control` configuration in your experiment YAML files.

## Features
✅ **Per-method weight control** for both real and fake data  
✅ **Default equal sampling** when weights aren't specified  
✅ **Automatic weight normalization** if they don't sum to 1.0  
✅ **Detailed logging** showing final weight distribution  
✅ **Property-balanced sampling** within each method group  

## Configuration

### Basic Structure
```yaml
lesson_data_control:
  enabled: true
  
  real_method_groups:
    group_name_1:
      methods: ["method1", "method2"]
      weight: 0.4  # Optional: 40% of real samples
    group_name_2:
      methods: ["method3"]
      weight: 0.6  # Optional: 60% of real samples
  
  fake_method_groups:
    group_name_1:
      methods: ["fake_method1"]
      weight: 0.3  # Optional: 30% of fake samples
    group_name_2:
      methods: ["fake_method2", "fake_method3"]
      weight: 0.7  # Optional: 70% of fake samples
```

### Example 1: Equal Sampling (No Weights Specified)
```yaml
lesson_data_control:
  enabled: true
  
  real_method_groups:
    R_youtube:
      methods: ["youtube_real"]
      # No weight specified - will use equal weighting
    R_celeb:
      methods: ["celeb_real"]
      # No weight specified - will use equal weighting
    R_other:
      methods: ["faceforensics++", "external_youtube_avspeech"]
      # No weight specified - will use equal weighting
```
**Result**: Each group gets 33.33% sampling probability (1/3 each)

### Example 2: Boosting YouTube Real 2x
```yaml
lesson_data_control:
  enabled: true
  
  real_method_groups:
    R_youtube_high:
      methods: ["youtube_real"]
      weight: 0.5  # 50% of all real samples
    R_celeb:
      methods: ["celeb_real"]
      weight: 0.25  # 25% of all real samples
    R_other:
      methods: ["faceforensics++", "external_youtube_avspeech", "real_social_12_09"]
      weight: 0.25  # 25% of all real samples
```
**Result**: YouTube is sampled 2x more than the other groups

### Example 3: Mixed - Some Weighted, Some Default
```yaml
lesson_data_control:
  enabled: true
  
  real_method_groups:
    R_youtube:
      methods: ["youtube_real"]
      weight: 0.5  # Explicitly set to 50%
    R_celeb:
      methods: ["celeb_real"]
      # No weight - will get equal share of remaining 50%
    R_ff:
      methods: ["faceforensics++"]
      # No weight - will get equal share of remaining 50%
```
**Result**: 
- YouTube: 50%
- Celeb: 25% (equal share of remaining 50%)
- FF++: 25% (equal share of remaining 50%)

### Example 4: Your Current exp1.yaml (Fake Methods)
```yaml
lesson_data_control:
  enabled: true
  
  fake_method_groups:
    L_danet:
      methods: ["danet"]
      weight: 0.12  # 12% of fake samples
    L_mcnet:
      methods: ["mcnet"]
      weight: 0.12
    L_neuraltextures:
      methods: ["neuraltextures"]
      weight: 0.12
    L_veo3_creations:
      methods: ["veo3_creations"]
      weight: 0.12
    L_fomm:
      methods: ["fomm"]
      weight: 0.09
    # ... etc
```

## Logging Output

When your training starts, you'll see detailed logging like this:

```
--- Using DYNAMIC method grouping for REAL data stream. ---
  - Group 'R_youtube_real': Found 15,432 frames from 1 method(s).
  - Group 'R_celeb_real': Found 8,234 frames from 1 method(s).
  - Group 'R_other': Found 12,567 frames from 3 method(s).

=== REAL METHOD SAMPLING SUMMARY ===
  Group 'R_youtube_real':
    Methods: youtube_real
    Frame count: 15,432
    Specified weight: 0.5
    Final sampling weight: 0.5000 (50.00%)
  Group 'R_celeb_real':
    Methods: celeb_real
    Frame count: 8,234
    Specified weight: 0.25
    Final sampling weight: 0.2500 (25.00%)
  Group 'R_other':
    Methods: faceforensics++, external_youtube_avspeech, real_social_12_09
    Frame count: 12,567
    Specified weight: 0.25
    Final sampling weight: 0.2500 (25.00%)
Total real streams: 3
Total real weight: 1.000000
========================================

--- Using DYNAMIC method grouping for FAKE data stream. ---
  - Group 'L_danet': Found 5,678 frames across 1 methods.
  - Group 'L_mcnet': Found 4,321 frames across 1 methods.
  ...

=== FAKE METHOD SAMPLING SUMMARY ===
  Group 'L_danet':
    Methods: danet
    Frame count: 5,678
    Specified weight: 0.12
    Final sampling weight: 0.1200 (12.00%)
  ...
========================================
```

## Important Notes

1. **Frame Count ≠ Sampling Frequency**: The frame count shows how much data is available, but the final sampling weight controls how often each method is actually sampled during training.

2. **Property Balancing**: Within each method group, frames are still balanced by their sharpness properties (q1, q2, q3, q4 buckets).

3. **Automatic Normalization**: If your weights don't sum to exactly 1.0, they will be automatically normalized. This is useful when you want to specify relative weights (e.g., `1, 2, 1` becomes `0.25, 0.5, 0.25`).

4. **Backward Compatibility**: If you don't specify `lesson_data_control`, the system falls back to the original pooled sampling behavior.

5. **Group Names**: Group names (like `R_youtube_real`, `L_danet`) are just labels for logging - they don't affect the behavior. Choose descriptive names.

## How It Works Internally

1. **Filtering**: All frames for each group's methods are collected
2. **Property Bucketing**: Frames are grouped by sharpness bucket (q1-q4)
3. **Stream Creation**: A property-balanced stream is created for each group
4. **Hierarchical Sampling**: The `CustomSampleMultiplexerDataPipe` uses your weights as probabilities when selecting which group's stream to sample from
5. **Final Balance**: Real vs Fake ratio is controlled by `real_label_ratio` (default 0.5)

## Troubleshooting

**Q: My weights sum to more than 1.0, what happens?**  
A: The system will automatically normalize them and show a warning message.

**Q: What if I omit weights for all groups?**  
A: They will all get equal weighting (e.g., 3 groups = 0.333 each).

**Q: Can I mix weighted and unweighted groups?**  
A: Yes! Unweighted groups default to 1.0 and everything gets normalized together.

**Q: Does this affect validation?**  
A: No, only training data is affected. Validation uses standard per-method loaders.

**Q: How is this different from the old `real_category_weights`?**  
A: The old system used method_category (like 'identity', 'expression'). The new system gives you direct control over specific methods and can group multiple methods together.
