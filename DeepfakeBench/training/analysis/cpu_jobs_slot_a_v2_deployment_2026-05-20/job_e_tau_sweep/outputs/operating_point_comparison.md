# Operating-point comparison — calibrated-τ summary
Panel: 800-frame manual canary (`analysis/manual_canary_2026-05-20/frames_meta.parquet`)
Calibration: τ s.t. real_fpr on `teams_real_all_dev` ≤ target_dev_fpr.
NOTE: `lockbox_real_fpr_proper_clean` is on `proper_real_clean_lockbox` (HDTF-style cleans), not the production Teams lockbox. The 29-suite scorecard uses `teams_real_all_lockbox` which is NOT in this panel.

## At target_dev_fpr ≤ 0.05
| ckpt | τ_cal | dev_fpr | dev_target_reached | lockbox_fake_recall | proper_clean_lockbox_fpr | viso_recall_dev | deeplive_recall_dev |
|---|---:|---:|:---:|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.9900 | 0.1520 | ❌ | 0.2700 | 0.0000 | 0.0000 | 0.0800 |
| T5C_PERIODIC_STEP3500 | 0.9250 | 0.0500 | ✅ | 0.1600 | 0.0000 | 0.0000 | 0.1000 |
| SLOT_A_V2_STEP3500 | 0.8950 | 0.0440 | ✅ | 0.5400 | 0.0000 | 0.0000 | 0.1800 |
| SLOT_1_6AXIS_ANCHOR_STEP1500 | 0.9550 | 0.0460 | ✅ | 0.1300 | 0.0000 | 0.0000 | 0.4200 |
| SLOT_1_6AXIS_ANCHOR_STEP2500 | 0.9200 | 0.0300 | ✅ | 0.1100 | 0.0000 | 0.0000 | 0.1000 |
| SLOT_1_6AXIS_ANCHOR_STEP3500 | 0.9000 | 0.0300 | ✅ | 0.1700 | 0.0000 | 0.0000 | 0.2600 |
| SLOT_2_LORA_L8_L9_STEP2500 | 0.9350 | 0.0180 | ✅ | 0.0400 | 0.0000 | 0.0000 | 0.0400 |
| SLOT_3_5AXIS_NOLUMA_STEP3500 | 0.8900 | 0.0320 | ✅ | 0.1300 | 0.0000 | 0.0000 | 0.0000 |

## At target_dev_fpr ≤ 0.10
| ckpt | τ_cal | dev_fpr | dev_target_reached | lockbox_fake_recall | proper_clean_lockbox_fpr | viso_recall_dev | deeplive_recall_dev |
|---|---:|---:|:---:|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.9900 | 0.1520 | ❌ | 0.2700 | 0.0000 | 0.0000 | 0.0800 |
| T5C_PERIODIC_STEP3500 | 0.9100 | 0.0940 | ✅ | 0.4000 | 0.0000 | 0.0000 | 0.2000 |
| SLOT_A_V2_STEP3500 | 0.8800 | 0.1000 | ✅ | 0.6000 | 0.0000 | 0.0000 | 0.3400 |
| SLOT_1_6AXIS_ANCHOR_STEP1500 | 0.9400 | 0.0960 | ✅ | 0.4500 | 0.0000 | 0.0000 | 0.7000 |
| SLOT_1_6AXIS_ANCHOR_STEP2500 | 0.9100 | 0.0880 | ✅ | 0.2900 | 0.0000 | 0.0000 | 0.3000 |
| SLOT_1_6AXIS_ANCHOR_STEP3500 | 0.8900 | 0.0680 | ✅ | 0.2000 | 0.0000 | 0.0000 | 0.3600 |
| SLOT_2_LORA_L8_L9_STEP2500 | 0.9150 | 0.1000 | ✅ | 0.5600 | 0.0000 | 0.0200 | 0.3400 |
| SLOT_3_5AXIS_NOLUMA_STEP3500 | 0.8700 | 0.1000 | ✅ | 0.2600 | 0.0000 | 0.0000 | 0.0600 |

## At target_dev_fpr ≤ 0.20
| ckpt | τ_cal | dev_fpr | dev_target_reached | lockbox_fake_recall | proper_clean_lockbox_fpr | viso_recall_dev | deeplive_recall_dev |
|---|---:|---:|:---:|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.9850 | 0.1900 | ✅ | 0.3400 | 0.0000 | 0.0000 | 0.1600 |
| T5C_PERIODIC_STEP3500 | 0.8300 | 0.2000 | ✅ | 0.6900 | 0.0000 | 0.0600 | 0.7200 |
| SLOT_A_V2_STEP3500 | 0.8150 | 0.1980 | ✅ | 0.6900 | 0.0000 | 0.0400 | 0.7000 |
| SLOT_1_6AXIS_ANCHOR_STEP1500 | 0.8150 | 0.1980 | ✅ | 0.7300 | 0.0000 | 0.3800 | 1.0000 |
| SLOT_1_6AXIS_ANCHOR_STEP2500 | 0.8550 | 0.2000 | ✅ | 0.5800 | 0.0000 | 0.0600 | 0.7600 |
| SLOT_1_6AXIS_ANCHOR_STEP3500 | 0.8300 | 0.2000 | ✅ | 0.4800 | 0.0000 | 0.0200 | 0.7200 |
| SLOT_2_LORA_L8_L9_STEP2500 | 0.8700 | 0.1960 | ✅ | 0.7300 | 0.0000 | 0.1400 | 0.7000 |
| SLOT_3_5AXIS_NOLUMA_STEP3500 | 0.7900 | 0.2000 | ✅ | 0.5800 | 0.0000 | 0.0800 | 0.6200 |
