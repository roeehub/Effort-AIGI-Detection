# clean_eval_v1 + shortcut_probe_v1 BUILD REPORT

Generated 2026-04-28 by `analysis/clean_eval_2026-04-29/build_clean_eval_and_probe.py`.

Source: `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` (n=7334)

## Quality gates

- face_area_ratio >= 0.10
- NOT is_pose_extreme
- NOT is_no_face
- decode_ok

## clean_eval_v1 — deployment-honest substrate

- **Total frames**: 54
- Real: 28 frames across 8 identities
- Fake: 26 frames across 13 methods
- Capture modes restricted to {normal_photo, phone_screen} (deployment-realistic).
- dor_shkedi excluded from real (anchor-pool overlap).

### Real composition

- Capture modes: {'normal_photo': 14, 'phone_screen': 14}
- Identities: {'Test_Cam__s73': 4, 'PC_Generator__s13': 4, 'Test_Cam__s41': 4, 'Test_Cam__s76': 4, 'bla_bla_chow__s2': 4, 'PC_Generator__s8': 4, 'Md_noyn_Sharker__s15': 2, 'Xiang_Xiang2_Feng__s23': 2}

### Fake composition

- Capture modes: {'normal_photo': 15, 'phone_screen': 11}
- Methods: {'teams_capture_cam_test_s33': 2, 'deeplive_enhanced': 2, 'teams_capture_pc_generator_s3': 2, 'teams_capture_test_cam_s73': 2, 'teams_capture_cam_test_s46': 2, 'teams_capture_test_cam_s53': 2, 'teams_capture_test_cam_s76': 2, 'teams_capture_dor_shkedi_s16': 2, 'teams_capture_pc_generator_s9': 2, 'teams_capture_cam_test_s35': 2, 'teams_capture_cam_test_s32': 2, 'teams_capture_pc_generator_s15': 2, 'teams_capture_pc_generator_s4': 2}

## shortcut_probe_v1 — same-face-different-pipeline pairs

- **Total pairs**: 11
- Each pair: same identity_key, label=real, two distinct clip_capture_mode values.
- Includes webcam (the test is precisely whether the model reads mode as label).

- Identity coverage: {'Md_noyn_Sharker__s15': 1, 'PC_Generator__s13': 1, 'PC_Generator__s45': 1, 'Test_Cam__s41': 2, 'Xiang_Xiang2_Feng': 2, 'Xiang_Xiang2_Feng__s23': 1, 'bla_bla_chow': 2, 'dor_shkedi': 1}
- Mode-combo coverage: {'phone_screen <-> webcam': 2, 'normal_photo <-> phone_screen': 3, 'normal_photo <-> webcam': 3, 'screen_recording <-> webcam': 1, 'screen <-> webcam': 1, 'phone_screen <-> screen': 1}

## Day-4 usage

- **Axis 2**: score model on clean_eval_v1; require fake recall ≥ 80% at the same operating threshold used for the 90/5 axis.
- **Axis 3**: score on shortcut_probe_v1 pair members; compute |prob_fake(a) - prob_fake(b)| per pair; take max across pairs. Require max Δ ≤ 0.15.

