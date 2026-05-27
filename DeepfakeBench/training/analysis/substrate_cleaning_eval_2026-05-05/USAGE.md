# substrate_cleaning_eval — usage

When the new ckpt finishes (Packet A or Packet C-codec), run the existing
batch-inference flow to produce per-frame score CSVs over `teams_real_all_dev`,
`visomaster_enhanced_macro_dev`, `deeplive_enhanced_dev`, and `teams_fake_all_dev`
(same set Job 14 used). The CSVs need columns `frame_path, frame_prob, video_id`.
Then invoke:

```bash
python3 analysis/substrate_cleaning_eval_2026-05-05/run_clean_eval.py \
  --ckpt-name <packet_a_or_c_codec_label> \
  --real-csv  /path/to/teams_real_all_dev_<ckpt>_frames_report.csv \
  --fake-csv  visomaster_enhanced_macro_dev=/path/to/visomaster_enhanced_macro_dev_<ckpt>_frames_report.csv \
  --fake-csv  deeplive_enhanced_dev=/path/to/deeplive_enhanced_dev_<ckpt>_frames_report.csv \
  --fake-csv  teams_fake_all_dev=/path/to/teams_fake_all_dev_<ckpt>_frames_report.csv \
  --out-dir   analysis/substrate_cleaning_eval_2026-05-05
```

This (re-)writes `cleaned_substrate_manifest.json` (frozen F4 rules), then
emits `<ckpt>_real_cleaned.csv`, `<ckpt>_per_filter_fpr.csv`,
`<ckpt>_per_suite_recall_lift.csv`, `<ckpt>_summary.json`. The summary lists
recall at F4-calibrated tau for both FPR=5% and FPR=10%, plus n frames before
and after the F4 filter. To regression-check the pipeline first, run
`python3 analysis/substrate_cleaning_eval_2026-05-05/run_clean_eval.py --reference-run`
— it must print `=== Overall: PASS ===`.
