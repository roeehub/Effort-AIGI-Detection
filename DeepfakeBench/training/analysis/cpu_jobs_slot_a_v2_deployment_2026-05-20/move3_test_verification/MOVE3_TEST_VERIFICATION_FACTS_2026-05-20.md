# Move 3 verification — canary + LoRA-load fixes test results (FACTS)

## 1. Question

The 2026-05-20 T5C_TRIPLE session surfaced two HIGH-severity infra bugs and produced working-tree fixes:

1. `canary-silence-when-multi-axis-grl-active` — fixed by passing `inference=True` to `self.model(data_dict)` in `trainer/mixins/canary_probe.py`.
2. `lora-enabled-not-propagated-by-load-model` — fixed by installing LoRA layers in `batch_inference_gcs.load_model` based on cfg block, with a state-dict-key fallback for older ckpts.

Both fixes shipped with new test files (untracked in git, ?? status):
- `tests/test_canary_with_grl_wiring.py`
- `tests/test_lora_adapter.py`
- `tests/test_lora_ckpt_roundtrip.py`

Question: do the tests pass end-to-end on the current working tree, against the as-shipped fix code?

## 2. Method

```bash
cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training
python -m pytest tests/test_canary_with_grl_wiring.py \
                 tests/test_lora_adapter.py \
                 tests/test_lora_ckpt_roundtrip.py \
                 -v --tb=short
```

Python 3.11.7, pytest 7.4.0, pluggy 1.0.0, anyio 4.7.0.

## 3. Numbers

### 3.1 Per-file test counts

| Test file | Tests | Passed | Failed |
|---|---:|---:|---:|
| `test_canary_with_grl_wiring.py` | 3 | **3** | 0 |
| `test_lora_adapter.py` | 10 | **10** | 0 |
| `test_lora_ckpt_roundtrip.py` | 3 | **3** | 0 |
| **Total** | **16** | **16** | **0** |

### 3.2 Per-test outcomes

```
tests/test_canary_with_grl_wiring.py::test_forward_with_inference_true_skips_grl_branch                        PASSED
tests/test_canary_with_grl_wiring.py::test_forward_without_inference_kwarg_hits_grl_and_raises                 PASSED
tests/test_canary_with_grl_wiring.py::test_canary_mixin_call_pattern_succeeds_with_inference_true              PASSED
tests/test_lora_adapter.py::test_zero_init_lora_delta_is_exactly_zero                                          PASSED
tests/test_lora_adapter.py::test_zero_init_forward_output_matches_base_layer_weight                            PASSED
tests/test_lora_adapter.py::test_zero_init_full_resblock_forward_parity                                        PASSED
tests/test_lora_adapter.py::test_only_lora_params_receive_gradients_after_freeze                               PASSED
tests/test_lora_adapter.py::test_higher_rank_yields_more_lora_parameters                                       PASSED
tests/test_lora_adapter.py::test_lora_stacks_on_svd_in_proj_routing                                            PASSED
tests/test_lora_adapter.py::test_apply_lora_to_openclip_visual_indexes_by_layer                                PASSED
tests/test_lora_adapter.py::test_freeze_base_clip_encoder_freezes_everything_but_lora                          PASSED
tests/test_lora_adapter.py::test_apply_lora_rejects_out_of_range_layer                                         PASSED
tests/test_lora_adapter.py::test_apply_lora_rejects_double_install                                             PASSED
tests/test_lora_ckpt_roundtrip.py::test_lora_save_load_round_trip_zero_unexpected                              PASSED
tests/test_lora_ckpt_roundtrip.py::test_lora_skip_install_drops_state_dict_keys_as_unexpected                  PASSED
tests/test_lora_ckpt_roundtrip.py::test_save_ckpt_embeds_lora_block_in_model_config                            PASSED
```

Wall-clock: 5.05 s.

## 4. Mechanical pass/fail

| Bar | Definition | Result |
|---|---|---|
| **Bar 1 — Canary fix coverage** | All 3 tests in `test_canary_with_grl_wiring.py` pass, including the "forward without inference_kwarg hits GRL and raises" regression test that proves the bug existed | **MET** (3/3) |
| **Bar 2 — LoRA adapter integration** | All 10 unit tests covering apply_lora_to_openclip_visual, zero-init parity, rank/layer indexing, freeze policy, double-install rejection | **MET** (10/10) |
| **Bar 3 — LoRA ckpt roundtrip** | `test_lora_save_load_round_trip_zero_unexpected` verifies trainer save_ckpt embeds lora cfg in model_config; `test_lora_skip_install_drops_state_dict_keys_as_unexpected` verifies the warning fires when LoRA install is skipped | **MET** (3/3) |
| **Bar 4 — Wall-clock budget** | < 60 s total | **MET** (5.05 s) |

## 5. Commit status

As of session start (2026-05-20 PM):

```
M  trainer/mixins/canary_probe.py        +23/−2 (inference=True + 'prob' key fallback)
M  batch_inference_gcs.py                +77/−0 (LoRA install + state-dict fallback + unexpected-key warning)
?? tests/test_canary_with_grl_wiring.py  153 lines
?? tests/test_lora_adapter.py            364 lines
?? tests/test_lora_ckpt_roundtrip.py     177 lines
```

Two source-code fixes + three test files = 5 file changes in the diff. All 16 tests pass against the fix code. The fixes are not committed; the open loops `canary-silence-when-multi-axis-grl-active` (HIGH) and `lora-enabled-not-propagated-by-load-model` (HIGH) remain in `OPEN_LOOPS.md`.

## 6. Artifacts

| Path | Contents |
|---|---|
| `move3_pytest_output.txt` | Full pytest output (this dir) |

## 7. Cross-references

- Threads: `threads/in_training_canary_signal.md`, `threads/wandb_yaml_propagation_bugs.md`
- 2026-05-20 open loops surfaced: `canary-silence-when-multi-axis-grl-active`, `lora-enabled-not-propagated-by-load-model`
