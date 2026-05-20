# Slot 3 relaunch addendum — 2026-05-21

> Update to `MORNING_DECISION_TREE_2026-05-21.md`. Slot 3 was relaunched
> after the original failed on an image-currency race condition.

## What happened

| event | time UTC | detail |
|---|---|---|
| Cloud Build start (image 1.3.295) | 22:17:17 | source tarball captured |
| Slot 1 yaml created | 22:14:43 | (in tarball) ✓ |
| Slot 2 yaml created | 22:15:34 | (in tarball) ✓ |
| Slot 3 yaml created | 22:18:00 | (NOT in tarball) ✗ |
| Cloud Build finish | 22:19:06 | image pushed at 22:19:50 |
| Vertex Slot 3 submitted | 22:24:15 | `5637646623317164032` |
| Vertex Slot 3 FAILED | 22:27:41 | `FileNotFoundError` on yaml |
| Cloud Build start (image 1.3.296) | 22:37:08 | rebuild |
| Cloud Build finish | 22:39:36 | image pushed |
| Vertex Slot 3 relaunched | 22:41:05 | `7516210617884082176` ✓ |

## Updated Vertex job table

| Slot | Region | Vertex Job ID | Image | Submitted | Anchor | codec_p | FT base |
|---|---|---|---|---|---|---:|---|
| 1 | us-west4 | `5839421592722997248` | 1.3.295 | 22:23:47 UTC | ON | 0.40 | Slot A v2 step3500 |
| 2 | us-east1 | `1506065191836581888` | 1.3.295 | 22:24:02 UTC | ON | 0.20 | Slot A v2 step3500 |
| 3 | us-central1 | `7516210617884082176` | **1.3.296** | **22:41:05 UTC** | OFF | 0.40 | T5C step3500 |

Original failed Slot 3 job `5637646623317164032` is in JOB_STATE_FAILED — do not action; it is the proximate evidence in the open-loop thread.

## Implications for morning verdict

- All 3 packets are budgeted: user authorized "up to 3 GPU jobs" and we've used 3 distinct slots (the failed one doesn't count against the budget because no useful training happened — only ~50s of pod startup before the FileNotFoundError).
- Slot 3 starts ~17 min later than Slots 1+2; if all 3 take ~4h training, expected SUCCEEDED:
  - Slot 1: ~26:24 UTC (or earlier — actual start depends on PENDING→RUNNING time)
  - Slot 2: ~26:24 UTC
  - Slot 3: ~26:41 UTC
- Monitor `tasks/blbxkk5u3` watches the CORRECT 3 IDs (Slot 1, Slot 2, Slot 3-relaunched). The original `tasks/ba1ur5e2z` carried the failed Slot 3 ID and is now superseded.

## What does NOT change

- Decision tree branches A-E remain valid (the 2×2 design and decomposition logic is identical).
- CPU gate results unchanged (gates were on FT bases T5C / Slot A v2, both image-independent).
- may6+codec results unchanged (CPU experiment, no Vertex dependency).
- Hypothesis under test, falsifiers, confirmation criteria all unchanged.

## References

- Open loop: `docs/packet_retrospectives/threads/image_currency_check_race_condition_2026-05-21.md`
- Failed job: `gcloud ai custom-jobs describe projects/700371397073/locations/us-central1/customJobs/5637646623317164032`
- Relaunched job: `gcloud ai custom-jobs describe projects/700371397073/locations/us-central1/customJobs/7516210617884082176`
- Memory: `~/.claude/projects/.../memory/feedback_image_currency_race_condition.md`
