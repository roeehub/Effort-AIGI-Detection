# Image-currency check has a race condition when yaml mtime falls between Cloud Build source-tarball time and image push time

> **Severity**: medium (recoverable, but silent training-job failure if not caught)
> **First seen**: 2026-05-21 ~22:27 UTC
> **Last verified**: 2026-05-21 ~22:30 UTC
> **Status**: open

## Pattern (FACTS)

`scripts/launch/check_image_currency.sh` compares each yaml's filesystem mtime to the image's `gcloud artifacts docker images list ... --format='value(createTime)'`. The check PASSES when yaml mtime < image createTime.

But Cloud Build's source tarball is captured at the START of the build (`gcloud builds submit` uploads the directory before running Dockerfile steps), not at the END. The image `createTime` reported by Artifact Registry is the image push time, which is AFTER the Dockerfile finishes — typically ~2 minutes after the source tarball was created.

**Race window**: if a yaml is created during a Cloud Build run, the yaml's mtime can fall between source-tarball-time and image-createTime. The currency check passes (mtime < createTime) but the yaml is NOT in the image (because the source tarball was already submitted).

## Concrete incident (2026-05-21)

Source tarball: `2026-05-20T22:17:17Z` (Cloud Build start)
Image createTime: `2026-05-20T22:19:50Z` (image push, +2m 33s)
Yaml mtime: `2026-05-20T22:18:00Z` (created 43s after source tarball, 110s before push)

Yaml: `experiments/phase2_round13/R13_T5C_CODEC_ONLY_NO_ANCHOR_2026-05-21.yaml`

Result: check_image_currency.sh PASSED. Vertex job submitted with this yaml as `--param-config /workspace/...`. Job died with `FileNotFoundError: [Errno 2] No such file or directory: '/workspace/experiments/phase2_round13/R13_T5C_CODEC_ONLY_NO_ANCHOR_2026-05-21.yaml'` at `train_sweep.py:134`.

Vertex job: `5637646623317164032` (us-central1). Image: `1.3.295`. Wall time of failed job: ~3 min from submit to FAILED (3 retries, each ~50s).

## Mitigation already applied (2026-05-21)

1. Triggered fresh `./dev.sh build-prod -y` after the 3 yaml writes. New image `1.3.296` contains all 3 yamls (build started AFTER all 3 yamls were on disk).
2. Relaunched Slot 3 on the new image. (See STATE.md for relaunch Job ID.)

## Open loop — close criterion

Either:
- (a) `check_image_currency.sh` is updated to use `gcloud builds describe <build_id> --format='value(createTime)'` (the Cloud Build *start* time, when the source tarball was created) instead of Artifact Registry's `createTime` (the image push time, after Dockerfile finishes).
- (b) An explicit Cloud Build `_source_create_time` substitution is plumbed through and the check uses it.
- (c) The launcher refuses to launch if any param-config yaml was modified within N seconds (e.g., 600s) of an image push, with the explicit message "yaml modification timestamp is within the source-tarball race window; rebuild required".

Until one of (a-c) lands, the workaround is:
- Always WAIT N seconds (~120s buffer) between writing a new yaml and triggering a build.
- OR always rebuild *after* writing all yamls but *before* launching anything.
