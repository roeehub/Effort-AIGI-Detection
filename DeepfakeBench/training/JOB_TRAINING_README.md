# Effort-AIGI ► Vertex AI Launch Guide
Run single experiments or hyper-parameter sweeps on A100/H100 GPUs with built-in
W-and-B logging and automated cleanup.

---

## 0  Prerequisites
| Item | Notes |
|------|-------|
| **GCP project** with Vertex AI & Artifact Registry enabled | `gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com` |
| **Service account** that can run Vertex jobs | Grant `Vertex AI Custom Job Runner`, `Artifact Registry Reader`, and `Storage Object Viewer` |
| **Docker** installed locally **or** Cloud Build enabled | Local build is default; Cloud Build YAML included if you prefer |
| **WandB account** | Get an API key: **Settings ➜ API Keys** |

---

## 1  Build & push the container

```bash
PROJECT_ID=$(gcloud config get-value project)
IMAGE_URI="us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:latest"

# Local build
docker build -t "${IMAGE_URI}" .
docker push "${IMAGE_URI}"

# ──OR──
# Cloud Build (uses cloudbuild.yaml)
# gcloud builds submit --config cloudbuild.yaml
```


## 2 Single training run (default)
```bash
WANDB_API_KEY=<your-wandb-key> \
./launch_experiment.sh \
  --job-name=exp-resnet-baseline \
  --gpu=NVIDIA_A100_40GB \
  -- \
  --config DeepfakeBench/training/config/train_config.yaml
```
Anything after -- is passed straight to train.py.
Logs & checkpoints appear in W&B under the run name you set in your script.

## 3 Vertex Hyper-parameter Tuning
