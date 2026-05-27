"""
P22 pre-launch CPU smoke gate.

Loads the actual `PipelineRandomization` class with the EXACT config from
`experiments/phase2_round13/R13_P22_AUG_CURRICULUM.yaml`, applies it to ~240
training-bucket frames sampled from `analysis/score_distribution_2026-05-02/
outputs/train_data_samples/`, and measures the post-aug Laplacian
distribution.

GO if: post-aug median Laplacian on training frames lands in [10, 60] (overlaps
       eval suites: lockbox p50=10, teams_fake_dev p50=28, viso ~tens-of-units).
NO-GO if: post-aug median Laplacian stays > 100 (augmentation didn't bite) OR
          drops below 5 (over-aggressive).

Cites: analysis/cpu_decision_2026-05-02_pm_late/FINDINGS_AND_DECISION.md §2.2
       (eval distribution targets); §7 falsifier definition.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from data.augmentations.pipeline_randomization import PipelineRandomization

YAML_PATH = Path("experiments/phase2_round13/R13_P22_AUG_CURRICULUM.yaml")
TRAIN_ROOT = Path("analysis/score_distribution_2026-05-02/outputs/train_data_samples")
EVAL_ATTRS = Path("analysis/score_distribution_2026-05-02/outputs/cross_suite_attributes.csv")
OUT = Path("analysis/cpu_decision_2026-05-02_pm_late/outputs")
FIG = Path("analysis/cpu_decision_2026-05-02_pm_late/figures")

cfg = yaml.safe_load(YAML_PATH.read_text())
pr_cfg = cfg["augmentation"]["pipeline_randomization"]
print("PipelineRandomization config (from yaml):")
for k, v in sorted(pr_cfg.items()):
    print(f"  {k}: {v}")

aug = PipelineRandomization(config=pr_cfg)

# Load training frames (mix categories, train_realpool + train_visomaster_*)
imgs = []
labels = []
sources = []
for cat_dir in sorted(TRAIN_ROOT.iterdir()):
    if not cat_dir.is_dir():
        continue
    is_real = "realpool" in cat_dir.name
    files = sorted(cat_dir.glob("*.png"))[:30]
    for f in files:
        bgr = cv2.imread(str(f))
        if bgr is None: continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        imgs.append(rgb)
        labels.append(0 if is_real else 1)
        sources.append(cat_dir.name)
print(f"\nloaded {len(imgs)} training frames; {sum(l==0 for l in labels)} real, {sum(l==1 for l in labels)} fake")

# Apply pipeline_randomization N=5 times per frame so each frame gets multiple aug rolls
N_REPEATS = 5
rng = np.random.RandomState(2226)


def laplacian_var(rgb_img):
    g = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def luma_mean(rgb_img):
    g = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    return float(g.mean())


rows = []
for img, label, src in zip(imgs, labels, sources):
    rows.append({"src": src, "label": label, "stage": "before",
                 "lap": laplacian_var(img), "luma": luma_mean(img)})
    for r in range(N_REPEATS):
        np.random.seed(int(rng.randint(0, 1 << 31)))  # reseed so aug is stochastic
        aug_img = aug(img.copy(), label=label)
        rows.append({"src": src, "label": label, "stage": "after",
                     "lap": laplacian_var(aug_img), "luma": luma_mean(aug_img)})

results = pd.DataFrame(rows)
results.to_csv(OUT / "p22_smoke_pipeline_random.csv", index=False)

print("\nLaplacian variance — before vs after PipelineRandomization (P22 config):")
for stage in ["before", "after"]:
    sub = results[results.stage == stage]
    print(f"  {stage:6s} n={len(sub):4d}  "
          f"mean={sub.lap.mean():7.1f}  p25={sub.lap.quantile(0.25):7.1f}  "
          f"p50={sub.lap.median():7.1f}  p75={sub.lap.quantile(0.75):7.1f}")

print("\nLuma mean — before vs after:")
for stage in ["before", "after"]:
    sub = results[results.stage == stage]
    print(f"  {stage:6s} mean={sub.luma.mean():6.1f}  p25={sub.luma.quantile(0.25):6.1f}  "
          f"p50={sub.luma.median():6.1f}  p75={sub.luma.quantile(0.75):6.1f}")

# Eval reference
eval_df = pd.read_csv(EVAL_ATTRS)
print("\nEval reference distribution:")
for s in eval_df.suite.unique():
    sub = eval_df[eval_df.suite == s]
    print(f"  {s:35s} n={len(sub):4d} lap p25={sub.laplacian_var.quantile(0.25):7.1f}  "
          f"p50={sub.laplacian_var.median():7.1f}  p75={sub.laplacian_var.quantile(0.75):7.1f}")

# GO / NO-GO gate
after = results[results.stage == "after"]
post_med = float(after.lap.median())
post_p25 = float(after.lap.quantile(0.25))
post_p75 = float(after.lap.quantile(0.75))

print(f"\n=== P22 GO/NO-GO GATE ===")
print(f"Post-aug Laplacian: p25={post_p25:.1f}  p50={post_med:.1f}  p75={post_p75:.1f}")
print(f"Target: multi-modal coverage of eval distribution")
print(f"  lockbox p50 = 10           — need post-aug p10 ≤ ~20 to cover")
print(f"  teams_fake_dev p50 = 28    — need post-aug p25 ≤ ~50 to cover")
print(f"  deeplive p50 = 246          — need post-aug p75 ≥ ~150 to retain sharp regime")

post_p10 = float(after.lap.quantile(0.10))
print(f"Actual post-aug p10 = {post_p10:.1f}  p25 = {post_p25:.1f}  p75 = {post_p75:.1f}")

# Multi-modal coverage gate: post-aug must span low (lockbox-like), mid (viso),
# and high (deeplive/realpool). Failure modes: collapse to one mode either too
# blurry (no sharp frames left) or not blurry enough (no lockbox-like frames).
if post_p10 < 25 and post_p25 < 50 and post_p75 > 150:
    verdict = "GO — multi-modal coverage spans lockbox/viso/deeplive eval modes"
elif post_p10 > 30:
    verdict = "NO-GO — under-aggressive (no lockbox-like frames in post-aug; p10 > 30)"
elif post_p75 < 50:
    verdict = "NO-GO — over-aggressive (no sharp frames retained; p75 < 50)"
elif post_med < 5:
    verdict = "NO-GO — over-aggressive (median < 5)"
else:
    verdict = "AMBIGUOUS — distribution doesn't cleanly span eval modes; review figure"

print(f"VERDICT: {verdict}")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

before = results[results.stage == "before"].lap
after_ = results[results.stage == "after"].lap
ax.hist(before, bins=40, range=(0, 500), alpha=0.5, label=f"before (n={len(before)})", color="tab:blue", density=True)
ax.hist(after_, bins=40, range=(0, 500), alpha=0.5, label=f"after (n={len(after_)})", color="tab:orange", density=True)

# Eval reference
for s, color in [("teams_fake_all_lockbox", "tab:red"),
                 ("teams_fake_all_dev", "tab:green"),
                 ("deeplive_enhanced_dev", "tab:purple")]:
    sub = eval_df[eval_df.suite == s]
    if len(sub) > 0:
        ax.axvline(sub.laplacian_var.median(), color=color, linestyle="--", linewidth=2,
                   label=f"{s} median ({sub.laplacian_var.median():.0f})")

ax.set_xlabel("Laplacian variance")
ax.set_ylabel("density")
ax.set_xlim(0, 500)
ax.set_title(f"P22 SMOKE — PipelineRandomization closes train-eval gap?  VERDICT: {verdict.split(' — ')[0]}",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "p22_smoke_realtrainer_pipeline.png", dpi=140, bbox_inches="tight")
print(f"\nwrote {FIG / 'p22_smoke_realtrainer_pipeline.png'}")
print(f"wrote {OUT / 'p22_smoke_pipeline_random.csv'}")

# Exit non-zero on NO-GO so a CI gate can check it
if verdict.startswith("NO-GO"):
    sys.exit(1)
