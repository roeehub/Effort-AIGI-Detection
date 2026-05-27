"""Post-training automation: fill the auto_mode_2026-05-16 ckpt map placeholders
with actual periodic step1500/step3500 paths from W&B run IDs.

Usage:
  python3 fill_ckpt_map.py <slot_a_run_id> <slot_b_run_id>

Or auto-discover by reading the Vertex job descriptions:
  python3 fill_ckpt_map.py --auto
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


SLOT_A_JOB = "projects/700371397073/locations/us-central1/customJobs/8471062799328477184"  # v2 after .gcloudignore fix
SLOT_B_JOB = "projects/700371397073/locations/us-west4/customJobs/3559896493831749632"

MAP_PATH = Path(__file__).resolve().parent.parent.parent.parent / "arena" / "checkpoint_maps" / "teams_target_domain.auto_mode_2026-05-16.yaml"


def get_wandb_run_id_from_logs(job_path: str) -> str | None:
    """Stream a few lines of logs and extract the wandb run ID."""
    result = subprocess.run(
        ['gcloud', 'ai', 'custom-jobs', 'stream-logs', job_path],
        capture_output=True, text=True, timeout=15,
    )
    text = (result.stdout or "") + (result.stderr or "")
    import re
    m = re.search(r'wandb/run-\d+_\d+-([a-z0-9]{8})', text)
    if m:
        return m.group(1)
    m = re.search(r'wandb\.ai/dtect-vision/phase2-round13/runs/([a-z0-9]{8})', text)
    if m:
        return m.group(1)
    return None


def list_periodic_ckpts(run_id: str) -> dict[int, str]:
    """List periodic_effort_*.pth under gs://training-job-outputs/best_checkpoints/{run_id}/
    and parse step numbers."""
    import re
    result = subprocess.run(
        ['gsutil', 'ls', f'gs://training-job-outputs/best_checkpoints/{run_id}/'],
        capture_output=True, text=True,
    )
    out = {}
    for line in result.stdout.split('\n'):
        line = line.strip()
        if not line: continue
        m = re.search(r'periodic_effort_\d+_step(\d+)_[^/]+\.pth$', line)
        if m:
            out[int(m.group(1))] = line
    return out


def fill_map(slot_a_run: str, slot_b_run: str) -> None:
    a_ckpts = list_periodic_ckpts(slot_a_run)
    b_ckpts = list_periodic_ckpts(slot_b_run)
    print(f'Slot A run {slot_a_run}: steps={sorted(a_ckpts.keys())}')
    print(f'Slot B run {slot_b_run}: steps={sorted(b_ckpts.keys())}')

    def closest(d, target):
        if not d: return None
        return d[min(d.keys(), key=lambda k: abs(k - target))]

    a_1500 = closest(a_ckpts, 1500)
    a_3500 = closest(a_ckpts, 3500)
    b_1500 = closest(b_ckpts, 1500)
    b_3500 = closest(b_ckpts, 3500)

    text = MAP_PATH.read_text()
    import re
    text = re.sub(r'SLOT_A_ANCHOR_AWARE_STEP1500:\s+".*"',
                  f'SLOT_A_ANCHOR_AWARE_STEP1500:  "{a_1500}"', text)
    text = re.sub(r'SLOT_A_ANCHOR_AWARE_STEP3500:\s+".*"',
                  f'SLOT_A_ANCHOR_AWARE_STEP3500:  "{a_3500}"', text)
    text = re.sub(r'SLOT_B_REAL_REBAL_STEP1500:\s+".*"',
                  f'SLOT_B_REAL_REBAL_STEP1500:    "{b_1500}"', text)
    text = re.sub(r'SLOT_B_REAL_REBAL_STEP3500:\s+".*"',
                  f'SLOT_B_REAL_REBAL_STEP3500:    "{b_3500}"', text)
    MAP_PATH.write_text(text)
    print(f'Updated {MAP_PATH}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--auto', action='store_true', help='Auto-discover from Vertex jobs')
    ap.add_argument('--slot-a-run', help='W&B run ID for Slot A')
    ap.add_argument('--slot-b-run', help='W&B run ID for Slot B')
    args = ap.parse_args()

    if args.auto:
        a = get_wandb_run_id_from_logs(SLOT_A_JOB)
        b = get_wandb_run_id_from_logs(SLOT_B_JOB)
        print(f'Auto-discovered: Slot A={a}, Slot B={b}')
        if not (a and b):
            print('FAIL: could not auto-discover. Run with --slot-a-run and --slot-b-run.', file=sys.stderr)
            sys.exit(1)
        fill_map(a, b)
    elif args.slot_a_run and args.slot_b_run:
        fill_map(args.slot_a_run, args.slot_b_run)
    else:
        ap.print_help()
        sys.exit(2)


if __name__ == '__main__':
    main()
