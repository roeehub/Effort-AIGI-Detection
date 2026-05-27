"""Fast top-up fetch of any remaining frames per group.

Reads outputs/_uri_lists/{group}_remaining.txt, writes to
_frame_cache/{group}/<safe_name>. Higher worker count (16, I/O-bound).
"""
from __future__ import annotations
import subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DIAG_ROOT = Path(__file__).resolve().parent.parent
CACHE = DIAG_ROOT / "_frame_cache"
URI_LISTS = DIAG_ROOT / "outputs" / "_uri_lists"

WORKERS = 16  # I/O-bound; gsutil cp is its own process, not joblib

GROUPS = ["eval_real", "train_viso_fake", "train_viso_real"]

def safe_name(uri: str) -> str:
    return uri.replace("gs://", "").replace("/", "__")

def fetch(uri: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            ["gsutil", "-q", "cp", uri, str(dest)],
            capture_output=True, timeout=60,
        )
        return r.returncode == 0 and dest.exists() and dest.stat().st_size > 0
    except Exception:
        return False

def main():
    t0 = time.time()
    for g in GROUPS:
        rfile = URI_LISTS / f"{g}_remaining.txt"
        if not rfile.exists():
            print(f"[{g}] no remaining file"); continue
        uris = [l.strip() for l in rfile.read_text().splitlines() if l.strip()]
        gdir = CACHE / g
        gdir.mkdir(parents=True, exist_ok=True)
        ok = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(fetch, u, gdir / safe_name(u)): u for u in uris}
            for i, fut in enumerate(as_completed(futs)):
                if fut.result(): ok += 1
                if (i + 1) % 100 == 0:
                    print(f"[{g}] {i+1}/{len(uris)} ok={ok}  ({time.time()-t0:.0f}s)", flush=True)
        print(f"[{g}] done; ok={ok}/{len(uris)}  cumulative={time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
