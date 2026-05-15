"""
run_all.py — Execute every figure-generating script in `scripts/` in
sequence. Writes one PNG per script into `figures/`. Total wall-clock
on a single modern CPU core is roughly 20-30 minutes.

Usage:
    python run_all.py                # run every script
    python run_all.py --only 00,03   # only run scripts 00 and 03
"""

import argparse
import os
import subprocess
import sys
import time


SCRIPTS = [
    "00_quickstart.py",
    "01_cta_alignment.py",
    "02_hkpa_gating.py",
    "03_car_routing.py",
    "04_stage_ablation.py",
    "05_tsne_features.py",
    "06_loss_sensitivity.py",
    "07_hierarchy_levels.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated 2-digit prefixes to run, e.g. '00,03'.",
    )
    args = parser.parse_args()

    selected = SCRIPTS
    if args.only:
        prefixes = set(p.strip() for p in args.only.split(","))
        selected = [s for s in SCRIPTS if s[:2] in prefixes]
        if not selected:
            print(f"No scripts match prefixes {prefixes}.")
            sys.exit(1)

    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
    failures = []
    t_start = time.time()
    for s in selected:
        print(f"\n{'=' * 70}\n>>> {s}\n{'=' * 70}")
        t0 = time.time()
        ret = subprocess.call([sys.executable, os.path.join(script_dir, s)])
        elapsed = time.time() - t0
        status = "OK" if ret == 0 else f"FAILED (exit code {ret})"
        print(f"--- {s}: {status}  ({elapsed:.1f}s)")
        if ret != 0:
            failures.append(s)

    print(f"\n{'=' * 70}")
    print(f"all done in {time.time() - t_start:.1f}s — {len(selected) - len(failures)} ok,"
          f" {len(failures)} failed")
    if failures:
        print("failed:", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
