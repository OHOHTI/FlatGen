#!/usr/bin/env python
#SBATCH --job-name=wysc_gen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
"""Constrained generation only (no screening).

    python run_sc_gen.py

Parameters come from config.py; override via environment variables.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from gen_screen import check_model, print_config, run_generation


def main():
    print_config(screening=False)
    check_model()
    cfg.SAVE_DIR.mkdir(parents=True, exist_ok=True)
    run_generation()
    print(f"\nDone. Results saved to: {cfg.SAVE_DIR}")


if __name__ == "__main__":
    main()
