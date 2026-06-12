#!/usr/bin/env python
#SBATCH --job-name=scigen_screen
#SBATCH --ntasks=1
"""Screening only (reads generated chunks from SAVE_DIR).

    python screen.py

Parameters come from config.py; override via environment variables.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from gen_screen import print_config, run_screening


def main():
    print_config(generation=False)
    cfg.SCREEN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_screening()
    print(f"\nDone. Screened outputs: {cfg.SCREEN_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
