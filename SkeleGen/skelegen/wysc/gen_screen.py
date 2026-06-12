#!/usr/bin/env python
#SBATCH --job-name=wysc_gen_screen
#SBATCH --ntasks=1
"""Full pipeline: constrained generation followed by screening.

Usage (inside the conda env, from this directory):

    python gen_screen.py

All parameters come from config.py and can be overridden via environment
variables, e.g. ``NUM_BATCHES=10 python gen_screen.py``.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


def print_config(generation=True, screening=True):
    print("=" * 40)
    print(" SkeleGen — Wyckoff + Sublattice pipeline")
    print("=" * 40)
    print(f"  Model:           {cfg.MODEL}")
    print(f"  Save dir:        {cfg.SAVE_DIR}")
    if screening:
        print(f"  Screen dir:      {cfg.SCREEN_OUTPUT_DIR}")
    print(f"  use_template:    {cfg.USE_TEMPLATE}")
    if generation:
        print(f"  step_lr:         {cfg.STEP_LR}")
        print(f"  t_stop_x / l:    {cfg.T_STOP_X} / {cfg.T_STOP_L}")
        print(f"  batch_size:      {cfg.BATCH_SIZE}")
        print(f"  num_batches:     {cfg.NUM_BATCHES}  (total samples = {cfg.BATCH_SIZE * cfg.NUM_BATCHES})")
        print(f"  seed:            {cfg.SEED}")
        print(f"  save_every:      {cfg.SAVE_EVERY}")
        print(f"  natm_range:      [{cfg.NATM_MIN}, {cfg.NATM_MAX}]")
        if not cfg.USE_TEMPLATE:
            print(f"  dp_temp:         {cfg.DP_TEMPERATURE}")
            print(f"  non_triclinic:   {cfg.NON_TRICLINIC_WEIGHT}")
            print(f"  skeleton_power:  {cfg.SKELETON_SYMMETRY_POWER}")
    if screening:
        print(f"  gen_cif:         {cfg.GEN_CIF}")
        print(f"  screen_mag:      {cfg.SCREEN_MAG}")
        print(f"  screen_flat:     {cfg.SCREEN_FLAT}")
        print(f"  flat_threshold:  {cfg.FLAT_THRESHOLD}")
    print("=" * 40)
    print()


def check_model():
    if not cfg.MODEL.is_dir():
        print(f"ERROR: model directory not found: {cfg.MODEL}")
        models_dir = cfg.PROJECT_DIR / "diffcsp" / "pretrained_models"
        if models_dir.is_dir():
            print("\nAvailable models:")
            for d in sorted(p for p in models_dir.iterdir() if p.is_dir()):
                print(f"  {d.relative_to(cfg.PROJECT_DIR)}")
        sys.exit(1)


def run_generation():
    cmd = [
        sys.executable, "-m", "wysc.sc_gen",
        "--model_path", str(cfg.MODEL),
        "--save_dir", str(cfg.SAVE_DIR),
        "--step_lr", str(cfg.STEP_LR),
        "--t_stop_x", str(cfg.T_STOP_X),
        "--t_stop_l", str(cfg.T_STOP_L),
        "--batch_size", str(cfg.BATCH_SIZE),
        "--num_batches", str(cfg.NUM_BATCHES),
        "--seed", str(cfg.SEED),
        "--save_every", str(cfg.SAVE_EVERY),
        "--natm_min", str(cfg.NATM_MIN),
        "--natm_max", str(cfg.NATM_MAX),
    ]
    if cfg.USE_TEMPLATE:
        cmd.append("--use_template")
    else:
        cmd += [
            "--temperature", str(cfg.DP_TEMPERATURE),
            "--non_triclinic_weight", str(cfg.NON_TRICLINIC_WEIGHT),
            "--skeleton_symmetry_power", str(cfg.SKELETON_SYMMETRY_POWER),
        ]
    subprocess.run(cmd, cwd=cfg.PROJECT_DIR, check=True)


def run_screening():
    cmd = [
        sys.executable, "-m", "wysc.screen_sublat",
        "--chunk_dir", str(cfg.SAVE_DIR),
        "--output_dir", str(cfg.SCREEN_OUTPUT_DIR),
        "--use_template", str(cfg.USE_TEMPLATE),
        "--gen_cif", str(cfg.GEN_CIF),
        "--screen_mag", str(cfg.SCREEN_MAG),
        "--screen_flat", str(cfg.SCREEN_FLAT),
        "--flat_threshold", str(cfg.FLAT_THRESHOLD),
    ]
    subprocess.run(cmd, cwd=cfg.PROJECT_DIR, check=True)


def main():
    print_config()
    check_model()
    cfg.SAVE_DIR.mkdir(parents=True, exist_ok=True)
    cfg.SCREEN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/2] Running constrained generation...")
    run_generation()

    print("\n[2/2] Running screening...")
    run_screening()

    print(f"\nDone. Generated chunks: {cfg.SAVE_DIR}")
    print(f"Done. Screened outputs: {cfg.SCREEN_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
