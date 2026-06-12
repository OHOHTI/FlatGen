"""Shared configuration for the generation / screening pipeline.

Defaults are the exact parameters used to generate the flatband_sym_1M dataset
(1 M symmetry-biased samples, screened to flat-band candidates).

Every value can be overridden with an environment variable of the same name,
e.g. ``NUM_BATCHES=10 python gen_screen.py`` for a quick smoke test.
"""

import os
from pathlib import Path


def _bool(v):
    return str(v).strip().lower() in ("1", "true", "yes")


# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent          # .../skelegen
RUN_NAME = os.environ.get("RUN_NAME", "flatband_sym_1M")
MODEL = Path(os.environ.get("MODEL", PROJECT_DIR / "diffcsp" / "pretrained_models" / "mp_gen"))
SAVE_DIR = Path(os.environ.get("SAVE_DIR", PROJECT_DIR / "outputs" / RUN_NAME))
SCREEN_OUTPUT_DIR = Path(os.environ.get("SCREEN_OUTPUT_DIR", SAVE_DIR / "screened"))

# Environment variables required by the diffcsp package (hydra config resolution).
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_DIR))
os.environ.setdefault("HYDRA_JOBS", str(PROJECT_DIR / "results"))
os.environ.setdefault("WANDB_DIR", str(PROJECT_DIR))
os.environ.setdefault("USE_WANDB_LOGGING", "0")

# ── Generation hyperparameters ───────────────────────────────────────────────
STEP_LR = float(os.environ.get("STEP_LR", 1e-5))
T_STOP_X = float(os.environ.get("T_STOP_X", 0.1))    # 0.0 = coord constraints always on
T_STOP_L = float(os.environ.get("T_STOP_L", 0.1))    # 0.0 = lattice constraints always on
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 100))
NUM_BATCHES = int(os.environ.get("NUM_BATCHES", 10000))  # total samples = BATCH_SIZE × NUM_BATCHES
SEED = int(os.environ.get("SEED", 102))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 100))      # save a chunk every N batches

# ── Dataset generation parameters ────────────────────────────────────────────
NATM_MIN = int(os.environ.get("NATM_MIN", 5))
NATM_MAX = int(os.environ.get("NATM_MAX", 20))
DP_TEMPERATURE = float(os.environ.get("DP_TEMPERATURE", 1.0))    # >1 = more diverse DP completions, <1 = sharper
NON_TRICLINIC_WEIGHT = float(os.environ.get("NON_TRICLINIC_WEIGHT", 5.0))      # 1.0 = uniform g_full sampling, >1 favors SG > 2
SKELETON_SYMMETRY_POWER = float(os.environ.get("SKELETON_SYMMETRY_POWER", 1.2))  # 0.0 = uniform skeleton sampling, >0 favors higher-SG skeletons
USE_TEMPLATE = _bool(os.environ.get("USE_TEMPLATE", "False"))    # False = DP sampler (used by flatband_sym_1M)

# ── Screening parameters ─────────────────────────────────────────────────────
GEN_CIF = _bool(os.environ.get("GEN_CIF", "True"))
SCREEN_MAG = _bool(os.environ.get("SCREEN_MAG", "False"))
SCREEN_FLAT = _bool(os.environ.get("SCREEN_FLAT", "True"))
FLAT_THRESHOLD = float(os.environ.get("FLAT_THRESHOLD", 0.5))
