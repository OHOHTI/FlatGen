# SkeleGen

SkeleGen combines Wyckoff-constrained diffusion (DiffCSP++ backbone) with
SCIGEN-style sublattice constraints to generate crystal structures, then screens them down to flat-band candidate materials. This repository contains all code, pretrained models, and precomputed data needed to generate the dataset.

## Pipeline

1. **Constrained generation** (`wysc/sc_gen.py`) — samples sublattice constraints
   (236 sublattice geometries from known flat-band materials + DP-sampled Wyckoff
   assignments, `skelegen/data/`) and runs DiffCSP++ diffusion with the pretrained
   `mp_gen` model, keeping the constrained sublattice atoms and lattice fixed.
2. **Screening** (`wysc/screen_sublat.py`) — SMACT validity → space occupation
   ratio → two e3nn GNN stability classifiers → GNN+RoBERTa flatness surrogate
   (trained with the top-level `flat_screen` module). Writes CIF files, logs,
   and score files.

## Setup

```bash
conda env create -f skelegen/environment.yml   # creates env "skelegen"
conda activate skelegen
```


## Run

All parameters live in [`skelegen/wysc/config.py`](skelegen/wysc/config.py);
the defaults are the exact flatband_sym_1M settings (seed 102,
100 × 10 000 samples, `non_triclinic_weight = 5.0`,
`skeleton_symmetry_power = 1.2`, `flat_threshold = 0.5`).

```bash
cd skelegen/wysc
python gen_screen.py      # generation + screening (the full pipeline)
python run_sc_gen.py      # generation only
python screen.py          # screening only (re-screens existing chunks)
```

Outputs land in `skelegen/outputs/flatband_sym_1M/` (generated chunks) and
`skelegen/outputs/flatband_sym_1M/screened/` (CIFs, logs, funnel summary).

Any parameter can be overridden via environment variables, e.g.
`NUM_BATCHES=10 python gen_screen.py` for a quick smoke test. The scripts also
work as SLURM batch jobs (`sbatch gen_screen.py` from an activated env).

Note: ~20 % of samples are skipped (no valid Wyckoff assignment exists for the
drawn sublattice), so raw counts land below batch_size × num_batches. Exact
bitwise reproduction depends on GPU model and library versions.

## Layout

```
skelegen/                Pipeline code (wysc/), DiffCSP++ package + mp_gen
                         checkpoint (diffcsp/), precomputed constraint data (data/)
SCIGEN/                  Structure utilities and bond-length KDE used by the pipeline
GNN_EVAL/models/         Trained GNN stability classifier weights
Flat_screen/             Flatness surrogate inference code
```

Paths are resolved relative to this structure — keep the four top-level folders
side by side, with the FlatGen repository root (containing `flat_screen/`) as
their parent.

`Flat_screen/model/` is a copy of the inference code from the top-level
`flat_screen` module (identical model definition). The weights are loaded
directly from `../flat_screen/results/flatness_pyg/best_model.pth`, which is
produced by running `flat_screen/model/train_pyg.py` (or downloaded from the
release assets and placed there).

## Pretrained weights

The flatness surrogate weights (~500 MB) are not stored in git. Download
`best_model.pth` from the [release assets](https://github.com/OHOHTI/FlatGen/releases)
and place it at `flat_screen/results/flatness_pyg/best_model.pth` in the
FlatGen repository root.

## Acknowledgements

The generation backbone is [DiffCSP++](https://github.com/jiaor17/DiffCSP-PP)
(Jiao et al., MIT License — see `skelegen/LICENSE`), and the sublattice-constraint
approach follows [SCIGEN](https://arxiv.org/abs/2407.04557) (Okabe et al. — see
`SCIGEN/LICENSE`).
