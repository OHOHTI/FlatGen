# Flat_screen

`Flat_screen` is the flatness-screening module used in the broader FlatGen workflow. It combines Materials Project label generation with a PyTorch Geometric surrogate model for predicting flatness scores from crystal structures. In the full pipeline, this module provides the fast screening stage used to rank candidate materials before higher-cost validation.

This release keeps only the maintained PyG workflow and the lightweight data needed to reproduce the screening stage. Large generated artifacts, cached downloads, and pretrained checkpoints are intentionally excluded.

## Contents

- flatness-score generation from Materials Project band structures and DOS
- structure preprocessing into PyG graph and text inputs
- model training and inference for labeled materials or external CIF files

## Setup

```bash
conda env create -f environment.yml
conda activate flat_screen
```

## Materials Project API

Use your own Materials Project API key:

```bash
export MP_API_KEY=""
```

Fill in your key between the quotes before running any download or labeling step.

Scripts that require `MP_API_KEY`:

- `scripts/download_mp_summary.py`
- `scripts/preprocess_pyg.py`
- `scripts/MP_sbands_sdos.py`

## Minimal Workflow

1. Generate flatness labels:

```bash
cd scripts
python MP_sbands_sdos.py --batch_index 1
python stotal.py
python scores_stat_complete_csv.py
```

2. Build the PyG dataset:

```bash
cd scripts
python preprocess_pyg.py
```

3. Train the model:

```bash
cd model
python train_pyg.py
```

4. Run inference:

```bash
cd model
python prediction_pyg.py
python predict_cif.py --cif_dir /path/to/cifs
```

## Note

Review `model/configs.py` before full reproduction, since some default paths and output locations may need adjustment for a new environment.
