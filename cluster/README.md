# cluster

Self-supervised GNN encoder for crystallographic sublattices, with downstream clustering and novelty screening. The pipeline trains a contrastive graph neural network on sublattice structures from the Materials Project, produces fixed-length embeddings, then clusters and ranks them to identify novel flat-band sublattice geometries.


## Setup

```bash
conda env create -f environment.yml
conda activate cluster
```

Key dependencies: `torch`, `torch_geometric`, `pymatgen`, `ase`, `mp-api`, `umap-learn`, `hdbscan`, `adjustText`, `numba`.

## Configuration

All shared settings live in `config.py`:

- `DATASET`, `MODEL_NAME` -- dataset and model identifiers
- `API_KEY` -- Materials Project API key (required for step 1)
- `SUPERCELL_SIZE_1`, `SUPERCELL_SIZE_2` -- supercell expansion factors (>= 3)
- `BATCH_SIZE`, `NUM_EPOCHS` -- GNN training hyperparameters
- `GRAPH_GENERATION`, `TRAINING` -- flags to skip pipeline stages
- Clustering paths (`EMBEDDINGS_PATH`, `CLUSTER_OUTPUT_DIR`, etc.)

## Usage

Run the full training pipeline:

```bash
python run_pipeline.py
```

Run clustering on the resulting embeddings:

```bash
python cluster_screen.py
```

Run inference on new CIF files (place them in `Materials_gen/`):

```bash
python infer_pipeline.py
```

## Required Inputs

- **Phase 1** expects a Materials Project API key and optionally an MP ID list at `INPUT_FILE_MP_LIST`.
- **Phase 2** expects upstream data products:
  - Embedding tensors in `Embeddings/`
  - Label tensors in `Labels/`
  - Sublattice structure files in `Materials/MaterialsProject/*.xyz`
  - Flat-band lookup tables from `../crystal_net/`:
    - `TB_search_flat_bands_lat_ids.xlsx`
    - `crystal_net_results/duplicate_id_species_rcsr_mismatch.csv`
