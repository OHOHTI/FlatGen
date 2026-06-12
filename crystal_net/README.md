# crystal_net

`crystal_net` is the crystal-sublattice tight-binding stage used in the broader FlatGen workflow. It takes a list of candidate materials, builds element-resolved nearest-neighbor tight-binding models, filters for non-trivial flat-band sublattices, and then groups the resulting crystal nets with Systre and graph invariants.

This release keeps the files needed to rerun the maintained screening and post-processing workflow. 

## Included Files

- search and post-processing scripts: `main.py`, `collect_and_output_new.py`, `get_graph_invariants.py`, `output_graph_invariants.py`
- runtime modules: `config.py`, `io_utils.py`, `physics_engine.py`, `plot_utils.py`
- batch helper: `submit_screening.sh`
- search inputs and external classifier: `MP_95.txt`, `Systre-exp-fix2.jar`
- environment files: `environment.yml`, `requirements.txt`

## Setup

```bash
conda env create -f environment.yml
conda activate crynet
```

This module also requires a working Java installation because Systre is invoked through `Systre-exp-fix2.jar`.

## Materials Project Access

The published copy keeps the Materials Project key empty in [config.py](/home/yihao/vaebo/Flatgen/crystal_net/config.py). Add your own local key there only if you need to download structures instead of using an existing cache.

The search code expects one of the following:

- cached metadata and structures already present as `all_mp_IDs.txt`, `all_mp_names.txt`, `all_mp_nsites.txt`, and `mp_structs/*.cif`
- or a fresh initialization from the included `MP_95.txt` list using the download block in [main.py](/home/yihao/vaebo/Flatgen/crystal_net/main.py)

## Minimal Workflow

1. Run the tight-binding search:

```bash
python main.py
```

This writes per-run outputs under `crystal_net_results/TB_dmax*_decay*/`.

2. Aggregate and classify the screened sublattices:

```bash
bash submit_screening.sh
```

This runs:

- `collect_and_output_new.py`
- `get_graph_invariants.py`
- `output_graph_invariants.py`

## Generated Outputs

The workflow will create data and summary artifacts such as:

- `crystal_net_results/`
- `common_lats/`
- `collision_graph_invariants_sorted_unif/`
- `coll_pie.png`

These generated directories are excluded from this curated copy on purpose.

## Notes

- `get_graph_invariants.py` in this copy has been trimmed to depend on the maintained local modules instead of the older legacy `main_search.py`.
- `submit_screening.sh` now changes into its own directory before running, so it can be launched from outside `Flatgen/crystal_net`.
