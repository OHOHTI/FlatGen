"""
screen_sublat.py — Screen generated crystal structures.

Loads eval_gen*.pt files (either eval_gen.pt from DiffCSP++ generation or
eval_gen_wysc_*.pt chunks from sc_gen.py), runs screening (SMACT validity,
space occupation ratio, GNN stability, optional flatness), generates CIF
files for passed structures, and writes sublattice ID mapping.

Usage:
    python -m wysc.screen_sublat --chunk_dir <path> --output_dir <path>
"""

import os
import sys
import glob
import json
import argparse
import zipfile
import numpy as np
import pandas as pd
import torch
from torch import nn
from os.path import join, basename
from pathlib import Path
from collections import defaultdict
from io import StringIO

from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

# ── Path setup ───────────────────────────────────────────────────────────────

SKELEGEN_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, SKELEGEN_ROOT)

from config_skelegen import (
    gnn_eval_path, flat_screen_path, flat_model_path,
    stab_pred_name_A, stab_pred_name_B, mag_pred_name,
)

SCIGEN_ROOT = str(Path(__file__).resolve().parent.parent.parent / 'SCIGEN')
SCIGEN_SCRIPT = join(SCIGEN_ROOT, 'script')
sys.path.insert(0, SCIGEN_SCRIPT)

sys.path.append(gnn_eval_path)
sys.path.append(join(flat_screen_path, 'model'))

from mat_utils import (
    get_pstruct_list, lattice_params_to_matrix_torch,
    vol_density, charge_neutrality, ase2pmg,
)
from gnn_eval.utils.model_class import GraphNetworkClassifier
from gnn_eval.utils.model_class_mag import GraphNetworkClassifierMag
from gnn_eval.utils.data import Dataset_Cls
from gnn_eval.utils.output import generate_dataframe

torch.set_default_dtype(torch.float64)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Screen generated crystal structures from sc_gen.py chunks.",
    )
    p.add_argument('--chunk_dir', required=True,
                   help="Directory containing eval_gen*.pt files (eval_gen.pt from DiffCSP++ or eval_gen_wysc_*.pt from sc_gen.py)")
    p.add_argument('--output_dir', required=True,
                   help="Base directory for screening results (CIF dirs, logs, mapping file)")
    p.add_argument('--use_template', type=lambda x: x.lower() == 'true', default=None,
                   help="Which files to screen: True = template chunks (eval_gen_wysc_tmpl_*.pt), "
                        "False = non-template chunks (eval_gen_wysc_*.pt + eval_gen.pt), "
                        "omit = all eval_gen*.pt (default)")
    p.add_argument('--gen_cif', type=lambda x: x.lower() == 'true', default=True,
                   help="Generate CIF files for passed structures (default: True)")
    p.add_argument('--screen_mag', type=lambda x: x.lower() == 'true', default=False,
                   help="Screen for magnetic materials (default: False)")
    p.add_argument('--screen_flat', type=lambda x: x.lower() == 'true', default=True,
                   help="Screen by flatness score (default: True)")
    p.add_argument('--flat_threshold', type=float, default=0.5,
                   help="Minimum flatness score to keep a structure (default: 0.5)")
    return p.parse_args()


# ── Screening helper functions ───────────────────────────────────────────────
# Adapted from SCIGEN/script/eval_funcs.py to work with sc_gen.py's
# integer atom_types format (already argmax+1, not probability vectors).

def load_gnn_model(model_name, param_dict, device, logger):
    """Load a GNN stability classifier."""
    logger.info(f"Loading model: {model_name}")
    model = GraphNetworkClassifier(
        mul=param_dict['mul'],
        irreps_out=param_dict['irreps_out'],
        lmax=param_dict['lmax'],
        nlayers=param_dict['nlayers'],
        number_of_basis=param_dict['number_of_basis'],
        radial_layers=param_dict['radial_layers'],
        radial_neurons=param_dict['radial_neurons'],
        node_dim=param_dict['node_dim'],
        node_embed_dim=param_dict['node_embed_dim'],
        input_dim=param_dict['input_dim'],
        input_embed_dim=param_dict['input_embed_dim'],
    )
    model_file = join(gnn_eval_path, 'models', f"{model_name}.torch")
    model.load_state_dict(torch.load(model_file, map_location=device)['state'])
    return model.to(device).eval()


def load_gnn_model_mag(model_name, param_dict, device, logger):
    """Load a GNN magnetic classifier."""
    logger.info(f"Loading model (mag): {model_name}")
    model = GraphNetworkClassifierMag(
        mul=param_dict['mul'],
        irreps_out=param_dict['irreps_out'],
        lmax=param_dict['lmax'],
        nlayers=param_dict['nlayers'],
        number_of_basis=param_dict['number_of_basis'],
        radial_layers=param_dict['radial_layers'],
        radial_neurons=param_dict['radial_neurons'],
        node_dim=param_dict['node_dim'],
        node_embed_dim=param_dict['node_embed_dim'],
        input_dim=param_dict['input_dim'],
        input_embed_dim=param_dict['input_embed_dim'],
        num_classes=param_dict['num_classes'],
    )
    model_file = join(gnn_eval_path, 'models', f"{model_name}.torch")
    model.load_state_dict(torch.load(model_file, map_location=device)['state'])
    return model.to(device).eval()


def process_chunk(chunk, logger):
    """Convert a sc_gen.py chunk dict to a list of ASE Atoms structures.

    sc_gen.py saves atom_types as integers (already argmax+1), so we use
    atom_type_prob=False in get_pstruct_list.
    """
    frac_coords = chunk['frac_coords']
    atom_types = chunk['atom_types']
    lengths = chunk['lengths']
    angles = chunk['angles']
    num_atoms = chunk['num_atoms']

    logger.info(f"Lengths stats — min: {lengths.min():.4f}, max: {lengths.max():.4f}, "
                f"NaN: {torch.isnan(lengths).sum()}, <= 0: {(lengths <= 0).sum()}")
    logger.info(f"Angles stats — min: {angles.min():.4f}, max: {angles.max():.4f}, "
                f"NaN: {torch.isnan(angles).sum()}, == 0: {(angles == 0).sum()}, == 180: {(angles == 180).sum()}")

    lattices = lattice_params_to_matrix_torch(lengths, angles).to(dtype=torch.float32)
    logger.info(f"Lattices NaN count: {torch.isnan(lattices).sum()}, total elements: {lattices.numel()}")

    # atom_type_prob=False because sc_gen.py already did argmax+1
    pstruct_list = get_pstruct_list(
        num_atoms, frac_coords, atom_types, lattices, atom_type_prob=False,
    )
    astruct_list = [Atoms(AseAtomsAdaptor().get_atoms(ps)) for ps in pstruct_list]
    logger.info(f"Processed {len(astruct_list)} structures from chunk.")
    return astruct_list


def build_df(astruct_list, logger):
    """Build a screening DataFrame from ASE Atoms list."""
    rows = []
    for i, struct in enumerate(astruct_list):
        rows.append({
            'mpid': f"{i:05}",
            'structure': struct,
            'f_energy': 0.0,
            'ehull': 0.0,
            'label': 0,
            'smact_valid': charge_neutrality(struct),
            'occupy_ratio': vol_density(struct),
        })
    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} structures into DataFrame.")
    return df


def classify_stability(model, dataset, loss_fn, scaler, batch_size, device, logger):
    """Classify stability using a GNN model."""
    from torch_geometric.loader import DataLoader
    logger.info("Classifying material stability...")
    te_set = torch.utils.data.Subset(dataset, range(len(dataset)))
    loader = DataLoader(te_set, batch_size=batch_size)
    return generate_dataframe(model, loader, loss_fn, scaler, device)


def write_cif_files(df, cif_dir, logger):
    """Generate CIF files for filtered structures."""
    logger.info("Generating CIF files...")
    os.makedirs(cif_dir, exist_ok=True)
    for _, row in df.iterrows():
        mpid, astruct = row['mpid'], row['structure']
        pstruct = ase2pmg(astruct)
        filename = join(cif_dir, f"{mpid}.cif")
        try:
            pstruct.to(fmt="cif", filename=filename)
        except Exception as e:
            logger.error(f"Error generating CIF for {mpid}: {e}")

    zip_name = f"{cif_dir}.zip"
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for file in os.listdir(cif_dir):
            if file.endswith('.cif'):
                zipf.write(join(cif_dir, file), arcname=file)
    logger.info("CIF generation and zipping completed.")


def run_flatness_screening(df, threshold, batch_size, device, logger):
    """Screen structures by predicted flatness score."""
    from predict_cif import prepare_data, AtomsDataset, collate_fn, predict
    from model import Main_model
    from configs import get_cfg_defaults
    from torch.utils.data import DataLoader

    logger.info("Starting flatness screening...")

    atoms_dict = {row['mpid']: row['structure'] for _, row in df.iterrows()}
    data = prepare_data(atoms_dict)
    logger.info(f"Built graphs for {len(data)}/{len(atoms_dict)} structures.")

    dataset = AtomsDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, collate_fn=collate_fn)

    cfg = get_cfg_defaults()
    model = Main_model(**cfg).to(device)
    model.load_state_dict(torch.load(flat_model_path, map_location=device))
    model.eval()
    logger.info(f"Loaded flatness model from {flat_model_path}")

    keys, predictions = predict(model, data_loader, device)
    scores_dict = {k: float(p) for k, p in zip(keys, predictions)}

    if predictions:
        logger.info(f"Flatness scores — min: {min(predictions):.4f}, max: {max(predictions):.4f}, "
                     f"mean: {sum(predictions)/len(predictions):.4f}")

    keep_ids = [k for k, p in zip(keys, predictions) if p >= threshold]
    df_filtered = df[df['mpid'].isin(keep_ids)].reset_index(drop=True)
    logger.info(f"Flatness filter (threshold={threshold}): kept {len(df_filtered)}/{len(df)} structures.")

    return df_filtered, scores_dict


# ── Per-chunk screening pipeline ─────────────────────────────────────────────

def screen_chunk(astruct_list, model_list, model_names, loss_fn, device, logger,
                 screen_mag=False, screen_flat=True, flat_threshold=0.5):
    """Run all screening steps on one chunk. Returns (final_df, step_counts)."""
    step_counts = {}  # {step_num: (desc, passed, total)}

    df = build_df(astruct_list, logger)

    # [1] SMACT validity
    df1 = df[df['smact_valid']].reset_index(drop=True)
    logger.info(f"[1] Filtered {len(df1)}/{len(df)} materials by SMACT validity.")
    step_counts[1] = ('Filtered by SMACT validity', len(df1), len(df))

    # [2] Space occupation ratio
    ratios = df1['occupy_ratio']
    logger.info(f"[2] occupy_ratio stats — min: {ratios.min():.4f}, max: {ratios.max():.4f}, "
                f"median: {ratios.median():.4f}, mean: {ratios.mean():.4f}, NaN: {ratios.isna().sum()}")
    df2 = df1[df1['occupy_ratio'] < 1.7].reset_index(drop=True)
    logger.info(f"[2] Filtered {len(df2)}/{len(df1)} materials by space occupation ratio.")
    step_counts[2] = ('Filtered by space occupation ratio', len(df2), len(df1))

    # [3+] GNN stability classifiers
    dfs = [df2]
    for i, model_dict in enumerate(model_list):
        model = model_dict['model']
        param_dict, batch_size = model_dict['param'], model_dict['batch_size']
        dataset = Dataset_Cls(dfs[-1], **param_dict)
        result_df = classify_stability(
            model, dataset, loss_fn, param_dict['scaler'], batch_size, device, logger,
        )
        id_stable = result_df[result_df['pred'] == 1]['id'].values
        step_num = i + 3
        logger.info(f"[{step_num}] Model {model_names[i]} classified "
                     f"{len(id_stable)}/{len(dfs[-1])} materials as stable.")
        step_counts[step_num] = (
            f'Model {model_names[i]} stability',
            len(id_stable), len(dfs[-1]),
        )

        # Log confidence statistics
        raw_scores = np.array([np.atleast_1d(v).flatten()[0] for v in result_df['output'].values])
        if raw_scores.size > 0:
            logger.info(f"[{step_num}] Confidence stats — min: {raw_scores.min():.4f}, max: {raw_scores.max():.4f}, "
                         f"mean: {raw_scores.mean():.4f}, median: {np.median(raw_scores):.4f}, std: {raw_scores.std():.4f}")
            bins = np.arange(0, 1.1, 0.1)
            counts, _ = np.histogram(raw_scores, bins=bins)
            hist_str = ' '.join([f'[{bins[j]:.1f}-{bins[j+1]:.1f}]:{counts[j]}' for j in range(len(counts))])
            logger.info(f"[{step_num}] Score distribution: {hist_str}")

        df_ = dfs[-1][dfs[-1]['mpid'].isin(id_stable)].reset_index(drop=True)
        dfs.append(df_)

    # [Flatness] Optional flatness screening
    flat_scores = None
    if screen_flat:
        step_num = len(model_names) + 3
        df_flat, flat_scores = run_flatness_screening(
            dfs[-1], flat_threshold, batch_size=8, device=device, logger=logger,
        )
        logger.info(f"[{step_num}] Flatness screening kept {len(df_flat)}/{len(dfs[-1])} structures "
                     f"(threshold={flat_threshold}).")
        step_counts[step_num] = (
            f'Flatness screening (threshold={flat_threshold})',
            len(df_flat), len(dfs[-1]),
        )

        if flat_scores:
            flat_vals = np.array(list(flat_scores.values()))
            logger.info(f"[{step_num}] Flatness confidence stats — min: {flat_vals.min():.4f}, "
                         f"max: {flat_vals.max():.4f}, mean: {flat_vals.mean():.4f}, "
                         f"median: {np.median(flat_vals):.4f}, std: {flat_vals.std():.4f}")

        dfs.append(df_flat)

    return dfs[-1], step_counts, flat_scores


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Discover chunk files ─────────────────────────────────────────────
    if args.use_template is True:
        pt_files = sorted(glob.glob(join(args.chunk_dir, 'eval_gen_wysc_tmpl_*.pt')))
        mode_label = 'template (eval_gen_wysc_tmpl_*.pt)'
    elif args.use_template is False:
        all_files = sorted(glob.glob(join(args.chunk_dir, 'eval_gen*.pt')))
        pt_files = [f for f in all_files if 'tmpl' not in basename(f)]
        mode_label = 'non-template (eval_gen_wysc_*.pt + eval_gen.pt)'
    else:
        pt_files = sorted(glob.glob(join(args.chunk_dir, 'eval_gen*.pt')))
        mode_label = 'all (eval_gen*.pt)'
    print(f'Screening mode: {mode_label}')
    print(f'Found {len(pt_files)} chunk files in {args.chunk_dir}')
    if not pt_files:
        print('No chunk files found. Check --chunk_dir path and --use_template setting.')
        return

    # ── Build folder tag ─────────────────────────────────────────────────
    folder_tag = 'filtered'
    if args.screen_mag:
        folder_tag += '_mag'
    if args.screen_flat:
        folder_tag += '_flat'

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load GNN models once ─────────────────────────────────────────────
    # Set up logger (explicit handler setup — basicConfig is a no-op if root
    # logger was already configured by a transitive import like gnn_eval.utils.record)
    import logging
    log_buffer = StringIO()
    logger = logging.getLogger('screen_sublat')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    buffer_handler = logging.StreamHandler(log_buffer)
    buffer_handler.setFormatter(fmt)
    logger.addHandler(console_handler)
    logger.addHandler(buffer_handler)

    model_names = [stab_pred_name_A, stab_pred_name_B]
    if args.screen_mag:
        model_names.append(mag_pred_name)

    loss_fn = nn.BCEWithLogitsLoss(reduce=False)

    model_list = []
    for model_name in model_names:
        param_dict_path = join(gnn_eval_path, 'models', f'{model_name}_config.json')
        with open(param_dict_path, 'r') as f:
            param_dict = json.load(f)

        mag_model = False
        try:
            model = load_gnn_model(model_name, param_dict, device, logger)
        except Exception:
            model = load_gnn_model_mag(model_name, param_dict, device, logger)
            mag_model = True

        model_list.append({
            'model': model,
            'param': {k: param_dict[k] for k in ['r_max', 'descriptor', 'scaler']},
            'batch_size': param_dict['batch_size'],
        })
        model_list[-1]['param']['nearest'] = param_dict['nearest_neighbor']
        model_list[-1]['param']['target'] = 'label' if mag_model else param_dict['target']

    # ── Process each chunk ───────────────────────────────────────────────
    aggregate_counts = defaultdict(lambda: {'passed': 0, 'total': 0, 'desc': ''})
    sublat_id_records = []

    for pt_file in pt_files:
        # e.g. eval_gen_wysc_0003.pt -> gen_wysc_0003, eval_gen.pt -> gen
        fname = basename(pt_file).replace('.pt', '')
        if fname == 'eval_gen':
            label = 'gen'
        else:
            label = 'gen_' + fname.replace('eval_gen_', '')
        use_name = label
        cif_dir = join(args.output_dir, use_name + '_' + folder_tag)

        print(f'\n{"="*60}')
        print(f'Processing {basename(pt_file)} ...')
        print(f'{"="*60}')

        # Reset log buffer for this chunk
        log_buffer.truncate(0)
        log_buffer.seek(0)

        # Load chunk
        chunk = torch.load(pt_file, map_location='cpu', weights_only=False)
        logger.info(f"Loaded {basename(pt_file)}: {len(chunk['num_atoms'])} structures")

        # Process chunk -> ASE Atoms
        astruct_list = process_chunk(chunk, logger)

        # Run screening pipeline
        df_final, step_counts, flat_scores = screen_chunk(
            astruct_list, model_list, model_names, loss_fn, device, logger,
            screen_mag=args.screen_mag,
            screen_flat=args.screen_flat,
            flat_threshold=args.flat_threshold,
        )

        # Aggregate counts
        for step, (desc, passed, total) in step_counts.items():
            aggregate_counts[step]['passed'] += passed
            aggregate_counts[step]['total'] += total
            if not aggregate_counts[step]['desc']:
                aggregate_counts[step]['desc'] = desc

        # Generate CIF files
        if args.gen_cif:
            write_cif_files(df_final, cif_dir, logger)

        # Save flatness scores if screening was applied
        if args.screen_flat and flat_scores:
            os.makedirs(cif_dir, exist_ok=True)
            scores_file = join(cif_dir, f"{use_name}_flatness_scores.txt")
            with open(scores_file, 'w') as f:
                f.write("mpid\tflatness_score\n")
                for mpid, score in sorted(flat_scores.items()):
                    f.write(f"{mpid}\t{score:.6f}\n")
            logger.info(f"Flatness scores saved to {scores_file}")

        # Write per-chunk log
        os.makedirs(cif_dir, exist_ok=True)
        log_file = join(cif_dir, f"{use_name}.log")
        logger.info(f"Save log to {log_file}: {len(df_final)} materials.")
        with open(log_file, 'w') as f:
            f.write(log_buffer.getvalue())

        # Collect sublat_id mapping for this chunk
        sublat_ids = chunk.get('sublat_ids', None)
        if sublat_ids is None:
            print(f'Warning: no sublat_ids in {pt_file}, skipping ID mapping')
        elif args.gen_cif:
            cif_files = sorted(glob.glob(join(cif_dir, '*.cif')))
            for cif_file in cif_files:
                mpid = basename(cif_file).replace('.cif', '')
                idx = int(mpid)
                if idx < len(sublat_ids):
                    sublat_id_records.append((use_name, mpid, sublat_ids[idx]))

        print(f'{label} done — {len(df_final)} structures passed screening')

    # ── Aggregate summary ────────────────────────────────────────────────
    print()
    print('=' * 60)
    print('AGGREGATE SCREENING SUMMARY')
    print('=' * 60)
    for step in sorted(aggregate_counts):
        s = aggregate_counts[step]
        print(f"[{step}] {s['desc']} {s['passed']}/{s['total']}")
    print('=' * 60)

    # ── Write sublattice ID mapping ──────────────────────────────────────
    mapping_file = join(args.output_dir, 'sublat_id_mapping.txt')
    with open(mapping_file, 'w') as f:
        f.write("chunk\tmpid\tsublat_id\n")
        for chunk_name, mpid, sublat_id in sublat_id_records:
            f.write(f"{chunk_name}\t{mpid}\t{sublat_id}\n")
    print(f'Saved sublattice ID mapping for {len(sublat_id_records)} structures to {mapping_file}')


if __name__ == "__main__":
    main()
