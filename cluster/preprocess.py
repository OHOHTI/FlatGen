# Preprocessing pipeline: sublattice generation → supercell generation → graph generation
#
# Can be run standalone or called from run_pipeline.py.

import os
import re
import glob
import time
import numpy as np
import torch
import pandas as pd
import ast
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from ase.io import read, write
from ase.build import make_supercell
from joblib import Parallel, delayed, parallel_backend
from pymatgen.core import Structure
from monty.serialization import MontyDecoder

import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)

from config import (
    DATASET,
    API_KEY,
    INPUT_FILE_MP_LIST,
    SUPERCELL_SIZE_1,
    SUPERCELL_SIZE_2,
    MODEL_NAME,
)

from data_utils import (
    natural_sort_key,
    parse_pbc_from_xyz,
    perturb_graphs,
    process_single_xyz_to_graph,
)


# =========================================
# Constants
# =========================================

DELTA = 0.1              # Tolerance for nearest-neighbour cut-off (Å)
PERTURBATION_SIZE = 0.15 # Max per-atom random perturbation (Å)
MASKING_RATE = 0.3       # Node / edge masking rate for augmented views
CHUNK_SIZE = 5000        # Structures per disk chunk during graph generation

MATERIALS_FOLDER = f"Materials/{DATASET}"
SUPERCELL_FOLDER_1 = f"Supercells/{DATASET}_{SUPERCELL_SIZE_1}x{SUPERCELL_SIZE_1}"
SUPERCELL_FOLDER_2 = f"Supercells/{DATASET}_{SUPERCELL_SIZE_2}x{SUPERCELL_SIZE_2}"
OUTPUT_FOLDER = f"Graphs/{MODEL_NAME}"
CHUNKED_FOLDER = "Graphs/temp_chunked"


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Sublattice generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_sublattices(output_folder=None, max_atoms=40):
    """
    Download structures from Materials Project and split them into
    single-element sublattice .xyz files.
    """
    if output_folder is None:
        output_folder = MATERIALS_FOLDER

    if not API_KEY or not isinstance(API_KEY, str) or len(API_KEY.strip()) == 0:
        print("[Sublattice] No valid API_KEY provided. Skipping.")
        return

    os.makedirs(output_folder, exist_ok=True)

    df_ids = pd.read_csv(INPUT_FILE_MP_LIST, sep=',')
    mp_ids = df_ids['material_id'].tolist()
    print(mp_ids[:5], len(mp_ids))

    sublattices = {}
    for i, row in df_ids.iterrows():
        mp_id = row['material_id']
        try:
            struct_str = row['structure']
            struct_dict = ast.literal_eval(struct_str)
            struct = MontyDecoder().process_decoded(struct_dict)
            for element in struct.composition.elements:
                specific_sites = [site for site in struct if element in site.species]
                sub_lattice = Structure.from_sites(specific_sites)
                sublat_id = f"{mp_id}_{element.symbol}"
                sublattices[sublat_id] = sub_lattice
        except Exception as e:
            print(f"[Sublattice] Could not process {mp_id}: {e}")

    print(f"[Sublattice] Found {len(sublattices)} sublattices. Storing structures...")

    def _save_xyz(sublat_id, structure):
        try:
            if len(structure.sites) > max_atoms:
                return f"Skipped: {sublat_id} ({len(structure.sites)} atoms > {max_atoms})"
            lattice = structure.lattice.matrix
            pbc = structure.pbc
            xyz_filename = os.path.join(output_folder, f"{sublat_id}.xyz")
            with open(xyz_filename, "w") as f:
                f.write(f"{len(structure.sites)}\n")
                f.write(
                    f'Lattice="{lattice[0][0]} {lattice[0][1]} {lattice[0][2]} '
                    f'{lattice[1][0]} {lattice[1][1]} {lattice[1][2]} '
                    f'{lattice[2][0]} {lattice[2][1]} {lattice[2][2]}" '
                )
                f.write('Properties=species:S:1:pos:R:3 ')
                f.write(f'pbc="{str(pbc[0])[0]} {str(pbc[1])[0]} {str(pbc[2])[0]}"\n')
                for site in structure.sites:
                    symbol = site.species_string
                    x, y, z = site.coords
                    f.write(f"{symbol} {x:.8f} {y:.8f} {z:.8f}\n")
            return f"Saved: {xyz_filename}"
        except Exception as e:
            return f"Failed: {sublat_id}, Error: {e}"

    num_workers = os.cpu_count() or 1
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(
            lambda item: _save_xyz(item[0], item[1]), sublattices.items()
        )
    for res in results:
        print(res)
    elapsed = time.time() - start_time
    print(f"[Sublattice] Completed in {elapsed:.2f} seconds!")


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Supercell generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_supercells():
    """
    Expand each sublattice .xyz file into supercells for each configured size.
    """
    sizes = list(set([SUPERCELL_SIZE_1, SUPERCELL_SIZE_2]))

    output_dirs = {}
    for size in sizes:
        out_dir = f"Supercells/{DATASET}_{size}x{size}"
        os.makedirs(out_dir, exist_ok=True)
        output_dirs[size] = out_dir

    def _process_file(xyz_file):
        filepath = os.path.join(MATERIALS_FOLDER, xyz_file)
        try:
            atoms = read(filepath, format='extxyz')
        except Exception as e:
            print(f"Could not read {xyz_file}: {e}")
            return

        pbc_array = parse_pbc_from_xyz(filepath)
        atoms.pbc = pbc_array
        material_id = os.path.splitext(xyz_file)[0]

        for size in sizes:
            transform_matrix = np.eye(3, dtype=int)
            for i in range(3):
                if pbc_array[i]:
                    transform_matrix[i, i] = size
            supercell = make_supercell(atoms, transform_matrix)
            out_path = os.path.join(output_dirs[size], f"supercell_{material_id}.xyz")
            write(out_path, supercell, format='extxyz')

    num_workers = os.cpu_count() or 1
    print(f"[Supercell] Using {num_workers} workers for parallel processing.")

    xyz_files = sorted(
        [f for f in os.listdir(MATERIALS_FOLDER) if f.endswith('.xyz')],
        key=natural_sort_key,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(_process_file, xyz_files))

    print("[Supercell] All supercells written to their respective folders.")


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Graph generation (chunked)
# ═══════════════════════════════════════════════════════════════════════════

def _process_folder_in_chunks(
    folder_path,
    supercell_size,
    n_jobs,
    output_prefix,
    *,
    save_unperturbed=True,
):
    """
    Read .xyz files from *folder_path* in chunks, build unperturbed and
    perturbed graph lists, and save each chunk to disk.
    """
    os.makedirs(CHUNKED_FOLDER, exist_ok=True)

    all_filenames = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.xyz')],
        key=natural_sort_key,
    )

    for chunk_idx in range(0, len(all_filenames), CHUNK_SIZE):
        chunk_fnames = all_filenames[chunk_idx: chunk_idx + CHUNK_SIZE]
        print(f"=== Processing chunk {chunk_idx}, {len(chunk_fnames)} files ===")

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_xyz_to_graph)(
                fname, folder_path, DELTA, supercell_size
            )
            for fname in chunk_fnames
        )
        chunk_graphs, chunk_atoms, chunk_pbcs = zip(*results)
        chunk_graphs = list(chunk_graphs)
        chunk_atoms = list(chunk_atoms)
        chunk_pbcs = list(chunk_pbcs)

        chunk_perturbed = perturb_graphs(
            chunk_graphs, chunk_atoms, chunk_pbcs,
            PERTURBATION_SIZE, DELTA, MASKING_RATE, n_jobs=n_jobs,
        )

        if save_unperturbed:
            unpert_file = os.path.join(
                CHUNKED_FOLDER, f"{output_prefix}_unpert_chunk_{chunk_idx}.pt"
            )
            torch.save(chunk_graphs, unpert_file)
            print(f"    Saved unperturbed to: {unpert_file}")

        pert_file = os.path.join(
            CHUNKED_FOLDER, f"{output_prefix}_pert_chunk_{chunk_idx}.pt"
        )
        torch.save(chunk_perturbed, pert_file)
        print(f"    Saved perturbed   to: {pert_file}")

        del chunk_graphs, chunk_atoms, chunk_pbcs, chunk_perturbed

    print("=== Done processing all chunks. ===")


def _collect_and_save_combined_lists(prefix_1, prefix_2):
    """
    Gather chunk files for two passes and combine them into four final
    .pt graph lists, then delete the temporary chunks.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    def _combine(pattern, out_path):
        chunk_files = sorted(glob.glob(pattern))
        if not chunk_files:
            print(f"No chunk files matching {pattern} — skipping.")
            return
        combined = []
        for cf in chunk_files:
            graphs = torch.load(cf, weights_only=False)
            for g in graphs:
                for attr in ("atoms", "pbc_array"):
                    if hasattr(g, attr):
                        delattr(g, attr)
            combined.extend(graphs)
        torch.save(combined, out_path)
        print(f"Saved {len(combined)} graphs → {out_path}")
        for cf in chunk_files:
            os.remove(cf)
        del combined

    _combine(
        os.path.join(CHUNKED_FOLDER, f"{prefix_1}_unpert_chunk_*.pt"),
        os.path.join(OUTPUT_FOLDER, "graph_list_unperturbed_1.pt"),
    )
    _combine(
        os.path.join(CHUNKED_FOLDER, f"{prefix_2}_unpert_chunk_*.pt"),
        os.path.join(OUTPUT_FOLDER, "graph_list_unperturbed_2.pt"),
    )
    _combine(
        os.path.join(CHUNKED_FOLDER, f"{prefix_1}_pert_chunk_*.pt"),
        os.path.join(OUTPUT_FOLDER, "graph_list_set_1.pt"),
    )
    _combine(
        os.path.join(CHUNKED_FOLDER, f"{prefix_2}_pert_chunk_*.pt"),
        os.path.join(OUTPUT_FOLDER, "graph_list_set_2.pt"),
    )
    print("All final graph lists combined and chunk files removed.")


def generate_graphs():
    """
    Build unperturbed + perturbed graph lists from the supercell folders.
    Handles the special case where both supercell sizes are equal.
    """
    n_jobs = os.cpu_count()
    os.makedirs(CHUNKED_FOLDER, exist_ok=True)

    same_size = (SUPERCELL_SIZE_1 == SUPERCELL_SIZE_2)
    prefix_1 = f"graph_list_{SUPERCELL_SIZE_1}x{SUPERCELL_SIZE_1}_pass1"
    prefix_2 = f"graph_list_{SUPERCELL_SIZE_2}x{SUPERCELL_SIZE_2}_pass2"

    with parallel_backend("loky", inner_max_num_threads=1):
        # Pass 1 — always
        _process_folder_in_chunks(
            folder_path=SUPERCELL_FOLDER_1,
            supercell_size=SUPERCELL_SIZE_1,
            n_jobs=n_jobs,
            output_prefix=prefix_1,
            save_unperturbed=True,
        )

        if same_size:
            # Re-perturb the unperturbed chunks from pass 1
            unpert_chunk_paths = sorted(
                glob.glob(os.path.join(
                    CHUNKED_FOLDER, f"{prefix_1}_unpert_chunk_*.pt"
                ))
            )
            for cf in unpert_chunk_paths:
                graphs = torch.load(cf, weights_only=False)
                atoms_list = [g.atoms for g in graphs]
                pbc_list = [g.pbc_array for g in graphs]
                new_pert = perturb_graphs(
                    graphs, atoms_list, pbc_list,
                    PERTURBATION_SIZE, DELTA, MASKING_RATE, n_jobs=n_jobs,
                )
                torch.save(
                    new_pert,
                    cf.replace(f"{prefix_1}_unpert", f"{prefix_2}_pert"),
                )
                del graphs, atoms_list, pbc_list, new_pert
        else:
            # Pass 2 — different supercell size
            _process_folder_in_chunks(
                folder_path=SUPERCELL_FOLDER_2,
                supercell_size=SUPERCELL_SIZE_2,
                n_jobs=n_jobs,
                output_prefix=prefix_2,
                save_unperturbed=True,
            )

        # Combine all chunks into final lists
        _collect_and_save_combined_lists(prefix_1, prefix_2)

    print("=== Graph Generation Completed ===")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("---- Step 1: Sublattice Generation ----")
    generate_sublattices()

    print("\n---- Step 2: Supercell Generation ----")
    generate_supercells()

    print("\n---- Step 3: Graph Generation ----")
    generate_graphs()

    print("\n=== Preprocessing Pipeline Completed ===")
