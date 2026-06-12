"""
Inference Pipeline: CIF files → sublattices → supercells → graphs → embeddings

Reads CIF files from Materials_gen/, splits into sublattices, builds supercells
and graphs, then runs inference with a trained GNN encoder to produce embeddings.

All intermediate files are stored in *_infer folders.
"""

import os
import re
import numpy as np
import torch
from torch_geometric.data import DataLoader
from pymatgen.core import Structure
from ase.io import read, write
from ase.build import make_supercell
from joblib import Parallel, delayed

from data_utils import (
    parse_pbc_from_xyz,
    create_graph_from_structure,
)
from gnn import GNNEncoder

# ─── Settings ───────────────────────────────────────────────────────────────
CIF_FOLDER = "Materials_gen"
SUBLATTICE_FOLDER = "Materials_gen_infer"
SUPERCELL_FOLDER = "Supercells_infer"
GRAPH_FOLDER = "Graphs_infer"
EMBEDDING_FOLDER = "Embeddings_infer"
LABEL_FOLDER = "Labels_infer"

SUPERCELL_SIZE = 3
DELTA = 0.1
BATCH_SIZE = 16
CHECKPOINT_PATH = "Models/MaterialsProject_V1_sublattice.pth"

# Model hyperparameters (must match training)
EMBEDDING_DIMENSION = 128
HIDDEN_DIMENSION = 256
LAYER_NUMBER = 4
DROPOUT_RATE = 0.1


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: CIF → Sublattice XYZ files
# ═══════════════════════════════════════════════════════════════════════════

def step1_generate_sublattices():
    print("=" * 60)
    print("Step 1: Generating sublattice XYZ files from CIF files")
    print("=" * 60)
    os.makedirs(SUBLATTICE_FOLDER, exist_ok=True)

    cif_files = sorted([f for f in os.listdir(CIF_FOLDER) if f.endswith('.cif')])
    print(f"Found {len(cif_files)} CIF files in {CIF_FOLDER}/")

    total_sublattices = 0
    for cif_file in cif_files:
        cif_path = os.path.join(CIF_FOLDER, cif_file)
        cif_stem = os.path.splitext(cif_file)[0]

        try:
            struct = Structure.from_file(cif_path)
        except Exception as e:
            print(f"  Could not read {cif_file}: {e}")
            continue

        for element in struct.composition.elements:
            specific_sites = [site for site in struct if element in site.species]
            sub_lattice = Structure.from_sites(specific_sites)

            sublat_id = f"{cif_stem}_{element.symbol}"
            xyz_path = os.path.join(SUBLATTICE_FOLDER, f"{sublat_id}.xyz")

            lattice = sub_lattice.lattice.matrix
            with open(xyz_path, "w") as f:
                f.write(f"{len(sub_lattice.sites)}\n")
                f.write(
                    f'Lattice="{lattice[0][0]} {lattice[0][1]} {lattice[0][2]} '
                    f'{lattice[1][0]} {lattice[1][1]} {lattice[1][2]} '
                    f'{lattice[2][0]} {lattice[2][1]} {lattice[2][2]}" '
                    f'Properties=species:S:1:pos:R:3 '
                    f'pbc="T T T"\n'
                )
                for site in sub_lattice.sites:
                    symbol = site.species_string
                    x, y, z = site.coords
                    f.write(f"{symbol} {x:.8f} {y:.8f} {z:.8f}\n")

            total_sublattices += 1

    print(f"Generated {total_sublattices} sublattice XYZ files in {SUBLATTICE_FOLDER}/")
    return total_sublattices


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: XYZ → Supercells
# ═══════════════════════════════════════════════════════════════════════════

def step2_generate_supercells():
    print("\n" + "=" * 60)
    print("Step 2: Generating supercells from sublattice XYZ files")
    print("=" * 60)
    os.makedirs(SUPERCELL_FOLDER, exist_ok=True)

    xyz_files = sorted([f for f in os.listdir(SUBLATTICE_FOLDER) if f.endswith('.xyz')])
    print(f"Found {len(xyz_files)} XYZ files in {SUBLATTICE_FOLDER}/")

    count = 0
    for xyz_file in xyz_files:
        filepath = os.path.join(SUBLATTICE_FOLDER, xyz_file)
        try:
            atoms = read(filepath, format='extxyz')
        except Exception as e:
            print(f"  Could not read {xyz_file}: {e}")
            continue

        pbc_array = parse_pbc_from_xyz(filepath)
        atoms.pbc = pbc_array

        transform_matrix = np.eye(3, dtype=int)
        for i in range(3):
            if pbc_array[i]:
                transform_matrix[i, i] = SUPERCELL_SIZE

        supercell = make_supercell(atoms, transform_matrix)

        material_id = os.path.splitext(xyz_file)[0]
        out_path = os.path.join(SUPERCELL_FOLDER, f"supercell_{material_id}.xyz")
        write(out_path, supercell, format='extxyz')
        count += 1

    print(f"Generated {count} supercells in {SUPERCELL_FOLDER}/")
    return count


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Supercells → Graphs (unperturbed only)
# ═══════════════════════════════════════════════════════════════════════════

def _process_single_file(fname, folder_path, delta, supercell_size):
    fpath = os.path.join(folder_path, fname)
    pbc_array = parse_pbc_from_xyz(fpath)
    atoms = read(fpath)
    atoms.pbc = pbc_array

    fname_no_ext = os.path.splitext(fname)[0]
    prefix = "supercell_"
    label = fname_no_ext[len(prefix):] if fname_no_ext.startswith(prefix) else fname_no_ext

    graph = create_graph_from_structure(atoms, delta, supercell_size)
    graph.label = label
    return graph


def step3_generate_graphs():
    print("\n" + "=" * 60)
    print("Step 3: Generating unperturbed graphs from supercells")
    print("=" * 60)
    os.makedirs(GRAPH_FOLDER, exist_ok=True)

    all_filenames = sorted(
        [f for f in os.listdir(SUPERCELL_FOLDER) if f.endswith('.xyz')],
        key=lambda s: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', s)]
    )
    print(f"Found {len(all_filenames)} supercell files in {SUPERCELL_FOLDER}/")

    n_jobs = os.cpu_count() or 1
    graph_list = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_file)(fname, SUPERCELL_FOLDER, DELTA, SUPERCELL_SIZE)
        for fname in all_filenames
    )

    out_path = os.path.join(GRAPH_FOLDER, "graph_list_unperturbed.pt")
    torch.save(graph_list, out_path)
    print(f"Saved {len(graph_list)} graphs to {out_path}")
    return graph_list


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Inference with trained GNN
# ═══════════════════════════════════════════════════════════════════════════

def step4_inference(graph_list):
    print("\n" + "=" * 60)
    print("Step 4: Running inference with trained GNN encoder")
    print("=" * 60)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"No checkpoint found at {CHECKPOINT_PATH}")

    os.makedirs(EMBEDDING_FOLDER, exist_ok=True)
    os.makedirs(LABEL_FOLDER, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_node_features = graph_list[0].x.size(1)
    num_edge_features = graph_list[0].edge_attr.size(1)

    model = GNNEncoder(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=HIDDEN_DIMENSION,
        embedding_dim=EMBEDDING_DIMENSION,
        num_layers=LAYER_NUMBER,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False))
    print(f"Loaded model weights from {CHECKPOINT_PATH}")

    model.eval()
    loader = DataLoader(graph_list, batch_size=BATCH_SIZE, shuffle=False)

    embeddings_list = []
    labels_list = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            embeddings = model(data)
            embeddings_list.append(embeddings.cpu())
            labels_list.extend(data.label)

    all_embeddings = torch.cat(embeddings_list, dim=0)
    print(f"Generated embeddings shape: {all_embeddings.shape}")
    print(f"Number of labels: {len(labels_list)}")
    print(f"Sample labels: {labels_list[:5]}")

    emb_path = os.path.join(EMBEDDING_FOLDER, "embeddings.pt")
    lab_path = os.path.join(LABEL_FOLDER, "labels.pt")
    torch.save(all_embeddings, emb_path)
    torch.save(labels_list, lab_path)
    print(f"Embeddings saved to {emb_path}")
    print(f"Labels saved to {lab_path}")

    return all_embeddings, labels_list


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    step1_generate_sublattices()
    step2_generate_supercells()
    graph_list = step3_generate_graphs()
    embeddings, labels = step4_inference(graph_list)
    print("\n" + "=" * 60)
    print("Inference pipeline complete!")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Labels:     {len(labels)} entries")
    print("=" * 60)
