# Shared data utilities for the GNN sublattice pipeline.
# Contains: graph construction, perturbation, I/O helpers, and the paired dataset.

import os
import re
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from ase.io import read
from joblib import Parallel, delayed
import numba as nb
from torch.utils.data import Dataset


# =========================================
# I/O and sorting helpers
# =========================================

def natural_sort_key(s):
    """Sort filenames so that numeric parts are compared numerically."""
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', s)]


def parse_pbc_from_xyz(xyz_path):
    """
    Return [pbc_x, pbc_y, pbc_z] parsed from the second line of an
    extended .xyz file.  Defaults to 3-D periodic if not found.
    """
    pbc = [True, True, True]
    try:
        with open(xyz_path, "r") as fh:
            _ = fh.readline()           # atom count
            header = fh.readline()      # comment / metadata line
    except OSError:
        return pbc

    m = re.search(r'pbc="?([TF])\s+([TF])\s+([TF])"?', header, re.I)
    if m:
        pbc = [c.upper() == "T" for c in m.groups()]
    return pbc


# =========================================
# Low-level graph building
# =========================================

@nb.njit(cache=True, fastmath=True)
def wrap_delta(delta_scaled, pbc):
    """Wrap fractional-coordinate displacements back into [-0.5, 0.5]."""
    delta_scaled = delta_scaled.copy()
    for ax, periodic in enumerate(pbc):
        if periodic:
            delta_scaled[..., ax] -= np.round(delta_scaled[..., ax])
    return delta_scaled


def compute_edge_index(atoms, delta):
    """
    Build an undirected edge index containing nearest-neighbour (NN)
    and next-nearest-neighbour (NNN) edges, with tolerance *delta* (Å).
    """
    n = len(atoms)
    edge_set = set()

    for i in range(n):
        d = atoms.get_distances(i, range(n), mic=True)
        d[i] = np.inf

        d1 = np.min(d)
        if not np.isfinite(d1):
            continue

        mask_nnn = d > (d1 + delta)
        radius = d1 + delta
        if np.any(mask_nnn):
            d2 = np.min(d[mask_nnn])
            if np.isfinite(d2):
                radius = d2 + delta

        for j in np.where(d <= radius)[0]:
            edge_set.add((min(i, j), max(i, j)))

    if not edge_set:
        return torch.empty((2, 0), dtype=torch.long)

    edge_array = np.fromiter(
        (v for pair in edge_set for v in pair), dtype=np.int64
    ).reshape(-1, 2)
    return torch.from_numpy(edge_array.T).contiguous()


def compute_edge_attr_fast(scaled_positions, cell, edge_index, pbc):
    """Edge length in Å, respecting periodic boundary conditions."""
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()

    delta_scaled = wrap_delta(scaled_positions[row] - scaled_positions[col], pbc)
    displacement_vectors = delta_scaled @ cell
    distances = np.linalg.norm(displacement_vectors, axis=1)
    return torch.from_numpy(distances).float().unsqueeze(1)


def compute_node_features(atoms, supercell_size):
    """
    Rotationally-invariant node features: per-unit-cell relative Cartesian
    coordinates expressed in the crystal-fixed orthonormal basis.
    """
    a, b, c = atoms.get_cell().array
    e1 = a / np.linalg.norm(a)
    b_orth = b - np.dot(b, e1) * e1
    e2 = b_orth / np.linalg.norm(b_orth)
    e3 = np.cross(e1, e2)
    R = np.vstack([e1, e2, e3]).T

    num_atoms = len(atoms)
    dim = int(sum(atoms.pbc))
    blocks = [supercell_size] * dim + [1] * (3 - dim)
    total_cells = np.prod(blocks)
    atoms_per_uc = num_atoms // total_cells
    if atoms_per_uc * total_cells != num_atoms:
        raise ValueError("Atom count not divisible by #unit cells.")

    pos = atoms.get_positions()
    rel_pos = np.zeros_like(pos)
    for start in range(0, num_atoms, atoms_per_uc):
        stop = start + atoms_per_uc
        ref = pos[start]
        rel_pos[start:stop] = (pos[start:stop] - ref) @ R

    return torch.tensor(rel_pos, dtype=torch.float)


def create_graph_from_structure(atoms, delta, supercell_size):
    """
    Build a PyTorch Geometric ``Data`` object from an ASE Atoms object.
    """
    edge_index_full = compute_edge_index(atoms, delta)
    scaled_positions = atoms.get_scaled_positions()
    cell = atoms.get_cell()

    edge_attr_full = compute_edge_attr_fast(
        scaled_positions, cell, edge_index_full, atoms.pbc
    )
    node_features = compute_node_features(atoms, supercell_size)

    graph = Data(
        x=node_features,
        edge_index=edge_index_full,
        edge_attr=edge_attr_full,
    )
    graph.supercell_size = supercell_size
    return graph


# =========================================
# Masking and perturbation
# =========================================

NODE_ATTRS = {"x", "pos", "batch", "y", "label"}


def apply_masking(graph: Data, rate: float, *, drop_nodes: bool = True) -> Data:
    """Randomly mask a fraction *rate* of nodes and edges."""
    g = graph.clone()

    n_nodes = g.num_nodes
    node_drop = torch.rand(n_nodes, device=g.x.device) < rate

    if node_drop.any() and drop_nodes:
        keep = ~node_drop
        g.edge_index, g.edge_attr = subgraph(
            keep, g.edge_index, g.edge_attr,
            relabel_nodes=True, num_nodes=n_nodes,
        )
        for key in NODE_ATTRS:
            val = getattr(g, key, None)
            if isinstance(val, torch.Tensor) and val.size(0) == n_nodes:
                setattr(g, key, val[keep])
    elif node_drop.any():
        g.x = g.x.clone()
        g.x[node_drop] = 0.0

    n_edges = g.edge_attr.size(0)
    edge_drop = torch.rand(n_edges, device=g.edge_attr.device) < rate
    if edge_drop.any():
        keep_e = ~edge_drop
        g.edge_index = g.edge_index[:, keep_e]
        g.edge_attr = g.edge_attr[keep_e]

    return g


def _perturb_single(graph, atoms, pbc_array, perturbation_size, delta, masking_rate):
    """
    Randomly perturb atomic positions, recompute graph attributes,
    and apply node/edge masking.
    """
    pert_graph = graph.clone()
    positions = atoms.get_positions().copy()

    if perturbation_size > 0:
        perturbation = np.random.uniform(
            -perturbation_size, perturbation_size, positions.shape
        )
        positions += perturbation

    perturbed_atoms = atoms.copy()
    perturbed_atoms.set_positions(positions)
    perturbed_atoms.pbc = pbc_array

    scaled_positions = perturbed_atoms.get_scaled_positions()
    cell = perturbed_atoms.get_cell()

    pert_graph.edge_attr = compute_edge_attr_fast(
        scaled_positions, cell, pert_graph.edge_index, pbc_array
    )
    pert_graph.x = compute_node_features(perturbed_atoms, pert_graph.supercell_size)
    pert_graph = apply_masking(pert_graph, masking_rate)

    for attr in ("atoms", "pbc_array"):
        if hasattr(pert_graph, attr):
            delattr(pert_graph, attr)
    return pert_graph


def perturb_graphs(graph_list, atoms_list, pbc_list,
                   perturbation_size, delta, masking_rate, n_jobs=4):
    """Perturb a list of graphs in parallel."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(_perturb_single)(g, a, p, perturbation_size, delta, masking_rate)
        for g, a, p in zip(graph_list, atoms_list, pbc_list)
    )
    return list(results)


# =========================================
# Single-file processing (used by preprocess)
# =========================================

def process_single_xyz_to_graph(fname, folder_path, delta, supercell_size):
    """Read one supercell .xyz file, build its graph, return (graph, atoms, pbc)."""
    fpath = os.path.join(folder_path, fname)
    pbc_array = parse_pbc_from_xyz(fpath)
    atoms = read(fpath)
    atoms.pbc = pbc_array

    fname_no_ext = os.path.splitext(fname)[0]
    prefix = "supercell_"
    label = fname_no_ext[len(prefix):] if fname_no_ext.startswith(prefix) else fname_no_ext

    graph = create_graph_from_structure(atoms, delta, supercell_size)
    graph.label = label
    graph.atoms = atoms
    graph.pbc_array = np.array(pbc_array, dtype=bool)
    return graph, atoms, pbc_array


# =========================================
# Paired dataset for contrastive training
# =========================================

class PairedGraphDataset(Dataset):
    """Pairs two graph lists element-wise for contrastive learning."""

    def __init__(self, set_1, set_2):
        assert len(set_1) == len(set_2), "Both sets must have the same length."
        self.set_1 = set_1
        self.set_2 = set_2

    def __len__(self):
        return len(self.set_1)

    def __getitem__(self, idx):
        return self.set_1[idx], self.set_2[idx]
