"""
Combined Pipeline: Steps 1–4 → WyckoffSubLatDataset

Loads sublattices from step2_candidates.pkl, runs Step 3 (Wyckoff assignment),
fits/loads the DP sampler (Step 4), and builds a dataset where each sample has:

  - DiffCSP++ Wyckoff fields: spacegroup, ops, anchor_index, ops_inv
  - SCIGEN structural constraint fields: frac_coords, atom_types,
    lattice_known, lengths, angles, constrained_indices, spectator_indices
  - Metadata: sublat_id, num_known, num_atoms

Lattice lengths are sampled via KDE (matching SCIGEN's sublat_sc.py), and
atom types for constrained atoms are randomly drawn from known_species.

Usage:
    python build_dataset.py                          # build and save from wysc/
    python -m wysc.build_dataset                     # build and save
    python -m wysc.build_dataset --save_path out.pt  # custom output path
"""

from __future__ import annotations

import os
import pickle
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from pyxtal.symmetry import Group

try:
    from .dp_sampling import WyckoffDPSampler, wyckoff_multiplicity
    from .wysc_utils import find_all_valid_assignments
except ImportError:
    # Allow direct execution via `python build_dataset.py` from this directory.
    from dp_sampling import WyckoffDPSampler, wyckoff_multiplicity
    from wysc_utils import find_all_valid_assignments

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR       = Path(__file__).resolve().parent.parent / "data"
CANDIDATES_PKL = DATA_DIR / "step2_candidates.pkl"
WYCKOFF_CSV    = Path(__file__).resolve().parent / "wyckoff_data.csv"
DEFAULT_OUTPUT = DATA_DIR / "wysc_dataset.pt"
DEFAULT_SAMPLER = DATA_DIR / "wyckoff_sampler.json"
DEFAULT_PRECOMPUTE = DATA_DIR / "precomputed_assignments.pkl"
KDE_BOND_PKL   = Path(__file__).resolve().parent.parent.parent / "SCIGEN" / "data" / "kde_bond.pkl"

# ── Atom symbol table (matches DiffCSP++ / SCIGEN convention) ────────────────
chemical_symbols = [
    'X',
    'H', 'He',
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og',
]
rev_chemical_symbols = {ch: i for i, ch in enumerate(chemical_symbols)}

MAX_ATOMIC_NUM = 100

DEFAULT_NON_TRICLINIC_WEIGHT = 5.0
DEFAULT_SKELETON_SYMMETRY_POWER = 1.2

# ── Known species list (copied from sublat_sc.py) ───────────────────────────
KNOWN_SPECIES = [
    "O", "S", "Se", "Te",             # Chalcogens
    "N", "P", "As", "Sb", "Bi",       # Pnictides
    "F", "Cl", "Br", "I",             # Halogens
    "B", "C", "Si", "Ge",             # Metalloids / Light non-metals
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
    "La", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    "Zn", "Hg", "Ce", "Cd", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm",
]

# ── Metallic radius (copied from sublat_sc.py) ──────────────────────────────
metallic_radius = {
    "Sc": 1.641, "Ti": 1.462, "V": 1.347, "Cr": 1.285, "Mn": 1.292,
    "Fe": 1.277, "Co": 1.250, "Ni": 1.246, "Cu": 1.278, "Y": 1.800,
    "Zr": 1.602, "Nb": 1.473, "Mo": 1.402, "Tc": 1.363, "Ru": 1.339,
    "Rh": 1.345, "Pd": 1.376, "Ag": 1.445, "La": 1.877, "Yb": 1.940,
    "Lu": 1.735, "Hf": 1.580, "Ta": 1.470, "W": 1.410, "Re": 1.375,
    "Os": 1.352, "Ir": 1.357, "Pt": 1.387, "Au": 1.442, "Zn": 1.249,
    "Hg": 1.440, "Ce": 1.824, "Cd": 1.413, "Pr": 1.828, "Nd": 1.821,
    "Pm": 1.811, "Sm": 1.804, "Eu": 2.041, "Gd": 1.802, "Tb": 1.781,
    "Dy": 1.773, "Ho": 1.765, "Er": 1.756, "Tm": 1.747,
}

# ── KDE bond length distributions (loaded once) ─────────────────────────────
_kde_dict = None

def _get_kde_dict():
    global _kde_dict
    if _kde_dict is None:
        with open(KDE_BOND_PKL, "rb") as f:
            _kde_dict = pickle.load(f)
    return _kde_dict


# ── Lattice helpers (matches SCIGEN convention: a along x, b in xy-plane) ───
def lattice_params_to_matrix(lengths: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Convert (a,b,c), (alpha,beta,gamma) in degrees to 3×3 matrix.
    Batched: lengths (B,3), angles (B,3) → (B,3,3).
    Matches lattice_params_to_matrix_xy_torch from SCIGEN."""
    angles_r = torch.deg2rad(angles)
    alpha, beta, gamma = angles_r[..., 0], angles_r[..., 1], angles_r[..., 2]
    cos_a = torch.cos(alpha)
    cos_b = torch.cos(beta)
    cos_g = torch.cos(gamma)
    sin_g = torch.sin(gamma)

    va = torch.stack([lengths[..., 0],
                      torch.zeros_like(lengths[..., 0]),
                      torch.zeros_like(lengths[..., 0])], dim=-1)
    vb = torch.stack([lengths[..., 1] * cos_g,
                      lengths[..., 1] * sin_g,
                      torch.zeros_like(lengths[..., 1])], dim=-1)
    vc_x = lengths[..., 2] * cos_b
    vc_y = lengths[..., 2] * (cos_a - cos_b * cos_g) / sin_g
    vc_z = lengths[..., 2] * torch.sqrt(
        torch.clamp(torch.sin(beta)**2 - ((cos_a - cos_b * cos_g) / sin_g)**2, min=0)
    )
    vc = torch.stack([vc_x, vc_y, vc_z], dim=-1)

    return torch.stack([va, vb, vc], dim=-2)


def nn_distance(structure) -> float:
    """Nearest-neighbor distance in a pymatgen Structure (matches sublat_sc.py)."""
    neighbors = structure.get_neighbors(structure[0], 7.0)
    if neighbors:
        return min(n.nn_distance for n in neighbors)
    return 0.0


def sample_bond_length(species: str, rng: random.Random,
                       bond_sigma_per_mu: float | None = None,
                       use_min_bond_len: bool = False) -> float:
    """Sample a bond length for the given species, matching sublat_sc.py logic.

    If bond_sigma_per_mu is set: sample from N(2*r, 2*r*sigma_per_mu).
    Otherwise: sample from KDE distribution (default in SCIGEN).
    """
    if bond_sigma_per_mu is not None and species in metallic_radius:
        mu = 2 * metallic_radius[species]
        sigma = mu * bond_sigma_per_mu
        return rng.gauss(mu, sigma)
    else:
        kde = _get_kde_dict()
        if species in kde:
            bl = float(kde[species].resample(1).item())
        else:
            bl = 2.5  # fallback
        if use_min_bond_len and species in metallic_radius:
            bl = max(bl, 2 * metallic_radius[species])
        return bl


def sample_spectator_types_by_orbit(
    spectator_letters: Sequence[str],
    group: Group,
    rng: random.Random,
) -> list[int]:
    """Sample one species per spectator orbit and expand across multiplicity."""
    spectator_type_indices: list[int] = []
    for letter in spectator_letters:
        orbit_species = rng.choice(chemical_symbols[1:MAX_ATOMIC_NUM + 1])
        orbit_type_idx = rev_chemical_symbols[orbit_species]
        spectator_type_indices.extend([orbit_type_idx] * group[letter].multiplicity)
    return spectator_type_indices


# ── Build Wyckoff ops from assignment (DiffCSP++ format) ────────────────────
def build_wyckoff_ops(
    sg_number: int,
    wyckoff_letters: List[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Build DiffCSP++-compatible Wyckoff operation tensors.

    wyckoff_letters: one entry per orbit, e.g. ['f', 'a', 'b', 'f'].

    Returns (ops, anchor_index, ops_inv, num_atoms).
    """
    g = Group(sg_number)
    ops_list = []
    anchor_list = []
    num_atoms = 0

    for letter in wyckoff_letters:
        wp = g[letter]
        for op in wp:
            ops_list.append(op.affine_matrix)
            anchor_list.append(num_atoms)
        num_atoms += wp.multiplicity

    ops = torch.FloatTensor(np.array(ops_list))      # (M, 4, 4)
    anchor_index = torch.LongTensor(anchor_list)      # (M,)
    ops_inv = torch.linalg.pinv(ops[:, :3, :3])       # (M, 3, 3)
    return ops, anchor_index, ops_inv, num_atoms


# ── Core: build one sample ───────────────────────────────────────────────────
def build_sample(
    sublattice,                        # pymatgen Structure (sublattice geometry)
    sublat_id: str,
    type_known: str,                   # randomly sampled species for constrained atoms
    bond_len: float,                   # sampled bond length for lattice rescaling
    sg_full: int,
    constrained_assignments: List[dict],
    spectator_letters: List[str],
    rng: random.Random,
) -> Data:
    """
    Build a single Data sample combining Wyckoff + sublattice constraints.

    Lattice is rescaled by bond_len/nn_dist (matching SCIGEN's SubLatC_Full).
    Constrained atom types = type_known.
    Spectator atom types are sampled once per orbit and repeated across that
    orbit's multiplicity so symmetry-equivalent spectator sites share a species.
    """
    # ── Wyckoff letters: constrained orbits first, then spectators ───────────
    constrained_letters = [a["letter"] for a in constrained_assignments]
    all_letters = constrained_letters + spectator_letters

    group = Group(sg_full)
    num_constrained = sublattice.num_sites
    num_spectator = sum(group[l].multiplicity for l in spectator_letters)
    num_total = num_constrained + num_spectator

    # ── Wyckoff ops (DiffCSP++ format) ───────────────────────────────────────
    ops, anchor_index, ops_inv, num_atoms_check = build_wyckoff_ops(
        sg_full, all_letters,
    )
    assert num_atoms_check == num_total

    # ── Lattice rescaling (matches SubLatC_Full in sublat_sc.py) ─────────────
    # bond_len_sublat = nn_distance of the sublattice
    # a_scale = a / bond_len_sublat; then a_len = a_scale * bond_len
    nn_dist = nn_distance(sublattice)
    if nn_dist == 0:
        nn_dist = 1.0  # fallback
    lat = sublattice.lattice
    a_scale = lat.a / nn_dist
    b_scale = lat.b / nn_dist
    c_scale = lat.c / nn_dist

    lengths_t = torch.tensor(
        [a_scale * bond_len, b_scale * bond_len, c_scale * bond_len],
        dtype=torch.float,
    )
    angles_t = torch.tensor(
        [lat.alpha, lat.beta, lat.gamma],
        dtype=torch.float,
    )
    cell = lattice_params_to_matrix(
        lengths_t.unsqueeze(0), angles_t.unsqueeze(0),
    ).squeeze(0)  # (3, 3)

    # ── Fractional coordinates ───────────────────────────────────────────────
    # Use adjusted_coords from the Wyckoff assignment (snapped to ideal orbit
    # positions) rather than the raw sublattice fractional coordinates.  The
    # raw coords can be up to SNAP_TOL=0.1 off the ideal Wyckoff site, which
    # is enough to prevent spglib from detecting the intended space group.
    frac_coords = torch.zeros(num_total, 3)
    frac_known = torch.cat(
        [torch.tensor(a["adjusted_coords"], dtype=torch.float) % 1.0
         for a in constrained_assignments],
        dim=0,
    )  # (num_constrained, 3) — one row per constrained atom, Wyckoff-op order
    frac_coords[:num_constrained] = frac_known

    # ── Atom types ───────────────────────────────────────────────────────────
    # Constrained: all get type_known (randomly sampled species)
    type_idx_known = rev_chemical_symbols[type_known]
    # Spectator: sample one species per orbit, then expand across multiplicity
    spectator_type_indices = sample_spectator_types_by_orbit(
        spectator_letters=spectator_letters,
        group=group,
        rng=rng,
    )

    atom_types = torch.zeros(num_total, dtype=torch.long)
    atom_types[:num_constrained] = type_idx_known
    atom_types[num_constrained:] = torch.LongTensor(spectator_type_indices)

    # ── Constraint index sets ────────────────────────────────────────────────
    constrained_indices = torch.arange(num_constrained, dtype=torch.long)
    spectator_indices = torch.arange(num_constrained, num_total, dtype=torch.long)

    # ── Per-atom boolean mask (batches correctly in PyG via concatenation) ──
    is_constrained = torch.zeros(num_total, dtype=torch.bool)
    is_constrained[:num_constrained] = True

    # ── Assemble Data ────────────────────────────────────────────────────────
    data = Data(
        # DiffCSP++ Wyckoff fields
        spacegroup=torch.LongTensor([sg_full]),
        ops=ops,                               # (M, 4, 4)
        anchor_index=anchor_index,             # (M,)
        ops_inv=ops_inv,                       # (M, 3, 3)
        # Atom counts
        num_atoms=torch.LongTensor([num_total]),
        num_nodes=num_total,
        num_known=num_constrained,
        num_spectator=num_spectator,
        # Coordinates
        frac_coords=frac_coords,               # (N, 3) — known in [:num_known]
        # Atom types
        atom_types=atom_types,                 # (N,) — known + random spectators
        # Lattice (rescaled)
        lattice_known=cell.unsqueeze(0),       # (1, 3, 3)
        lengths=lengths_t.unsqueeze(0),        # (1, 3)
        angles=angles_t.unsqueeze(0),          # (1, 3)
        # Constraint masks and indices
        is_constrained=is_constrained,            # (N,) bool
        constrained_indices=constrained_indices,  # (num_known,)
        spectator_indices=spectator_indices,      # (num_spectator,)
        # Metadata
        sublat_id=sublat_id,
        type_known=type_known,
        wyckoff_letters=all_letters,
        constrained_wyckoff_letters=constrained_letters,
        spectator_wyckoff_letters=spectator_letters,
    )
    return data


# ── Dataset class ────────────────────────────────────────────────────────────
class WyckoffSubLatDataset(Dataset):
    """
    Dataset combining Wyckoff symmetry constraints (DiffCSP++) with
    sublattice structural constraints (SCIGEN).

    Each sample is a torch_geometric Data object ready for the combined
    diffusion process (Step 5).
    """

    def __init__(self, data_list: List[Data]):
        super().__init__()
        self.data_list = data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def save(self, path: str | Path) -> None:
        torch.save(self.data_list, path)
        print(f"Saved {len(self.data_list)} samples → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "WyckoffSubLatDataset":
        data_list = torch.load(path, weights_only=False)
        print(f"Loaded {len(data_list)} samples from {path}")
        return cls(data_list)


# ── Precompute all valid g_full assignments per sublattice ──────────────────
def precompute_assignments(
    entries: list,
    sampler: WyckoffDPSampler,
    natm_max: int,
    cache_path: Path | None = None,
) -> list[dict]:
    """
    For each sublattice entry, find ALL valid g_full Wyckoff assignments.

    Returns a list of dicts, each with:
        sublattice, sublat_id, num_known, valid_assignments
    Only sublattices with at least one valid assignment are included.
    Results are cached to cache_path if provided.
    """
    if cache_path and cache_path.exists():
        print(
            f"Loading precomputed assignments from {cache_path} "
            f"(already filtered by natm_max / assignment / sampler support)"
        )
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    precomputed = []
    stats = {
        "total": len(entries),
        "no_viable": 0,
        "with_viable": 0,
        "no_assignment": 0,
        "with_assignment": 0,
        "no_sampler_support": 0,
        "with_valid": 0,
    }

    for i, entry in enumerate(entries):
        sublattice = entry["sublattice"]
        frac_coords = sublattice.frac_coords
        candidates = entry["g_full_candidates"]
        sublat_id = entry["full_id"]
        num_known = sublattice.num_sites

        # Filter candidates by atom count budget
        viable_candidates = []
        for sg in candidates:
            g = Group(sg)
            min_mult = min(wp.multiplicity for wp in g)
            if num_known + min_mult <= natm_max:
                viable_candidates.append(sg)

        if not viable_candidates:
            stats["no_viable"] += 1
            continue
        stats["with_viable"] += 1

        # Find ALL valid assignments
        all_valid = find_all_valid_assignments(frac_coords, viable_candidates)
        if not all_valid:
            stats["no_assignment"] += 1
            continue
        stats["with_assignment"] += 1

        # Filter by DP sampler coverage and precompute constrained info
        valid_assignments = []
        for sg_full, assignments, adjusted_coords in all_valid:
            if sg_full not in sampler.sg_stats:
                continue

            constrained_letters = [a["letter"] for a in assignments]
            group = Group(sg_full)
            known_counts_labeled: Dict[str, int] = {}
            for letter, count in Counter(constrained_letters).items():
                wp = group[letter]
                label = f"{wp.multiplicity}{letter}"
                known_counts_labeled[label] = count

            valid_assignments.append({
                "sg_full": sg_full,
                "assignments": assignments,
                "adjusted_coords": adjusted_coords,
                "constrained_letters": constrained_letters,
                "known_counts_labeled": known_counts_labeled,
            })

        if not valid_assignments:
            stats["no_sampler_support"] += 1
            continue
        stats["with_valid"] += 1

        precomputed.append({
            "sublattice": sublattice,
            "sublat_id": sublat_id,
            "num_known": num_known,
            "valid_assignments": valid_assignments,
        })

        if (i + 1) % 50 == 0:
            print(f"  Precompute {i + 1}/{len(entries)}  |  "
                  f"viable={stats['with_viable']}  "
                  f"assigned={stats['with_assignment']}  "
                  f"kept={stats['with_valid']}")

    print("Precompute done:")
    print(f"  Total sublattices: {stats['total']}")
    print(f"  Within natm_max={natm_max}: {stats['with_viable']}")
    print(f"  Excluded by natm_max budget: {stats['no_viable']}")
    print(f"  With at least one Wyckoff assignment: {stats['with_assignment']}")
    print(f"  Excluded by assignment failure: {stats['no_assignment']}")
    print(f"  With sampler-supported assignments: {stats['with_valid']}")
    print(f"  Excluded by missing sampler support: {stats['no_sampler_support']}")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(precomputed, f)
        print(f"  Saved precomputed assignments → {cache_path}")

    return precomputed


# ── Full pipeline ────────────────────────────────────────────────────────────
def run_pipeline(
    candidates_pkl: str | Path = CANDIDATES_PKL,
    wyckoff_csv: str | Path = WYCKOFF_CSV,
    sampler_path: str | Path | None = DEFAULT_SAMPLER,
    precompute_cache: str | Path | None = DEFAULT_PRECOMPUTE,
    natm_range: tuple[int, int] = (5, 20),
    total_num: int = 10000,
    seed: int = 42,
    alpha: float = 1.0,
    tau: float = 0.1,
    temperature: float = 1.0,
    top_k: int = 30,
    bond_sigma_per_mu: float | None = None,
    use_min_bond_len: bool = False,
    known_species: List[str] | None = None,
    non_triclinic_weight: float = DEFAULT_NON_TRICLINIC_WEIGHT,
    skeleton_symmetry_power: float = DEFAULT_SKELETON_SYMMETRY_POWER,
) -> WyckoffSubLatDataset:
    """
    Run the full pipeline and produce the dataset.

    1. Precompute all valid g_full assignments per sublattice.
    2. Sample total_num sublattices with replacement.
    3. For each sample, pick one g_full, sample spectator completion.

    Parameters
    ----------
    candidates_pkl : Path to step2_candidates.pkl (Step 2 output)
    wyckoff_csv    : Path to wyckoff_data.csv (training data for DP sampler)
    sampler_path   : Path to save/load the fitted DP sampler
    precompute_cache : Path to cache precomputed assignments (None to skip)
    natm_range     : (min, max) total atom count
    total_num      : number of samples to draw (dataset size)
    seed           : random seed
    alpha, tau, temperature : DP sampler hyperparameters
    top_k          : candidate shortlist size for DP sampler
    bond_sigma_per_mu : if set, sample bond lengths from Gaussian; else use KDE
    use_min_bond_len  : enforce minimum bond lengths from metallic_radius
    known_species  : element pool for constrained atoms (default: KNOWN_SPECIES)
    non_triclinic_weight : relative weight for non-triclinic (SG > 2) assignments
        vs. triclinic (SG 1–2) when sampling g_full.  Default 5.0.  Setting
        1.0 restores uniform weighting.  If a sublattice has only triclinic
        assignments the triclinic SG is still used.
    skeleton_symmetry_power : exponent applied to the maximum sg_full available
        for each sublattice skeleton when drawing the skeleton sample.
        Default 1.2.  Setting 0.0 restores uniform skeleton sampling.
        Positive values upsample skeletons that have at least one high-symmetry
        valid assignment.  Skeletons whose only valid assignments are triclinic
        (max SG ≤ 2) still appear, just with much lower probability.
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    natm_min, natm_max = natm_range
    if known_species is None:
        known_species = KNOWN_SPECIES

    # ── Load Step 2 candidates ───────────────────────────────────────────────
    print("Loading Step 2 candidates...")
    with open(candidates_pkl, "rb") as f:
        entries = pickle.load(f)
    print(f"  {len(entries)} sublattice entries loaded.")

    # ── Fit or load DP sampler (Step 4 model) ────────────────────────────────
    sampler_path = Path(sampler_path) if sampler_path else None
    if sampler_path and sampler_path.exists():
        print(f"Loading pre-fitted DP sampler from {sampler_path}")
        sampler = WyckoffDPSampler.load(sampler_path)
    else:
        print(f"Fitting DP sampler on {wyckoff_csv}...")
        df = pd.read_csv(wyckoff_csv)
        sampler = WyckoffDPSampler().fit(df)
        if sampler_path:
            sampler_path.parent.mkdir(parents=True, exist_ok=True)
            sampler.save(sampler_path)
            print(f"  Saved sampler → {sampler_path}")
    print(f"  Sampler covers {len(sampler.sg_stats)} space groups.")

    # ── Precompute all valid g_full assignments ──────────────────────────────
    cache_path = Path(precompute_cache) if precompute_cache else None
    precomputed = precompute_assignments(entries, sampler, natm_max, cache_path)

    if not precomputed:
        print("No sublattices with valid assignments found.")
        return WyckoffSubLatDataset([])

    # ── Sample sublattices with replacement ──────────────────────────────────
    # Weight each skeleton by (max available sg_full) ** skeleton_symmetry_power
    # so rare high-symmetry skeletons are upsampled.  Power=0 → uniform.
    if skeleton_symmetry_power != 0.0:
        skeleton_weights = [
            max(va["sg_full"] for va in e["valid_assignments"]) ** skeleton_symmetry_power
            for e in precomputed
        ]
    else:
        skeleton_weights = None
    sampled_indices = rng.choices(range(len(precomputed)), weights=skeleton_weights, k=total_num)

    data_list: List[Data] = []
    skip_count = 0

    for i, idx in enumerate(sampled_indices):
        entry = precomputed[idx]
        sublattice = entry["sublattice"]
        sublat_id = entry["sublat_id"]
        num_known = entry["num_known"]

        # Sample one g_full, optionally up-weighting non-triclinic (SG > 2)
        valid = entry["valid_assignments"]
        if non_triclinic_weight != 1.0:
            weights = [
                non_triclinic_weight if v["sg_full"] > 2 else 1.0
                for v in valid
            ]
            va = rng.choices(valid, weights=weights, k=1)[0]
        else:
            va = rng.choice(valid)
        sg_full = va["sg_full"]
        assignments = va["assignments"]
        known_counts_labeled = va["known_counts_labeled"]

        # Sample type_known and bond_len (matching sublat_sc.py)
        type_known = rng.choice(known_species)
        bond_len = sample_bond_length(
            type_known, rng,
            bond_sigma_per_mu=bond_sigma_per_mu,
            use_min_bond_len=use_min_bond_len,
        )

        # Sample remaining atom count
        remaining_atoms_target = rng.randint(
            max(1, natm_min - num_known),
            natm_max - num_known,
        )

        # DP sampler completion (single attempt)
        candidate_labels = sampler.shortlist_candidates(sg_full, top_k=top_k)
        try:
            additional = sampler.sample_completion(
                sg=sg_full,
                known_counts=known_counts_labeled,
                remaining_atoms=remaining_atoms_target,
                candidate_labels=candidate_labels,
                alpha=alpha,
                tau=tau,
                temperature=temperature,
                rng=rng,
            )
        except (ValueError, RuntimeError):
            skip_count += 1
            continue

        # Convert additional counts to spectator letter list
        spectator_letters = []
        for label, count in additional.items():
            if count <= 0:
                continue
            letter = label[-1]  # '8f' → 'f'
            for _ in range(count):
                spectator_letters.append(letter)

        if not spectator_letters:
            skip_count += 1
            continue

        group = Group(sg_full)
        total_atoms = num_known + sum(
            group[l].multiplicity for l in spectator_letters
        )
        if total_atoms < natm_min or total_atoms > natm_max:
            skip_count += 1
            continue

        sample = build_sample(
            sublattice=sublattice,
            sublat_id=sublat_id,
            type_known=type_known,
            bond_len=bond_len,
            sg_full=sg_full,
            constrained_assignments=assignments,
            spectator_letters=spectator_letters,
            rng=rng,
        )
        data_list.append(sample)

        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{total_num}  |  "
                  f"built={len(data_list)}  skipped={skip_count}")

    print(f"\nPipeline complete:")
    print(
        "  Sublattices retained after natm_max / assignment / sampler filters: "
        f"{len(precomputed)}"
    )
    print(f"  Requested samples: {total_num}")
    print(f"  Built samples: {len(data_list)}")
    print(f"  Skipped (DP failure/atom count): {skip_count}")

    return WyckoffSubLatDataset(data_list)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build WyckoffSubLatDataset from sublattice candidates."
    )
    parser.add_argument("--candidates", default=str(CANDIDATES_PKL))
    parser.add_argument("--wyckoff_csv", default=str(WYCKOFF_CSV))
    parser.add_argument("--sampler_path", default=str(DEFAULT_SAMPLER))
    parser.add_argument("--precompute_cache", default=str(DEFAULT_PRECOMPUTE))
    parser.add_argument("--save_path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--natm_min", type=int, default=5)
    parser.add_argument("--natm_max", type=int, default=20)
    parser.add_argument("--total_num", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--bond_sigma_per_mu", type=float, default=None,
                        help="If set, sample bond lengths from Gaussian; else use KDE")
    parser.add_argument("--use_min_bond_len", action="store_true")
    parser.add_argument("--non_triclinic_weight", type=float, default=DEFAULT_NON_TRICLINIC_WEIGHT,
                        help="Relative weight for non-triclinic (SG>2) vs triclinic assignments. "
                             "Default 5.0; set 1.0 for uniform weighting.")
    parser.add_argument("--skeleton_symmetry_power", type=float, default=DEFAULT_SKELETON_SYMMETRY_POWER,
                        help="Exponent on max-SG for skeleton sampling weight. "
                             "Default 1.2; set 0.0 for uniform skeleton sampling.")
    args = parser.parse_args()

    t0 = time.time()
    dataset = run_pipeline(
        candidates_pkl=args.candidates,
        wyckoff_csv=args.wyckoff_csv,
        sampler_path=args.sampler_path,
        precompute_cache=args.precompute_cache,
        natm_range=(args.natm_min, args.natm_max),
        total_num=args.total_num,
        seed=args.seed,
        alpha=args.alpha,
        tau=args.tau,
        temperature=args.temperature,
        top_k=args.top_k,
        bond_sigma_per_mu=args.bond_sigma_per_mu,
        use_min_bond_len=args.use_min_bond_len,
        non_triclinic_weight=args.non_triclinic_weight,
        skeleton_symmetry_power=args.skeleton_symmetry_power,
    )

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    dataset.save(args.save_path)

    print(f"\nTotal time: {time.time() - t0:.1f}s")

    # ── Quick demo ───────────────────────────────────────────────────────────
    print(f"\n── Demo: first 3 samples ──")
    for idx in range(min(3, len(dataset))):
        d = dataset[idx]
        print(f"\nSample {idx}:")
        print(f"  sublat_id              : {d.sublat_id}")
        print(f"  type_known             : {d.type_known}")
        print(f"  spacegroup             : {d.spacegroup.item()}")
        print(f"  num_atoms              : {d.num_atoms.item()}")
        print(f"  num_known              : {d.num_known}")
        print(f"  num_spectator          : {d.num_spectator}")
        print(f"  wyckoff_letters        : {d.wyckoff_letters}")
        print(f"  constrained_wyckoff    : {d.constrained_wyckoff_letters}")
        print(f"  spectator_wyckoff      : {d.spectator_wyckoff_letters}")
        print(f"  ops shape              : {tuple(d.ops.shape)}")
        print(f"  anchor_index           : {d.anchor_index.tolist()}")
        print(f"  lengths                : {[f'{x:.3f}' for x in d.lengths.squeeze().tolist()]}")
        print(f"  angles                 : {[f'{x:.1f}' for x in d.angles.squeeze().tolist()]}")
        print(f"  constrained_indices    : {d.constrained_indices.tolist()}")
        print(f"  spectator_indices      : {d.spectator_indices.tolist()}")
        print(f"  atom_types (known)     : {[chemical_symbols[t] for t in d.atom_types[:d.num_known].tolist()]}")
        print(f"  atom_types (spectator) : {[chemical_symbols[t] for t in d.atom_types[d.num_known:].tolist()]}")


if __name__ == "__main__":
    main()
