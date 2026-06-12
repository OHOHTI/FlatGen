"""
wysc_utils.py 

  Step 1 — Extract Sublattice and Determine Its Space Group
    parse_outliers, detect_spacegroup, main_step1

  Step 2 — Enumerate Compatible Subgroups
    SUBGROUP_TABLE (precomputed at import), sg_to_family, FAMILY_NAMES,
    standardized_lattice_params, is_metric_compatible,
    enumerate_compatible_subgroups, sample_subgroup, main_step2

  Step 3 — Wyckoff Assignment
    find_wyckoff_assignment_solutions, assign_with_snap_solutions,
    find_all_valid_assignments, compute_wyckoff_counts,
    process_sublattice, main_step3

Note: SUBGROUP_TABLE is built at import time (~0.8 s via PyXtal BFS).
"""

from __future__ import annotations

import math
import os
import pickle
import random
from collections import Counter
from itertools import product

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group
from tqdm import tqdm

# ── Shared paths ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(__file__)
_DATA = os.path.join(_HERE, "..", "data")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Extract Sublattice and Determine Its Space Group
# ══════════════════════════════════════════════════════════════════════════════

# Step 1 configuration
STEP1_OUTLIERS_FILE = os.path.join(_DATA, "outliers.txt")
STEP1_OUTPUT_FILE   = os.path.join(_DATA, "sublattices.pkl")
MP_API_KEY          = os.environ.get("MP_API_KEY", "")
STEP1_SYMPREC       = 0.1   # generous: sublattices from real materials have small distortions


def parse_outliers(path: str) -> list[tuple[str, str, str]]:
    """Return list of (full_id, material_id, element_symbol) from outliers.txt."""
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Material_ID"):
                continue
            entry = line.split()[0]          # e.g. "mp-cgadx_O"
            *id_parts, element = entry.rsplit("_", 1)
            material_id = "_".join(id_parts)  # e.g. "mp-cgadx"
            tasks.append((entry, material_id, element))
    return tasks


def detect_spacegroup(sublattice: Structure, symprec: float) -> tuple[int, str]:
    """Return (number, symbol) for the sublattice's space group."""
    analyzer = SpacegroupAnalyzer(sublattice, symprec=symprec)
    number   = analyzer.get_space_group_number()
    symbol   = analyzer.get_space_group_symbol()
    return number, symbol


def main_step1() -> None:
    """Fetch sublattices from Materials Project and write sublattices.pkl."""
    from mp_api.client import MPRester

    tasks = parse_outliers(STEP1_OUTLIERS_FILE)
    print(f"Loaded {len(tasks)} entries from {STEP1_OUTLIERS_FILE}")

    results: list[dict] = []
    failed:  list[str]  = []

    with MPRester(MP_API_KEY) as mpr:
        for full_id, material_id, element_symbol in tqdm(tasks, desc="Fetching structures"):
            try:
                structure = mpr.get_structure_by_material_id(material_id)
            except Exception as e:
                print(f"  Warning: could not fetch {material_id}: {e}")
                failed.append(full_id)
                continue

            specific_sites = [site for site in structure
                              if site.specie.symbol == element_symbol]
            if not specific_sites:
                print(f"  Warning: no {element_symbol} sites in {material_id}, skipping.")
                failed.append(full_id)
                continue

            sublattice = Structure.from_sites(specific_sites)

            try:
                sg_number, sg_symbol = detect_spacegroup(sublattice, STEP1_SYMPREC)
            except Exception as e:
                print(f"  Warning: spacegroup detection failed for {full_id}: {e}")
                failed.append(full_id)
                continue

            results.append({
                "full_id":           full_id,
                "material_id":       material_id,
                "element":           element_symbol,
                "sublattice":        sublattice,
                "spacegroup_number": sg_number,
                "spacegroup_symbol": sg_symbol,
                "symprec":           STEP1_SYMPREC,
            })

    print(f"\nSuccessfully processed: {len(results)}")
    print(f"Failed:                 {len(failed)}")
    if failed:
        print("  Failed IDs:", failed)

    os.makedirs(os.path.dirname(STEP1_OUTPUT_FILE), exist_ok=True)
    with open(STEP1_OUTPUT_FILE, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved to {STEP1_OUTPUT_FILE}")

    sg_counts = Counter(r["spacegroup_number"] for r in results)
    print("\nTop 10 space groups detected:")
    for sg, count in sg_counts.most_common(10):
        symbol = next(r["spacegroup_symbol"] for r in results
                      if r["spacegroup_number"] == sg)
        print(f"  SG {sg:3d}  {symbol:<12s}  {count} sublattices")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Enumerate Compatible Subgroups
# ══════════════════════════════════════════════════════════════════════════════

# Step 2 configuration
STEP2_SUBLATTICES_FILE = os.path.join(_DATA, "sublattices.pkl")
STEP2_OUTPUT_FILE      = os.path.join(_DATA, "step2_candidates.pkl")
STEP2_RANDOM_SEED      = 42
STEP2_N_DEMO           = 5      # how many sublattices to show in demo mode
TOL_ANGLE              = 5.0    # degrees — tolerance for lattice angle constraints
TOL_LEN_REL            = 0.10   # relative tolerance for equal-length constraints


def _build_subgroup_table() -> dict[int, frozenset[int]]:
    """
    BFS from each SG over PyXtal's maximal-subgroup graph to collect every
    transitive subgroup.  Returns SUBGROUP_TABLE[sg] = frozenset of all SGs
    that are proper subgroups of sg (sg itself excluded).
    """
    print("Precomputing subgroup table via PyXtal …", flush=True)
    table: dict[int, frozenset[int]] = {}
    for sg in range(1, 231):
        visited: set[int] = set()
        queue = [sg]
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for s in np.unique(Group(cur).get_max_subgroup_numbers()):
                if int(s) not in visited:
                    queue.append(int(s))
        visited.discard(sg)
        table[sg] = frozenset(visited)
    print("Done — table covers SG 1–230.\n")
    return table


SUBGROUP_TABLE: dict[int, frozenset[int]] = _build_subgroup_table()


def sg_to_family(sg: int) -> int:
    """Map space group number to DiffCSP++ crystal family index (1–6)."""
    if 195 <= sg <= 230: return 6   # cubic
    if 143 <= sg <= 194: return 5   # hexagonal / trigonal
    if  75 <= sg <= 142: return 4   # tetragonal
    if  16 <= sg <=  74: return 3   # orthorhombic
    if   3 <= sg <=  15: return 2   # monoclinic
    return 1                        # triclinic


FAMILY_NAMES = {
    1: "triclinic",
    2: "monoclinic",
    3: "orthorhombic",
    4: "tetragonal",
    5: "hexagonal/trigonal",
    6: "cubic",
}


def standardized_lattice_params(
    sublattice: Structure,
    symprec: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Return lattice parameters from a conventional standardized cell.

    Step 1 stores the raw sublattice basis, which may be a primitive or skewed
    representation of the same Bravais lattice.  Standardizing here makes the
    family sanity check invariant to that basis choice.
    """
    analyzer = SpacegroupAnalyzer(sublattice, symprec=symprec)
    lat = analyzer.get_conventional_standard_structure().lattice
    return lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma


def near_angle(x: float, target: float) -> bool:
    return abs(x - target) < TOL_ANGLE


def near_len(x: float, y: float) -> bool:
    return abs(x - y) < TOL_LEN_REL * max(x, y)


def is_metric_compatible(a, b, c, alpha, beta, gamma, family: int) -> bool:
    """
    Check whether the sublattice metric satisfies the hard constraints
    imposed by the given crystal family (DiffCSP++ Table 1 / crystal_family.py).

    Family 6 — cubic:          a=b=c,  α=β=γ=90°
    Family 5 — hexagonal:      a=b,    α=β=90°, γ=120°
    Family 4 — tetragonal:     a=b,    α=β=γ=90°
    Family 3 — orthorhombic:            α=β=γ=90°
    Family 2 — monoclinic:              α=γ=90°  (β free)
    Family 1 — triclinic:      no constraints
    """
    if family == 6:
        return (near_len(a, b) and near_len(b, c)
                and near_angle(alpha, 90) and near_angle(beta, 90) and near_angle(gamma, 90))
    if family == 5:
        return (near_len(a, b)
                and near_angle(alpha, 90) and near_angle(beta, 90) and near_angle(gamma, 120))
    if family == 4:
        return (near_len(a, b)
                and near_angle(alpha, 90) and near_angle(beta, 90) and near_angle(gamma, 90))
    if family == 3:
        return near_angle(alpha, 90) and near_angle(beta, 90) and near_angle(gamma, 90)
    if family == 2:
        return near_angle(alpha, 90) and near_angle(gamma, 90)
    return True   # triclinic: always compatible


def enumerate_compatible_subgroups(
    sublattice: Structure,
    sg_sub: int,
    symprec: float = 0.1,
) -> list[int]:
    """
    Return all SG candidates for G_full satisfying:
      (a) G_full ∈ SUBGROUP_TABLE[sg_sub] ∪ {sg_sub}  [G_full ≤ G_sub, from PyXtal]
      (b) G_full's crystal family constraints are satisfiable by the sublattice metric
    """
    a, b, c, alpha, beta, gamma = standardized_lattice_params(sublattice, symprec)

    sg_full_candidates = SUBGROUP_TABLE[sg_sub] | {sg_sub}

    candidates = []
    for sg_full in sg_full_candidates:
        fam_full = sg_to_family(sg_full)
        if is_metric_compatible(a, b, c, alpha, beta, gamma, fam_full):
            candidates.append(sg_full)
    return sorted(candidates)


def sample_subgroup(candidates: list[int], rng: random.Random) -> int | None:
    return rng.choice(candidates) if candidates else None


def _print_step2_result(entry: dict, candidates: list[int], sampled: int | None) -> None:
    sub  = entry["sublattice"]
    a, b, c, alpha, beta, gamma = standardized_lattice_params(
        sub,
        entry.get("symprec", 0.1),
    )
    fam_sub  = sg_to_family(entry["spacegroup_number"])
    fam_samp = sg_to_family(sampled) if sampled else None

    print("=" * 60)
    print(f"  ID       : {entry['full_id']}")
    print(f"  Element  : {entry['element']}")
    print(f"  Lattice  : a={a:.3f} b={b:.3f} c={c:.3f}  "
          f"α={alpha:.1f}° β={beta:.1f}° γ={gamma:.1f}°")
    print(f"  G_sub    : SG {entry['spacegroup_number']}  "
          f"({entry['spacegroup_symbol']})  "
          f"[{FAMILY_NAMES[fam_sub]}]")
    print(f"  Candidates: {len(candidates)} subgroups (G_full ≤ G_sub, metric-filtered)")
    if sampled:
        print(f"  Sampled G_full : SG {sampled}  [{FAMILY_NAMES[fam_samp]}]")
        fam_counts: dict[int, int] = {}
        for sg in candidates:
            f = sg_to_family(sg)
            fam_counts[f] = fam_counts.get(f, 0) + 1
        breakdown = "  |  ".join(
            f"{FAMILY_NAMES[f]}: {fam_counts[f]}" for f in sorted(fam_counts)
        )
        print(f"  Breakdown : {breakdown}")
    else:
        print("  Sampled G_full : None (no compatible subgroups found)")
    print()


def main_step2() -> None:
    """Read sublattices.pkl, enumerate compatible subgroups, write step2_candidates.pkl."""
    with open(STEP2_SUBLATTICES_FILE, "rb") as f:
        sublattices = pickle.load(f)
    print(f"Loaded {len(sublattices)} sublattices from Step 1.")
    print(f"Computing compatible subgroups for all {len(sublattices)} entries …\n", flush=True)

    results = []
    empty_count = 0
    for i, entry in enumerate(sublattices):
        candidates = enumerate_compatible_subgroups(
            entry["sublattice"],
            entry["spacegroup_number"],
            entry.get("symprec", 0.1),
        )
        if not candidates:
            empty_count += 1
        result = dict(entry)
        result["g_full_candidates"] = candidates
        results.append(result)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(sublattices)} done …", flush=True)

    os.makedirs(os.path.dirname(STEP2_OUTPUT_FILE), exist_ok=True)
    with open(STEP2_OUTPUT_FILE, "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved {len(results)} records → {STEP2_OUTPUT_FILE}")
    print(f"Entries with no candidates: {empty_count}")

    print(f"\n── Demo: {STEP2_N_DEMO} random samples ──")
    rng = random.Random(STEP2_RANDOM_SEED)
    for entry in rng.sample(results, min(STEP2_N_DEMO, len(results))):
        sampled = sample_subgroup(entry["g_full_candidates"], rng)
        _print_step2_result(entry, entry["g_full_candidates"], sampled)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Wyckoff Assignment
# ══════════════════════════════════════════════════════════════════════════════

# Step 3 configuration
STEP3_INPUT_FILE      = os.path.join(_DATA, "step2_candidates.pkl")
STEP3_OUTPUT_FILE     = os.path.join(_DATA, "step3_compatible_templates.pkl")
MP_COMPLETE_CSV       = os.path.join(_DATA, "mp_complete_summary.csv")
MP_COMPLETE_INDEX     = os.path.join(_DATA, "mp_complete_wyckoff_index.json")

FRAC_TOL  = 0.05   # tolerance for matching fractional coordinates
SNAP_TOL  = 0.10   # max snap adjustment when atoms are slightly off-Wyckoff
MAX_ATOMS = 52     # upper bound on template unit-cell size
MIN_ATOMS = 1      # lower bound
STEP3_N_DEMO = 5   # entries to display in the summary


# ── Coordinate helpers ────────────────────────────────────────────────────────

def frac_dist(f1: np.ndarray, f2: np.ndarray) -> float:
    diff = f1 - f2
    diff = diff - np.round(diff)
    return float(np.linalg.norm(diff))


def frac_delta(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    diff = np.asarray(f1, dtype=float) - np.asarray(f2, dtype=float)
    return diff - np.round(diff)


def max_frac_residual(f1: np.ndarray, f2: np.ndarray) -> float:
    return float(np.max(np.abs(frac_delta(f1, f2))))


def match_orbit_to_atoms(
    orbit_coords: np.ndarray,
    atom_coords: np.ndarray,
    unassigned: set,
    tol: float,
    required_matches: dict | None = None,
) -> list | None:
    M = len(orbit_coords)
    required_matches = required_matches or {}
    unassigned_sorted = sorted(unassigned)
    adjacency: list[list[tuple]] = []

    for orbit_idx, orbit_pos in enumerate(orbit_coords):
        neighbors: list[tuple] = []
        if orbit_idx in required_matches:
            atom_idx = required_matches[orbit_idx]
            if atom_idx not in unassigned:
                return None
            d = frac_dist(orbit_pos, atom_coords[atom_idx])
            if d > tol:
                return None
            neighbors.append((atom_idx, d))
        else:
            for atom_idx in unassigned_sorted:
                d = frac_dist(orbit_pos, atom_coords[atom_idx])
                if d <= tol:
                    neighbors.append((atom_idx, d))
        neighbors.sort(key=lambda item: (item[1], item[0]))
        if not neighbors:
            return None
        adjacency.append(neighbors)

    orbit_order = sorted(
        range(M),
        key=lambda oi: (len(adjacency[oi]), adjacency[oi][0][1], oi),
    )
    atom_to_orbit: dict[int, int] = {}

    def augment(orbit_idx: int, seen_atoms: set) -> bool:
        for atom_idx, _ in adjacency[orbit_idx]:
            if atom_idx in seen_atoms:
                continue
            seen_atoms.add(atom_idx)
            prev = atom_to_orbit.get(atom_idx)
            if prev is None or augment(prev, seen_atoms):
                atom_to_orbit[atom_idx] = orbit_idx
                return True
        return False

    for oi in orbit_order:
        if not augment(oi, set()):
            return None

    matched: list = [None] * M
    for atom_idx, orbit_idx in atom_to_orbit.items():
        matched[orbit_idx] = atom_idx
    if any(v is None for v in matched):
        return None
    return [int(v) for v in matched]


def row_shift_values(row: np.ndarray, target: float) -> range:
    row = np.asarray(row, dtype=int)
    min_lhs = int(row[row < 0].sum())
    max_lhs = int(row[row > 0].sum())
    lower = math.floor(min_lhs - float(target))
    upper = math.ceil(max_lhs - float(target))
    return range(lower, upper + 1)


def solve_basic_coords(anchor_coord: np.ndarray, op, tol: float) -> list:
    R = np.asarray(op.rotation_matrix, dtype=float)
    t = np.asarray(op.translation_vector, dtype=float)
    target = (np.asarray(anchor_coord, dtype=float) - t) % 1.0
    shift_ranges = [row_shift_values(row, tj) for row, tj in zip(R.astype(int), target)]
    solutions = []
    for shifts in product(*shift_ranges):
        rhs = target + np.asarray(shifts, dtype=float)
        f_prime, _, _, _ = np.linalg.lstsq(R, rhs, rcond=None)
        if np.max(np.abs(R @ f_prime - rhs)) > tol:
            continue
        f_mod = f_prime % 1.0
        if max_frac_residual(R @ f_mod + t, anchor_coord) > tol:
            continue
        if any(max_frac_residual(ex, f_mod) <= tol for ex in solutions):
            continue
        solutions.append(f_mod)
    return solutions


def generate_orbit(f_prime: np.ndarray, wp) -> np.ndarray:
    orbit = []
    for op in wp:
        pos = op.rotation_matrix @ f_prime + op.translation_vector
        orbit.append(pos % 1.0)
    return np.array(orbit)


def orbit_signature(orbit: np.ndarray, decimals: int = 8):
    rows = [tuple(np.round(pos % 1.0, decimals=decimals)) for pos in orbit]
    return tuple(sorted(rows))


def assignment_signature(assignments: list) -> tuple:
    parts = []
    for a in assignments:
        parts.append((
            a["multiplicity"],
            a["letter"],
            tuple(sorted(a["atom_indices"])),
            orbit_signature(a["adjusted_coords"]),
        ))
    return tuple(sorted(parts))


def enumerate_assignment_candidates(
    frac_coords: np.ndarray,
    wps_sorted: list,
    unassigned: set,
    tol: float,
) -> list:
    anchor = min(unassigned)
    anchor_coord = frac_coords[anchor]
    candidates = []
    seen = set()

    for wp in wps_sorted:
        if wp.multiplicity > len(unassigned):
            continue
        for op_idx, op in enumerate(wp):
            for f_prime in solve_basic_coords(anchor_coord, op, tol):
                orbit = generate_orbit(f_prime, wp)
                matched = match_orbit_to_atoms(
                    orbit, frac_coords, unassigned, tol,
                    required_matches={op_idx: anchor},
                )
                if matched is None or len(matched) != wp.multiplicity:
                    continue
                residuals = [
                    frac_dist(orbit[oi], frac_coords[ai])
                    for oi, ai in enumerate(matched)
                ]
                adjusted = orbit.copy()
                sig = (wp.letter, wp.multiplicity,
                       tuple(sorted(matched)), orbit_signature(adjusted))
                if sig in seen:
                    continue
                seen.add(sig)
                candidates.append({
                    "letter": wp.letter,
                    "multiplicity": wp.multiplicity,
                    "basic_coord": f_prime % 1.0,
                    "atom_indices": matched,
                    "residuals": residuals,
                    "adjusted_coords": adjusted,
                })

    candidates.sort(key=lambda a: (
        -a["multiplicity"],
        float(sum(a["residuals"])),
        a["letter"],
        tuple(sorted(a["atom_indices"])),
    ))
    return candidates


def find_wyckoff_assignment_solutions(
    frac_coords: np.ndarray,
    sg_number: int,
    tol: float = FRAC_TOL,
    max_solutions: int | None = None,
) -> list:
    group = Group(sg_number)
    wps_sorted = sorted(group, key=lambda wp: wp.multiplicity, reverse=True)
    all_indices = set(range(len(frac_coords)))
    dead_states: set = set()
    candidate_cache: dict = {}

    def backtrack(unassigned: set) -> list:
        if not unassigned:
            return [[]]
        state = frozenset(unassigned)
        if state in dead_states:
            return []
        if state not in candidate_cache:
            candidate_cache[state] = enumerate_assignment_candidates(
                frac_coords, wps_sorted, unassigned, tol)
        candidates = candidate_cache[state]
        if not candidates:
            dead_states.add(state)
            return []
        solutions = []
        for candidate in candidates:
            remaining = set(unassigned)
            for ai in candidate["atom_indices"]:
                remaining.discard(ai)
            for suffix in backtrack(remaining):
                solutions.append([candidate] + suffix)
                if max_solutions is not None and len(solutions) >= max_solutions:
                    return solutions
        if not solutions:
            dead_states.add(state)
        return solutions

    raw = backtrack(all_indices)
    solutions = []
    seen = set()
    for assignments in raw:
        sig = assignment_signature(assignments)
        if sig in seen:
            continue
        seen.add(sig)
        solutions.append(assignments)
        if max_solutions is not None and len(solutions) >= max_solutions:
            break
    return solutions


def snap_solution_coords(frac_coords: np.ndarray, assignments: list) -> np.ndarray:
    snapped = frac_coords.copy()
    for a in assignments:
        for oi, ai in enumerate(a["atom_indices"]):
            snapped[ai] = a["adjusted_coords"][oi]
    return snapped


def assign_with_snap_solutions(
    frac_coords: np.ndarray,
    sg_number: int,
    tol: float = FRAC_TOL,
    snap_tol: float = SNAP_TOL,
    max_solutions: int | None = None,
) -> list:
    strict = find_wyckoff_assignment_solutions(frac_coords, sg_number, tol=tol,
                                               max_solutions=max_solutions)
    if strict:
        return [(a, frac_coords.copy()) for a in strict]
    relaxed = find_wyckoff_assignment_solutions(frac_coords, sg_number, tol=snap_tol,
                                                max_solutions=max_solutions)
    return [(a, snap_solution_coords(frac_coords, a)) for a in relaxed]


def assign_wyckoff_positions(
    frac_coords: np.ndarray,
    sg_number: int,
    tol: float = FRAC_TOL,
) -> list | None:
    """Return the first valid Wyckoff assignment for frac_coords in sg_number.

    Returns a list of assignment dicts (one per orbit), or None if no assignment
    was found.  Each dict contains 'letter', 'multiplicity', 'atom_indices', etc.
    """
    solutions = find_wyckoff_assignment_solutions(frac_coords, sg_number, tol=tol,
                                                  max_solutions=1)
    return solutions[0] if solutions else None


def find_all_valid_assignments(
    frac_coords: np.ndarray,
    viable_candidates: list,
    tol: float = FRAC_TOL,
    snap_tol: float = SNAP_TOL,
) -> list:
    """Return all valid Wyckoff assignments across multiple space group candidates.

    Parameters
    ----------
    frac_coords : (N, 3) fractional coordinates of constrained atoms
    viable_candidates : list of space group numbers to try
    tol : fractional-coordinate tolerance for strict matching
    snap_tol : tolerance for snapped (relaxed) matching

    Returns
    -------
    list of (sg_full, assignments, adjusted_coords) tuples, one per valid
    solution found across all candidate space groups.
    """
    results = []
    for sg_full in viable_candidates:
        solutions = assign_with_snap_solutions(frac_coords, sg_full, tol=tol,
                                               snap_tol=snap_tol)
        for assignments, adjusted_coords in solutions:
            results.append((sg_full, assignments, adjusted_coords))
    return results


def compute_wyckoff_counts(assignments: list) -> dict:
    """Convert a list of assignment dicts → {letter: orbit_count}."""
    return dict(Counter(a["letter"] for a in assignments))


def process_sublattice(entry: dict, finder) -> dict:
    """Find all compatible mp_20 templates for a single sublattice entry.

    For each candidate G_full:
      1. Fit the sublattice atom coordinates to Wyckoff positions (W_c).
      2. Search the mp_20 index for templates with the same space group
         whose config contains W_c.

    Returns the entry dict augmented with:
      wyckoff_constraints_by_gfull, compatible_templates, template_search_status.
    """
    frac_coords = entry["sublattice"].frac_coords
    candidates = entry.get("g_full_candidates", [])

    constraints_by_gfull: dict[int, dict] = {}
    all_templates: dict[str, dict] = {}   # keyed by material_id to deduplicate

    for sg in sorted(candidates, reverse=True):
        solutions = assign_with_snap_solutions(frac_coords, sg, max_solutions=1)
        if not solutions:
            continue
        assignments, _ = solutions[0]
        wc = compute_wyckoff_counts(assignments)
        constraints_by_gfull[sg] = wc

        templates = finder.find(
            space_group_number=sg,
            constrained_wyckoffs=wc,
            min_atoms=MIN_ATOMS,
            max_atoms=MAX_ATOMS,
        )
        for t in templates:
            all_templates[t.material_id] = t.to_dict()

    result = dict(entry)
    result["wyckoff_constraints_by_gfull"] = constraints_by_gfull
    result["compatible_templates"] = list(all_templates.values())

    if not constraints_by_gfull:
        result["template_search_status"] = "no_assignment"
    elif not all_templates:
        result["template_search_status"] = "no_templates"
    else:
        result["template_search_status"] = "success"

    return result


def main_step3() -> None:
    """Read step2_candidates.pkl, find Wyckoff templates, write step3_compatible_templates.pkl."""
    try:
        from .template_utils import load_mp_complete_wyckoff_index, TemplateFinder
    except ImportError:
        from template_utils import load_mp_complete_wyckoff_index, TemplateFinder

    if os.path.exists(STEP3_OUTPUT_FILE):
        print(f"Output already exists, skipping: {STEP3_OUTPUT_FILE}")
        return

    with open(STEP3_INPUT_FILE, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} sublattices from Step 2.")

    mp20_index = load_mp_complete_wyckoff_index(MP_COMPLETE_CSV, MP_COMPLETE_INDEX)
    finder = TemplateFinder(mp20_index)
    finder.summary()

    results = []
    status_counts = {"success": 0, "no_assignment": 0, "no_templates": 0}

    for i, entry in enumerate(data):
        result = process_sublattice(entry, finder)
        results.append(result)
        status_counts[result["template_search_status"]] += 1

        if (i + 1) % 50 == 0:
            print(
                f"  {i+1}/{len(data)}  "
                f"success={status_counts['success']}  "
                f"no_assign={status_counts['no_assignment']}  "
                f"no_templates={status_counts['no_templates']}",
                flush=True,
            )

    os.makedirs(os.path.dirname(STEP3_OUTPUT_FILE), exist_ok=True)
    with open(STEP3_OUTPUT_FILE, "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved {len(results)} records → {STEP3_OUTPUT_FILE}")
    print(f"  success      : {status_counts['success']}")
    print(f"  no_assignment: {status_counts['no_assignment']}")
    print(f"  no_templates : {status_counts['no_templates']}")

    print(f"\n── Demo: first {STEP3_N_DEMO} successful entries ──")
    shown = 0
    for entry in results:
        if entry["template_search_status"] != "success" or shown >= STEP3_N_DEMO:
            continue
        print("=" * 70)
        print(f"  ID              : {entry['full_id']}")
        print(f"  Element         : {entry['element']}")
        print(f"  G_sub           : SG {entry['spacegroup_number']}")
        print(f"  W_c by G_full   : {entry['wyckoff_constraints_by_gfull']}")
        print(f"  Compatible tmpl : {len(entry['compatible_templates'])}")
        for t in entry["compatible_templates"][:3]:
            print(f"    {t['material_id']}  SG={t['space_group']}  "
                  f"Wyckoff={t['wyckoff_counts']}  atoms={t['total_atoms']}")
        print()
        shown += 1


# ── Convenience: run individual steps from the command line ──────────────────

if __name__ == "__main__":
    import sys
    step = sys.argv[1] if len(sys.argv) > 1 else None
    if step == "1":
        main_step1()
    elif step == "2":
        main_step2()
    elif step == "3":
        main_step3()
    else:
        print("Usage: python wysc_utils.py {1|2|3}")
        sys.exit(1)
