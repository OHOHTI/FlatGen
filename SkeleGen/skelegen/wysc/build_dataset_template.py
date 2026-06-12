"""
Template-Based Pipeline: Step3 Templates → WyckoffSubLatDataset

Replaces the DP sampler in build_dataset.py with step3 compatible templates.
Each spectator (unconstrained) Wyckoff configuration is drawn directly from a
real mp-complete material that shares the sublattice's space group and contains
the sublattice's constrained Wyckoff positions as a subset.

Usage:
    python -m wysc.build_dataset_template                          # build and save
    python -m wysc.build_dataset_template --save_path out.pt       # custom output path
"""

from __future__ import annotations

import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from pyxtal.symmetry import Group

try:
    from .build_dataset import (
        build_sample,
        WyckoffSubLatDataset,
        sample_bond_length,
        KNOWN_SPECIES,
    )
    from .wysc_utils import assign_with_snap_solutions
    from .template_utils import WyckoffConfig
except ImportError:
    from build_dataset import (
        build_sample,
        WyckoffSubLatDataset,
        sample_bond_length,
        KNOWN_SPECIES,
    )
    from wysc_utils import assign_with_snap_solutions
    from template_utils import WyckoffConfig

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_TEMPLATES_PKL = DATA_DIR / "step3_compatible_templates.pkl"
DEFAULT_PRECOMPUTE_CACHE = DATA_DIR / "precomputed_template_assignments.pkl"
DEFAULT_OUTPUT = DATA_DIR / "wysc_template_dataset.pt"


# ── Precompute valid (sg_full, assignments, spectator_letters) triples ───────

def precompute_template_assignments(
    entries: list,
    natm_min: int,
    natm_max: int,
    cache_path: Path | None = None,
) -> list[dict]:
    """
    For each step3 entry with successful templates, find all valid
    (sg_full, assignments, spectator_letters) triples.

    step3 stores only letter counts in wyckoff_constraints_by_gfull, not
    atom-level assignments.  We re-run assign_with_snap_solutions for each
    sg_full to recover the atom_indices / adjusted_coords needed by
    build_sample().

    Returns a list of dicts, each with:
        sublattice, sublat_id, num_known, valid_entries
    Only sublattices with at least one valid triple are included.
    Results are optionally cached to cache_path.
    """
    if cache_path and cache_path.exists():
        print(f"Loading precomputed template assignments from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    precomputed = []
    stats = {
        "total": len(entries),
        "skipped_no_templates": 0,
        "no_assignment": 0,
        "with_valid": 0,
    }

    for i, entry in enumerate(entries):
        if entry.get("template_search_status") != "success":
            stats["skipped_no_templates"] += 1
            continue

        sublattice = entry["sublattice"]
        frac_coords = sublattice.frac_coords
        num_known = sublattice.num_sites
        sublat_id = entry["full_id"]
        wyckoff_constraints_by_gfull: Dict[int, dict] = entry["wyckoff_constraints_by_gfull"]
        raw_templates: list[dict] = entry["compatible_templates"]

        # Group compatible templates by their space group
        templates_by_sg: Dict[int, list[WyckoffConfig]] = {}
        for t in raw_templates:
            cfg = WyckoffConfig.from_dict(t)
            templates_by_sg.setdefault(cfg.space_group, []).append(cfg)

        valid_entries = []
        for sg_full, wc in wyckoff_constraints_by_gfull.items():
            # Re-run Wyckoff assignment to recover atom_indices + adjusted_coords
            solutions = assign_with_snap_solutions(frac_coords, sg_full, max_solutions=1)
            if not solutions:
                continue
            assignments, _ = solutions[0]

            sg_templates = templates_by_sg.get(sg_full, [])
            group = Group(sg_full)

            for template in sg_templates:
                # Spectator letters = template config minus constrained positions
                unconstrained = template.unconstrained_wyckoffs(wc)
                spectator_letters = []
                for letter, count in unconstrained.items():
                    spectator_letters.extend([letter] * count)

                if not spectator_letters:
                    continue

                num_spectator = sum(group[l].multiplicity for l in spectator_letters)
                total_atoms = num_known + num_spectator
                if not (natm_min <= total_atoms <= natm_max):
                    continue

                valid_entries.append({
                    "sg_full": sg_full,
                    "assignments": assignments,
                    "spectator_letters": spectator_letters,
                    "template_id": template.material_id,
                })

        if not valid_entries:
            stats["no_assignment"] += 1
            continue

        stats["with_valid"] += 1
        precomputed.append({
            "sublattice": sublattice,
            "sublat_id": sublat_id,
            "num_known": num_known,
            "valid_entries": valid_entries,
        })

        if (i + 1) % 50 == 0:
            print(
                f"  Precompute {i + 1}/{len(entries)}  |  "
                f"valid={stats['with_valid']}  "
                f"no_templates={stats['skipped_no_templates']}  "
                f"no_assignment={stats['no_assignment']}"
            )

    print("Precompute done:")
    print(f"  Total step3 entries          : {stats['total']}")
    print(f"  Skipped (no templates)       : {stats['skipped_no_templates']}")
    print(f"  Skipped (assignment failure) : {stats['no_assignment']}")
    print(f"  Retained with valid triples  : {stats['with_valid']}")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(precomputed, f)
        print(f"  Saved precomputed assignments → {cache_path}")

    return precomputed


# ── Full pipeline ────────────────────────────────────────────────────────────

def run_pipeline_template(
    templates_pkl: str | Path = DEFAULT_TEMPLATES_PKL,
    precompute_cache: str | Path | None = DEFAULT_PRECOMPUTE_CACHE,
    natm_range: tuple[int, int] = (5, 20),
    total_num: int = 10000,
    seed: int = 42,
    bond_sigma_per_mu: float | None = None,
    use_min_bond_len: bool = False,
    known_species: List[str] | None = None,
) -> WyckoffSubLatDataset:
    """
    Run the template-based pipeline and produce the dataset.

    Instead of the DP sampler, spectator Wyckoff positions are drawn directly
    from step3 compatible templates (real mp-complete materials).

    Parameters
    ----------
    templates_pkl     : Path to step3_compatible_templates.pkl (Step 3 output)
    precompute_cache  : Path to cache precomputed template assignments (None to skip)
    natm_range        : (min, max) total atom count per sample
    total_num         : number of samples to draw (dataset size)
    seed              : random seed
    bond_sigma_per_mu : if set, sample bond lengths from Gaussian; else use KDE
    use_min_bond_len  : enforce minimum bond lengths from metallic_radius
    known_species     : element pool for constrained atoms (default: KNOWN_SPECIES)
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    natm_min, natm_max = natm_range
    if known_species is None:
        known_species = KNOWN_SPECIES

    # ── Load step3 templates ─────────────────────────────────────────────────
    print(f"Loading step3 templates from {templates_pkl}...")
    with open(templates_pkl, "rb") as f:
        entries = pickle.load(f)
    print(f"  {len(entries)} sublattice entries loaded.")

    # ── Precompute valid triples ─────────────────────────────────────────────
    cache_path = Path(precompute_cache) if precompute_cache else None
    precomputed = precompute_template_assignments(entries, natm_min, natm_max, cache_path)

    if not precomputed:
        print("No sublattices with valid template assignments found.")
        return WyckoffSubLatDataset([])

    # ── Sample with replacement ──────────────────────────────────────────────
    sampled_indices = rng.choices(range(len(precomputed)), k=total_num)

    data_list = []
    skip_count = 0

    for i, idx in enumerate(sampled_indices):
        entry = precomputed[idx]
        sublattice = entry["sublattice"]
        sublat_id = entry["sublat_id"]

        # Pick one (sg_full, assignments, spectator_letters) triple at random
        ve = rng.choice(entry["valid_entries"])
        sg_full = ve["sg_full"]
        assignments = ve["assignments"]
        spectator_letters = ve["spectator_letters"]

        # Sample constrained atom type and bond length (same as run_pipeline)
        type_known = rng.choice(known_species)
        bond_len = sample_bond_length(
            type_known, rng,
            bond_sigma_per_mu=bond_sigma_per_mu,
            use_min_bond_len=use_min_bond_len,
        )

        try:
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
        except Exception:
            skip_count += 1
            continue

        data_list.append(sample)

        if (i + 1) % 2000 == 0:
            print(
                f"  {i + 1}/{total_num}  |  "
                f"built={len(data_list)}  skipped={skip_count}"
            )

    print(f"\nPipeline complete:")
    print(f"  Sublattices with valid template triples: {len(precomputed)}")
    print(f"  Requested samples : {total_num}")
    print(f"  Built samples     : {len(data_list)}")
    print(f"  Skipped           : {skip_count}")

    return WyckoffSubLatDataset(data_list)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build WyckoffSubLatDataset using step3 templates (no DP sampler)."
    )
    parser.add_argument("--templates", default=str(DEFAULT_TEMPLATES_PKL))
    parser.add_argument("--precompute_cache", default=str(DEFAULT_PRECOMPUTE_CACHE))
    parser.add_argument("--save_path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--natm_min", type=int, default=5)
    parser.add_argument("--natm_max", type=int, default=20)
    parser.add_argument("--total_num", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bond_sigma_per_mu", type=float, default=None,
                        help="If set, sample bond lengths from Gaussian; else use KDE")
    parser.add_argument("--use_min_bond_len", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    dataset = run_pipeline_template(
        templates_pkl=args.templates,
        precompute_cache=args.precompute_cache,
        natm_range=(args.natm_min, args.natm_max),
        total_num=args.total_num,
        seed=args.seed,
        bond_sigma_per_mu=args.bond_sigma_per_mu,
        use_min_bond_len=args.use_min_bond_len,
    )

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    dataset.save(args.save_path)
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
