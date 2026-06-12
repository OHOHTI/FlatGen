"""
Wyckoff template utilities for constrained crystal generation.

Pipeline context:
  step1_extract_sublattices.py  →  ../data/sublattices.pkl
  step2_enumerate_subgroups.py  →  ../data/step2_candidates.pkl
  step3_wyckoff_assignment.py   →  ../data/step3_compatible_templates.pkl
                                    (compatible mp_20 templates per sublattice)

Key public API
--------------
  WyckoffConfig               — dataclass: space_group, wyckoff_counts, total_atoms, material_id
  load_mp20_wyckoff_index()   — build/load the mp_20 Wyckoff index (cached as JSON)
  TemplateFinder              — search the index for compatible templates

Usage:
  # Build / reuse the mp_20 index and search it
  from template_utils import load_mp20_wyckoff_index, TemplateFinder
  index = load_mp20_wyckoff_index("../data/mp_20", "../data/mp_20_wyckoff_index.json")
  finder = TemplateFinder(index)
  templates = finder.find(space_group_number=225, constrained_wyckoffs={"a": 1, "c": 2})
"""

import json
import logging
import argparse
import re
from pathlib import Path
from typing import Optional
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class WyckoffConfig:
    """A Wyckoff configuration: mapping from Wyckoff letter to count of occupied orbits.

    wyckoff_counts uses bare letters (e.g. {"a": 1, "b": 2}) without multiplicity
    prefix, matching PyXtal's convention.
    """
    space_group: int
    wyckoff_counts: dict = field(default_factory=dict)
    # total atoms = sum of (multiplicity × orbit_count) for each letter
    total_atoms: int = 0
    # mp material_id, e.g. "mp-1234"
    material_id: str = ""

    def contains(self, partial: dict) -> bool:
        """Return True if every letter in *partial* has count ≤ this config's count."""
        for letter, count in partial.items():
            if self.wyckoff_counts.get(letter, 0) < count:
                return False
        return True

    def unconstrained_wyckoffs(self, partial: dict) -> dict:
        """Return orbits not consumed by the constrained (sublattice) assignment."""
        remaining = {}
        for letter, count in self.wyckoff_counts.items():
            leftover = count - partial.get(letter, 0)
            if leftover > 0:
                remaining[letter] = leftover
        return remaining

    def to_dict(self) -> dict:
        return {
            "space_group": self.space_group,
            "wyckoff_counts": self.wyckoff_counts,
            "total_atoms": self.total_atoms,
            "material_id": self.material_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WyckoffConfig":
        return cls(
            space_group=d["space_group"],
            wyckoff_counts=d["wyckoff_counts"],
            total_atoms=d["total_atoms"],
            material_id=d.get("material_id", ""),
        )


# ── Index building helpers ────────────────────────────────────────────────────

def _wyckoff_letter(pymatgen_label: str) -> str:
    """Strip the multiplicity prefix from a pymatgen Wyckoff label.

    e.g. '4b' → 'b',  '8c' → 'c',  'a' → 'a'
    """
    return re.sub(r"^\d+", "", pymatgen_label)



def load_mp_complete_wyckoff_index(
    csv_path: str,
    cache_path: str,
    symprec: float = 0.1,
) -> dict[int, list[WyckoffConfig]]:
    """Load the full MP Wyckoff index from a JSON cache, or build and cache it.

    Reads *csv_path* (mp_complete_summary.csv) which contains a ``structure``
    column (pymatgen Structure JSON) and a ``symmetry`` column with a pre-computed
    space group number.  Building the index calls SpacegroupAnalyzer on every
    row to obtain Wyckoff symbols; the result is cached as JSON at *cache_path*.

    Args:
        csv_path  : path to mp_complete_summary.csv.
        cache_path: path for the JSON cache file.
        symprec   : symmetry tolerance passed to SpacegroupAnalyzer.

    Returns:
        dict mapping int space group → list[WyckoffConfig].
    """
    cache = Path(cache_path)

    if cache.exists():
        logger.info(f"Loading MP-complete Wyckoff index from {cache}")
        return _load_index_json(cache)

    logger.info(f"Building MP-complete Wyckoff index from {csv_path} (this may take a while)…")
    index = _build_index_from_mp_complete_csv(csv_path, symprec)
    _save_index_json(index, cache)
    total = sum(len(v) for v in index.values())
    logger.info(f"Index built: {total} entries across {len(index)} space groups → {cache}")
    return index


def _build_index_from_mp_complete_csv(
    csv_path: str,
    symprec: float,
) -> dict[int, list[WyckoffConfig]]:
    import ast
    import pandas as pd
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    index: dict[int, list[WyckoffConfig]] = {}
    df = pd.read_csv(csv_path, usecols=["material_id", "structure", "symmetry"])
    logger.info(f"  Processing {len(df)} rows from {csv_path} …")

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 10000 == 0 and i > 0:
            logger.info(f"    {i}/{len(df)} …")
        try:
            struct_dict = ast.literal_eval(row["structure"])
            sym_dict = ast.literal_eval(row["symmetry"])
            spg_number = int(sym_dict["number"])
            structure = Structure.from_dict(struct_dict)
            analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
            sym = analyzer.get_symmetrized_structure()
            letters = [_wyckoff_letter(lbl) for lbl in sym.wyckoff_symbols]
            wyckoff_counts = dict(Counter(letters))
            total_atoms = sum(len(sites) for sites in sym.equivalent_sites)
            config = WyckoffConfig(
                space_group=spg_number,
                wyckoff_counts=wyckoff_counts,
                total_atoms=total_atoms,
                material_id=str(row["material_id"]),
            )
            index.setdefault(config.space_group, []).append(config)
        except Exception as exc:
            logger.debug(f"Skipping {row.get('material_id', '?')}: {exc}")

    return index



def _load_index_json(path: Path) -> dict[int, list[WyckoffConfig]]:
    with open(path) as f:
        raw = json.load(f)
    return {
        int(spg): [WyckoffConfig.from_dict(c) for c in configs]
        for spg, configs in raw.items()
    }


def _save_index_json(index: dict[int, list[WyckoffConfig]], path: Path) -> None:
    raw = {
        str(spg): [c.to_dict() for c in configs]
        for spg, configs in index.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(raw, f)


# ── Step-3 record helpers ─────────────────────────────────────────────────────

def get_wyckoff_config_from_step3_entry(entry: dict) -> Optional[WyckoffConfig]:
    """Extract a representative WyckoffConfig from a new step3 record.

    Pulls the first entry from ``compatible_templates`` (if present), so
    callers can treat step3 records uniformly with other WyckoffConfig sources.

    Returns None when no compatible templates were found.
    """
    templates = entry.get("compatible_templates")
    if templates:
        return WyckoffConfig.from_dict(templates[0])
    return None


# ── TemplateFinder ────────────────────────────────────────────────────────────

class TemplateFinder:
    """Search a Wyckoff index for templates compatible with given constraints.

    Construct from a pre-built index dict (returned by load_mp20_wyckoff_index)
    or from a JSON cache file path.

    Examples
    --------
    index = load_mp20_wyckoff_index("../data/mp_20", "../data/mp_20_wyckoff_index.json")
    finder = TemplateFinder(index)
    results = finder.find(225, {"a": 1, "c": 2})
    """

    def __init__(self, index: dict[int, list[WyckoffConfig]]):
        """
        Args:
            index: dict mapping space_group → list[WyckoffConfig].
        """
        self.index = index

    @classmethod
    def from_json(cls, cache_path: str) -> "TemplateFinder":
        """Load directly from a previously saved JSON index file."""
        return cls(_load_index_json(Path(cache_path)))

    def find(
        self,
        space_group_number: int,
        constrained_wyckoffs: dict,
        max_atoms: int = 52,
        min_atoms: int = 1,
        max_results: int = 100,
    ) -> list[WyckoffConfig]:
        """Return all templates whose config contains *constrained_wyckoffs* as a subset.

        Args:
            space_group_number  : G_full (the target space group).
            constrained_wyckoffs: {letter: orbit_count} for sublattice atoms.
            max_atoms           : upper bound on template unit-cell size.
            min_atoms           : lower bound.
            max_results         : cap on returned results.
        """
        candidates = self.index.get(space_group_number, [])
        if not candidates:
            logger.warning(f"No mp_20 entries found for space group {space_group_number}")
            return []

        results = []
        for config in candidates:
            if not (min_atoms <= config.total_atoms <= max_atoms):
                continue
            if config.contains(constrained_wyckoffs):
                results.append(config)
            if len(results) >= max_results:
                break
        return results

    def find_across_subgroups(
        self,
        subgroup_numbers: list[int],
        constrained_wyckoffs_per_subgroup: dict[int, dict],
        max_atoms: int = 52,
        min_atoms: int = 1,
        max_results_per_group: int = 50,
    ) -> dict[int, list[WyckoffConfig]]:
        """Find templates across multiple candidate G_full values.

        Args:
            subgroup_numbers                 : candidate space group numbers.
            constrained_wyckoffs_per_subgroup: {spg: {letter: count}} mapping.
        """
        all_results = {}
        for spg in subgroup_numbers:
            constrained = constrained_wyckoffs_per_subgroup.get(spg, {})
            results = self.find(
                space_group_number=spg,
                constrained_wyckoffs=constrained,
                max_atoms=max_atoms,
                min_atoms=min_atoms,
                max_results=max_results_per_group,
            )
            if results:
                all_results[spg] = results
                logger.info(f"SG {spg}: {len(results)} compatible templates")
            else:
                logger.info(f"SG {spg}: no compatible templates")
        return all_results

    def get_template_wyckoff_for_generation(
        self,
        template: WyckoffConfig,
        constrained_wyckoffs: dict,
    ) -> dict:
        """Split a template into constrained and unconstrained Wyckoff parts.

        Returns
        -------
        dict with keys:
          space_group, total_atoms, constrained, unconstrained, full_config, material_id
        """
        return {
            "space_group": template.space_group,
            "total_atoms": template.total_atoms,
            "constrained": constrained_wyckoffs,
            "unconstrained": template.unconstrained_wyckoffs(constrained_wyckoffs),
            "full_config": template.wyckoff_counts,
            "material_id": template.material_id,
        }

    def summary(self) -> None:
        total = sum(len(v) for v in self.index.values())
        print(f"Index: {total} templates across {len(self.index)} space groups")
        print("Top 10 space groups by count:")
        for spg, configs in sorted(self.index.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"  SG {spg:>3d}: {len(configs):>6d} templates")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_constrained_wyckoffs(s: str) -> dict:
    """Parse 'a:1,c:2' → {'a': 1, 'c': 2}."""
    result = {}
    for pair in s.split(","):
        letter, count = pair.strip().split(":")
        result[letter.strip()] = int(count.strip())
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Search MP-complete for Wyckoff-compatible templates"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="../data/mp_complete_summary.csv",
        help="Path to mp_complete_summary.csv",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="../data/mp_complete_wyckoff_index.json",
        help="Path to JSON cache for the MP-complete Wyckoff index",
    )
    parser.add_argument("--spg", type=int, required=True, help="Target space group (G_full)")
    parser.add_argument(
        "--constrained",
        type=str,
        required=True,
        help='Constrained Wyckoff positions, e.g. "a:1,c:2"',
    )
    parser.add_argument("--max-atoms", type=int, default=52)
    parser.add_argument("--min-atoms", type=int, default=1)
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--symprec", type=float, default=0.1)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    constrained = parse_constrained_wyckoffs(args.constrained)
    print(f"Searching SG {args.spg} with W_c = {constrained}")

    index = load_mp_complete_wyckoff_index(args.csv, args.cache, symprec=args.symprec)
    finder = TemplateFinder(index)
    finder.summary()

    results = finder.find(
        space_group_number=args.spg,
        constrained_wyckoffs=constrained,
        max_atoms=args.max_atoms,
        min_atoms=args.min_atoms,
        max_results=args.max_results,
    )

    print(f"\nFound {len(results)} compatible templates:\n")
    for i, config in enumerate(results):
        scaffold = finder.get_template_wyckoff_for_generation(config, constrained)
        print(f"  [{i+1}] {config.material_id}")
        print(f"      Wyckoff config : {config.wyckoff_counts}")
        print(f"      Total atoms    : {config.total_atoms}")
        print(f"      Unconstrained  : {scaffold['unconstrained']}")
        print()


if __name__ == "__main__":
    main()
