from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd


def parse_wyckoff_positions(raw: str) -> List[str]:
    """Parse entries like 'Cs:1a,Cs:1b,Al:1b' into ['1a', '1b', '1b']."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text:
        return []

    tokens: List[str] = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" in piece:
            _, wyck = piece.split(":", 1)
            tokens.append(wyck.strip())
        else:
            tokens.append(piece)
    return tokens


def wyckoff_multiplicity(label: str) -> int:
    """Extract multiplicity from label, e.g. '4c' -> 4, 'a' -> 1."""
    digits = []
    for ch in label:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return int("".join(digits)) if digits else 1


@dataclass
class SGStats:
    n_rows: int
    vocabulary: List[str]
    max_count: Dict[str, int]
    count_hist: Dict[str, Dict[int, int]]
    present_count: Dict[str, int]
    co_present_count: Dict[str, Dict[str, int]]


class WyckoffDPSampler:
    def __init__(self, lambda_smooth: float = 1e-3, mu_smooth: float = 1.0) -> None:
        self.lambda_smooth = lambda_smooth
        self.mu_smooth = mu_smooth
        self.sg_stats: Dict[int, SGStats] = {}

    def fit(self, df: pd.DataFrame) -> "WyckoffDPSampler":
        grouped: Dict[int, List[Counter[str]]] = defaultdict(list)

        for _, row in df.iterrows():
            sg = int(row["spacegroup_number"])
            tokens = parse_wyckoff_positions(row["wyckoff_positions"])
            grouped[sg].append(Counter(tokens))

        self.sg_stats = {}
        for sg, rows in grouped.items():
            vocab = sorted({w for counts in rows for w in counts})
            n_rows = len(rows)

            max_count: Dict[str, int] = {w: 0 for w in vocab}
            count_hist: Dict[str, Dict[int, int]] = {w: defaultdict(int) for w in vocab}
            present_count: Dict[str, int] = {w: 0 for w in vocab}
            co_present_count: Dict[str, Dict[str, int]] = {u: defaultdict(int) for u in vocab}

            for counts in rows:
                present_labels = {w for w, c in counts.items() if c > 0}
                for w in vocab:
                    c = int(counts.get(w, 0))
                    count_hist[w][c] += 1
                    if c > max_count[w]:
                        max_count[w] = c
                    if c > 0:
                        present_count[w] += 1

                for u in present_labels:
                    for w in present_labels:
                        co_present_count[u][w] += 1

            self.sg_stats[sg] = SGStats(
                n_rows=n_rows,
                vocabulary=vocab,
                max_count=max_count,
                count_hist={k: dict(v) for k, v in count_hist.items()},
                present_count=present_count,
                co_present_count={k: dict(v) for k, v in co_present_count.items()},
            )

        return self

    def save(self, path: str | Path) -> None:
        """Serialize fitted model to a JSON file."""
        data = {
            "lambda_smooth": self.lambda_smooth,
            "mu_smooth": self.mu_smooth,
            "sg_stats": {},
        }
        for sg, stats in self.sg_stats.items():
            data["sg_stats"][str(sg)] = {
                "n_rows": stats.n_rows,
                "vocabulary": stats.vocabulary,
                "max_count": stats.max_count,
                "count_hist": {k: {str(ki): vi for ki, vi in v.items()} for k, v in stats.count_hist.items()},
                "present_count": stats.present_count,
                "co_present_count": {k: dict(v) for k, v in stats.co_present_count.items()},
            }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "WyckoffDPSampler":
        """Load a fitted model from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        obj = cls(lambda_smooth=data["lambda_smooth"], mu_smooth=data["mu_smooth"])
        obj.sg_stats = {}
        for sg_str, sd in data["sg_stats"].items():
            obj.sg_stats[int(sg_str)] = SGStats(
                n_rows=sd["n_rows"],
                vocabulary=sd["vocabulary"],
                max_count=sd["max_count"],
                count_hist={k: {int(ki): vi for ki, vi in v.items()} for k, v in sd["count_hist"].items()},
                present_count=sd["present_count"],
                co_present_count=sd["co_present_count"],
            )
        return obj

    def _p_final(self, sg: int, label: str, final_count: int) -> float:
        stats = self.sg_stats[sg]
        cmax = stats.max_count[label]
        if final_count < 0 or final_count > cmax:
            return 0.0

        hist = stats.count_hist[label]
        numerator = hist.get(final_count, 0) + self.lambda_smooth
        denom = sum(hist.get(c, 0) + self.lambda_smooth for c in range(cmax + 1))
        return numerator / denom if denom > 0 else 0.0

    def _p_occ(self, sg: int, u: str, w: str) -> float:
        stats = self.sg_stats[sg]
        n_u = stats.present_count.get(u, 0)
        n_uw = stats.co_present_count.get(u, {}).get(w, 0)
        return (n_uw + self.mu_smooth) / (n_u + 2.0 * self.mu_smooth)

    def rho(self, sg: int, known_counts: Dict[str, int], candidate_w: str) -> float:
        support = [u for u, c in known_counts.items() if c > 0]
        if not support:
            return 0.0

        values = [math.log(self._p_occ(sg, u, candidate_w)) for u in support]
        return sum(values) / len(values)

    def shortlist_candidates(self, sg: int, top_k: int | None = None) -> List[str]:
        if sg not in self.sg_stats:
            return []
        vocab = self.sg_stats[sg].vocabulary
        if top_k is None or top_k >= len(vocab):
            return list(vocab)
        return list(vocab[:top_k])

    def _unary_weight(
        self,
        sg: int,
        known_counts: Dict[str, int],
        label: str,
        a: int,
        alpha: float,
        tau: float,
        rho_cache: Dict[str, float],
        temperature: float = 1.0,
    ) -> float:
        stats = self.sg_stats[sg]
        c_known = int(known_counts.get(label, 0))
        cmax = stats.max_count[label]
        if a < 0 or a > (cmax - c_known):
            return 0.0

        final_count = c_known + a
        p_final = self._p_final(sg, label, final_count)
        if p_final <= 0:
            return 0.0

        if a > 0:
            bonus = alpha * rho_cache[label] - tau
            raw = p_final * math.exp(bonus)
        else:
            raw = p_final

        if temperature == 1.0:
            return raw
        return raw ** (1.0 / temperature)

    def sample_completion(
        self,
        sg: int,
        known_counts: Dict[str, int],
        remaining_atoms: int,
        candidate_labels: Sequence[str],
        alpha: float = 1.0,
        tau: float = 0.1,
        temperature: float = 1.0,
        rng: random.Random | None = None,
    ) -> Dict[str, int]:
        if rng is None:
            rng = random.Random()
        if sg not in self.sg_stats:
            raise ValueError(f"Space group {sg} is not available in fitted stats.")

        labels = list(candidate_labels)
        M = len(labels)
        multiplicities = [wyckoff_multiplicity(w) for w in labels]

        rho_cache = {w: self.rho(sg, known_counts, w) for w in labels}

        # Z[i][r] corresponds to Z(i+1, r) with 0-based i.
        Z = [[0.0 for _ in range(remaining_atoms + 1)] for _ in range(M + 1)]
        Z[M][0] = 1.0

        for i in range(M - 1, -1, -1):
            w = labels[i]
            m = multiplicities[i]
            c_known = int(known_counts.get(w, 0))
            cmax = self.sg_stats[sg].max_count[w]
            add_cap = max(cmax - c_known, 0)

            for r in range(remaining_atoms + 1):
                a_max = min(add_cap, r // m)
                total = 0.0
                for a in range(a_max + 1):
                    fw = self._unary_weight(sg, known_counts, w, a, alpha, tau, rho_cache, temperature)
                    if fw <= 0:
                        continue
                    total += fw * Z[i + 1][r - a * m]
                Z[i][r] = total

        if Z[0][remaining_atoms] <= 0:
            raise ValueError(
                "No feasible weighted completion found. Try larger candidate set or different known/remaining settings."
            )

        sampled_additional: Dict[str, int] = {}
        r = remaining_atoms
        for i in range(M):
            w = labels[i]
            m = multiplicities[i]
            c_known = int(known_counts.get(w, 0))
            cmax = self.sg_stats[sg].max_count[w]
            add_cap = max(cmax - c_known, 0)
            a_max = min(add_cap, r // m)

            choices: List[int] = []
            probs: List[float] = []
            for a in range(a_max + 1):
                fw = self._unary_weight(sg, known_counts, w, a, alpha, tau, rho_cache, temperature)
                if fw <= 0:
                    continue
                p = fw * Z[i + 1][r - a * m]
                if p <= 0:
                    continue
                choices.append(a)
                probs.append(p)

            if not choices:
                raise RuntimeError(f"Sampling failed at state (i={i}, r={r})")

            norm = sum(probs)
            probs = [p / norm for p in probs]
            a_star = rng.choices(choices, weights=probs, k=1)[0]

            sampled_additional[w] = a_star
            r -= a_star * m

        if r != 0:
            raise RuntimeError(f"Sampling ended with residual atoms r={r}, expected 0")

        return sampled_additional


def pick_demo_problem(df: pd.DataFrame, rng: random.Random) -> Tuple[int, Dict[str, int], int, Dict[str, int]]:
    """
    Build a small synthetic completion task from one row:
    - full_counts from the row
    - randomly keep some known counts
    - derive remaining atom budget from held-out counts
    """
    idx = rng.randrange(len(df))
    row = df.iloc[idx]
    sg = int(row["spacegroup_number"])

    full_tokens = parse_wyckoff_positions(row["wyckoff_positions"])
    full_counts = Counter(full_tokens)

    known_counts: Dict[str, int] = {}
    for w, c in full_counts.items():
        keep = rng.randint(0, c)
        if keep > 0:
            known_counts[w] = keep

    # Ensure at least one unknown token remains.
    if known_counts == dict(full_counts):
        w = next(iter(full_counts))
        known_counts[w] = max(known_counts[w] - 1, 0)
        if known_counts[w] == 0:
            del known_counts[w]

    remaining_counts = Counter(full_counts) - Counter(known_counts)
    remaining_atoms = sum(wyckoff_multiplicity(w) * c for w, c in remaining_counts.items())

    return sg, known_counts, remaining_atoms, dict(full_counts)


def format_counts(d: Dict[str, int]) -> str:
    if not d:
        return "{}"
    items = sorted((k, v) for k, v in d.items() if v > 0)
    return "{" + ", ".join(f"{k}:{v}" for k, v in items) + "}"


def main() -> None:
    parser = argparse.ArgumentParser(description="DP sampler for Wyckoff spectator completion.")
    parser.add_argument("--csv", default="wyckoff_data.csv", help="Path to wyckoff_data.csv")
    parser.add_argument("--nrows", type=int, default=200, help="Only read first N rows for quick checks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=5, help="Number of completion samples to draw")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (>1 = more diverse, <1 = sharper)")
    parser.add_argument("--lambda_smooth", type=float, default=1e-3)
    parser.add_argument("--mu_smooth", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=30, help="Candidate shortlist size within the SG vocab")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, nrows=args.nrows if args.nrows > 0 else None)
    if len(df) == 0:
        raise ValueError("CSV is empty after loading.")

    model = WyckoffDPSampler(lambda_smooth=args.lambda_smooth, mu_smooth=args.mu_smooth).fit(df)

    rng = random.Random(args.seed)
    sg, known_counts, remaining_atoms, full_counts = pick_demo_problem(df, rng)

    candidates = model.shortlist_candidates(sg, top_k=args.top_k)

    print(f"Loaded rows: {len(df)}")
    print(f"Demo space group: {sg}")
    print(f"Known counts K: {format_counts(known_counts)}")
    print(f"Remaining atom budget R: {remaining_atoms}")
    print(f"Candidate labels: {len(candidates)}")

    print("\nSamples (additional-count vectors):")
    for i in range(args.samples):
        sampled = model.sample_completion(
            sg=sg,
            known_counts=known_counts,
            remaining_atoms=remaining_atoms,
            candidate_labels=candidates,
            alpha=args.alpha,
            tau=args.tau,
            temperature=args.temperature,
            rng=rng,
        )

        sampled_nonzero = {k: v for k, v in sampled.items() if v > 0}
        final_counts = Counter(known_counts) + Counter(sampled)

        print(f"sample {i + 1}: add={format_counts(sampled_nonzero)}")
        print(f"           final={format_counts(dict(final_counts))}")

    print("\nReference full counts from selected row:")
    print(format_counts(full_counts))


if __name__ == "__main__":
    main()
