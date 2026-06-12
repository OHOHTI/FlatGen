"""Standalone flat-band cluster screening workflow.

This script extracts the main logic from ``screened_cluster.ipynb`` and
reproduces the two final notebook figures:

- ``novelty_score_distribution.png``
- ``UMAP_Discovery_Plot.png``

It also saves the same intermediate arrays used by the notebook under the
``Clusters_fb/<MODEL_NAME>`` folder.
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from pathlib import Path
from typing import Any

TMP_CACHE_ROOT = Path(os.environ.get("FLATGEN_TMP_CACHE", "/tmp/flatgen-cluster"))
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(TMP_CACHE_ROOT / "numba"))

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import hdbscan
except ModuleNotFoundError:
    hdbscan = None

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    import umap.umap_ as umap
except ModuleNotFoundError:
    umap = None

try:
    from adjustText import adjust_text
except ModuleNotFoundError:
    adjust_text = None

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)

from config import (
    CLUSTER_OUTPUT_DIR,
    CLUSTER_PLOT_DIR,
    DUP_LIST,
    EMBEDDINGS_PATH,
    EMBEDDING_UMAP_PATH,
    FB_LIST,
    LABELS_PATH,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FB_LIST = FB_LIST
DEFAULT_DUP_LIST = DUP_LIST

DIMENSIONALITY_REDUCTION_ALGORITHM = "UMAP"
SPECIAL_SUBLATTICES = [
    "mp-bizmh_Cs",
    "mp-xpb_O",
    "mp-bewys_Fe",
]
SPECIAL_SUBLATTICE_COLOR = "deepskyblue"


def ensure_dependencies() -> None:
    missing = []
    if torch is None:
        missing.append("torch")
    if umap is None:
        missing.append("umap-learn")
    if hdbscan is None:
        missing.append("hdbscan")
    if adjust_text is None:
        missing.append("adjustText")
    if missing:
        raise SystemExit(
            "Missing dependencies: "
            + ", ".join(missing)
            + ". Run this script in the same environment used for the notebook "
            + "(for example the 'cluster' conda env)."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce cluster screening plots from screened_cluster.ipynb."
    )
    parser.add_argument(
        "--fb-list",
        type=Path,
        default=DEFAULT_FB_LIST,
        help="Excel file containing the screened flat-band sublattice IDs.",
    )
    parser.add_argument(
        "--dup-list",
        type=Path,
        default=DEFAULT_DUP_LIST,
        help="CSV file used to resolve duplicate RCSR assignments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR,
        help="Directory where the final PNGs and outliers.txt are written.",
    )
    return parser.parse_args()


def _pick_unique_rcsr(group: pd.DataFrame) -> pd.Series:
    group = group.sort_values("NNN_dist", ascending=False)
    non_none = group[~group["rcsr_is_none"]]
    chosen = non_none.iloc[0] if len(non_none) > 0 else group.iloc[0]
    return pd.Series(
        {
            "rcsr_name": chosen["rcsr_names"],
            "NNN_dist": chosen["NNN_dist"],
        }
    )


def _parse_rcsr_cell(raw: object) -> str:
    """Parse notebook-style strings like ``[np.str_('hca')]`` into ``hca``."""
    value = str(raw).strip()
    matches = re.findall(r"np\.str_\('([^']+)'\)", value)
    if matches:
        for match in matches:
            if match != "None":
                return match
        return "None"
    if value in ("", "nan", "NaN", "None", "none"):
        return "None"
    return value


def load_filtered_embeddings(
    fb_list: Path,
    dup_list: Path,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    embeddings = torch.load(EMBEDDINGS_PATH, map_location=torch.device("cpu")).numpy()
    labels = torch.load(LABELS_PATH, map_location=torch.device("cpu"))

    df_sublats = pd.read_excel(fb_list, sheet_name="Sheet1")
    print(f"Number of flatband sublattices: {len(df_sublats)}")

    lower_cols = [column.lower() for column in df_sublats.columns]
    id_col = next((column for column, lower in zip(df_sublats.columns, lower_cols) if "id" in lower), None)
    species_col = next(
        (column for column, lower in zip(df_sublats.columns, lower_cols) if "spec" in lower),
        None,
    )
    if id_col is None or species_col is None:
        raise ValueError(
            f"Could not find id/species columns in flat-band list. Columns: {list(df_sublats.columns)}"
        )

    sublat_id_list = [f"{row[id_col]}_{row[species_col]}" for _, row in df_sublats.iterrows()]
    print(sublat_id_list[:5])

    filtered_indices = [index for index, label in enumerate(labels) if label in sublat_id_list]
    data_for_clustering = embeddings[filtered_indices]
    labels_for_clustering = [labels[index] for index in filtered_indices]
    print(f"Number of embeddings after filtering: {len(data_for_clustering)}")
    print(labels_for_clustering[:5])

    dup_df = pd.read_csv(dup_list).copy()
    dup_df["rcsr_names"] = dup_df["rcsr_names"].fillna("").astype(str).str.strip()
    dup_df["rcsr_is_none"] = dup_df["rcsr_names"].isin(["", "None", "none", "nan", "NaN"])
    unique_rcsr_df = (
        dup_df.groupby(["mp_id", "species"], as_index=False)
        .apply(_pick_unique_rcsr, include_groups=False)
        .reset_index(drop=True)
    )

    df_sublats = df_sublats.copy()
    lower_to_original = {column.lower(): column for column in df_sublats.columns}
    rcsr_col = next((lower_to_original[column] for column in lower_to_original if "rcsr" in column), None)
    nnn_col = next((lower_to_original[column] for column in lower_to_original if "nnn" in column), None)
    if rcsr_col is None:
        raise ValueError(
            f"Could not find RCSR column in flat-band list. Columns: {list(df_sublats.columns)}"
        )

    df_sublats["_label"] = (
        df_sublats[id_col].astype(str).str.strip() + "_" + df_sublats[species_col].astype(str).str.strip()
    )
    df_sublats["_rcsr"] = df_sublats[rcsr_col].apply(_parse_rcsr_cell)
    df_sublats["_rcsr_is_none"] = df_sublats["_rcsr"] == "None"

    dup_map = unique_rcsr_df.set_index(["mp_id", "species"])["rcsr_name"].to_dict()

    def _pick_rcsr(group: pd.DataFrame) -> str:
        non_none = group[~group["_rcsr_is_none"]]
        unique_values = non_none["_rcsr"].unique().tolist()
        if len(unique_values) == 1:
            return unique_values[0]
        if len(unique_values) == 0:
            return "None"

        key = (
            group[id_col].astype(str).str.strip().iloc[0],
            group[species_col].astype(str).str.strip().iloc[0],
        )
        mapped = dup_map.get(key, "")
        if mapped:
            return mapped
        if nnn_col is not None and len(non_none) > 0:
            return non_none.sort_values(nnn_col, ascending=False).iloc[0]["_rcsr"]
        return unique_values[-1]

    label_to_rcsr = (
        df_sublats.groupby("_label")
        .apply(_pick_rcsr, include_groups=False)
        .to_dict()
    )
    rcsr_name_list = [label_to_rcsr.get(label, "None") for label in labels_for_clustering]
    rcsr_name_list = [value if value not in ("", "nan", "NaN") else "None" for value in rcsr_name_list]

    known = sum(1 for value in rcsr_name_list if value != "None")
    print(f"RCSR names: {known} known, {len(rcsr_name_list) - known} unknown (out of {len(rcsr_name_list)})")

    return data_for_clustering, labels_for_clustering, rcsr_name_list, np.asarray(filtered_indices)


def compute_embedding_umap(data_for_clustering: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=10,
        metric="euclidean",
        random_state=42,
    )
    embedding_umap = reducer.fit_transform(data_for_clustering)
    np.save(EMBEDDING_UMAP_PATH, embedding_umap)
    print(f"Saved embedding_umap to {EMBEDDING_UMAP_PATH}")
    return embedding_umap


def inspect_distance_distribution(embedding_umap: np.ndarray) -> None:
    dists = pairwise_distances(embedding_umap, metric="euclidean")
    upper = dists[np.triu_indices_from(dists, k=1)]

    plt.figure(figsize=(8, 4))
    plt.hist(upper, bins=200)
    plt.axvline(upper.mean() - upper.std(), color="r", linestyle="--", label="mean - 1std")
    plt.axvline(upper.mean() - 2 * upper.std(), color="g", linestyle="--", label="mean - 2std")
    plt.show()
    plt.close()

    print(f"Min: {upper.min():.3f}")
    print(f"Max: {upper.max():.3f}")
    print(f"Mean: {upper.mean():.3f}")
    print(f"Std: {upper.std():.3f}")


def hdbscan_clustering(data: np.ndarray) -> tuple[Any, np.ndarray, np.ndarray]:
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=8,
        min_samples=4,
        cluster_selection_epsilon=0.1,
        metric="euclidean",
    )
    cluster_labels = hdb.fit_predict(data)
    outlier_scores = hdb.outlier_scores_
    return hdb, cluster_labels, outlier_scores


def save_clusters_with_labels(
    cluster_output_folder: Path,
    cluster_labels: np.ndarray,
    embedding_labels: list[str],
    filename: str = "clustered_labels.npy",
) -> None:
    combined_array = np.column_stack((embedding_labels, cluster_labels))
    out_path = cluster_output_folder / filename
    np.save(out_path, combined_array)
    print(f"Clusters and labels saved to {out_path}")


def load_clusters_with_labels(
    cluster_output_folder: Path,
    filename: str = "clustered_labels.npy",
) -> tuple[np.ndarray, np.ndarray]:
    in_path = cluster_output_folder / filename
    loaded_array = np.load(in_path, allow_pickle=True)
    embedding_labels = loaded_array[:, 0]
    cluster_labels = loaded_array[:, 1]
    print(f"Clusters and labels loaded from {in_path}")
    return embedding_labels, cluster_labels


def umap_plot(
    cluster_output_folder: Path,
    plot_output_folder: Path,
    data: np.ndarray,
    color_labels: list[str] | np.ndarray,
    title: str,
    filename: str,
) -> np.ndarray:
    if data.shape[1] > 2:
        print("Running UMAP...")
        umap_reducer = umap.UMAP(n_neighbors=20, n_components=2, min_dist=0.5, random_state=42, init="pca")
        data_umap_2d = umap_reducer.fit_transform(data)
    else:
        data_umap_2d = data

    np.save(cluster_output_folder / "data_umap_2d.npy", data_umap_2d)

    color_labels = np.array(color_labels)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if not np.issubdtype(color_labels.dtype, np.number):
        unique_vals, counts = np.unique(color_labels, return_counts=True)
        valid_mask = unique_vals != "None"
        valid_vals = unique_vals[valid_mask]
        valid_counts = counts[valid_mask]
        sorted_indices = np.argsort(-valid_counts)
        top_labels = valid_vals[sorted_indices][:10]

        is_top = np.isin(color_labels, top_labels)
        bg_indices = ~is_top
        ax.scatter(
            data_umap_2d[bg_indices, 0],
            data_umap_2d[bg_indices, 1],
            c="#E0E0E0",
            s=15,
            alpha=0.3,
            linewidth=0,
            label="Other/None",
            zorder=1,
        )

        cmap = plt.get_cmap("tab10") if len(top_labels) <= 10 else plt.get_cmap("tab20")
        legend_elements: list[Line2D] = []
        for index, label in enumerate(top_labels):
            mask = color_labels == label
            color = cmap(index)
            ax.scatter(
                data_umap_2d[mask, 0],
                data_umap_2d[mask, 1],
                color=color,
                s=15,
                alpha=0.8,
                linewidth=0,
                zorder=2,
            )
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=label,
                    markerfacecolor=color,
                    markersize=10,
                )
            )

        legend = ax.legend(
            handles=legend_elements,
            title="RCSR Categories",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
            fontsize=10,
        )
        legend.get_title().set_fontweight("bold")
    else:
        background_points = color_labels == -1
        ax.scatter(
            data_umap_2d[background_points, 0],
            data_umap_2d[background_points, 1],
            c="#E0E0E0",
            s=10,
            alpha=0.3,
            label="Noise",
            zorder=1,
        )

        scatter = ax.scatter(
            data_umap_2d[~background_points, 0],
            data_umap_2d[~background_points, 1],
            c=color_labels[~background_points],
            cmap="Spectral",
            s=10,
            alpha=0.8,
            zorder=2,
        )
        colorbar = plt.colorbar(scatter)
        colorbar.set_label("Cluster ID")

        unique_labels = np.unique(color_labels)
        for label in unique_labels:
            if label == -1:
                continue
            label_points = data_umap_2d[color_labels == label]
            centroid = np.mean(label_points, axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                str(label),
                fontsize=9,
                fontweight="bold",
                color="black",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
            )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_output_folder / filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return data_umap_2d


def full_clustering_workflow(
    cluster_output_folder: Path,
    plot_output_folder: Path,
    embedding_umap: np.ndarray,
    labels_for_clustering: list[str],
    rcsr_name_list: list[str],
    dimensionality_reduction: str = DIMENSIONALITY_REDUCTION_ALGORITHM,
) -> tuple[Any, np.ndarray]:
    print("Clustering with HDBSCAN...")
    hdb, cluster_labels, outlier_scores = hdbscan_clustering(embedding_umap)

    save_clusters_with_labels(cluster_output_folder, cluster_labels, labels_for_clustering, filename="clustered_labels.npy")

    if dimensionality_reduction in ("UMAP", "both"):
        print("Visualizing with UMAP...")
        umap_plot(
            cluster_output_folder,
            plot_output_folder,
            embedding_umap,
            cluster_labels,
            title="UMAP Visualization of HDBSCAN Clusters in 2D",
            filename="UMAP Clusters.png",
        )
        umap_plot(
            cluster_output_folder,
            plot_output_folder,
            embedding_umap,
            rcsr_name_list,
            title="UMAP Visualization Colored by RCSR Name",
            filename="UMAP RCSR.png",
        )

    clustered_mask = cluster_labels != -1
    noise_mask = cluster_labels == -1

    print("Total number of data points:", len(cluster_labels))
    print("Number of clustered points:", np.sum(clustered_mask))
    print("Number of noise points:", np.sum(noise_mask))

    valid_indices = ~np.isnan(outlier_scores)
    clustered_mask &= valid_indices
    noise_mask &= valid_indices

    clustered_outlier_scores = outlier_scores[clustered_mask]
    noise_outlier_scores = outlier_scores[noise_mask]

    if clustered_outlier_scores.size > 0:
        print(f"Average Outlier Score (Clustered): {np.mean(clustered_outlier_scores):.4f}")
    else:
        print("No valid outlier scores for clustered points.")

    if noise_outlier_scores.size > 0:
        print(f"Average Outlier Score (Noise): {np.mean(noise_outlier_scores):.4f}")
    else:
        print("No valid outlier scores for noise points.")

    return hdb, cluster_labels


def calculate_multi_centroid_novelty(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    rcsr_labels: list[str] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rcsr_labels = np.array(rcsr_labels)

    clustered_mask = cluster_labels != -1
    known_mask = rcsr_labels != "None"

    if np.any(clustered_mask):
        dists_clustered = euclidean_distances(embeddings, embeddings[clustered_mask])
        min_dist_clustered = np.min(dists_clustered, axis=1)
    else:
        min_dist_clustered = np.full(len(embeddings), np.inf)

    if np.any(known_mask):
        dists_known = euclidean_distances(embeddings, embeddings[known_mask])
        min_dist_known = np.min(dists_known, axis=1)
    else:
        min_dist_known = np.full(len(embeddings), np.inf)

    min_dist = np.minimum(min_dist_clustered, min_dist_known)
    return min_dist, min_dist_clustered, min_dist_known


def write_outliers_file(
    output_dir: Path,
    scores: np.ndarray,
    labels_for_clustering: list[str],
) -> None:
    scores_sorted = np.sort(scores)[::-1]
    ids_sorted = [labels_for_clustering[index] for index in np.argsort(scores)][::-1]

    scores_reduced = scores[scores > 0.01]
    threshold = np.percentile(scores_reduced, 50)
    scores_selected = scores_sorted[scores_sorted > threshold]
    ids_selected = ids_sorted[: len(scores_selected)]

    with open(output_dir / "outliers.txt", "w", encoding="utf-8") as handle:
        handle.write(f"{'Material_ID'} {'Score'}\n")
        for name, score in zip(ids_selected, scores_selected):
            handle.write(f"{name} {score:<10.4f}\n")


def plot_novelty_score_distribution(output_dir: Path, scores: np.ndarray) -> np.ndarray:
    scores_reduced = scores[scores > 0.01]
    plt.figure(figsize=(10, 6))

    counts, _, _ = plt.hist(
        scores_reduced,
        bins=50,
        color="#4c72b0",
        alpha=0.7,
        edgecolor="black",
        label="Novelty Score",
    )

    threshold = np.percentile(scores_reduced, 50)
    plt.axvline(
        threshold,
        color="#c44e52",
        linestyle="--",
        linewidth=2,
        label=f"50th Percentile Cutoff ({threshold:.2f})",
    )

    plt.xlabel("Novelty Score (Euclidean Distance)", fontsize=13)
    plt.ylabel("Number of Sublattices", fontsize=13)
    plt.legend()
    plt.text(
        threshold * 1.1,
        max(counts) * 0.85,
        "True Novelty Candidates\n(Select these for generation)",
        color="#c44e52",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "novelty_score_distribution.png", dpi=300)
    plt.show()
    plt.close()

    scores_selected = scores_reduced[scores_reduced > threshold]
    print(f"Number of sublattices selected: {len(scores_selected)}")
    return scores_reduced


def plot_publication_umap_candidates(
    plot_output_folder: Path,
    data_2d: np.ndarray,
    embedding_labels: list[str] | np.ndarray,
    rcsr_labels: list[str] | np.ndarray,
    scores: np.ndarray,
    cluster_labels: np.ndarray,
    filename: str,
    rcsr_min_count: int = 3,
    rcsr_min_fraction: float = 0.3,
    annotate_special_ids: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    embedding_labels = np.array(embedding_labels)
    rcsr_labels = np.array(rcsr_labels)
    cluster_labels = np.array(cluster_labels)

    noise_mask = cluster_labels == -1
    ax.scatter(
        data_2d[noise_mask, 0],
        data_2d[noise_mask, 1],
        c="#E0E0E0",
        s=16,
        alpha=0.3,
        linewidth=0,
        label="Noise",
        zorder=1,
    )

    clustered_mask = ~noise_mask
    scatter = ax.scatter(
        data_2d[clustered_mask, 0],
        data_2d[clustered_mask, 1],
        c=cluster_labels[clustered_mask],
        cmap="Spectral",
        s=16,
        alpha=0.8,
        linewidth=0,
        zorder=2,
    )
    colorbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    colorbar.set_label("Cluster ID", fontsize=13)

    unique_clusters = sorted(set(cluster_labels) - {-1})
    texts = []
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        if cluster_mask.sum() == 0:
            continue
        centroid = data_2d[cluster_mask].mean(axis=0)

        cluster_rcsr = rcsr_labels[cluster_mask]
        valid = cluster_rcsr[cluster_rcsr != "None"]
        label_text = str(cluster_id)
        if len(valid) >= rcsr_min_count:
            values, counts = np.unique(valid, return_counts=True)
            best_index = np.argmax(counts)
            if counts[best_index] >= rcsr_min_count and counts[best_index] / cluster_mask.sum() >= rcsr_min_fraction:
                label_text = f"{cluster_id}: {values[best_index]}"

        text = ax.text(
            centroid[0],
            centroid[1],
            label_text,
            fontsize=7,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
            zorder=6,
        )
        texts.append(text)

    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5),
        force_text=(0.8, 0.8),
        force_points=(0.3, 0.3),
        expand=(1.2, 1.4),
    )

    top10_idx = np.argsort(scores)[-10:][::-1]
    ax.scatter(
        data_2d[top10_idx, 0],
        data_2d[top10_idx, 1],
        marker="*",
        s=150,
        c="gold",
        edgecolor="black",
        linewidth=0.8,
        zorder=7,
    )

    special_mask = np.isin(embedding_labels, SPECIAL_SUBLATTICES)
    special_idx = np.where(special_mask)[0]
    if len(special_idx) > 0:
        ax.scatter(
            data_2d[special_idx, 0],
            data_2d[special_idx, 1],
            marker="*",
            s=150,
            c=SPECIAL_SUBLATTICE_COLOR,
            edgecolor="black",
            linewidth=0.8,
            zorder=8,
        )
        if annotate_special_ids:
            for index in special_idx:
                ax.annotate(
                    embedding_labels[index],
                    (data_2d[index, 0], data_2d[index, 1]),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    color=SPECIAL_SUBLATTICE_COLOR,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1),
                    zorder=9,
                )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP Dimension 1", fontsize=14)
    ax.set_ylabel("UMAP Dimension 2", fontsize=14)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Noise", markerfacecolor="#E0E0E0", markersize=8),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label="Top-10 Outliers",
            markerfacecolor="gold",
            markeredgecolor="black",
            markersize=12,
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label="Selected Sublattices",
            markerfacecolor=SPECIAL_SUBLATTICE_COLOR,
            markeredgecolor="black",
            markersize=13,
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, framealpha=0.8, fontsize=11)

    plt.tight_layout()
    plt.savefig(plot_output_folder / filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dependencies()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_output_folder = CLUSTER_OUTPUT_DIR
    plot_output_folder = CLUSTER_PLOT_DIR
    cluster_output_folder.mkdir(parents=True, exist_ok=True)
    plot_output_folder.mkdir(parents=True, exist_ok=True)

    data_for_clustering, labels_for_clustering, rcsr_name_list, _ = load_filtered_embeddings(
        args.fb_list,
        args.dup_list,
    )
    embedding_umap = compute_embedding_umap(data_for_clustering)
    inspect_distance_distribution(embedding_umap)

    _, cluster_labels = full_clustering_workflow(
        cluster_output_folder,
        plot_output_folder,
        embedding_umap,
        labels_for_clustering,
        rcsr_name_list,
    )
    load_clusters_with_labels(cluster_output_folder, "clustered_labels.npy")

    scores, _, _ = calculate_multi_centroid_novelty(embedding_umap, cluster_labels, rcsr_name_list)
    np.save(cluster_output_folder / "novelty_scores.npy", scores)
    write_outliers_file(output_dir, scores, labels_for_clustering)
    plot_novelty_score_distribution(output_dir, scores)

    umap_reducer = umap.UMAP(n_neighbors=20, n_components=2, min_dist=0.5, random_state=42, init="pca")
    data_umap_2d = umap_reducer.fit_transform(embedding_umap)
    plot_publication_umap_candidates(
        plot_output_folder,
        data_umap_2d,
        labels_for_clustering,
        rcsr_name_list,
        scores,
        cluster_labels,
        "UMAP_Discovery_Plot.png",
    )
    plot_publication_umap_candidates(
        plot_output_folder,
        data_umap_2d,
        labels_for_clustering,
        rcsr_name_list,
        scores,
        cluster_labels,
        "UMAP_Discovery_Plot_id.png",
        annotate_special_ids=True,
    )


if __name__ == "__main__":
    main()
