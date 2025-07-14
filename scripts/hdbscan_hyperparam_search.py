# hdbscan_hyperparam_search.py
"""
Automated grid search over HDBSCAN hyperparameters for one or more embedding files.
For each combination of (min_cluster_size, min_samples, cluster_selection_epsilon, pca_dims),
run HDBSCAN, compute
  - #clusters (excluding noise)
  - #noise points
  - silhouette (cosine) on L2-normalized vectors after PCA (if used)
  - Davies-Bouldin Index (Euclidean) on PCA-reduced or L2-normalized vectors
Results are collected into a single CSV/Excel report.

Usage example:
    python hdbscan_hyperparam_search.py \
        --inputs doc_vectors_e5.parquet doc_vectors_BAAI.parquet \
        --out report_hdbscan_search.xlsx

Options:
  --inputs: One or more Parquet files with columns 'doc_id', 'vector'.
  --out: Path to write the final Excel report.
  --min_sizes: List of min_cluster_size values to try (default: 5,10,15,20).
  --min_samples: List of min_samples values to try (default: 5, 10).
  --eps_vals: List of cluster_selection_epsilon values (default: 0.01,0.02,0.05).
  --pca_dims: List of PCA dims to try (default: 0, 50, 100).

Requires: numpy, pandas, pyarrow, scikit-learn, hdbscan, openpyxl
"""
import argparse
import itertools
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import hdbscan
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from pathlib import Path


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def load_vectors(parquet_path: str):
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    if "doc_id" not in df.columns or "vector" not in df.columns:
        raise ValueError(f"Parquet {parquet_path} must have columns 'doc_id' and 'vector'.")
    vectors = np.vstack(df["vector"].to_numpy())
    doc_ids = df["doc_id"].to_numpy()
    return doc_ids, vectors


def compute_metrics(vectors: np.ndarray, labels: np.ndarray):
    # Exclude noise for silhouette/DBI calculations
    mask = labels != -1
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        # Not enough clusters or points
        sil = np.nan
        dbi = np.nan
    else:
        sil = silhouette_score(vectors[mask], labels[mask], metric="cosine")
        dbi = davies_bouldin_score(vectors[mask], labels[mask])
    n_clusters = len([l for l in set(labels) if l != -1])
    n_noise = (labels == -1).sum()
    return silhouette_score, davies_bouldin_score, n_clusters, n_noise


def main():
    parser = argparse.ArgumentParser(description="Grid search over HDBSCAN hyperparameters.")
    parser.add_argument(
        "--inputs", nargs='+', required=True,
        help="One or more doc_vectors_*.parquet files to cluster"
    )
    parser.add_argument(
        "--out", type=str, default="hdbscan_search_report.xlsx",
        help="Path to write Excel report"
    )
    parser.add_argument(
        "--min_sizes", nargs='+', type=int, default=[5,10,15,20],
        help="List of min_cluster_size values to test"
    )
    parser.add_argument(
        "--min_samples", nargs='+', type=int, default=[5,10],
        help="List of min_samples values to test"
    )
    parser.add_argument(
        "--eps_vals", nargs='+', type=float, default=[0.01, 0.02, 0.05],
        help="List of cluster_selection_epsilon values to test"
    )
    parser.add_argument(
        "--pca_dims", nargs='+', type=int, default=[0,50,100],
        help="List of PCA dims to test (0 = no PCA)"
    )
    args = parser.parse_args()

    records = []

    for input_path in args.inputs:
        base = Path(input_path).stem
        print(f"\n=== Processing embeddings from: {base} ===")
        doc_ids, original_vectors = load_vectors(input_path)

        # Always L2-normalize original before optional PCA
        normed = l2_normalize(original_vectors)

        for pca_dim, min_size, min_sample, eps in itertools.product(
            args.pca_dims, args.min_sizes, args.min_samples, args.eps_vals
        ):
            print(f"Testing pca={pca_dim}, min_size={min_size}, min_samples={min_sample}, eps={eps} ...", end=' ')

            # 1) maybe PCA
            if pca_dim > 0 and pca_dim < normed.shape[1]:
                pca = PCA(n_components=pca_dim, random_state=42)
                vecs = pca.fit_transform(normed)
            else:
                vecs = normed

            # 2) HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_size,
                min_samples=min_sample,
                metric="euclidean",
                cluster_selection_epsilon=eps,
                core_dist_n_jobs=-1,
            )
            labels = clusterer.fit_predict(vecs)

            # 3) compute metrics
            # silhouette & DBI on the space we clustered
            mask = labels != -1
            if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
                sil, dbi = np.nan, np.nan
            else:
                sil = silhouette_score(vecs[mask], labels[mask], metric="cosine")
                dbi = davies_bouldin_score(vecs[mask], labels[mask])
            n_clusters = len([l for l in set(labels) if l != -1])
            n_noise = int((labels == -1).sum())

            records.append({
                "embedding": base,
                "pca_dim": pca_dim,
                "min_cluster_size": min_size,
                "min_samples": min_sample,
                "eps": eps,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "silhouette": sil,
                "davies_bouldin": dbi,
            })
            print(f"n_cl={n_clusters}, noise={n_noise}")

    df = pd.DataFrame.from_records(records)
    # Sort rows so that best silhouette appears on top per embedding
    df = df.sort_values(["embedding", "silhouette"], ascending=[True, False])
    print(f"\nSaving full report to {args.out} …")
    # Write to Excel (one sheet) or CSV if you prefer
    df.to_excel(args.out, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
