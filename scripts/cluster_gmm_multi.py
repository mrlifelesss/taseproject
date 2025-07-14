#!/usr/bin/env python3
"""
cluster_gmm_multi.py

Run Gaussian Mixture clustering for several component counts (50, 48, 44, 38)
with fixed hyper‑parameters:
  • covariance_type = 'tied'
  • PCA dimensions   = 50

Given a Parquet of pooled document embeddings (`doc_id`, `vector`), this script:
  1. Loads the embeddings
  2. L2‑normalises the vectors
  3. Reduces them to 50 dimensions via PCA (performed **once**)
  4. For each requested ``n_components`` value it:
       a. Fits a GaussianMixture(covariance_type='tied')
       b. Predicts a cluster label for every document
       c. Writes a CSV ``{output_prefix}_comp{N}.csv`` with columns: doc_id, cluster

Usage example
-------------
    python cluster_gmm_multi.py \
        --input  doc_vectors_e5.parquet \
        --output_prefix gmm_clusters_covtied_pca50

The command above will create:
    gmm_clusters_covtied_pca50_comp50.csv
    gmm_clusters_covtied_pca50_comp48.csv
    gmm_clusters_covtied_pca50_comp44.csv
    gmm_clusters_covtied_pca50_comp38.csv

All other script behaviour (normalisation, PCA, etc.) mirrors the previous
``cluster_gmm.py`` implementation.
"""
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


DEFAULT_COMPONENTS: List[int] = [50, 48, 44, 38]


def parse_args():
    parser = argparse.ArgumentParser(
        description="GMM clustering for multiple n_components values"
    )
    parser.add_argument(
        "--input", required=True, type=str,
        help="Input Parquet file with columns ('doc_id', 'vector')"
    )
    parser.add_argument(
        "--output_prefix", required=True, type=str,
        help="Prefix for output CSV files; a suffix _comp<N>.csv will be added"
    )
    parser.add_argument(
        "--components", nargs="+", type=int, default=DEFAULT_COMPONENTS,
        help=f"List of mixture component counts to run (default: {' '.join(map(str, DEFAULT_COMPONENTS))})"
    )
    return parser.parse_args()


def l2_normalise(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    # ------------------------------------------------------------------
    # 1. Load embeddings
    # ------------------------------------------------------------------
    table = pq.read_table(str(input_path))
    df = table.to_pandas()
    if {"doc_id", "vector"} - set(df.columns):
        raise ValueError("Parquet must contain 'doc_id' and 'vector' columns")

    doc_ids = df["doc_id"].to_numpy()
    vectors = np.vstack(df["vector"].to_numpy())  # shape (N, D)

    # ------------------------------------------------------------------
    # 2. L2‑normalise
    # ------------------------------------------------------------------
    vectors = l2_normalise(vectors)

    # ------------------------------------------------------------------
    # 3. PCA (once!)
    # ------------------------------------------------------------------
    print("Fitting PCA (50 dims)…")
    pca = PCA(n_components=50, random_state=42)
    vectors_reduced = pca.fit_transform(vectors)
    print("PCA explained variance ratio (first 10 dims):",
          np.round(pca.explained_variance_ratio_[:10], 4))

    # ------------------------------------------------------------------
    # 4. Run GMM for each n_components value
    # ------------------------------------------------------------------
    for n in args.components:
        print(f"\n=== Fitting GMM with n_components={n} ===")
        gmm = GaussianMixture(n_components=n, covariance_type="tied", random_state=42)
        labels = gmm.fit_predict(vectors_reduced)

        # Save results
        out_df = pd.DataFrame({"doc_id": doc_ids, "cluster": labels})
        out_path = Path(f"{args.output_prefix}_comp{n}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved clusters to {out_path}")

        # Print summary statistics
        counts = out_df["cluster"].value_counts().sort_index()
        print("Cluster label counts:")
        for lbl, cnt in counts.items():
            print(f"  {lbl:>3}: {cnt}")


if __name__ == "__main__":
    main()
