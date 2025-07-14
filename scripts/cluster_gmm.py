#!/usr/bin/env python3
"""
cluster_gmm_specific.py

Run Gaussian Mixture clustering with fixed hyperparameters:
  • n_components = 20
  • covariance_type = 'tied'
  • PCA dimensions = 50

Given a Parquet of pooled document embeddings (`doc_id`, `vector`), this script:
  1. Loads the embeddings
  2. L2-normalizes the vectors
  3. Reduces to 50 dimensions via PCA
  4. Fits a GaussianMixture(n_components=20, covariance_type='tied')
  5. Predicts cluster labels for each document
  6. Writes out a CSV with columns: doc_id, cluster

Usage:
    python cluster_gmm_specific.py \
        --input  doc_vectors_e5.parquet \
        --output gmm_clusters_comp20_covtied_pca50.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def parse_args():
    parser = argparse.ArgumentParser(
        description="GMM clustering with n_components=20, cov_type='tied', pca_dims=50"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input Parquet file with columns ('doc_id', 'vector')"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output CSV file for (doc_id, cluster)"
    )
    return parser.parse_args()


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def main():
    args = parse_args()

    # Load embeddings
    table = pq.read_table(args.input)
    df = table.to_pandas()
    if 'doc_id' not in df.columns or 'vector' not in df.columns:
        raise ValueError("Parquet must contain 'doc_id' and 'vector' columns")

    doc_ids = df['doc_id'].to_numpy()
    vectors = np.vstack(df['vector'].to_numpy())  # shape (N, D)

    # L2-normalize
    vectors = l2_normalize(vectors)

    # PCA to 50 dims
    pca = PCA(n_components=50, random_state=42)
    vectors_reduced = pca.fit_transform(vectors)

    # Fit GMM
    gmm = GaussianMixture(
        n_components=20,
        covariance_type='tied',
        random_state=42
    )
    labels = gmm.fit_predict(vectors_reduced)

    # Save results
    out_df = pd.DataFrame({'doc_id': doc_ids, 'cluster': labels})
    out_df.to_csv(args.output, index=False)

    # Print summary
    counts = out_df['cluster'].value_counts().sort_index()
    print("Cluster label counts:")
    for lbl, cnt in counts.items():
        print(f"  {lbl}: {cnt}")

if __name__ == '__main__':
    main()
