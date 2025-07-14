#!/usr/bin/env python3
"""
gmm_hyperparam_search.py

Grid‑search over Gaussian Mixture Model clustering hyperparameters.

Given one or more pooled document‑embedding Parquet files
(with columns `doc_id` and `vector`), this script will:

  1. Load the embedding matrix X (shape N×D) from each Parquet.
  2. L2‑normalized the vectors so Euclidean ≈ cosine.
  3. Optionally PCA‑reduce to `pca_dims` dimensions.
  4. For each combination of {
       n_components (number of mixture components),
       covariance_type ("full","tied","diag","spherical"),
       pca_dims
     }:
     a. Fit a GaussianMixture model on the processed vectors.
     b. Predict hard labels for all points.
     c. Compute metrics:
        • n_clusters (unique labels)
        • silhouette_score (cosine)
        • davies_bouldin_score (Euclidean)
     d. Append a row of results.
  5. Write a combined Excel & CSV summary report.

Usage example:
  python gmm_hyperparam_search.py \
    --inputs doc_vectors_e5.parquet doc_vectors_BAAI.parquet \
    --n_components 5 10 15 20 \
    --cov_types full tied diag spherical \
    --pca_dims 0 50 100 \
    --output_prefix gmm_search_
"""

import argparse
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture


def parse_args():
    p = argparse.ArgumentParser(
        description="Grid search over GaussianMixture hyperparameters."
    )
    p.add_argument(
        '--inputs', nargs='+', required=True,
        help='One or more Parquet files with columns [doc_id, vector].'
    )
    p.add_argument(
        '--n_components', nargs='+', type=int, default=list(range(30,39,1)),
        help='List of mixture component counts to try (default: 5 10 15 20).'
    )
    p.add_argument(
        '--cov_types', nargs='+', type=str,
        default=['full','tied'],
        help='Covariance types for GaussianMixture (default: full tied diag spherical).'
    )
    p.add_argument(
        '--pca_dims', nargs='+', type=int, default=list(range(30,65,5)),
        help='PCA dims before clustering (0 to skip PCA).'
    )
    p.add_argument(
        '--output_prefix', type=str, default='gmm_search_2',
        help='Prefix for output CSV/XLSX report files.'
    )
    return p.parse_args()


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def load_vectors(parquet_path: str):
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    if 'doc_id' not in df.columns or 'vector' not in df.columns:
        raise ValueError(f"Parquet {parquet_path} must contain 'doc_id' and 'vector'.")
    doc_ids = df['doc_id'].to_numpy()
    vectors = np.vstack(df['vector'].to_numpy())
    return doc_ids, vectors


def main():
    args = parse_args()
    results = []

    for input_path in args.inputs:
        stem = Path(input_path).stem
        print(f"\n▶ Processing {stem}...")
        doc_ids, raw_vectors = load_vectors(input_path)
        N, D = raw_vectors.shape
        print(f"  • Loaded {N} docs, dim = {D}")

        # L2-normalize
        vectors_norm = l2_normalize(raw_vectors)

        # grid search
        combos = list(itertools.product(
            args.n_components,
            args.cov_types,
            args.pca_dims
        ))
        total = len(combos)

        for idx, (n_comp, cov, pca_dim) in enumerate(combos, 1):
            print(f"  [{idx}/{total}] n_comp={n_comp}, cov={cov}, pca={pca_dim}...", end=' ')

            # PCA
            if pca_dim > 0 and pca_dim < D:
                pca = PCA(n_components=pca_dim, random_state=42)
                X = pca.fit_transform(vectors_norm)
            else:
                X = vectors_norm

            # fit GMM
            try:
                gmm = GaussianMixture(
                    n_components=n_comp,
                    covariance_type=cov,
                    random_state=42
                )
                labels = gmm.fit_predict(X)
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            # metrics
            unique = np.unique(labels)
            n_clusters = len(unique)
            try:
                sil = silhouette_score(vectors_norm, labels, metric='cosine')
            except Exception:
                sil = np.nan
            try:
                dbi = davies_bouldin_score(X, labels)
            except Exception:
                dbi = np.nan

            combo = f"comp{n_comp}_cov{cov}_pca{pca_dim}"
            results.append({
                'input': stem,
                'combo': combo,
                'n_clusters': n_clusters,
                'silhouette': sil,
                'davies_bouldin': dbi
            })
            print(f"clusters={n_clusters}, sil={sil:.3f}, dbi={dbi:.3f}")

    # compile results
    df_res = pd.DataFrame(results)
    csv_out = args.output_prefix + 'report.csv'
    xlsx_out = args.output_prefix + 'report.xlsx'
    df_res.to_csv(csv_out, index=False)
    df_res.to_excel(xlsx_out, index=False)
    print(f"\n✅ Saved CSV: {csv_out}\n✅ Saved Excel: {xlsx_out}")


if __name__ == '__main__':
    main()
