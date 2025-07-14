#!/usr/bin/env python3
"""
meta_cluster_hyperparam_search.py

Grid‐search over base‐ and meta‐clustering hyperparameters for `meta_cluster_docs.py`.

Given one or more pooled document‐embedding Parquet files (with columns `doc_id` and
`vector`), this script will:

  1. Load the embedding matrix X (shape N×D).
  2. For each combination of {k_base, hdb_min, k_meta}:
     a. Run three base clusterers on X:
        - MiniBatchKMeans(n_clusters=k_base)
        - HDBSCAN(min_cluster_size=hdb_min)
        - SpectralClustering(n_clusters=k_base)
     b. Stack their label vectors → (N×3) matrix
     c. Perform meta‐clustering: MiniBatchKMeans(n_clusters=k_meta) on that N×3 matrix
     d. Evaluate the meta labels on X via:
        • silhouette_score(X, meta_labels, metric="cosine")
        • davies_bouldin_score(X, meta_labels)
     e. Record: input_file, k_base, hdb_min, k_meta, n_meta_clusters,
        silhouette_cosine, davies_bouldin

At the end, writes:
  • <output_prefix>report.csv
  • <output_prefix>report.xlsx

Usage:
  python meta_cluster_hyperparam_search.py \
    --inputs doc_vectors_e5.parquet doc_vectors_BAAI.parquet \
    --k_base_list 10 20 30 \
    --hdb_min_list 5 12 20 \
    --k_meta_list 5 10 15 20 \
    --output_prefix meta_search_
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import hdbscan


def parse_args():
    p = argparse.ArgumentParser(
        description="Grid search hyperparams for meta‐clustering pipeline"
    )
    p.add_argument(
        "--inputs", nargs='+', required=True,
        help="One or more doc_vectors.parquet files (columns: doc_id, vector)"
    )
    p.add_argument(
        "--k_base_list", nargs='+', type=int, default=[10,20,30],
        help="List of k_base values for base KMeans/Spectral (default: 10 20 30)"
    )
    p.add_argument(
        "--hdb_min_list", nargs='+', type=int, default=[5,12,20],
        help="List of HDBSCAN min_cluster_size values (default: 5 12 20)"
    )
    p.add_argument(
        "--k_meta_list", nargs='+', type=int, default=[5,10,15,20],
        help="List of k_meta values for meta‐KMeans (default: 5 10 15 20)"
    )
    p.add_argument(
        "--output_prefix", type=str, default="meta_search_",
        help="Prefix for output report files."
    )
    return p.parse_args()


def load_embeddings(parquet_path):
    tbl = pq.read_table(parquet_path)
    df = tbl.to_pandas()
    if 'doc_id' not in df.columns or 'vector' not in df.columns:
        raise ValueError(f"{parquet_path} must contain 'doc_id' and 'vector' columns")
    X = np.vstack(df['vector'].to_numpy())
    doc_ids = df['doc_id'].to_numpy()
    return doc_ids, X


def run_base_clusterings(X, k_base, hdb_min):
    # 1) KMeans
    km = MiniBatchKMeans(n_clusters=k_base, batch_size=1024, random_state=42)
    labels_km = km.fit_predict(X)
    # 2) HDBSCAN
    hdb = hdbscan.HDBSCAN(min_cluster_size=hdb_min,
                          metric='euclidean',core_dist_n_jobs=-1)
    labels_hdb = hdb.fit_predict(X)
    # 3) Spectral
    spc = SpectralClustering(n_clusters=k_base,
                             affinity='nearest_neighbors',
                             assign_labels='kmeans', random_state=42)
    labels_spc = spc.fit_predict(X)
    return labels_km, labels_hdb, labels_spc


def run_meta_clustering(label_mat, k_meta):
    # Map noise (-1) to a new label = max+1
    L = label_mat.copy()
    L[L < 0] = L.max() + 1
    mb = MiniBatchKMeans(n_clusters=k_meta, random_state=123)
    return mb.fit_predict(L)


def main():
    args = parse_args()
    results = []

    for parquet_path in args.inputs:
        input_stem = Path(parquet_path).stem
        print(f"\n==== Processing {input_stem} ====")
        doc_ids, X = load_embeddings(parquet_path)
        N, D = X.shape
        print(f"Loaded {N} docs, dim={D}")

        # Grid search
        for k_base, hdb_min, k_meta in itertools.product(
            args.k_base_list, args.hdb_min_list, args.k_meta_list
        ):
            combo = f"kb{k_base}_hdb{hdb_min}_km{str(k_meta)}"
            print(f"Combo {combo}…", end=' ')

            # Base layer
            lb_km, lb_hdb, lb_spc = run_base_clusterings(X, k_base, hdb_min)
            label_mat = np.vstack([lb_km, lb_hdb, lb_spc]).T  # (N,3)

            # Meta layer
            meta_labels = run_meta_clustering(label_mat, k_meta)
            # Metrics:
            # #meta clusters
            n_meta = len(set(meta_labels.tolist()))
            # silhouette (cosine)
            try:
                sil = silhouette_score(X, meta_labels, metric='cosine')
            except Exception:
                sil = np.nan
            # DBI
            try:
                dbi = davies_bouldin_score(X, meta_labels)
            except Exception:
                dbi = np.nan

            results.append({
                'input': input_stem,
                'k_base': k_base,
                'hdb_min': hdb_min,
                'k_meta': k_meta,
                'n_meta_clusters': n_meta,
                'silhouette_cosine': sil,
                'davies_bouldin': dbi,
            })
            print(f"n={n_meta}, sil={sil:.3f}, dbi={dbi:.3f}")

    # Save report
    df = pd.DataFrame(results)
    csv_out = args.output_prefix + 'report.csv'
    xlsx_out = args.output_prefix + 'report.xlsx'
    df.to_csv(csv_out, index=False)
    df.to_excel(xlsx_out, index=False, sheet_name='meta_search')
    print(f"\n✅ Saved report to {csv_out} and {xlsx_out}")


if __name__ == '__main__':
    main()
