#!/usr/bin/env python3
"""
meta_cluster_docs.py  –  Option A “cluster the cluster-labels”

Input  : doc_vectors.parquet   (doc_id, vector)
Output : doc_meta_clusters.csv (doc_id, meta_cluster)

Steps
-----
1. Run three diverse base clusterers on the document vectors.
2. Treat every doc as a 3-D vector of its three labels.
3. Run a *meta* MiniBatch-KMeans on those label-vectors.

Why this works
--------------
If two algorithms give the same (or close) label to many docs, meta-clustering
groups them even if the *numerical* vector space disagrees, producing a coarse
but robust consensus.

Usage
-----
python meta_cluster_docs.py \
       --file   doc_vectors.parquet \
       --k-base 30 \
       --k-meta 15 \
       --out    doc_meta_clusters.csv
"""

import argparse, warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from sklearn.cluster import MiniBatchKMeans, SpectralClustering
import hdbscan


# ──────────────────────────────────────────────────────────
def load_vectors(path: Path):
    tbl = pq.read_table(path)
    return np.stack(tbl["vector"].to_numpy()), tbl["doc_id"].to_numpy()


def base_clusterings(vecs, k_base, hdb_min):
    """Return list of 3 label arrays: KM, HDB, Spectral."""
    print("[KM     ]")
    km = MiniBatchKMeans(n_clusters=k_base, batch_size=1024, random_state=42)
    labels_km = km.fit_predict(vecs)

    print("[HDBSCAN]")
    hdb = hdbscan.HDBSCAN(min_cluster_size=hdb_min,
                          metric="euclidean", prediction_data=False)
    labels_hdb = hdb.fit_predict(vecs)

    print("[Spectrl]")
    spc = SpectralClustering(n_clusters=k_base, affinity="nearest_neighbors",
                             assign_labels="kmeans", random_state=42)
    labels_spc = spc.fit_predict(vecs)

    return labels_km, labels_hdb, labels_spc


def meta_cluster(label_matrix, k_meta):
    """MiniBatch-KMeans on 3-D label space; noise −1 mapped to big positive."""
    X = label_matrix.copy()
    X[X < 0] = X.max() + 1           # map −1 noise to a separate high value
    km_meta = MiniBatchKMeans(n_clusters=k_meta, random_state=123)
    return km_meta.fit_predict(X)


# ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=Path("doc_vectors.parquet"))
    ap.add_argument("--k-base", type=int, default=25,
                    help="k for KMeans & Spectral in base layer")
    ap.add_argument("--hdb-min", type=int, default=12,
                    help="min_cluster_size for HDBSCAN")
    ap.add_argument("--k-meta", type=int, default=12,
                    help="k for meta MiniBatch-KMeans")
    ap.add_argument("--out", type=Path, default=Path("doc_meta_clusters.csv"))
    args = ap.parse_args()

    print("▶ loading document vectors …")
    vecs, doc_ids = load_vectors(args.file)

    print("▶ base clusterings …")
    km, hdb, spc = base_clusterings(vecs, args.k_base, args.hdb_min)

    label_mat = np.vstack([km, hdb, spc]).T    # shape (n_docs, 3)

    print("▶ meta-clustering those label vectors …")
    meta_labels = meta_cluster(label_mat, args.k_meta)

    pd.DataFrame({"doc_id": doc_ids,
                  "meta_cluster": meta_labels}) \
      .to_csv(args.out, index=False)

    print("✅ results written to", args.out)
    print("cluster sizes:", dict(Counter(meta_labels)))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
