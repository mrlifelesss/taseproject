#!/usr/bin/env python3
"""
Ensemble-vote clustering on document vectors (no legacy deps).

Input : doc_vectors.parquet   (columns: doc_id, vector)
Output: doc_clusters.csv      (doc_id, cluster_label)

Base clusterers:
  • MiniBatch-KMeans      (fast, centroid-based)
  • HDBSCAN               (density, variable size, noise label = −1)
  • Spectral Clustering   (captures non-convex shapes)

Fusion:
  Majority vote across the three labels.
  Ties → pick the smallest positive label (noise/−1 discarded if any POS label).
"""

import argparse, warnings, json, math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from sklearn.cluster import MiniBatchKMeans, SpectralClustering
import hdbscan
from scipy.stats import mode


# ── CLI args ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=Path, default=Path("doc_vectors.parquet"))
    p.add_argument("--kmeans",   type=int, default=25,  help="k for MiniBatch-KMeans")
    p.add_argument("--spectral", type=int, default=25,  help="k for Spectral clustering")
    p.add_argument("--hdb-min",  type=int, default=12,  help="HDBSCAN min_cluster_size")
    p.add_argument("--out",      type=Path, default=Path("doc_clusters.csv"))
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────
def load_vectors(pq_file: Path):
    tbl = pq.read_table(pq_file)
    return np.stack(tbl["vector"].to_numpy()), tbl["doc_id"].to_numpy()


def majority_vote(label_matrix: np.ndarray) -> np.ndarray:
    """
    label_matrix  shape: (n_docs, n_clusterers)
    Rule:
      • Ignore noise (−1) when any positive label exists.
      • Otherwise use the (possibly −1) majority / tie-break lowest label.
    """
    votes = []
    for row in label_matrix:
        pos = row[row >= 0]
        if len(pos):
            # vote among positive labels
            lab, cnt = mode(pos, keepdims=False)
            votes.append(int(lab))
        else:
            # all −1  (noise); vote among them → −1
            votes.append(-1)
    return np.array(votes, dtype=int)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    print(f"▶ loading vectors from {args.file} …")
    vecs, doc_ids = load_vectors(args.file)

    base_labels = []

    print("[KMeans] …")
    km = MiniBatchKMeans(n_clusters=args.kmeans, batch_size=1024, random_state=42)
    base_labels.append(km.fit_predict(vecs))

    print("[HDBSCAN] …")
    hdb = hdbscan.HDBSCAN(min_cluster_size=args.hdb_min,
                          metric="euclidean",
                          prediction_data=False)
    base_labels.append(hdb.fit_predict(vecs))

    print("[Spectral] …")
    spc = SpectralClustering(n_clusters=args.spectral,
                             affinity="nearest_neighbors",
                             assign_labels="kmeans",
                             random_state=42)
    base_labels.append(spc.fit_predict(vecs))

    label_mat = np.vstack(base_labels).T   # shape (n_docs, 3)

    print("▶ majority vote fusion …")
    final = majority_vote(label_mat)

    # write CSV
    pd.DataFrame({"doc_id": doc_ids, "cluster": final}).to_csv(args.out, index=False)
    print("✅ saved clusters →", args.out)

    # mini summary
    sizes = Counter(final)
    print("cluster sizes (label: count) :", dict(sizes))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
