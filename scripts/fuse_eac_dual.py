#!/usr/bin/env python3
"""
fuse_eac_dual.py  (HDBSCAN or KMeans fusion)

Perform Evidence‐Accumulation (EAC) on two existing clusterings (AlephBERT + E5),
then fuse either via HDBSCAN or via KMeans on the co‐association similarity.

Inputs:
  • --aleph_csv  (doc_id, meta_cluster or cluster)
  • --e5_csv     (doc_id, meta_cluster or cluster)

Options:
  • --method {hdbscan,kmeans}   (default: hdbscan)
  • --n_clusters <int>          (only required if method=kmeans)
  • --min_size <int>            (only used if method=hdbscan; default=10)

Outputs:
  • If method=hdbscan → {output} with columns:
        doc_id, fused_cluster, dist_to_medoid
  • If method=kmeans  → {output} with columns:
        doc_id, fused_cluster, sim_to_centroid

Usage examples:
  # 1) HDBSCAN fusion (default):
  python fuse_eac_dual.py \
      --aleph_csv doc_meta_clusters_aleph.csv \
      --e5_csv   doc_meta_clusters_e5.csv \
      --method   hdbscan \
      --min_size 10 \
      --output   fuse_eac_hdb.csv

  # 2) KMeans fusion (specify number of final clusters):
  python fuse_eac_dual.py \
      --aleph_csv  doc_meta_clusters_aleph.csv \
      --e5_csv    doc_meta_clusters_e5.csv \
      --method    kmeans \
      --n_clusters 20 \
      --output    fuse_eac_km.csv
"""

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
import hdbscan


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--aleph_csv",
        type=str,
        required=True,
        help="CSV with columns (doc_id, meta_cluster or cluster) for AlephBERT view",
    )
    p.add_argument(
        "--e5_csv",
        type=str,
        required=True,
        help="CSV with columns (doc_id, meta_cluster or cluster) for E5 view",
    )
    p.add_argument(
        "--method",
        choices=["hdbscan", "kmeans"],
        default="hdbscan",
        help="Fusion method: 'hdbscan' or 'kmeans' (default: hdbscan)",
    )
    p.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Number of clusters for KMeans fusion (required if method=kmeans)",
    )
    p.add_argument(
        "--min_size",
        type=int,
        default=10,
        help="HDBSCAN min_cluster_size (only used if method=hdbscan)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="fuse_eac_final.csv",
        help="Output CSV path",
    )
    return p.parse_args()


def load_labels(path1: str, path2: str):
    """
    Load two CSVs, merging on doc_id. Each CSV must have either:
      • "meta_cluster" or "cluster".
    Rename the found column to "A" and "B" respectively.
    Returns:
      doc_ids : List[str]    (length N)
      labels  : np.ndarray   shape (N, 2) of ints
    """
    def _load_one(path: str, newcol: str):
        df = pd.read_csv(path)
        if "meta_cluster" in df.columns:
            df = df.rename(columns={"meta_cluster": newcol})
        elif "cluster" in df.columns:
            df = df.rename(columns={"cluster": newcol})
        else:
            raise KeyError(f"File {path!r} must contain 'meta_cluster' or 'cluster' column.")
        return df[["doc_id", newcol]]

    df1 = _load_one(path1, "A")
    df2 = _load_one(path2, "B")
    merged = df1.merge(df2, on="doc_id", how="inner")
    merged = merged.sort_values("doc_id").reset_index(drop=True)
    doc_ids = merged["doc_id"].astype(str).tolist()
    labels = merged[["A", "B"]].to_numpy(dtype=int)  # shape (N, 2)
    return doc_ids, labels


def build_coassociation(labels: np.ndarray, ignore_label: int = -1):
    """
    Build a COO sparse matrix of shape (N, N) whose nonzero entries count how many
    partitions (1 or 2) each pair (i,j) appeared together in (i.e., same cluster).
    Skip any docs with label == ignore_label in that partition.
    """
    N = labels.shape[0]
    rows, cols = [], []

    for p in range(labels.shape[1]):  # p=0 (Aleph), p=1 (E5)
        idx_by_cluster = defaultdict(list)
        for i, lab in enumerate(labels[:, p]):
            if lab == ignore_label:
                continue
            idx_by_cluster[lab].append(i)

        for idx_list in idx_by_cluster.values():
            idx_arr = np.array(idx_list, dtype=int)
            k = idx_arr.shape[0]
            if k == 0:
                continue
            ii = np.repeat(idx_arr, k)
            jj = np.tile(idx_arr, k)
            rows.append(ii)
            cols.append(jj)

    if not rows:
        return sp.coo_matrix((N, N), dtype=int)

    all_rows = np.concatenate(rows, axis=0)
    all_cols = np.concatenate(cols, axis=0)
    data = np.ones_like(all_rows, dtype=int)
    co = sp.coo_matrix((data, (all_rows, all_cols)), shape=(N, N), dtype=int)
    return co


def fuse_hdbscan(co: sp.coo_matrix, min_cluster_size: int):
    """
    1) Convert co to CSR, zero diag, sum duplicates.
    2) Normalize co.data /= 2.0 → similarity ∈ {0.0,0.5,1.0}.
    3) distance matrix D = 1.0 - similarity.
    4) Run HDBSCAN(metric="precomputed", min_cluster_size=min_cluster_size).
    5) For each non-noise cluster, find medoid (minimal sum-of-distances) 
       and record each doc’s distance_to_medoid.
    Returns (fused_labels, distance_to_medoid).
    """
    co = co.tocsr()
    co.setdiag(0)
    co.sum_duplicates()

    co.data = co.data.astype(float) / 2.0
    dense_sim = co.toarray().astype(np.float64)
    dense_dist = 1.0 - dense_sim

    clusterer = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=min_cluster_size)
    fused_labels = clusterer.fit_predict(dense_dist)

    N = dense_dist.shape[0]
    dist_to_medoid = np.zeros(N, dtype=float)
    label_to_indices = defaultdict(list)
    for i, lab in enumerate(fused_labels):
        label_to_indices[lab].append(i)

    for lab, indices in label_to_indices.items():
        if lab == -1:
            for i in indices:
                dist_to_medoid[i] = np.nan
            continue
        sub_idx = np.array(indices, dtype=int)
        sub_dist = dense_dist[np.ix_(sub_idx, sub_idx)]
        sum_dist = sub_dist.sum(axis=1)
        medoid_idx = sub_idx[np.argmin(sum_dist)]
        for i in sub_idx:
            dist_to_medoid[i] = dense_dist[i, medoid_idx]

    return fused_labels, dist_to_medoid


def fuse_kmeans(co: sp.coo_matrix, n_clusters: int):
    """
    1) Convert co to CSR, zero diag, sum duplicates.
    2) Normalize co.data /= 2.0 → dense similarity matrix S (N×N).
    3) Treat each row of S as an N-dimensional feature-vector.
    4) Run MiniBatchKMeans(n_clusters) on these row-vectors → fused_labels.
    5) Compute each cluster’s centroid in embedding‐space S, then record each doc’s
       cosine‐distance to its assigned centroid (as sim_to_centroid).
    Returns (fused_labels, sim_to_centroid).
    """
    co = co.tocsr()
    co.setdiag(0)
    co.sum_duplicates()

    co.data = co.data.astype(float) / 2.0  # similarity ∈ {0.0,0.5,1.0}
    dense_sim = co.toarray().astype(np.float64)  # shape (N, N)

    # Run MiniBatchKMeans on rows of dense_sim
    mbk = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
    fused_labels = mbk.fit_predict(dense_sim)

    # Compute cosine distance to the centroid of each doc’s cluster
    # Centroid_k is mbk.cluster_centers_[k], shape (N,).
    # We measure cosine distance as 1 - (x·centroid / (||x||·||centroid||))
    centroids = mbk.cluster_centers_  # shape (n_clusters, N)
    norms_c = np.linalg.norm(centroids, axis=1, keepdims=True)  # shape (n_clusters,1)
    norms_c[norms_c == 0] = 1.0

    # Precompute row norms
    norms_x = np.linalg.norm(dense_sim, axis=1, keepdims=True)  # shape (N,1)
    norms_x[norms_x == 0] = 1.0

    # sim_to_centroid[i] = cosine similarity to chosen centroid
    sim_to_centroid = np.zeros(dense_sim.shape[0], dtype=float)
    for i in range(dense_sim.shape[0]):
        k = fused_labels[i]
        sim = (dense_sim[i : i + 1] @ centroids[k : k + 1].T) / (
            norms_x[i] * norms_c[k]
        )  # shape (1,1)
        sim_to_centroid[i] = float(sim)

    # Convert similarity to “distance”: dist = 1 - sim
    dist_to_centroid = 1.0 - sim_to_centroid
    return fused_labels, dist_to_centroid


def main():
    args = parse_args()

    # 1) Load & merge
    print("▶ Loading and merging clustering CSVs …")
    doc_ids, labels = load_labels(args.aleph_csv, args.e5_csv)
    N = len(doc_ids)
    print(f"  ⇒ {N} documents to fuse.\n")

    # 2) Build co‐association
    print("▶ Building sparse co‐association matrix …")
    co = build_coassociation(labels, ignore_label=-1)
    print(f"  ⇒ {co.count_nonzero()} nonzero co‐occurrence entries\n")

    # 3) Fuse via chosen method
    if args.method == "hdbscan":
        print(f"▶ Fusing with HDBSCAN (min_cluster_size={args.min_size}) …")
        fused_labels, dist_to_medoid = fuse_hdbscan(co, min_cluster_size=args.min_size)
        out_df = pd.DataFrame(
            {"doc_id": doc_ids, "cluster": fused_labels, "dist_to_medoid": dist_to_medoid}
        )
    else:
        if args.n_clusters is None:
            raise ValueError("`--n_clusters` must be specified when method=kmeans.")
        print(f"▶ Fusing with KMeans (n_clusters={args.n_clusters}) …")
        fused_labels, dist_to_centroid = fuse_kmeans(co, n_clusters=args.n_clusters)
        out_df = pd.DataFrame(
            {"doc_id": doc_ids, "cluster": fused_labels, "dist_to_centroid": dist_to_centroid}
        )

    # 4) Write output
    out_df.to_csv(args.output, index=False)
    print(f"\n✅ Wrote fused labels to: {args.output}\n")

    # 5) Print cluster‐size summary
    sizes = pd.Series(fused_labels).value_counts().sort_index()
    print("Cluster sizes (label: count):")
    for lab, cnt in sizes.items():
        print(f"  {lab:>3} : {cnt}")


if __name__ == "__main__":
    main()
