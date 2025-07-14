#!/usr/bin/env python3
"""
fuse_hierarchical_dual.py

Fuse two existing clusterings via hierarchical clustering on the co-association distance.

Inputs:
  --aleph_csv    path/to/doc_meta_clusters_aleph.csv  (columns: doc_id, meta_cluster or cluster)
  --e5_csv       path/to/doc_meta_clusters_e5.csv     (columns: doc_id, meta_cluster or cluster)

Options:
  --n_clusters   int     (final number of clusters; required)
  --distance_threshold float (alternative: cut dendrogram at this distance; optional)
  --output       path/to/fused.csv   (default: fused_hier.csv)

Outputs:
  fused.csv containing columns:
    doc_id, fused_cluster
"""

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import AgglomerativeClustering


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--aleph_csv", type=str, required=True,
        help="CSV with columns doc_id, meta_cluster (or cluster) for AlephBERT view"
    )
    p.add_argument(
        "--e5_csv", type=str, required=True,
        help="CSV with columns doc_id, meta_cluster (or cluster) for E5 view"
    )
    p.add_argument(
        "--n_clusters", type=int, default=None,
        help="Number of final clusters (if None, use distance_threshold instead)"
    )
    p.add_argument(
        "--distance_threshold", type=float, default=None,
        help="Distance threshold to cut dendrogram (if None, must specify n_clusters)"
    )
    p.add_argument(
        "--output", type=str, default="fused_hier.csv",
        help="Output CSV path"
    )
    return p.parse_args()


def load_labels(path1: str, path2: str):
    """
    Load two CSVs, merging on doc_id. Each file must have either
    'meta_cluster' or 'cluster' as the label column.
    Returns:
      doc_ids : list[str]     (length N)
      labels  : np.ndarray    shape (N, 2)
    """
    def _load_one(path: str, newcol: str):
        df = pd.read_csv(path)
        if "meta_cluster" in df.columns:
            df = df.rename(columns={"meta_cluster": newcol})
        elif "cluster" in df.columns:
            df = df.rename(columns={"cluster": newcol})
        else:
            raise KeyError(f"{path!r} must have 'meta_cluster' or 'cluster' column.")
        return df[["doc_id", newcol]]

    df1 = _load_one(path1, "A")
    df2 = _load_one(path2, "B")
    merged = df1.merge(df2, on="doc_id", how="inner")
    merged = merged.sort_values("doc_id").reset_index(drop=True)

    doc_ids = merged["doc_id"].astype(str).tolist()
    labels = merged[["A", "B"]].to_numpy(dtype=int)
    return doc_ids, labels


def build_coassociation(labels: np.ndarray, ignore_label: int = -1):
    """
    Build sparse co-association counts in a COO matrix of shape (N,N).
    co[i,j] = number of partitions (0..2) in which i and j shared a non-ignored label.
    """
    N = labels.shape[0]
    rows, cols = [], []

    for p in range(labels.shape[1]):  # p=0,1
        idx_by_cluster = defaultdict(list)
        for i, lab in enumerate(labels[:, p]):
            if lab == ignore_label:
                continue
            idx_by_cluster[lab].append(i)

        for idx_list in idx_by_cluster.values():
            arr = np.array(idx_list, dtype=int)
            k = arr.shape[0]
            if k == 0:
                continue
            ii = np.repeat(arr, k)
            jj = np.tile(arr, k)
            rows.append(ii)
            cols.append(jj)

    if not rows:
        return sp.coo_matrix((N, N), dtype=int)

    all_rows = np.concatenate(rows, axis=0)
    all_cols = np.concatenate(cols, axis=0)
    data = np.ones_like(all_rows, dtype=int)

    return sp.coo_matrix((data, (all_rows, all_cols)), shape=(N, N), dtype=int)


def main():
    args = parse_args()

    # 1) Load and merge
    doc_ids, labels = load_labels(args.aleph_csv, args.e5_csv)
    N = len(doc_ids)
    print(f"Loaded {N} documents.")

    # 2) Build co-association
    co = build_coassociation(labels, ignore_label=-1)
    co = co.tocsr()
    co.setdiag(0)
    co.sum_duplicates()

    # 3) Normalize to similarity {0.0,0.5,1.0}, then convert to distance
    co.data = co.data.astype(float) / 2.0
    dense_sim = co.toarray().astype(np.float64)
    dense_dist = 1.0 - dense_sim

    # 4) Hierarchical clustering on precomputed distances
    if args.distance_threshold is None and args.n_clusters is None:
        raise ValueError("Specify either --n_clusters or --distance_threshold.")

    hc = AgglomerativeClustering(
        linkage="average",
        **(
            {"n_clusters": args.n_clusters}
            if args.n_clusters is not None
            else {"distance_threshold": args.distance_threshold, "n_clusters": None}
        )
    )
    fused_labels = hc.fit_predict(dense_dist)

    # 5) Output CSV
    out_df = pd.DataFrame({"doc_id": doc_ids, "cluster": fused_labels})
    out_df.to_csv(args.output, index=False)
    print(f"Written fused labels to {args.output}")

    # 6) Print cluster-size summary
    from collections import Counter
    sizes = Counter(fused_labels)
    print("Cluster sizes (label: count):")
    for lab, cnt in sorted(sizes.items()):
        print(f"  {lab:>3} : {cnt}")


if __name__ == "__main__":
    main()
