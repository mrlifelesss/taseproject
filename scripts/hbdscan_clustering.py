#!/usr/bin/env python3
"""
hdbscan_on_parquets.py

Load one or more “doc_vectors_*.parquet” files, L2‐normalize the embeddings,
optionally PCA‐reduce, then run HDBSCAN to produce (doc_id, cluster) CSVs.

Usage example:
  python hdbscan_on_parquets.py \
      --inputs  doc_vectors_e5.parquet doc_vectors_BAAI.parquet \
      --output_prefix  hdb_clusters_  \
      --min_size 15 \
      --min_samples 5 \
      --cluster_eps 0.02 \
      --pca_dims 100

What it does for each input Parquet:
  1) Read columns (“doc_id”, “vector”) into a Pandas DataFrame.
  2) Stack “vector” into an (N × D) NumPy array.
  3) L2-normalize every row so that Euclidean ≈ cosine.
  4) (If requested) PCA-reduce from D → `--pca_dims`.
  5) Run HDBSCAN(min_cluster_size=`--min_size`, min_samples=`--min_samples`,
      cluster_selection_epsilon=`--cluster_eps`, metric="euclidean").
  6) Write out CSV:  {output_prefix}{basename(input)}.csv, containing two columns:
     – doc_id  
     – cluster  (integer label, −1 = noise)

Example output files (assuming `--output_prefix hdb_clusters_`):
  • hdb_clusters_doc_vectors_e5.csv  
  • hdb_clusters_doc_vectors_BAAI.csv  
"""

import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import hdbscan
from sklearn.decomposition import PCA
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(
        description="Normalize + PCA + HDBSCAN on one or more doc_vectors_*.parquet files."
    )
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more Parquet files containing columns ['doc_id','vector'].",
    )
    p.add_argument(
        "--output_prefix",
        type=str,
        default="hdb_clusters_",
        help="Prefix for the CSV output files. Final name = <prefix><input_basename>.csv",
    )
    p.add_argument(
        "--min_size",
        type=int,
        default=15,
        help="HDBSCAN `min_cluster_size` (default: 15).",
    )
    p.add_argument(
        "--min_samples",
        type=int,
        default=5,
        help="HDBSCAN `min_samples` (default: 5). If not set, it would default to `min_size`.",
    )
    p.add_argument(
        "--cluster_eps",
        type=float,
        default=0.02,
        help="HDBSCAN `cluster_selection_epsilon` (default: 0.02).",
    )
    p.add_argument(
        "--pca_dims",
        type=int,
        default=100,
        help="PCA target dimensions before clustering (set to 0 to skip PCA). Default = 100.",
    )
    return p.parse_args()


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row‐wise L2 normalization: ensure each vector has unit norm."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def process_parquet(
    parquet_path: str,
    output_csv: str,
    min_size: int,
    min_samples: int,
    cluster_eps: float,
    pca_dims: int,
):
    """
    1) Load doc_id + vector from `parquet_path`.
    2) Stack vectors → (N, D) array.
    3) L2‐normalize each row.
    4) (Optional) PCA → (N, pca_dims).
    5) HDBSCAN.
    6) Write out (doc_id, cluster) to `output_csv`.
    """

    print(f"\n▶ Processing `{parquet_path}` …")

    # --- 1) load
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    if "doc_id" not in df.columns or "vector" not in df.columns:
        raise ValueError(
            f"Parquet `{parquet_path}` must contain columns 'doc_id' and 'vector'."
        )

    doc_ids = df["doc_id"].to_numpy()
    vectors = np.vstack(df["vector"].to_numpy())  # shape = (N, D)
    N, D = vectors.shape
    print(f"  • Loaded {N} documents, embedding dimension = {D}")

    # --- 2) L2‐normalize (so Euclidean ≈ cosine)
    print("  • L2‐normalizing embeddings …")
    vectors = l2_normalize(vectors)

    # --- 3) PCA reduction if requested
    if pca_dims > 0 and pca_dims < D:
        print(f"  • Applying PCA: {D}→{pca_dims} dims …")
        pca = PCA(n_components=pca_dims, random_state=42)
        vectors = pca.fit_transform(vectors)
        print(f"    → New shape = {vectors.shape}")
    else:
        print("  • Skipping PCA (either pca_dims=0 or ≥ original dim).")

    # --- 4) HDBSCAN clustering
    print(
        f"  • Running HDBSCAN( min_cluster_size={min_size}, "
        f"min_samples={min_samples}, eps={cluster_eps}, metric='euclidean' ) …"
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_epsilon=cluster_eps,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(vectors)

    # --- 5) Save CSV of (doc_id, cluster)
    out_df = pd.DataFrame({"doc_id": doc_ids, "cluster": labels})
    out_df.to_csv(output_csv, index=False)
    print(f"  → Wrote: {output_csv}")

    # --- 6) Print a brief summary of label counts
    counts = out_df["cluster"].value_counts().sort_index()
    print("  • Cluster label counts (label : size):")
    for lab, cnt in counts.items():
        print(f"    {lab:>3} : {cnt}")
    n_noise = (labels == -1).sum()
    print(f"  • Noise points (-1): {n_noise} ({n_noise/len(labels):.0%} of all docs)")


def main():
    args = parse_args()

    for input_path in args.inputs:
        # Derive output CSV name:
        base = Path(input_path).stem  # e.g. "doc_vectors_e5"
        out_csv = args.output_prefix + base + ".csv"
        process_parquet(
            parquet_path=input_path,
            output_csv=out_csv,
            min_size=args.min_size,
            min_samples=args.min_samples,
            cluster_eps=args.cluster_eps,
            pca_dims=args.pca_dims,
        )


if __name__ == "__main__":
    main()
