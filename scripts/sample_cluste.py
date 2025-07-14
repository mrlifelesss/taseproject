#!/usr/bin/env python3
"""
sample_cluster_examples.py

For each cluster in your clustering, sample two sets of K examples from 3 regions:
  • center (closest to centroid)
  • mid-range (around median distance)
  • edge (furthest from centroid)

Each set consists of K examples per region (default K=4) for each seed, and is assigned a seed.
The script outputs a CSV where each row is:
    seed, cluster_label, region, doc_id

Usage:
  python sample_cluster_examples.py \
      --doc_vectors   doc_vectors_e5.parquet \
      --cluster_csv   doc_clusters.csv \
      --cluster_col   cluster \
      --doc_id_col    doc_id \
      --seeds         0 1 \
      --samples_per_region 4 \
      --output        sampled_examples.csv

Output:
  CSV with columns: seed,cluster_label,region,doc_id
"""
import argparse
import csv
import random
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def parse_args():
    p = argparse.ArgumentParser(
        description="Sample docs per cluster region with identification."
    )
    p.add_argument(
        "--doc_vectors", type=str, required=True,
        help="Path to Parquet with columns [doc_id, vector]."
    )
    p.add_argument(
        "--cluster_csv", type=str, required=True,
        help="CSV with columns [doc_id, cluster]."
    )
    p.add_argument(
        "--cluster_col", type=str, default="cluster",
        help="Name of cluster column in cluster_csv."
    )
    p.add_argument(
        "--doc_id_col", type=str, default="doc_id",
        help="Name of doc_id column in both files."
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[0,1],
        help="Random seeds to generate different sets (default 0 1)."
    )
    p.add_argument(
        "--samples_per_region", type=int, default=4,
        help="Number of examples per region per seed (default=4)."
    )
    p.add_argument(
        "--output", type=str, default="sampled_examples.csv",
        help="Output CSV path."
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 1) Load embeddings
    table = pq.read_table(args.doc_vectors)
    df_vec = table.to_pandas()
    if args.doc_id_col not in df_vec.columns or 'vector' not in df_vec.columns:
        print(f"ERROR: doc_vectors must contain '{args.doc_id_col}' and 'vector'", file=sys.stderr)
        sys.exit(1)

    # Build id2vec with string keys
    id2vec = {}
    for row in df_vec.itertuples(index=False):
        key = str(getattr(row, args.doc_id_col))
        vec = getattr(row, 'vector')
        id2vec[key] = np.array(vec)

    # 2) Load cluster assignments
    df_clust = pd.read_csv(args.cluster_csv)
    if args.doc_id_col not in df_clust.columns or args.cluster_col not in df_clust.columns:
        print(f"ERROR: cluster_csv must contain '{args.doc_id_col}' and '{args.cluster_col}'", file=sys.stderr)
        sys.exit(1)

    # 3) Open output CSV
    with open(args.output, 'w', newline='', encoding='utf8') as fout:
        writer = csv.writer(fout)
        writer.writerow(['seed','cluster_label','region','doc_id'])

        # 4) For each seed, each cluster, sample per region
        for seed in args.seeds:
            random.seed(seed)
            for cluster_label, grp in df_clust.groupby(args.cluster_col):
                # convert doc_ids to strings for matching
                doc_ids = [str(d) for d in grp[args.doc_id_col].tolist()]

                # Filter out missing vectors
                available = [doc for doc in doc_ids if doc in id2vec]
                missing = set(doc_ids) - set(available)
                if missing:
                    print(f"Warning: {len(missing)} docs in cluster {cluster_label} missing vectors, skipping them.", file=sys.stderr)
                if not available:
                    continue

                vectors = np.vstack([id2vec[doc] for doc in available])

                # Compute centroid and distances
                centroid = vectors.mean(axis=0, keepdims=True)
                dists = np.linalg.norm(vectors - centroid, axis=1)
                pairs = list(zip(available, dists))
                pairs.sort(key=lambda x: x[1])

                N = len(pairs)
                q1 = N // 4
                q3 = (3 * N) // 4
                regions = {
                    'center': [doc for doc,_ in pairs[:q1]],
                    'mid':    [doc for doc,_ in pairs[q1:q3]],
                    'edge':   [doc for doc,_ in pairs[q3:]],
                }

                # Sample per region
                K = args.samples_per_region
                for region, cands in regions.items():
                    if not cands:
                        continue
                    samples = cands if len(cands) <= K else random.sample(cands, K)
                    for doc in samples:
                        writer.writerow([seed, cluster_label, region, doc])

    print(f"✅ Wrote sampled examples to {args.output}")


if __name__ == '__main__':
    main()
