#!/usr/bin/env python3
"""
evaluate_json_clustering.py

Compute silhouette and Davies–Bouldin scores for a clustering defined in a JSON file,
ignoring specified categories and excluding clusters below a minimum size.
"""
import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import silhouette_score, davies_bouldin_score


def load_embeddings(parquet_path: Path):
    table = pq.read_table(parquet_path)
    # Convert doc_id values to int for consistency
    doc_ids = table["doc_id"].to_pandas().astype(int).tolist()
    vectors = np.stack(table["vector"].to_numpy())
    return doc_ids, vectors


def load_cluster_mapping(json_path: Path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    mapping = {}
    # final categories
    for cat, info in data.get("final_categories", {}).items():
        for doc_id in info.get("document_ids", []):
            mapping[int(doc_id)] = cat
    # provisional groups
    for cat, info in data.get("provisional_candidate_groups", {}).items():
        for doc_id in info.get("document_ids", []):
            mapping[int(doc_id)] = cat
    return mapping


def filter_docs(doc_ids, vectors, mapping, ignore):
    labels = []
    vecs = []
    for doc_id, vec in zip(doc_ids, vectors):
        cat = mapping.get(doc_id)
        # skip if unmapped or in ignored categories
        if cat is None or cat in ignore:
            continue
        labels.append(cat)
        vecs.append(vec)
    if not labels:
        print("No documents to evaluate after initial filtering.", file=sys.stderr)
        sys.exit(1)
    return labels, np.stack(vecs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate clustering from JSON assignments.")
    parser.add_argument("--emb", type=Path, required=True,
                        help="Path to doc_vectors.parquet")
    parser.add_argument("--json", type=Path, required=True,
                        help="Path to clustering JSON file")
    parser.add_argument("--ignore", nargs="+",
                        default=["UNCLASSIFIED_NOISE", "MISSING_ID_METADATA"],
                        help="Category names to ignore in the evaluation")
    parser.add_argument("--min-cluster-size", type=int, default=10,
                        help="Minimum number of documents per cluster to include")
    args = parser.parse_args()

    # Load data
    doc_ids, vectors = load_embeddings(args.emb)
    mapping = load_cluster_mapping(args.json)

    # Initial filter by ignore categories
    labels, X = filter_docs(doc_ids, vectors, mapping, set(args.ignore))

    # Exclude clusters smaller than min_cluster_size
    counts = Counter(labels)
    valid_clusters = {cat for cat, cnt in counts.items() if cnt >= args.min_cluster_size}
    if not valid_clusters:
        print(f"No clusters with size >= {args.min_cluster_size}.", file=sys.stderr)
        sys.exit(1)

    filtered_labels = []
    filtered_vecs = []
    for lab, vec in zip(labels, X):
        if lab in valid_clusters:
            filtered_labels.append(lab)
            filtered_vecs.append(vec)
    if not filtered_labels:
        print("No documents left after excluding small clusters.", file=sys.stderr)
        sys.exit(1)
    labels = filtered_labels
    X = np.stack(filtered_vecs)

    # Encode string labels to integers
    unique_labels = sorted(valid_clusters)
    label_to_int = {lab: idx for idx, lab in enumerate(unique_labels)}
    y = np.array([label_to_int[lab] for lab in labels])

    # Compute metrics
    sil = silhouette_score(X, y, metric="cosine")
    dbi = davies_bouldin_score(X, y)

    print(f"Documents evaluated: {len(labels)} (excluded categories: {args.ignore})")
    print(f"Clusters evaluated ({len(unique_labels)}): {unique_labels}")
    print(f"Silhouette score (cosine distance): {sil:.4f}")
    print(f"Davies–Bouldin index (Euclidean): {dbi:.4f}")


if __name__ == "__main__":
    main()
