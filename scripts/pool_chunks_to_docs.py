#!/usr/bin/env python3
"""
pool_chunks_to_docs.py
──────────────────────
Collapse chunk-level embeddings → one vector per document.

Example
-------
python pool_chunks_to_docs.py \
       --src embeddings \
       --dst doc_vectors.parquet \
       --method mean        # mean | tfidf | medoid
"""

import argparse, json, math, glob, os
from pathlib import Path
from collections import defaultdict

import pyarrow.parquet as pq
import numpy as np
from tqdm import tqdm

# ─── optional TF-IDF weighting (only if you pass --method tfidf) ───────────────
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("embeddings"),
                   help="Folder that contains *.parquet with doc_id, chunk_id, vector")
    p.add_argument("--dst", type=Path, default=Path("doc_vectors.parquet"),
                   help="Output Parquet file (doc_id, vector)")
    p.add_argument("--method", choices=["mean", "tfidf", "medoid"],
                   default="mean", help="Pooling strategy")
    p.add_argument("--texts", type=Path,
                   help="Optional folder with chunk *.jsonl (needed for tfidf)")
    return p.parse_args()

# ------------------------------------------------------------------------------
def load_vectors(src_dir: Path):
    """
    Return dict: doc_id -> list[np.ndarray]
    """
    buckets = defaultdict(list)
    for f in tqdm(sorted(src_dir.glob("*.parquet")), desc="load"):
        tbl = pq.read_table(f)
        for did, vec in zip(tbl["doc_id"].to_numpy(),
                            tbl["vector"].to_numpy()):
            buckets[did].append(vec)
    return buckets

def pool_mean(vectors):
    return np.mean(vectors, axis=0)

def pool_medoid(vectors):
    """
    Return the vector whose mean distance to all others is minimal.
    Good when documents have multi-topic chunks.
    """
    vecs = np.stack(vectors)
    dists = np.linalg.norm(vecs[:, None, :] - vecs[None, :, :], axis=-1)
    return vecs[np.argmin(dists.sum(axis=1))]

def pool_tfidf(doc_id, vectors, text_lookup):
    """
    Weight each chunk vector by its TF-IDF norm (needs raw chunk texts).
    """
    texts = text_lookup[doc_id]               # list[str] same order as vectors
    tfidf = TfidfVectorizer().fit_transform(texts).toarray()
    weights = tfidf.sum(axis=1) + 1e-9        # avoid /0
    weights /= weights.sum()
    vecs = np.stack(vectors)
    return (vecs * weights[:, None]).sum(axis=0)

# ------------------------------------------------------------------------------
def main():
    args = parse_args()
    buckets = load_vectors(args.src)

    # optional: load chunk texts for TF-IDF weighting
    text_lookup = defaultdict(list)
    if args.method == "tfidf":
        if not args.texts:
            raise SystemExit("Need --texts folder containing original chunk JSONL")
        for jf in tqdm(sorted(args.texts.glob("*.jsonl")), desc="load texts"):
            for line in jf.open(encoding="utf-8"):
                o = json.loads(line)
                text_lookup[o["doc_id"]].append(o["text"])

    pooled = {}
    for did, vecs in tqdm(buckets.items(), desc="pool"):
        if args.method == "mean":
            pooled[did] = pool_mean(vecs)
        elif args.method == "medoid":
            pooled[did] = pool_medoid(vecs)
        else:  # tfidf
            pooled[did] = pool_tfidf(did, vecs, text_lookup)

    # write to Parquet
    import pyarrow as pa
    doc_ids = list(pooled.keys())
    matrix  = np.stack([pooled[d] for d in doc_ids])
    table   = pa.Table.from_pydict({"doc_id": doc_ids,
                                    "vector": list(matrix)})
    pq.write_table(table, args.dst)
    print(f"✅ pooled vectors → {args.dst} (docs: {len(doc_ids)})")

if __name__ == "__main__":
    main()
