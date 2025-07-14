#!/usr/bin/env python3
"""
embed_bge_m3.py

Embed all chunks in the single JSONL file “1590620_1.jsonl” (located in cleaned_text/)
using the BAAI/bge-m3 SentenceTransformer model, and save the results as a Parquet.

Output will be written to:
    embeddings_bge-m3/1590620_1.parquet

Each row in the Parquet will contain:
    • doc_id    (string)
    • chunk_id  (int)
    • vector    (list of floats)

Usage:
    python embed_bge_m3.py

Dependencies:
    • sentence-transformers
    • torch
    • pyarrow
    • tqdm
"""

import json
import os
import sys
from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_NAME    = "BAAI/bge-m3"
SRC_DIR       = Path("cleaned_text")
SRC_FILE_NAME = "1590620_1.jsonl"
OUT_DIR       = Path("embeddings_bge-m3")
BATCH_SIZE    = 16
# ────────────────────────────────────────────────────────────────────────────────

def embed_in_batches(model: SentenceTransformer, texts: List[str], batch_size: int) -> List[List[float]]:
    """
    Given a SentenceTransformer model and a list of texts, run encode() in sub-batches,
    normalize embeddings, and return a list-of-lists (one embedding per text).
    """
    all_vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        sub_texts = texts[i : i + batch_size]
        # encode returns a NumPy array of shape (len(sub_texts), embedding_dim)
        embs = model.encode(
            sub_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # Convert to python list-of-lists
        all_vecs.extend(embs.tolist())
    return all_vecs

def main():
    # Ensure source file exists
    src_path = SRC_DIR / SRC_FILE_NAME
    if not src_path.exists():
        print(f"[ERROR] Source file not found: {src_path}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if missing
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_parquet = OUT_DIR / f"{src_path.stem}.parquet"

    # Detect device (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ℹ] Using device: {device}")

    # 1) Load the BAAI/bge-m3 SentenceTransformer
    try:
        model = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=True)
    except Exception as e:
        print(f"[ERROR] Failed to load model '{MODEL_NAME}': {e}", file=sys.stderr)
        sys.exit(1)

    # 2) Read JSONL file, collect doc_id, chunk_id, and text
    doc_ids   = []
    chunk_ids = []
    texts     = []

    print(f"[ℹ] Reading {src_path} …")
    try:
        with src_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                # Expect each record to have keys: "doc_id", "chunk_id", "text"
                doc_ids.append(rec["doc_id"])
                chunk_ids.append(rec["chunk_id"])
                texts.append(rec["text"])
    except Exception as e:
        print(f"[ERROR] Failed to read '{src_path}': {e}", file=sys.stderr)
        sys.exit(1)

    if len(texts) == 0:
        print(f"[ERROR] No records found in '{src_path}'!", file=sys.stderr)
        sys.exit(1)

    print(f"[ℹ] Found {len(texts)} chunks to embed.")

    # 3) Embed all texts in batches
    print("[ℹ] Computing embeddings…")
    embeddings = embed_in_batches(model, texts, BATCH_SIZE)
    if len(embeddings) != len(texts):
        print(
            f"[ERROR] Embedding count ({len(embeddings)}) != number of texts ({len(texts)})",
            file=sys.stderr,
        )
        sys.exit(1)

    # 4) Build a PyArrow table and write to Parquet
    print(f"[ℹ] Writing embeddings to {out_parquet} …")
    try:
        table = pa.Table.from_pydict({
            "doc_id":   doc_ids,
            "chunk_id": chunk_ids,
            "vector":   embeddings,
        })
        pq.write_table(table, out_parquet)
    except Exception as e:
        print(f"[ERROR] Failed to write Parquet '{out_parquet}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[✔] Successfully wrote: {out_parquet}")

    # 5) If using CUDA, free GPU memory
    if device.startswith("cuda"):
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
