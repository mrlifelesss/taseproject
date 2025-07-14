#!/usr/bin/env python3
# dual_embed_chunks_local.py (revised)
"""
Embed each chunk **twice** (or more) with one or more
HF/Sentence-Transformers models in a single pass.

• If a model requires authentication (e.g. gated HF repo), pass --token <HF_TOKEN>.
• If a model fails to load/download, the script warns and continues with the next.
• Outputs one subfolder per model inside <dst> (default "embeddings_dual").
• Each subfolder contains one Parquet per input *.jsonl, with columns:
    doc_id, chunk_id, vector

Usage Examples:
--------------

# 1) Default run (two models pre-configured):
python dual_embed_chunks_local.py

# 2) Pass 3 models and a smaller batch size:
python dual_embed_chunks_local.py \
       --models intfloat/multilingual-e5-large \
                intfloat/multilingual-e5-base \
       --batch 8 \
       --token   hf_XXXXXXXXXXXXXXXXXXXXX

# 3) Use a custom output folder and only one model:
python dual_embed_chunks_local.py \
       --models onlplab/alephbert-base \
       --dst     custom_embeddings \
       --batch   16
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def sanitize(name: str) -> str:
    """
    Turn a model path (e.g. "intfloat/multilingual-e5-large") into a safe folder name:
      - replace "/" or ":" with "_"
      - drop any characters other than alphanumerics, hyphens, or underscores
    """
    name = name.replace("/", "_").replace(":", "_")
    return re.sub(r"[^0-9A-Za-z_\-]+", "_", name)


def embed_in_batches(
    model: SentenceTransformer, texts: List[str], batch_size: int
) -> List[List[float]]:
    """
    Yield embeddings (as numpy lists) for the input list of texts,
    using SentenceTransformer.encode in sub-batches of size batch_size.
    """
    all_vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        sub = texts[i : i + batch_size]
        emb = model.encode(
            sub,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # convert to python list-of-lists (parquet can accept NumPy arrays directly too)
        all_vecs.extend(emb.tolist())
    return all_vecs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        default=[
            "BAAI/bge-m3",
        ],
        help=(
            "One or more HF model names (Sentence-Transformers compatible). "
            "E.g. intfloat/multilingual-e5-large or onlplab/alephbert-base"
        ),
    )
    ap.add_argument(
        "--src",
        type=Path,
        default=Path("cleaned_text"),
        help="Folder containing input *.jsonl files (each with doc_id, chunk_id, text).",
    )
    ap.add_argument(
        "--dst",
        type=Path,
        default=Path("embeddings_dual"),
        help="Base output folder. Subfolders will be created per model.",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for embedding (tune down if you run out of GPU memory).",
    )
    ap.add_argument(
        "--token",
        type=str,
        default=None,
        help=(
            "Optional HF authentication token if you must download gated models. "
            "Equivalent to use_auth_token in SentenceTransformer."
        ),
    )
    args = ap.parse_args()

    # Basic sanity checks
    if not args.src.exists() or not any(args.src.glob("*.jsonl")):
        print(f"[ERROR] No JSONL files found in {args.src!r}.", file=sys.stderr)
        sys.exit(1)

    args.dst.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Iterate over each model specified
    for model_name in args.models:
        sanitized = sanitize(model_name)
        model_folder = args.dst / sanitized
        model_folder.mkdir(parents=True, exist_ok=True)

        print(f"▶ Embedding with model: '{model_name}' …")
        t0_model = time.time()

        # Attempt to load the SentenceTransformer model
        try:
            if args.token:
                model = SentenceTransformer(
                    model_name,
                    trust_remote_code=True,
                    device=device,
                    use_auth_token=args.token,
                )
            else:
                model = SentenceTransformer(
                    model_name, trust_remote_code=True, device=device
                )
        except Exception as e:
            print(f"[WARN] Could not load model '{model_name}': {e}", file=sys.stderr)
            print("       Skipping this model.\n")
            continue

        # Iterate over every JSONL file in the source folder
        for jfile in tqdm(sorted(args.src.glob("*.jsonl")), desc=f"Files ({sanitized})"):
            out_parquet = model_folder / f"{jfile.stem}.parquet"
            # If we’ve already embedded this file for this model, skip it
            if out_parquet.exists():
                tqdm.write(f"[skipped] {out_parquet.name} (already exists)")
                continue

            # Read in all lines, collect doc_ids / chunk_ids / texts
            doc_ids = []
            chunk_ids = []
            texts = []
            try:
                with jfile.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        rec = json.loads(line)
                        # Expect keys: doc_id, chunk_id, text
                        doc_ids.append(rec["doc_id"])
                        chunk_ids.append(rec["chunk_id"])
                        texts.append(rec["text"])
            except Exception as e:
                tqdm.write(f"[ERROR] Failed to read {jfile.name}: {e}")
                continue

            # Embed in batches
            try:
                vecs = embed_in_batches(model, texts, args.batch)
            except Exception as e:
                tqdm.write(f"[ERROR] Embedding {jfile.name} failed: {e}")
                continue

            # Build a PyArrow table and write to Parquet
            try:
                table = pa.Table.from_pydict(
                    {
                        "doc_id": doc_ids,
                        "chunk_id": chunk_ids,
                        "vector": vecs,
                    }
                )
                pq.write_table(table, out_parquet)
            except Exception as e:
                tqdm.write(f"[ERROR] Writing {out_parquet.name} failed: {e}")
                continue

            tqdm.write(f"[ok] {out_parquet.name}")

        # Done with this model
        dur = time.time() - t0_model
        print(f"\n Finished '{model_name}' in {dur:.1f}s → folder: {model_folder}\n")

        # Free GPU memory if using CUDA before loading next model
        if device.startswith("cuda"):
            del model
            torch.cuda.empty_cache()

    print("▶ All models processed. Exiting.\n")


if __name__ == "__main__":
    main()
