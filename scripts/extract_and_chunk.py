# dynamo_to_chunks_s3_semantic.py  
# ───────────────────────────────────────────────────────────────
# DynamoDB → S3 → Clean‑and‑Semantic‑Chunk  (semantic = **default**)
# • Scan table **CompanyDisclosuresHebrew** for items whose `pdf_count == 0`
# • Each item holds one‑or‑more HTML/HTM keys in attribute `attachedFiles`
#   (plain list[str]) that live in bucket **summery-prompts-data**
# • Download, NFC‑normalise, strip Hebrew niqqud, collapse whitespace
# • **Semantic chunking** (sentence‑aware, embedding‑driven) is the default;
#   pass `--fixed` to revert to simple char windows.
# • Output ⇒ cleaned_text/<docId>_<idx>.jsonl – each line = {doc_id, chunk_id, text}
#
# Requirements (GPU optional but recommended):
#   pip install --upgrade "torch==2.*" transformers accelerate \
#                        sentence-transformers \
#                        llama-index-core llama-index-embeddings-huggingface \
#                        boto3 beautifulsoup4 tqdm

import argparse, json, re, unicodedata
from pathlib import Path
from typing import List, Dict, Iterator

import boto3
from bs4 import BeautifulSoup
from tqdm import tqdm
import torch

# ── LlamaIndex semantic splitter pieces ─────────────────────────
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Settings   # replaces deprecated ServiceContext
from llama_index.core import Settings, Document 

# ── Hebrew helpers ───────────────────────────────────────────────
_DIACRITICS = {chr(c) for c in range(0x0591, 0x05C8)}

def clean_he(text: str) -> str:
    """NFC, strip niqqud, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if ch not in _DIACRITICS)
    return " ".join(text.split())

# ── Basic char‑window fallback splitter ─────────────────────────

def char_chunks(text: str, size: int, stride: int) -> Iterator[str]:
    step = size - stride
    for i in range(0, len(text), step):
        yield text[i:i + size]

# ── HTML helpers ────────────────────────────────────────────────

def html_to_text(raw: str) -> str:
    return BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)

# ── DynamoDB scan ───────────────────────────────────────────────

def scan_items(table, pdf_attr: str):
    from boto3.dynamodb.conditions import Attr
    fe = Attr(pdf_attr).eq(0)
    kwargs = dict(FilterExpression=fe)
    while True:
        resp = table.scan(**kwargs)
        yield from resp["Items"]
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

# ── Main ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table",      default="CompanyDisclosuresHebrew")
    ap.add_argument("--region",     default="us-east-1")
    ap.add_argument("--bucket",     default="summery-prompts-data")
    ap.add_argument("--html-attr",  default="attachedFiles")
    ap.add_argument("--pdf-attr",   default="pdf_count")
    # semantic splitter params
    ap.add_argument("--chunk-size", type=int, default=450, help="target tokens")
    ap.add_argument("--overlap",    type=int, default=75,  help="token overlap")
    ap.add_argument("--model",      default="intfloat/multilingual-e5-large")
    # flags
    ap.add_argument("--fixed", action="store_true", help="use simple char windows")
    ap.add_argument("--char-size", type=int, default=1500)
    ap.add_argument("--char-stride", type=int, default=300)
    ap.add_argument("--out-dir",   type=Path, default=Path("cleaned_text"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # AWS clients
    dynamo = boto3.resource("dynamodb", region_name=args.region)
    table  = dynamo.Table(args.table)
    s3     = boto3.client("s3",    region_name=args.region)

    # ── Embedder + semantic splitter (unless --fixed) ───────────
    if not args.fixed:
        embed = HuggingFaceEmbedding(
            model_name=args.model,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        Settings.embed_model = embed   # global registration
        splitter = SemanticSplitterNodeParser(
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            buffer_size=1,
            embed_model=embed,
        )

    for item in tqdm(scan_items(table, args.pdf_attr), desc="items"):
        doc_id = str(item.get("reportId") or item.get("id") or item.get("pk") or "unknown")

        # list[str] of S3 keys
        keys = item.get(args.html_attr, [])
        if not isinstance(keys, list):
            continue  # skip malformed entry

        for idx, key in enumerate(keys):
            key = key.lstrip("/")
            try:
                body = s3.get_object(Bucket=args.bucket, Key=key)["Body"].read().decode("utf-8", "ignore")
            except Exception as e:
                tqdm.write(f"[ERR] {key}: {e}")
                continue

            plain = clean_he(html_to_text(body))
            if not plain:
                continue

            # choose splitter
            if args.fixed:
                chunks = char_chunks(plain, args.char_size, args.char_stride)
            else:
                # wrap text in Document so splitter expects correct type
                nodes = splitter.get_nodes_from_documents([Document(text=plain)])
                chunks = (n.text for n in nodes)

            out = args.out_dir / f"{doc_id}_{idx}.jsonl"
            with out.open("w", encoding="utf-8") as f:
                for c_idx, text in enumerate(chunks):
                    f.write(json.dumps({
                        "doc_id": doc_id,
                        "chunk_id": c_idx,
                        "text": text
                    }, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()