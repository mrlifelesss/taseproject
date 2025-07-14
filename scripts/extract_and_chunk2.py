# extract_and_chunk_final.py
# ───────────────────────────────────────────────────────────────
# DynamoDB → S3 → Clean‑and‑Semantic‑Chunk  (semantic = default)
# • Scan table CompanyDisclosuresHebrew for items whose pdf_count == 0
# • Each item holds HTML/HTM keys in attachedFiles
# • Download, clean HTML (remove header/footer), merge tables & narrative, strip Hebrew niqqud, collapse whitespace
# • Semantic chunking (sentence‑aware, embedding‑driven) is the default; use --fixed for char windows
# • Output → cleaned_text/<docId>_<idx>.jsonl – each line = {doc_id, chunk_id, text}

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterator

import boto3
from bs4 import BeautifulSoup
from tqdm import tqdm
import torch
from readability import Document

# ── Custom HTML cleaning before processing ─────────────────────
def clean_html_report(raw_html: str) -> str:
    """Remove header/footer around key phrases."""
    clean_html = re.sub(r'<!\[if[^\]]*\]>.*?<!\[endif\]>', '', raw_html, flags=re.DOTALL)
    soup = BeautifulSoup(clean_html, "html.parser")
    # Extract full text with line breaks
    full_text = soup.get_text(separator="\n")
    
    # 1. Remove everything above the first occurrence of "שעת שידור"
    start_phrase = "שעת שידור"
    start_idx = full_text.find(start_phrase)
    if start_idx != -1:
        full_text = full_text[start_idx:]
        
    # 2. Remove footer and everything below footer phrase
    first_footer_phrase = "פרטי החותמים המורשים לחתום בשם התאגיד:"
    footer_phrase = "הסבר: לפי תקנה 5 לתקנות"
    fallback_footer = "מספרי אסמכתאות של מסמכים קודמים בנושא"
    footer_idx = full_text.find(first_footer_phrase)
    if footer_idx == -1:
        footer_idx = full_text.find(footer_phrase)
    if footer_idx == -1:
        footer_idx = full_text.find(fallback_footer)
    if footer_idx != -1:
        full_text = full_text[:footer_idx]
    
    # 3. Clean up lines
    lines = [line.strip() for line in full_text.splitlines()]
    # Remove empty lines
    lines = [line for line in lines if line]
    
    return "\n".join(lines)
# ── Merge tables into descriptive sentences ────────────────────
def merge_tables(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    lines = []
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["th","td"])]
            if len(cells) >= 2:
                parts = [f"{cells[i]}: {cells[i+1]}" for i in range(0, len(cells)-1, 2)]
                lines.append("; ".join(parts))
    return "\n".join(lines)

# ── Narrative extraction via Readability ──────────────────────
def extract_narrative(raw_html: str) -> str:
    doc = Document(raw_html)
    summary = doc.summary()
    text = BeautifulSoup(summary, "html.parser").get_text(separator="\n", strip=True)
    return text

# ── Hebrew helpers ─────────────────────────────────────────────
_DIACRITICS = {chr(c) for c in range(0x0591, 0x05C8)}

def clean_he(text: str) -> str:
    """Normalize, strip niqqud, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if ch not in _DIACRITICS)
    return " ".join(text.split())

# ── Basic char‑window fallback splitter ─────────────────────────
def char_chunks(text: str, size: int, stride: int) -> Iterator[str]:
    step = size - stride
    for i in range(0, len(text), step):
        yield text[i:i + size]

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

# ── Main ───────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table",     default="CompanyDisclosuresHebrew")
    ap.add_argument("--region",    default="us-east-1")
    ap.add_argument("--bucket",    default="summery-prompts-data")
    ap.add_argument("--html-attr", default="attachedFiles")
    ap.add_argument("--pdf-attr",  default="pdf_count")
    ap.add_argument("--chunk-size", type=int, default=450)
    ap.add_argument("--overlap",    type=int, default=75)
    ap.add_argument("--model",      default="intfloat/multilingual-e5-large")
    ap.add_argument("--fixed",      action="store_true")
    ap.add_argument("--char-size", type=int, default=1500)
    ap.add_argument("--char-stride", type=int, default=300)
    ap.add_argument("--out-dir",   type=Path, default=Path("cleaned_text"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dynamo = boto3.resource("dynamodb", region_name=args.region)
    table  = dynamo.Table(args.table)
    s3     = boto3.client("s3", region_name=args.region)

    if not args.fixed:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.core import Settings, Document
        embed = HuggingFaceEmbedding(
            model_name=args.model,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        Settings.embed_model = embed
        splitter = SemanticSplitterNodeParser(
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            buffer_size=1,
            embed_model=embed,
        )

    for item in tqdm(scan_items(table, args.pdf_attr), desc="items"):
        doc_id = str(item.get("reportId") or item.get("id") or item.get("pk") or "unknown")
        keys   = item.get(args.html_attr, [])
        if not isinstance(keys, list):
            continue

        for idx, key in enumerate(keys):
            key = key.lstrip("/")
            try:
                body = s3.get_object(Bucket=args.bucket, Key=key)["Body"].read().decode("utf-8", "ignore")
            except Exception as e:
                tqdm.write(f"[ERR] {key}: {e}")
                continue

            # Preprocess: clean header/footer, merge tables & narrative, normalize
            header_trimmed = clean_html_report(body)
            table_txt      = merge_tables(header_trimmed)
            narrative_txt  = extract_narrative(header_trimmed)
            combined       = "\n\n".join(filter(None, [table_txt, narrative_txt]))
            plain          = clean_he(combined)
            if not plain:
                continue

            # Chunking
            if args.fixed:
                chunks = char_chunks(plain, args.char_size, args.char_stride)
            else:
                nodes  = splitter.get_nodes_from_documents([Document(text=plain)])
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
