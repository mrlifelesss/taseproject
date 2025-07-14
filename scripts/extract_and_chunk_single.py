#!/usr/bin/env python3
# extract_and_chunk_local.py
# ───────────────────────────────────────────────────────────────
# Local HTML → Clean → Semantic Chunk (sentence‑aware or fixed char windows)
# • Reads a single HTML/HTM file from disk
# • Cleans header/footer around key Hebrew phrases
# • Merges tables into descriptive sentences and extracts narrative via Readability
# • Normalizes Hebrew (strip niqqud, collapse whitespace)
# • Splits text into chunks (semantic or fixed)
# • Outputs JSONL where each line = {doc_id, chunk_id, text}

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Iterator
from bs4 import BeautifulSoup
from readability import Document
import torch

# ── HTML Cleaning ───────────────────────────────────────────────
def clean_html_report(raw_html: str) -> str:
    """Remove header/footer around key phrases."""
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator="\n")
    # Trim above 'שעת שידור'
    start = text.find("שעת שידור")
    if start != -1:
        text = text[start:]
    # Trim below footer phrases
    for footer in ("פרטי החותמים המורשים לחתום בשם התאגיד:",
                   "הסבר: לפי תקנה 5 לתקנות",
                   "מספרי אסמכתאות של מסמכים קודמים בנושא"):
        idx = text.find(footer)
        if idx != -1:
            text = text[:idx]
            break
    # Clean lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)

# ── Table Merge ─────────────────────────────────────────────────
def merge_tables(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    sentences = []
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["th","td"])]
            if len(cells) >= 2:
                parts = [f"{cells[i]}: {cells[i+1]}" for i in range(0, len(cells)-1, 2)]
                sentences.append("; ".join(parts))
    return "\n".join(sentences)

# ── Narrative Extraction ─────────────────────────────────────────
def extract_narrative(raw_html: str) -> str:
    doc = Document(raw_html)
    summary_html = doc.summary()
    return BeautifulSoup(summary_html, "html.parser").get_text("\n", strip=True)

# ── Hebrew Normalize ────────────────────────────────────────────
_DIACRITICS = {chr(c) for c in range(0x0591, 0x05C8)}

def normalize_he(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if ch not in _DIACRITICS)
    return " ".join(text.split())

# ── Chunking ────────────────────────────────────────────────────
def char_chunks(text: str, size: int, stride: int) -> Iterator[str]:
    step = size - stride
    for i in range(0, len(text), step):
        yield text[i:i+size]

# ── Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Process one Hebrew HTML file into semantic chunks.")
    parser.add_argument("input_file", type=Path, help="Path to input HTML/HTM file")
    parser.add_argument("--out_jsonl", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--chunk-size", type=int, default=450)
    parser.add_argument("--overlap", type=int, default=75)
    parser.add_argument("--model", default="intfloat/multilingual-e5-large")
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--char-size", type=int, default=1500)
    parser.add_argument("--char-stride", type=int, default=300)
    args = parser.parse_args()

    raw_html = args.input_file.read_text(encoding="utf-8", errors="ignore")

    # Preprocess
    cleaned_text = clean_html_report(raw_html)
    table_text   = merge_tables(raw_html)
    narrative    = extract_narrative(cleaned_text)
    combined     = "\n\n".join(filter(None, [table_text, narrative]))
    plain        = normalize_he(combined)

    # Prepare embedding model or char chunks
    if not args.fixed:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.core import Settings, Document
        embed = HuggingFaceEmbedding(
            model_name=args.model,
            device="cuda" if torch.cuda.is_available() else "cpu",)
        Settings.embed_model = embed
        splitter = SemanticSplitterNodeParser(
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            buffer_size=1,
            embed_model=embed,)
        chunks = (n.text for n in splitter.get_nodes_from_documents([Document(text=plain)]))
    else:
        chunks = char_chunks(plain, args.char_size, args.char_stride)

    # Write JSONL
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks):
            f.write(json.dumps({
                "doc_id": args.input_file.stem,
                "chunk_id": idx,
                "text": chunk}, ensure_ascii=False) + "\n")
    print(f"Wrote chunks to {args.out_jsonl}")

if __name__ == "__main__":
    main()
