#!/usr/bin/env python
"""
doc_pipeline.py
---------------
Locate key sections in *any* supported file (PDF / DOCX / HTML …),
export their page ranges **and** pull out any tables that live inside them.

The output structure is ready for RAG pipelines or direct LLM calls.
"""
from __future__ import annotations
import re, json, argparse, pathlib, logging
import pandas as pd
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

logging.getLogger("docling").setLevel(logging.ERROR)   # keep console quiet

# 1️⃣  Which headings mark our sections?  (compile once)
SECTION_PATTERNS = {
    "directors":  re.compile(r"(?:דוח|הסבר(?:י)?)\s+הדירקטוריון"),
    "financials": re.compile(r"דוחות?\s+כספיים"),
}
# labels that mark top-level headings in Docling 2.x
HEADER_LABELS = {"SECTION_HEADER", "TITLE"}
def rtl(s: str) -> str:                # reverse whole line (RTL stored LTR)
    return s[::-1]

# 2️⃣  Convert *any* file into a DoclingDocument
def convert(path: pathlib.Path):
    # tweak to taste: vision="lite", ocr=False is usually fine for born-digital PDFs
    pdf_opts = PdfPipelineOptions(vision="lite", ocr=False)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)
            # You can add DOCX / HTML entries here later if you need special settings
        }
    )
    return converter.convert(str(path)).document

# 3️⃣  Find first heading block that matches each pattern
def label_name(lbl):
    """Return the plain name whether lbl is an Enum or a raw string."""
    if lbl is None:
        return ""
    return lbl.name if hasattr(lbl, "name") else str(lbl)

def item_plain_text(it) -> str:
    """Return readable string regardless of Docling version."""
    for attr in ("export_to_text", "text", "content"):
        if hasattr(it, attr):
            fn = getattr(it, attr)
            return fn() if callable(fn) else fn
    return str(it)

def item_page_index(it) -> int:
    """
    0-based page index for any Docling item (works on v1.* and v2.*).

    • v1.*                →  it.page_number  
    • v1.2–1.3            →  it.page_index  
    • v2.* (current)      →  it.location.page_index  **or**  it.prov.page_no
    """
    if hasattr(it, "page_number"):
        return it.page_number
    if hasattr(it, "page_index"):
        return it.page_index
    if hasattr(it, "location") and hasattr(it.location, "page_index"):
        return it.location.page_index
    if hasattr(it, "prov"):
        prov = it.prov[0] if isinstance(it.prov, list) else it.prov
        if prov and hasattr(prov, "page_no"):
            return prov.page_no
    raise AttributeError("item carries no page index")
    
def detect_starts(doc):
    starts = {}
    for item, _ in doc.iterate_items():          # no keyword args in 2.x
        if label_name(item.label) not in HEADER_LABELS:
            continue

        txt = item_plain_text(item).strip()
        for key, rx in SECTION_PATTERNS.items():
            if key not in starts and (rx.search(txt) or rx.search(rtl(txt))):
                starts[key] = item_page_index(item) + 1     # 1-based
        if len(starts) == len(SECTION_PATTERNS):
            return starts

    raise RuntimeError("Could not find both headings – tweak patterns.")                            

# 4️⃣  Build inclusive page ranges
def make_ranges(starts, n_pages):
    ordered = sorted(starts.items(), key=lambda kv: kv[1])
    ranges = {}
    for (sec, s), (_, nxt) in zip(ordered, ordered[1:] + [(None, n_pages + 1)]):
        ranges[sec] = (s, nxt - 1)
    return ranges                                     # 1-based

# 5️⃣  Extract tables that *fall inside* those page ranges
def tables_by_section(doc, ranges):
    buckets: dict[str, list[pd.DataFrame]] = {k: [] for k in ranges}
    for tbl in doc.tables:
        p = tbl.page_number + 1                       # Docling 0-based → 1-based
        for sec, (a, b) in ranges.items():
            if a <= p <= b:
                buckets[sec].append(tbl.export_to_dataframe())
                break
    return buckets

# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="PDF/DOCX/HTML/… to process")
    ap.add_argument("--out-dir", default="output", help="where to save artefacts")
    ap.add_argument("--json", action="store_true", help="print JSON summary too")
    args = ap.parse_args()

    in_path  = pathlib.Path(args.file).expanduser().resolve()
    out_dir  = pathlib.Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    doc      = convert(in_path)
    ranges   = make_ranges(detect_starts(doc), doc.num_pages)
    buckets  = tables_by_section(doc, ranges)

    # ❶  Write Markdown chunks (or plain-text) per section
    for sec, (a, b) in ranges.items():
        md = doc.export_markdown(page_numbers=range(a-1, b))   # 0-based slice
        (out_dir / f"{in_path.stem}-{sec}.md").write_text(md, encoding="utf-8")

    # ❷  Save each table as CSV and HTML
    for sec, tbls in buckets.items():
        for i, df in enumerate(tbls, 1):
            stem = f"{in_path.stem}-{sec}-table{i}"
            df.to_csv(out_dir / f"{stem}.csv", index=False)
            df.to_html(out_dir / f"{stem}.html", index=False)

    # ❸  Optional JSON summary for pipelines
    summary = {
        "file": str(in_path),
        "page_ranges": ranges,
        "tables_per_section": {k: len(v) for k, v in buckets.items()},
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        for sec, (a, b) in ranges.items():
            print(f"{sec.capitalize():<10}: {a}-{b}  ({b-a+1} pages)   "
                  f"[tables: {len(buckets[sec])}]")

if __name__ == "__main__":
    main()
