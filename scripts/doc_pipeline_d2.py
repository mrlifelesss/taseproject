#!/usr/bin/env python
"""
doc_pipeline_d2.py  –  Docling 2.x only
------------------------------------------------------
* Finds the “directors” and “financials” sections in an ISA-style report.
* Works on PDF (and any other format Docling supports if you add options).
* Exports:
    - Markdown for each section
    - All tables inside each section as CSV + HTML
    - Console / JSON summary
"""

import re, json, argparse, pathlib, logging, warnings
warnings.filterwarnings("ignore", message="Parameter 'delim' has been deprecated")
from docling.document_converter import (
    DocumentConverter, PdfFormatOption, InputFormat
)
from docling.datamodel.pipeline_options import PdfPipelineOptions

logging.getLogger("docling").setLevel(logging.ERROR)

# ─────── regex patterns for the headings we care about ────────────
SECTION_PATTERNS = {
    "directors":  re.compile(r"(?:דוח|הסבר(?:י)?)\s+הדירקטוריון"),
    "financials": re.compile(r"דוחות?\s+כספיים"),
}
def rtl(s: str) -> str:
    return s[::-1]

# ─────── Docling conversion (PDF only for now) ────────────────────
def convert_to_doc(path: pathlib.Path):
    pdf_opts = PdfPipelineOptions(vision="full", ocr=True)
    conv     = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)
        }
    )
    return conv.convert(str(path)).document

# ─────── detect first page of each section ────────────────────────
HEADER_LABELS = {"SECTION_HEADER", "SUBSECTION_HEADER", "CHAPTER_TITLE","PART_HEADER", "TITLE"}


def page_of(item) -> int:
    """
    Return 0-based page index for any SectionHeaderItem produced by Docling 2.x.
    Priority:
      1. item.page_index
      2. item.page_no
      3. item.prov[0].page_no   (when .prov is a non-empty list)
    Raises AttributeError if none are present.
    """
    if hasattr(item, "page_index"):
        return item.page_index
    if hasattr(item, "page_no"):
        return item.page_no
    if hasattr(item, "prov") and item.prov:
        first = item.prov[0]
        if hasattr(first, "page_no"):
            return first.page_no
    raise AttributeError("no page information on item")


def detect_section_starts(doc) -> dict[str, int]:
    starts = {}
    for item, _ in doc.iterate_items():
        if item.label.name not in {"SECTION_HEADER", "SUBSECTION_HEADER", "TITLE"}:
            continue

        text = (item.text or item.content or "").strip()
        for key, rx in SECTION_PATTERNS.items():
            if key not in starts and (rx.search(text) or rx.search(rtl(text))):
                starts[key] = page_of(item) + 1        # 1-based now
        if len(starts) == len(SECTION_PATTERNS):
            return starts

    raise RuntimeError("Could not find both headings – adjust patterns.")


# ─────── build inclusive page ranges ──────────────────────────────
def make_ranges(starts: dict[str, int], total_pages: int):
    ordered = sorted(starts.items(), key=lambda kv: kv[1])
    ranges  = {}
    for (sec, a), (_, nxt) in zip(ordered, ordered[1:] + [(None, total_pages + 1)]):
        ranges[sec] = (a, nxt - 1)
    return ranges          # {'directors': (6,34), 'financials': (35,68)}

# ─────── grab tables inside each range (CSV + HTML) ───────────────
def extract_tables(
    doc,
    ranges: dict[str, tuple[int, int]],
    out_dir: pathlib.Path,
    stem: str,
    min_cols: int = 3,       # tweak if you want wider/narrower
) -> dict[str, int]:
    """
    Export every 'real' table that falls inside each section range.

    A table is considered real if:
      • it has at least 2 rows   AND
      • it has >= `min_cols` columns
    """
    buckets = {k: 0 for k in ranges}              # counter of saved tables

    for tbl in doc.tables:
        p = page_of(tbl) + 1                      # 1-based physical page

        # find which section this page belongs to
        target_sec = None
        for sec, (a, b) in ranges.items():
            if a <= p <= b:
                target_sec = sec
                break
        if target_sec is None:
            continue                               # table outside our sections

        # export to DataFrame and apply noise filter
        df = tbl.export_to_dataframe()
        if df.shape[0] < 2 or df.shape[1] < min_cols:
            continue                               # skip footer artefacts etc.

        # save as CSV + HTML
        buckets[target_sec] += 1
        idx  = buckets[target_sec]
        df.to_csv (out_dir / f"{stem}-{target_sec}-table{idx}.csv",  index=False)
        df.to_html(out_dir / f"{stem}-{target_sec}-table{idx}.html", index=False)

    return buckets


# ─────── CLI wrapper ──────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="PDF to process")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--json", action="store_true", help="print JSON summary")
    args = ap.parse_args()

    in_path  = pathlib.Path(args.file).resolve()
    out_dir  = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem     = in_path.stem

    doc       = convert_to_doc(in_path)
    total     = doc.num_pages()           # 2.x: num_pages() is a method
    ranges    = make_ranges(detect_section_starts(doc), total)
    #tbl_count = extract_tables(doc, ranges, out_dir, stem)

    # write Markdown per section
    for sec, (a, b) in ranges.items():
        md = doc.export_to_markdown(range(a - 1, b))    # 0-based slice
        (out_dir / f"{stem}-{sec}.md").write_text(md, encoding="utf-8")

    summary = {
        "file": str(in_path),
        "page_ranges": ranges,
        #"tables_per_section": tbl_count,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        w = max(len(k) for k in ranges)
        for sec, (a, b) in ranges.items():
            pages = b - a + 1
            print(f"{sec.capitalize():<{w}} : {a}-{b}  ({pages} pages)   "
                  #f"[tables: {tbl_count[sec]}]"
                  )
        print(f"✓ artefacts saved → {out_dir}")

if __name__ == "__main__":
    main()
