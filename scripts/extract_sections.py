"""
extract_sections_pypdf2.py
Isolate two sections in an ISA-style Hebrew report:
    • “דוחות כספיים”      (audited statements)
    • “דוח הדירקטוריון”  (board report)

USAGE (run from the repo root):
    python scripts/extract_sections_pypdf2.py inputs/1659067.pdf
    python scripts/extract_sections_pypdf2.py inputs/1659067.pdf --save-pdf
"""
from __future__ import annotations
import re, argparse, io, pathlib, typing as _t

import pdfplumber                      # text extraction
from PyPDF2 import PdfReader, PdfWriter # page slicing

# --- 1. Phrases we look for -------------------------------------------------
SECTION_PATTERNS: dict[str, str] = {
    "financials":  r"דוחות?\s+כספיים",
    "directors":   r"דוח\s+הדירקטוריון",
}
ALL_SECTIONS = list(SECTION_PATTERNS.keys())

# ---------------------------------------------------------------------------

def _rtl(line: str) -> str:
    "Reverse a whole line – helps us spot RTL text that was stored LTR."
    return line[::-1]

def _matches(line: str, rx: re.Pattern) -> bool:
    """True if *either* the line or its reversed version matches."""
    return bool(rx.search(line) or rx.search(_rtl(line)))

def find_starts(pdf_path: str | pathlib.Path) -> dict[str, int]:
    """
    Scan every page until each pattern is found once.
    Returns: { 'financials': page_index, 'directors': page_index }
    """
    compiled = {k: re.compile(p) for k, p in SECTION_PATTERNS.items()}
    found: dict[str, int] = {}

    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            for key, rx in compiled.items():
                if key not in found and _matches(txt, rx):
                    found[key] = i
                    if len(found) == len(ALL_SECTIONS):
                        return found
    return found   # may be incomplete → caller raises

def build_ranges(starts: dict[str, int], n_pages: int) -> dict[str, tuple[int,int]]:
    """
    Convert {'directors': 10, 'financials': 123}  →
            {'directors': (10,123), 'financials': (123, n_pages)}
    """
    if len(starts) < 2:
        raise ValueError("Could not locate both headings ‒ check the patterns or OCR")

    ordered = sorted(starts.items(), key=lambda kv: kv[1])
    ranges: dict[str, tuple[int,int]] = {}
    for (sec, start), (_, nxt) in zip(ordered, ordered[1:] + [(None, n_pages)]):
        ranges[sec] = (start, nxt)
    return ranges

def extract(pdf_path: str | pathlib.Path, save_pdf: bool = False, verbose: bool=False
           ) -> dict[str, bytes | str]:
    pdf_path = pathlib.Path(pdf_path)
    starts = find_starts(pdf_path)
    if verbose:
        print("Heading pages:", starts)

    reader = PdfReader(str(pdf_path))
    ranges = build_ranges(starts, len(reader.pages))

    outputs: dict[str, bytes | str] = {}
    for sec, (a, b) in ranges.items():
        if save_pdf:
            writer = PdfWriter()
            for p in range(a, b):
                writer.add_page(reader.pages[p])
            buf = io.BytesIO()
            writer.write(buf)
            outputs[sec] = buf.getvalue()
        else:
            # re-open with pdfplumber for text
            with pdfplumber.open(str(pdf_path)) as pdf:
                txt = "\n".join((pdf.pages[p].extract_text() or "") for p in range(a, b))
            outputs[sec] = txt
    return outputs

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="path to source PDF")
    ap.add_argument("--save-pdf", action="store_true",
                    help="write two mini-PDFs instead of plain text")
    ap.add_argument("--debug", action="store_true",
                    help="print the page numbers I find")
    args = ap.parse_args()

    result = extract(args.pdf, save_pdf=args.save_pdf, verbose=args.debug)

    # Write side-by-side with the original:  inputs/1659067/financials.txt …
    out_dir = pathlib.Path(args.pdf).with_suffix("")
    out_dir.mkdir(exist_ok=True)

    for key, data in result.items():
        ext = "pdf" if args.save_pdf else "txt"
        mode = "wb" if args.save_pdf else "w"
        fname = out_dir / f"{key}.{ext}"
        with open(fname, mode, encoding=None if args.save_pdf else "utf-8") as f:
            f.write(data)
    print(f"✓  extracted → {out_dir.resolve()}")

