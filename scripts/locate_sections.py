"""
locate_sections.py  –  v2 (first-line header detection)

Outputs the page ranges of:
    • דוח הדירקטוריון   (Board report)
    • דוחות כספיים       (Financial statements)

Run:
    python scripts/locate_sections.py inputs/1664899.pdf
    python scripts/locate_sections.py inputs/1664899.pdf --json
    python scripts/locate_sections.py inputs/1664899.pdf --out-dir output
"""
from __future__ import annotations
import argparse, json, re, pathlib
import pdfplumber
from PyPDF2 import PdfReader

# ──────────────────────────────────────────────────────────────────────────
PATTERNS = {
    "directors":  r"(?:דוח|הסברי)\s+הדירקטוריון",
    "financials": r"דוחות?\s+כספיים",
}
RE = {k: re.compile(p) for k, p in PATTERNS.items()}

# ─────────────────── helpers ──────────────────────────────────────────────
def _rtl(s: str) -> str:                       # reverse whole string
    return s[::-1]

def first_nonblank_line(page) -> str:
    txt = page.extract_text() or ""
    for line in txt.splitlines():
        if line.strip():
            return line.strip()
    return ""

def find_starts(pdf_path: pathlib.Path) -> dict[str, int]:
    """
    Pass 1 – header must be FIRST non-blank line  → high precision  
    Pass 2 – header may appear ANYWHERE on the page (still skips TOC pages)
    """
    starts: dict[str, int] = {}
    compiled = RE

    with pdfplumber.open(str(pdf_path)) as pdf:
        pages = list(pdf.pages)

        # ---------- pass 1 : first-line only ----------
        for i, page in enumerate(pages):
            first = first_nonblank_line(page)
            for key, rx in compiled.items():
                if key not in starts and (rx.search(first) or rx.search(_rtl(first))):
                    starts[key] = i
            if len(starts) == len(PATTERNS):
                return starts                           # success

        # ---------- pass 2 : anywhere on the page ----------
        for i, page in enumerate(pages):
            # quick skip – TOC is almost always before page 5
            if i < 5 and len(pages) > 10:
                continue
            text = page.extract_text() or ""
            for key, rx in compiled.items():
                if key not in starts and (rx.search(text) or rx.search(_rtl(text))):
                    starts[key] = i
            if len(starts) == len(PATTERNS):
                return starts

    return starts  # may be incomplete → caller raises


def build_ranges(starts: dict[str,int], total: int) -> dict[str, tuple[int,int]]:
    if len(starts) < 2:
        raise RuntimeError("Could not detect both headings – adjust patterns")
    ordered = sorted(starts.items(), key=lambda kv: kv[1])
    ranges = {}
    for (sec, s), (_, nxt) in zip(ordered, ordered[1:] + [(None, total)]):
        ranges[sec] = (s, nxt - 1)
    return ranges

def locate(pdf_file: str | pathlib.Path) -> dict[str, tuple[int,int]]:
    pdf_path = pathlib.Path(pdf_file)
    total_pages = len(PdfReader(str(pdf_path)).pages)
    return build_ranges(find_starts(pdf_path), total_pages)

# ─────────────────── CLI ─────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out-dir", default=None,
                    help="write result file(s) into this folder instead of stdout")
    args = ap.parse_args()

    ranges = locate(args.pdf)                       # {'directors': (23,33), ...}
    serialisable = {k: [a+1, b+1] for k,(a,b) in ranges.items()}  # 1-based

    # ───── decide destination ────────────────────────────────────────────
    if args.out_dir:
        out_dir = pathlib.Path(args.out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = pathlib.Path(args.pdf).stem
        ext  = "json" if args.json else "txt"
        out_file = out_dir / f"{stem}_sections.{ext}"
        with open(out_file, "w", encoding="utf-8") as fh:
            if args.json:
                json.dump(serialisable, fh, ensure_ascii=False, indent=2)
            else:
                w = max(len(k) for k in ranges)
                for k, (a,b) in ranges.items():
                    a1, b1 = a+1, b+1
                    fh.write(f"{k.capitalize():<{w}} : {a1}-{b1}   ({b1-a1+1} pages)\n")
        print(f"✓ ranges saved → {out_file}")
    else:
        if args.json:
            print(json.dumps(serialisable, ensure_ascii=False))
        else:
            w = max(len(k) for k in ranges)
            for k, (a,b) in ranges.items():
                a1, b1 = a+1, b+1
                print(f"{k.capitalize():<{w}} : {a1}-{b1}   ({b1-a1+1} pages)")