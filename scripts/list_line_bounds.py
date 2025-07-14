#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
list_line_bounds.py

Produce a list of every line’s normalized bounds and whether it’s “centered”.

Usage:
    python list_line_bounds.py report.pdf           # print to console
    python list_line_bounds.py report.pdf --txt out.txt
    python list_line_bounds.py report.pdf --csv out.csv

Dependencies:
    pip install pdfplumber pandas
"""
import argparse
import pathlib
import sys

import pdfplumber
import pandas as pd

def list_line_bounds(pdf_path: pathlib.Path) -> pd.DataFrame:
    rows = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for pnum, page in enumerate(pdf.pages, start=1):
            pw = page.width
            ph = page.height

            words = page.extract_words(use_text_flow=True)
            lines = {}
            for w in words:
                y = round(w["top"], 1)
                lines.setdefault(y, []).append(w)

            for y, ws in sorted(lines.items()):
                # absolute coords
                x0s = [float(w["x0"]) for w in ws]
                x1s = [float(w["x1"]) for w in ws]
                left, right = min(x0s), max(x1s)
                # normalized
                rel_x0 = left  / pw
                rel_x1 = right / pw
                rel_y  = y     / ph

                # centered if rel_x0 > 0.275 AND rel_x1 < 0.700
                centered = int(rel_x0 > 0.275 and rel_x1 < 0.700)

                text = " ".join(w["text"] for w in sorted(ws, key=lambda w: w["x0"]))
                rows.append({
                    "page":     pnum,
                    "rel_x0":   round(rel_x0, 3),
                    "rel_x1":   round(rel_x1, 3),
                    "rel_y":    round(rel_y, 3),
                    "centered": centered,
                    "text":     text
                })

    return pd.DataFrame(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("pdf", help="path to the PDF file")
    p.add_argument("--csv", help="save the results to CSV")
    p.add_argument("--txt", help="save the results to a text file (Markdown)")
    args = p.parse_args()

    pdf_file = pathlib.Path(args.pdf)
    if not pdf_file.exists():
        sys.exit(f"❌ File not found: {pdf_file}")

    df = list_line_bounds(pdf_file)
    if df.empty:
        print("No lines found.")
        return

    # Print to console
    print("\nNormalized line bounds (with centered flag):\n")
    print(df.to_markdown(index=False))

    # Optionally save CSV
    if args.csv:
        df.to_csv(args.csv, index=False, encoding="utf-8")
        print(f"\n✅ Saved CSV → {args.csv}")

    # Optionally save plain-text (Markdown table)    
    if args.txt:
        with open(args.txt, "w", encoding="utf-8") as f:
            f.write(df.to_markdown(index=False))
        print(f"\n✅ Saved text → {args.txt}")

if __name__ == "__main__":
    main()
