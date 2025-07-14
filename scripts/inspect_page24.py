#!/usr/bin/env python
"""
inspect_page24.py

Use this to see what Docling 2.x produces *after* OCR on page 24,
and write the dump to a text file.
"""
import sys
from pathlib import Path
from docling.document_converter import (
    DocumentConverter, PdfFormatOption, InputFormat
)
from docling.datamodel.pipeline_options import PdfPipelineOptions

def page_of(obj) -> int:
    # 0-based page index helper
    if hasattr(obj, "page_index"):
        return obj.page_index
    if hasattr(obj, "page_no"):
        return obj.page_no
    if hasattr(obj, "prov") and obj.prov:
        prov = obj.prov[0]
        if hasattr(prov, "page_no"):
            return prov.page_no
    raise AttributeError("no page info on item")

def main(pdf_path: str, out_path: str):
    pdf_path = Path(pdf_path)
    out_path = Path(out_path)

    # 1️⃣ Enable full OCR so items.text contains Hebrew
    opts = PdfPipelineOptions(vision="full", ocr=True)
    conv = DocumentConverter(
        format_options={ InputFormat.PDF: PdfFormatOption(pipeline_options=opts) }
    )
    doc = conv.convert(str(pdf_path)).document

    # 2️⃣ Collect every heading-like item on page 24 (index 23)
    target = 24
    lines = [f"### OCR’d Docling items on physical page {target+1}\n"]
    for item, _ in doc.iterate_items():  # Docling 2.x
        try:
            if page_of(item) != target:
                continue
        except AttributeError:
            continue

        # show only section/title items
        label = item.label.name if hasattr(item.label, "name") else str(item.label)


        text = (getattr(item, "text", None) or getattr(item, "content", "")).strip()
        lines.append(f"[{label:>15}]  {text}\n")

    # 3️⃣ Print to console
    print("".join(lines))

    # 4️⃣ Write to file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"✅ written inspect output → {out_path}")

if __name__ == "__main__":
    if not (2 <= len(sys.argv) <= 3):
        print("Usage: python inspect_page24.py path/to/your.pdf [path/to/out.txt]")
        sys.exit(1)
    pdf_file = sys.argv[1]
    if len(sys.argv) == 3:
        out_file = sys.argv[2]
    else:
        # default: place next to the PDF
        base = Path(pdf_file).with_suffix("")
        out_file = base / "page24_inspect.txt"
    main(pdf_file, out_file)
