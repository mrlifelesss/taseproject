#!/usr/bin/env python
"""
inspect_page24.py

Use this to see what Docling 2.x produces *after* OCR on page 24,
so you can discover the exact label and text.
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

def main(pdf_path):
    pdf_path = Path(pdf_path)
    # 1️⃣  Enable full OCR so items.text contains Hebrew
    opts = PdfPipelineOptions(vision="full", ocr=True)
    conv = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
        }
    )
    doc = conv.convert(str(pdf_path)).document

    # 2️⃣  Dump every heading-like item on page 24 (index 23)
    target = 23
    print(f"\n### OCR’d Docling items on physical page {target+1}\n")
    for item, _ in doc.iterate_items():  # Docling 2.x
        try:
            if page_of(item) != target:
                continue
        except AttributeError:
            continue

        # show only section/title items for clarity
        label = item.label.name if hasattr(item.label, "name") else str(item.label)
        if label not in {"SECTION_HEADER", "SUBSECTION_HEADER", "CHAPTER_TITLE", "PART_HEADER", "TITLE"}:
            continue

        text = (getattr(item, "text", None) or getattr(item, "content", "")).strip()
        print(f"[{label:>15}]  {text}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_page24.py path/to/your.pdf")
        sys.exit(1)
    main(sys.argv[1])
