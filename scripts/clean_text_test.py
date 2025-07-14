import argparse
import unicodedata
from pathlib import Path
from bs4 import BeautifulSoup
from readability import Document
import re

def clean_html_report(raw_html: str) -> str:
    # Read and parse HTML
    soup = BeautifulSoup(raw_html, 'html.parser')
    
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

# Narrative extraction via Readability
def extract_narrative(html: str) -> str:
    doc = Document(html)
    summary_html = doc.summary()
    return BeautifulSoup(summary_html, "html.parser").get_text("\n", strip=True)

# Merge table cells into sentences
def merge_tables(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    sentences = []
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
            if len(cells) >= 2:
                pairs = []
                for i in range(0, len(cells) - 1, 2):
                    pairs.append(f"{cells[i]}: {cells[i+1]}")
                sentences.append("; ".join(pairs))
    return "\n".join(sentences)

# Normalize Hebrew: strip niqqud, collapse whitespace
_DIACRITICS = {chr(c) for c in range(0x0591, 0x05C8)}
def clean_he(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if ch not in _DIACRITICS)
    return " ".join(text.split())

def main():
    parser = argparse.ArgumentParser(description="Merge table cells and narrative for Hebrew HTML.")
    parser.add_argument("input_file", type=Path, help="Path to HTML/HTM input file")
    parser.add_argument("output_file", type=Path, help="Path to write cleaned text")
    parser.add_argument("--encoding", default="windows-1255", help="File encoding")
    args = parser.parse_args()

    raw_html = args.input_file.read_text(encoding=args.encoding, errors="ignore")

    # 1) Clean header/footer
    trimmed_text = clean_html_report(raw_html)
    
    # 2) Merge tables from original HTML (so we keep the <table> structure)
    table_text = merge_tables(trimmed_text)

    # 3) Extract narrative prose from cleaned HTML
    narrative_text = extract_narrative(trimmed_text)

    # 4) Combine, then normalize Hebrew
    combined = "\n\n".join(filter(None, [table_text, narrative_text]))
    final = clean_he(combined)

    # 5) Write out
    args.output_file.write_text(final, encoding="utf-8")
    print(f"Written cleaned text to {args.output_file}")

if __name__ == "__main__":
    main()