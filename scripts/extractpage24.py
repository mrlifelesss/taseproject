from PyPDF2 import PdfReader, PdfWriter
r = PdfReader("inputs/1664899.pdf")
w = PdfWriter()
w.add_page(r.pages[22])      # zero-based
with open("page22.pdf","wb") as f: w.write(f)
# now run your old script:
# python find_centered_sections.py page24.pdf
