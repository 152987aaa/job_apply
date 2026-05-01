import PyPDF2
import sys

pdf_path = r'd:\Project_show_job\work\机器学习-实验六.pdf'

with open(pdf_path, 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    print(f"Total pages: {len(reader.pages)}")
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            print(f"\n=== Page {i+1} ===")
            print(text)
