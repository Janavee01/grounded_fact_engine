import pdfplumber
import sys

def debug_pdf(pdf_path):
    print(f"\nAnalyzing: {pdf_path}\n")
    print("="*60)
    
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}\n")
        
        for page_num, page in enumerate(pdf.pages[:5], 1):  
            text = page.extract_text()
            if text:
                print(f"Page {page_num} (first 500 chars):")
                print("-"*40)
                print(text[:500])
                print("-"*40)
                print(f"Total chars on page: {len(text)}\n")
            else:
                print(f"Page {page_num}: No text extracted!\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_pdf(sys.argv[1])
    else:
        print("Usage: python debug_pdf.py <path_to_pdf>")