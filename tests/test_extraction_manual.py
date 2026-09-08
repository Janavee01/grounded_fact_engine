import sys
from pathlib import Path

from src.extraction.pdf_extractor import PDFExtractor


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m tests.test_extraction_manual <pdf_path>")
        sys.exit(1)

    PDF = sys.argv[1]

    extractor = PDFExtractor()

    pages = extractor.extract_text(PDF)

    print(f"Pages: {len(pages)}")

    page = pages[0]

    chunks = extractor._chunk_page(page["text"])

    print(f"Page characters: {len(page['text'])}")
    print(f"Chunks on page: {len(chunks)}")

    for i, chunk in enumerate(chunks[:5], start=1):

        print(f"\n--- CHUNK {i} ---")
        print(f"Characters: {len(chunk)}")

        raw_facts = extractor._extract_facts_from_chunk(chunk)

        print(f"Raw facts: {len(raw_facts)}")

        for raw in raw_facts:
            fact = extractor._build_fact(
                raw,
                chunk,
                page["page"],
                Path(PDF).name,
            )

            if fact:
                print("\nFACT")
                print("Claim:", fact.text)
                print("Type:", fact.fact_type)
                print("Value:", fact.value)
                print("Unit:", fact.unit)
                print("Context:", fact.context)
                print("Evidence:", fact.source_snippet)
                print("Confidence:", fact.confidence)
            else:
                print("\nRejected fact")


if __name__ == "__main__":
    main()