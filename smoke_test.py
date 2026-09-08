import sys
from pathlib import Path

from src.extraction.pdf_extractor import PDFExtractor


def smoke_test(pdf_paths):
    extractor = PDFExtractor()

    for path in pdf_paths:
        print("\n" + "=" * 80)
        print(path)
        print("=" * 80)

        pages = extractor.extract_text(path)

        page = pages[0]
        chunks = extractor._chunk_page(page["text"])

        print(f"Page text: {len(page['text']):,} characters")
        print(f"Chunks on page 1: {len(chunks)}")

        facts_found = 0

        for i, chunk in enumerate(chunks, start=1):
            print(f"\n--- chunk {i}/{len(chunks)} ---")

            raw_facts = extractor._extract_facts_from_chunk(chunk)

            print(f"Raw facts returned: {len(raw_facts)}")

            for raw in raw_facts:
                fact = extractor._build_fact(
                    raw,
                    chunk,
                    page["page"],
                    path.split("/")[-1],
                )

                if fact:
                    facts_found += 1
                    print(f"FACT: {fact.text}")
                    print(f"  type:       {fact.fact_type.value}")
                    print(f"  value:      {fact.value}")
                    print(f"  unit:       {fact.unit}")
                    print(f"  confidence: {fact.confidence}")
                    print(f"  evidence:   {fact.source_snippet}")

        print(f"\nGrounded facts from first page: {facts_found}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python smoke_test.py <pdf_path> [pdf_path2 ...]")
        print("       python smoke_test.py <directory>")
        sys.exit(1)

    paths = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.pdf")))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"Warning: skipping {arg} (not a file or directory)")

    if not paths:
        print("No PDF files found.")
        sys.exit(1)

    smoke_test(str(p) for p in paths)
