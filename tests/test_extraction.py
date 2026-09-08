import sys

from src.extraction.pdf_extractor import PDFExtractor


def run_extraction(pdf_path):
    print(f"Testing extraction on: {pdf_path}\n")
    print("=" * 60)

    extractor = PDFExtractor()

    # Extract text first
    print("Extracting text...")
    pages = extractor.extract_text(pdf_path)
    print(f"Extracted {len(pages)} pages\n")

    # Show sample text from first page
    if pages:
        print("Sample text (first 1000 chars):")
        print("-" * 40)
        print(pages[0]["text"][:1000])
        print("-" * 40)

    # Extract facts
    print("\nExtracting facts...")
    facts = extractor.extract_facts(
        pdf_path,
        pdf_path.split("/")[-1],
    )

    print(f"\nFound {len(facts)} facts\n")

    if facts:
        print("Sample facts:")
        print("-" * 40)

        for fact in facts[:5]:
            print(f"Type: {fact.fact_type}")
            print(f"Text: {fact.text[:100]}...")
            print(f"Page: {fact.source_page}")
            print(f"Confidence: {fact.confidence}")
            print("-" * 40)

    return facts


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        print("Usage: python tests/test_extraction.py <path_to_pdf>")
        sys.exit(1)

    run_extraction(pdf_path)