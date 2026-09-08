from pathlib import Path
import sys

from src.extraction.pdf_extractor import PDFExtractor
from src.storage.database import Database


def ingest_directory(directory: str):
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pdfs = sorted(directory_path.glob("*.pdf"))

    if not pdfs:
        raise ValueError(f"No PDF files found in {directory}")

    extractor = PDFExtractor()
    db = Database()

    total_facts = 0
    failed = []

    print(f"Found {len(pdfs)} PDF(s) in {directory}")
    print()

    for index, pdf_path in enumerate(pdfs, start=1):
        print(f"[{index}/{len(pdfs)}] Processing: {pdf_path.name}")

        try:
            facts = extractor.extract_facts(
                str(pdf_path),
                source_document=pdf_path.name,
            )

            print(f"       Extracted: {len(facts)} facts")

            for fact in facts:
                db.save_fact(fact)

            total_facts += len(facts)
        except Exception as e:
            print(f"       FAILED: {e}")
            failed.append((pdf_path.name, str(e)))
            continue

    print()
    print("=" * 60)
    print(f"Documents processed: {len(pdfs) - len(failed)}/{len(pdfs)}")
    print(f"Total facts extracted: {total_facts}")
    if failed:
        print(f"Failed documents ({len(failed)}):")
        for name, error in failed:
            print(f"  - {name}: {error}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.ingest <directory>")
        sys.exit(1)

    ingest_directory(sys.argv[1])
