from pathlib import Path
import re

import pymupdf

from config import (
    PDF_DIR,
    OCR_TEXT_THRESHOLD,
)


def normalize_text(text: str) -> str:
    """
    Remove whitespace so we can accurately estimate
    whether a PDF page contains meaningful text.
    """
    return re.sub(r"\s+", "", text or "")


def audit_pdf(pdf_path: Path) -> dict:
    """
    Audit one PDF and determine which pages may require OCR.
    """

    doc = pymupdf.open(pdf_path)

    total_chars = 0
    ocr_pages = []

    for page_number, page in enumerate(doc, start=1):

        text = page.get_text("text")
        clean_text = normalize_text(text)

        char_count = len(clean_text)

        total_chars += char_count

        if char_count < OCR_TEXT_THRESHOLD:
            ocr_pages.append(page_number)

    result = {
        "file": pdf_path.name,
        "pages": len(doc),
        "text_chars": total_chars,
        "ocr_pages": ocr_pages,
        "ocr_count": len(ocr_pages),
    }

    doc.close()

    return result


def main():

    if not PDF_DIR.exists():
        print(f"ERROR: PDF directory does not exist: {PDF_DIR}")
        return

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {PDF_DIR}")
        return

    print("\n=== PDF AUDIT ===\n")

    results = []

    for pdf in pdf_files:

        result = audit_pdf(pdf)

        results.append(result)

        print(
            f"{result['file']}\t"
            f"pages={result['pages']}\t"
            f"text_chars={result['text_chars']}\t"
            f"ocr_pages={result['ocr_count']}"
        )

        if result["ocr_pages"]:
            print(
                f"  → OCR required on pages: "
                f"{result['ocr_pages']}"
            )

    print("\n=== SUMMARY ===\n")

    total_pages = sum(
        r["pages"] for r in results
    )

    total_chars = sum(
        r["text_chars"] for r in results
    )

    total_ocr = sum(
        r["ocr_count"] for r in results
    )

    print(f"PDF files : {len(results)}")
    print(f"Total pages: {total_pages}")
    print(f"Text chars : {total_chars}")
    print(f"OCR pages  : {total_ocr}")

    print("\n=== OCR STATUS ===\n")

    if total_ocr == 0:
        print(
            "No pages currently require OCR."
        )
    else:
        print(
            f"{total_ocr} page(s) will be sent "
            f"to Hugging Face OCR."
        )


if __name__ == "__main__":
    main()