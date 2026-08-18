import os
import sys

# Ensure Python can find the 'ingestion' module from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.loader import load_and_validate_documents

def run_phase_2():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    
    print("=" * 70)
    print("PHASE 2: DYNAMIC PDF EXTRACTION & OCR VALIDATION")
    print("=" * 70)
    print(f"Scanning directory: {data_dir}\n")
    
    try:
        results = load_and_validate_documents(data_dir)
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        return

    stats = results["stats"]
    valid_pages = results["valid_pages"]
    
    print("--- Per-File Statistics ---")
    for detail in stats["file_details"]:
        print(f"Document: {detail['file_name']}")
        print(f"  - Status: {detail['status']}")
        if detail['status'] == "Success":
            print(f"  - Total Pages: {detail['pages']}")
            print(f"  - Normal Text (PyMuPDF): {detail['pymupdf_count']} pages")
            print(f"  - Fallback OCR (HF): {detail['ocr_count']} pages")
            print(f"  - Total Characters: {detail['characters']:,}")
        print("-" * 50)
        
    print("\n" + "=" * 70)
    print("FINAL PHASE 2 SUMMARY")
    print("=" * 70)
    print(f"PDF Files Processed     : {stats['total_pdfs']}")
    print(f"Total Pages Processed   : {stats['total_pages_processed']}")
    print(f"Pages via PyMuPDF       : {stats['total_pymupdf_pages']}")
    print(f"Pages via HF OCR        : {stats['total_ocr_pages']}")
    if stats['total_ocr_errors'] > 0:
        print(f"OCR Errors Encountered  : {stats['total_ocr_errors']}")
    print("=" * 70)
    
    print("\n--- Validation Check: PageRecord Previews ---")
    previewed_files = set()
    for record in valid_pages:
        src = record.document_name
        if src not in previewed_files:
            previewed_files.add(src)
            print(f"[{src} | Page: {record.page_number} | Method: {record.extraction_method}]")
            # Create a clean preview string without line breaks
            preview_text = record.text[:150].replace('\n', ' ')
            print(f"Preview: {preview_text}...\n")

if __name__ == "__main__":
    run_phase_2()