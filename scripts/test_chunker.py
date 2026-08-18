import os
import sys

# Ensure Python can find the 'ingestion' module from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.loader import load_and_validate_documents
from ingestion.chunker import build_chunks

def run_phase_3():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    
    print("=" * 70)
    print("PHASE 3: TEXT CLEANING & SECTION-AWARE CHUNKING")
    print("=" * 70)
    
    print("1. Loading documents from Phase 2 (PyMuPDF + HF OCR)...")
    try:
        results = load_and_validate_documents(data_dir)
        valid_pages = results["valid_pages"]
        print(f"   -> Successfully extracted {len(valid_pages)} valid pages.\n")
    except Exception as e:
        print(f"[ERROR] Loading failed: {e}")
        return

    print("2. Cleaning, Sectionizing, and Chunking...")
    chunks = build_chunks(valid_pages)
    
    if not chunks:
        print("[ERROR] No chunks were created.")
        return
        
    token_counts = [c["token_count"] for c in chunks]
    
    print("\n" + "=" * 70)
    print("FINAL PHASE 3 SUMMARY")
    print("=" * 70)
    print(f"Total Chunks Created: {len(chunks)}")
    print(f"Token Range:          {min(token_counts)} - {max(token_counts)} tokens")
    print(f"Average Tokens/Chunk: {sum(token_counts) // len(token_counts)}")
    print("=" * 70)
    
    print("\n--- Validation Check: Sample Chunk Metadata ---")
    sample_chunk = chunks[0] # Preview the first chunk
    for key, value in sample_chunk.items():
        if key == "text":
            print(f"\nText Preview:\n{value[:300]}...")
        else:
            # Print keys in a clean format
            formatted_key = key.replace('_', ' ').title()
            print(f"{formatted_key.ljust(20)}: {value}")

if __name__ == "__main__":
    run_phase_3()