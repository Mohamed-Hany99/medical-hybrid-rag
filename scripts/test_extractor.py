import os
import sys
import json
import warnings
from dotenv import load_dotenv

# Ensure Python can find the 'ingestion' module from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.loader import load_and_validate_documents, create_hf_client
from ingestion.chunker import build_chunks
from ingestion.extractor import extract_graph_from_chunk

load_dotenv()

# قراءة الموديل المخصص للـ Extraction (تأكد من إضافته لملف .env)
HF_EXTRACTION_MODEL = os.getenv("HF_EXTRACTION_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")

def run_phase_4():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    
    print("=" * 70)
    print(f"PHASE 4: ENTITY & RELATIONSHIP EXTRACTION")
    print(f"USING HF MODEL: {HF_EXTRACTION_MODEL}")
    print("=" * 70)
    
    print("1. Initializing HF Client...")
    hf_client = create_hf_client()
    
    print("2. Running Pipeline (Load -> Clean -> Chunk)...")
    # Suppress output from earlier phases to keep the terminal clean
    with open(os.devnull, 'w') as devnull:
        original_stdout = sys.stdout
        sys.stdout = devnull
        try:
            results = load_and_validate_documents(data_dir)
            chunks = build_chunks(results["valid_pages"])
        finally:
            sys.stdout = original_stdout
    
    if not chunks:
        print("[ERROR] No chunks available.")
        return

    # Pick the first substantive chunk
    test_chunk = chunks[0]
    
    print("\n--- Testing Extraction on Chunk ---")
    print(f"Document: {test_chunk['document_name']}")
    print(f"Section:  {test_chunk['section_title']}")
    print(f"Tokens:   {test_chunk['token_count']}")
    print(f"Preview:  {test_chunk['text'][:150]}...")
    
    print("\n3. Sending to Hugging Face API for Extraction (Please wait)...")
    graph_data = extract_graph_from_chunk(hf_client, HF_EXTRACTION_MODEL, test_chunk['text'])
    
    print("\n--- LLM Extraction Result ---")
    print(json.dumps(graph_data, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 70)
    print("PHASE 4 VALIDATION CHECKLIST")
    print("=" * 70)
    print(f"Entities Extracted:      {len(graph_data.get('entities', []))}")
    print(f"Relationships Extracted: {len(graph_data.get('relationships', []))}")
    
    from ingestion.extractor import ALLOWED_ENTITY_TYPES, ALLOWED_RELATIONSHIPS
    invalid_entities = [e['type'] for e in graph_data.get('entities', []) if e.get('type') not in ALLOWED_ENTITY_TYPES]
    invalid_rels = [r['type'] for r in graph_data.get('relationships', []) if r.get('type') not in ALLOWED_RELATIONSHIPS]
    
    print(f"Invalid Entity Types:    {len(invalid_entities)} {invalid_entities}")
    print(f"Invalid Rel Types:       {len(invalid_rels)} {invalid_rels}")
    print("=" * 70)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    run_phase_4()