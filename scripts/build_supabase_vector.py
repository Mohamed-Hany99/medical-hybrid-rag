import os
import sys
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
import numpy as np
from supabase import create_client, Client
from huggingface_hub import InferenceClient

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.loader import load_and_validate_documents
from ingestion.chunker import build_chunks

load_dotenv()

# --- Config & Clients ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not all([SUPABASE_URL, SUPABASE_KEY, HF_TOKEN]):
    raise ValueError("Missing SUPABASE_URL, SUPABASE_KEY, or HF_TOKEN in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
hf_client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

MODEL_ID = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 10


def get_embedding(text: str) -> List[float]:
    """Generate 768-dim embedding using Hugging Face Inference API."""
    try:
        raw_res = hf_client.feature_extraction(text, model=MODEL_ID)
        
        if isinstance(raw_res, np.ndarray):
            vec = raw_res.tolist()
        else:
            vec = list(raw_res)
            
        if isinstance(vec[0], list):
            vec = vec[0]
            
        return [float(x) for x in vec]
    except Exception as e:
        print(f"\n[ERROR] Failed to generate embedding: {e}")
        time.sleep(2)
        return []


def ingest_chunks_to_supabase():
    print("=" * 70)
    print("PHASE 5.6: SUPABASE VECTOR INGESTION (BAAI/bge-base-en-v1.5)")
    print("=" * 70)

    # 1. Loading and Chunking
    print("1. Loading and Chunking ALL Documents...")
    
    try:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        raw_docs = load_and_validate_documents(data_dir)
        
        # Pass valid_pages to the chunker
        all_chunks = build_chunks(raw_docs["valid_pages"])
        
    except Exception as e:
        print(f"[WARN] Standard loading failed ({e}). Checking pages structure...")
        raise e

    total_chunks = len(all_chunks)
    print(f"[INFO] Found {total_chunks} chunks to process. Starting embedding and insertion...\n")

    # 2. Embedding & Ingestion
    success_count = 0
    records_to_upload: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(all_chunks, 1):
        # Extract data specifically based on the chunker's output keys
        chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
        chunk_text = chunk.get("text", "")
        source_doc = chunk.get("document_name", "unknown.pdf")
        source_page = str(chunk.get("page_number", "1"))

        print(f"[{idx}/{total_chunks}] Processing chunk: {chunk_id}...", end="\r")

        embedding = get_embedding(chunk_text)
        if not embedding or len(embedding) != 768:
            print(f"\n[WARN] Skipping chunk {chunk_id} (invalid embedding dimension).")
            continue

        records_to_upload.append({
            "id": str(chunk_id),
            "chunk_text": chunk_text,
            "source_document": source_doc,
            "source_page": source_page,
            "embedding": embedding
        })

        if len(records_to_upload) >= BATCH_SIZE or idx == total_chunks:
            try:
                supabase.table("medical_chunks").upsert(records_to_upload).execute()
                success_count += len(records_to_upload)
                records_to_upload = []
            except Exception as e:
                print(f"\n[ERROR] Failed to upsert batch to Supabase: {e}")

    print("\n" + "=" * 70)
    print(f"[COMPLETED] Successfully embedded and inserted {success_count}/{total_chunks} chunks into Supabase!")
    print("=" * 70)


if __name__ == "__main__":
    ingest_chunks_to_supabase()