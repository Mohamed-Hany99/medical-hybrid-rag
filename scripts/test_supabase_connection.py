import os
import random
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def test_supabase():
    print("1. Testing connection to Supabase...")
    
    # 768-dimensional dummy vector
    dummy_embedding = [random.uniform(-0.1, 0.1) for _ in range(768)]
    
    test_chunk = {
        "id": "test_chunk_001",
        "chunk_text": "Diabetes mellitus is a metabolic disorder characterized by elevated blood glucose.",
        "source_page": "1",
        "source_document": "sample_medical_doc.pdf",
        "embedding": dummy_embedding
    }
    
    # Insert / Upsert test record
    print("2. Inserting test chunk...")
    insert_res = supabase.table("medical_chunks").upsert(test_chunk).execute()
    print(f"   [SUCCESS] Inserted: {insert_res.data}")
    
    # Test RPC Search Function
    print("3. Testing match_medical_chunks RPC function...")
    rpc_res = supabase.rpc(
        "match_medical_chunks",
        {
            "query_embedding": dummy_embedding,
            "match_threshold": 0.5,
            "match_count": 1
        }
    ).execute()
    
    print(f"   [SUCCESS] Match result: {rpc_res.data}")
    
    # Cleanup dummy record (optional)
    print("4. Cleaning up test data...")
    supabase.table("medical_chunks").delete().eq("id", "test_chunk_001").execute()
    print("   [SUCCESS] Test chunk deleted. Supabase is fully configured!")

if __name__ == "__main__":
    test_supabase()