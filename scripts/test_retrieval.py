import os
from dotenv import load_dotenv
from supabase import create_client, Client
from huggingface_hub import InferenceClient

load_dotenv()

# --- Config ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "BAAI/bge-base-en-v1.5"

# --- Clients ---
if not all([SUPABASE_URL, SUPABASE_KEY, HF_TOKEN]):
    raise ValueError("Missing credentials in .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
hf_client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

def get_query_embedding(query: str):
    """Generate embedding for the user query."""
    print(f"1. Generating embedding for query: '{query}'...")
    res = hf_client.feature_extraction(query, model=MODEL_ID)
    
    # Extract flat list of floats
    if hasattr(res, "tolist"):
        vec = res.tolist()
    else:
        vec = list(res)
    if isinstance(vec[0], list):
        vec = vec[0]
        
    return [float(x) for x in vec]

def search_supabase(query: str, top_k: int = 3):
    print("="*70)
    print("TESTING SUPABASE VECTOR RETRIEVAL")
    print("="*70)
    
    # 1. Get embedding for the question
    query_vector = get_query_embedding(query)
    
    # 2. Call the RPC function in Supabase
    print(f"2. Searching Supabase for top {top_k} matches...")
    response = supabase.rpc(
        "match_medical_chunks",
        {
            "query_embedding": query_vector,
            "match_threshold": 0.3, # أقل نسبة تشابه مقبولة
            "match_count": top_k
        }
    ).execute()
    
    # 3. Print Results
    results = response.data
    
    if not results:
        print("\n[INFO] No relevant chunks found above the threshold.")
        return

    print(f"\n[SUCCESS] Found {len(results)} relevant chunks:\n")
    
    for i, match in enumerate(results, 1):
        similarity = match.get('similarity', 0) * 100
        doc_name = match.get('source_document', 'Unknown')
        page_num = match.get('source_page', '?')
        
        print(f"--- Match {i} (Similarity: {similarity:.2f}%) ---")
        print(f"Source: {doc_name} | Page: {page_num}")
        print(f"Text: {match.get('chunk_text')}\n")

if __name__ == "__main__":
    # غيّر السؤال ده بأي سؤال طبي له علاقة بمحتوى الـ PDF اللي رفعته
    test_query = "What are the primary risk factors for cardiovascular diseases according to the document?"
    search_supabase(test_query)