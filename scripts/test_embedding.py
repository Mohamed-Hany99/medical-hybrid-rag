import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Missing HF_TOKEN in .env file")

MODEL_ID = "BAAI/bge-base-en-v1.5"

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)

def test_hf_embedding():
    print(f"1. Testing Hugging Face Feature Extraction ({MODEL_ID})...")
    
    sample_text = "Diabetes mellitus is characterized by chronic hyperglycemia."
    
    # Generate Embedding
    embedding = client.feature_extraction(
        sample_text,
        model=MODEL_ID
    )
    
    # Handle response format (in case returned as list of floats or nested list)
    if isinstance(embedding, list) and isinstance(embedding[0], list):
        embedding = embedding[0]
        
    dim_len = len(embedding)
    print(f"   [SUCCESS] Received vector type: {type(embedding)}")
    print(f"   [SUCCESS] Vector Dimension: {dim_len}")
    
    if dim_len == 768:
        print("   [CONFIRMED] Dimension matches Supabase schema (768) perfectly!")
    else:
        print(f"   [WARNING] Unexpected dimension length: {dim_len}")

if __name__ == "__main__":
    test_hf_embedding()