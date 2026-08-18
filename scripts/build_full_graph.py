import os
import sys
import time
import warnings

# Ensure Python can find the modules from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.loader import load_and_validate_documents
from ingestion.chunker import build_chunks
from ingestion.extractor import extract_graph_from_chunk
from rag.graph import get_neo4j_driver, create_constraints, insert_graph_data

def get_processed_chunks(driver):
    """Fetches the IDs of chunks that are already inserted into Neo4j."""
    with driver.session() as session:
        result = session.run("MATCH (c:Chunk) RETURN c.id AS chunk_id")
        return set(record["chunk_id"] for record in result)

def build_full_knowledge_graph():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    
    print("=" * 70)
    print("PHASE 5.5: RESUMING FULL KNOWLEDGE GRAPH CONSTRUCTION (NEO4J via GROQ)")
    print("=" * 70)
    
    print("1. Connecting to Neo4j AuraDB...")
    try:
        driver = get_neo4j_driver()
        create_constraints(driver)
        
        # --- NEW: Get already processed chunks ---
        processed_chunk_ids = get_processed_chunks(driver)
        print(f"[INFO] Found {len(processed_chunk_ids)} chunks already in Neo4j. These will be skipped.")
        
    except Exception as e:
        print(f"[ERROR] Failed to connect to Neo4j: {e}")
        return

    print("\n2. Loading and Chunking ALL Documents...")
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

    # --- FILTER CHUNKS TO ONLY PROCESS NEW ONES ---
    chunks_to_process = [c for c in chunks if c['chunk_id'] not in processed_chunk_ids]

    if not chunks_to_process:
        print(f"\n[INFO] All {len(chunks)} chunks have already been processed and inserted! Nothing to do.")
        driver.close()
        return

    print(f"[INFO] Found {len(chunks_to_process)} NEW chunks to process out of {len(chunks)} total.\n")
    
    total_entities = 0
    total_relationships = 0

    # Loop through NEW chunks
    for i, chunk in enumerate(chunks_to_process):
        print(f"--- Processing New Chunk {i+1}/{len(chunks_to_process)} | ID: {chunk['chunk_id']} ---")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 1. Extract using Groq
                graph_data = extract_graph_from_chunk(chunk['text'])
                
                valid_entities = len(graph_data.get('entities', []))
                valid_rels = len(graph_data.get('relationships', []))
                
                # 2. Insert into Neo4j
                insert_graph_data(driver, chunk, graph_data)
                
                total_entities += valid_entities
                total_relationships += valid_rels
                
                print(f"    ✅ Success: Inserted {valid_entities} Entities & {valid_rels} Relationships.")
                break  # Break out of the retry loop if successful
                
            except Exception as e:
                print(f"    ⚠️ Attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    print("    ⏳ Waiting 5 seconds before retrying to avoid server overload...")
                    time.sleep(5)
                else:
                    print(f"    ❌ Failed completely to process chunk {chunk['chunk_id']} after {max_retries} attempts.")
        
        # API Rate Limiting pause for Groq
        if i < len(chunks_to_process) - 1:
            time.sleep(1.5)

    driver.close()
    print("\n" + "=" * 70)
    print(f"🎉 FULL GRAPH BUILT SUCCESSFULLY (RESUMED)!")
    print(f"Total New Entities Inserted: {total_entities}")
    print(f"Total New Relationships Inserted: {total_relationships}")
    print("=" * 70)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    build_full_knowledge_graph()