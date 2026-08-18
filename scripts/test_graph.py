import os
import sys
import warnings

# Ensure Python can find the modules from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.loader import load_and_validate_documents
from ingestion.chunker import build_chunks
from ingestion.extractor import extract_graph_from_chunk
from rag.graph import get_neo4j_driver, create_constraints, insert_graph_data

def run_phase_5():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    
    print("=" * 70)
    print("PHASE 5: KNOWLEDGE GRAPH INSERTION (NEO4J via GEMINI)")
    print("=" * 70)
    
    print("1. Connecting to Neo4j AuraDB...")
    try:
        driver = get_neo4j_driver()
        create_constraints(driver)
    except Exception as e:
        print(f"[ERROR] Failed to connect to Neo4j: {e}")
        return

    print("\n2. Processing Documents...")
    # Suppress output from earlier phases
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

    test_chunk = chunks[0]
    print(f"   -> Processing Chunk ID: {test_chunk['chunk_id']}")
    
    print("\n3. Extracting Entities & Relationships (Gemini 1.5 Flash)...")
    graph_data = extract_graph_from_chunk(test_chunk['text'])
    
    valid_entities = len(graph_data.get('entities', []))
    valid_rels = len(graph_data.get('relationships', []))
    print(f"   -> Extracted {valid_entities} valid Entities and {valid_rels} valid Relationships.")
    
    print("\n4. Inserting Data into Neo4j...")
    try:
        insert_graph_data(driver, test_chunk, graph_data)
        print("   -> ✅ Insertion Successful!")
    except Exception as e:
        print(f"   -> ❌ Insertion Failed: {e}")
    finally:
        driver.close()

    print("\n" + "=" * 70)
    print("Go to your Neo4j Aura Console and run this Cypher query:")
    print("MATCH (n) RETURN n LIMIT 50")
    print("=" * 70)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    run_phase_5()