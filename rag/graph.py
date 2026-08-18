import os
import sys
from neo4j import GraphDatabase

# Ensure Python can find the 'config' module from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

def get_neo4j_driver():
    """Initializes and returns the Neo4j driver."""
    if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
        raise ValueError("[ERROR] Neo4j credentials are missing in .env/config.py")
    
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def create_constraints(driver):
    """
    Creates unique constraints to ensure we don't duplicate Documents, Chunks, or Entities.
    """
    queries = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE"
    ]
    with driver.session() as session:
        for query in queries:
            session.run(query)
    print("[INFO] Neo4j Constraints verified successfully.")

def insert_graph_data(driver, chunk_metadata: dict, graph_data: dict):
    """
    Inserts Document, Chunk, Entities, and Relationships into Neo4j.
    Uses MERGE to avoid duplication.
    """
    doc_id = chunk_metadata["document_name"]
    chunk_id = chunk_metadata["chunk_id"]

    with driver.session() as session:
        # 1. Insert Document and Chunk, then link them
        session.run("""
            MERGE (d:Document {id: $doc_id})
            MERGE (c:Chunk {id: $chunk_id})
            SET c.text = $text, 
                c.page_number = $page, 
                c.section_title = $section
            MERGE (d)-[:HAS_CHUNK]->(c)
        """, doc_id=doc_id, chunk_id=chunk_id, 
             text=chunk_metadata.get("text", ""), 
             page=chunk_metadata.get("page_number", -1),
             section=chunk_metadata.get("section_title", ""))

        # 2. Insert Entities and link them to the Chunk (Provenance)
        for ent in graph_data.get("entities", []):
            ent_name = ent["name"].lower().strip() # Normalize to lowercase to improve merging
            ent_type = ent["type"]
            
            # Dynamically set the specific medical label (e.g., :Disease) along with generic :Entity
            query = f"""
            MERGE (e:Entity {{id: $ent_name}})
            SET e:{ent_type}, e.name = $ent_name, e.type = $ent_type
            WITH e
            MATCH (c:Chunk {{id: $chunk_id}})
            MERGE (c)-[:MENTIONS]->(e)
            """
            session.run(query, ent_name=ent_name, chunk_id=chunk_id, ent_type=ent_type)

        # 3. Insert Medical Relationships between Entities
        for rel in graph_data.get("relationships", []):
            src_name = rel["source"].lower().strip()
            tgt_name = rel["target"].lower().strip()
            rel_type = rel["type"] # e.g., REDUCES_RISK_OF
            
            # Dynamically set the relationship type
            query = f"""
            MATCH (source:Entity {{id: $src_name}})
            MATCH (target:Entity {{id: $tgt_name}})
            MERGE (source)-[:{rel_type}]->(target)
            """
            session.run(query, src_name=src_name, tgt_name=tgt_name)