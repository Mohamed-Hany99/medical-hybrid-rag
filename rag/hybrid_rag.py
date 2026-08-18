import os
import sys
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client
from neo4j import GraphDatabase
from huggingface_hub import InferenceClient
from groq import Groq

# Load environment variables
load_dotenv()

# --- Configurations & Keys ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://your-neo4j-aura-instance.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "openai/gpt-oss-120b"

# --- Clients Initializations ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
hf_client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)
groq_client = Groq(api_key=GROQ_API_KEY)
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


# ============================================================
# 1. Supabase Vector Retrieval
# ============================================================
def get_query_embedding(query: str) -> List[float]:
    """Generate 768-dim query embedding."""
    res = hf_client.feature_extraction(query, model=EMBEDDING_MODEL)
    if hasattr(res, "tolist"):
        vec = res.tolist()
    else:
        vec = list(res)
    if isinstance(vec[0], list):
        vec = vec[0]
    return [float(x) for x in vec]


def retrieve_vector_context(query: str, top_k: int = 3, threshold: float = 0.4) -> List[Dict[str, Any]]:
    """Retrieve top-k semantically relevant chunks from Supabase."""
    query_vector = get_query_embedding(query)
    response = supabase.rpc(
        "match_medical_chunks",
        {
            "query_embedding": query_vector,
            "match_threshold": threshold,
            "match_count": top_k
        }
    ).execute()
    return response.data or []


# ============================================================
# 2. Neo4j Knowledge Graph Retrieval
# ============================================================
def extract_query_keywords(query: str) -> List[str]:
    """Use fast LLM prompt to extract core entities/keywords from user query."""
    prompt = f"""Extract main medical entity names (diseases, symptoms, diets, metrics, therapies) from this query.
Return ONLY a JSON list of lowercase strings.
Query: {query}
Example output: ["diabetes", "hypertension"]"""
    
    try:
        res = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        data = json.loads(res.choices[0].message.content)
        return [k.lower() for k in data.get("entities", data.get("keywords", list(data.values())[0] if data else []))]
    except Exception:
        # Fallback to simple split if entity extraction fails
        return [w.lower() for w in query.split() if len(w) > 4]


def retrieve_graph_context(query: str) -> List[str]:
    """Query Neo4j for relationships connected to entities extracted from query."""
    keywords = extract_query_keywords(query)
    if not keywords:
        return []

    graph_facts = []
    cypher_query = """
    UNWIND $keywords AS kw
    MATCH (source)-[r]->(target)
    WHERE toLower(source.name) CONTAINS kw OR toLower(target.name) CONTAINS kw
    RETURN source.name AS src, type(r) AS rel, target.name AS tgt, labels(source)[0] AS src_type, labels(target)[0] AS tgt_type
    LIMIT 15
    """
    
    try:
        with neo4j_driver.session() as session:
            records = session.run(cypher_query, keywords=keywords)
            for rec in records:
                fact = f"({rec['src']} : {rec['src_type']}) -[{rec['rel']}]-> ({rec['tgt']} : {rec['tgt_type']})"
                graph_facts.append(fact)
    except Exception as e:
        print(f"[WARN] Neo4j query error: {e}")

    return graph_facts


# ============================================================
# 3. Hybrid Context Fusion & Generation
# ============================================================
def generate_hybrid_answer(query: str) -> str:
    print(f"\n[1/3] Retrieving semantic vectors from Supabase...")
    vector_results = retrieve_vector_context(query, top_k=3)
    
    print(f"[2/3] Retrieving relational facts from Neo4j...")
    graph_facts = retrieve_graph_context(query)
    
    # Build Structured Context
    vector_context_text = ""
    sources = set()
    for idx, match in enumerate(vector_results, 1):
        doc = match.get("source_document", "Unknown")
        page = match.get("source_page", "?")
        sources.add(f"{doc} (Page {page})")
        vector_context_text += f"\n--- Text Excerpt {idx} [{doc} | Page {page}] ---\n{match.get('chunk_text')}\n"

    graph_context_text = "\n".join([f"- {fact}" for fact in graph_facts]) if graph_facts else "No direct graph relationships found."

    system_prompt = """You are an expert Medical AI Assistant. 
Answer the user's question using ONLY the provided Knowledge Graph facts and Text Excerpts.
Be clear, clinically accurate, and mention the sources and relevant relations when answering.
If the context does not contain the answer, state that explicitly without fabricating facts."""

    user_message = f"""USER QUESTION:
{query}

============================================================
KNOWLEDGE GRAPH FACTS (Structured Relationships):
============================================================
{graph_context_text}

============================================================
TEXT EXCERPTS (Semantic Context):
============================================================
{vector_context_text}

Provide a comprehensive, well-structured answer with a citation list at the bottom."""

    print(f"[3/3] Generating final synthesis with Groq ({LLM_MODEL})...\n")
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    test_question = "What are the primary risk factors for cardiovascular diseases and how does diet help prevent them?"
    print("=" * 70)
    print(f"QUERY: {test_question}")
    print("=" * 70)
    
    final_output = generate_hybrid_answer(test_question)
    print(final_output)
    
    neo4j_driver.close()