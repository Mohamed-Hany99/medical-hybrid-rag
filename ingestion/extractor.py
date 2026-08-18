import os
import json
from typing import Dict, Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Set up Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

ALLOWED_ENTITY_TYPES = [
    "HealthMetric", "LifestyleBehavior", "DietPattern", "FoodItem", 
    "Disease", "Medication", "Therapy", "Recommendation"
]

ALLOWED_RELATIONSHIPS = [
    "LOWERS", "RAISES", "INCREASES_RISK_OF", "REDUCES_RISK_OF", 
    "ASSOCIATED_WITH", "RECOMMENDED_FOR", "EMPHASIZES", "LIMITS", 
    "TREATS", "ASSISTS_WITH", "IMPROVES", "WORSENS", "MEASURED_BY"
]

def extract_graph_from_chunk(chunk_text: str) -> Dict[str, Any]:
    """
    Extracts entities and relationships using Groq (GPT-OSS 120B).
    Forces strict JSON output and filters hallucinated types.
    """
    if not GROQ_API_KEY:
        print("[ERROR] GROQ_API_KEY is not set in .env")
        return {"entities": [], "relationships": []}

    prompt = f"""You are an expert medical data extraction algorithm. 
Extract medical entities and relationships from the input text and return a valid JSON object.

RULES:
1. ONLY extract explicitly stated facts. No assumptions.
2. Entities MUST use exact types: {', '.join(ALLOWED_ENTITY_TYPES)}.
3. Relationships MUST use exact types: {', '.join(ALLOWED_RELATIONSHIPS)}.

INPUT TEXT:
{chunk_text}

OUTPUT STRUCTURE:
{{
  "entities": [
    {{"name": "entity name", "type": "ExactTypeFromList"}}
  ],
  "relationships": [
    {{"source": "entity name", "type": "ExactTypeFromList", "target": "entity name"}}
  ]
}}
"""

    try:
        # Initialize the Groq client
        client = Groq(api_key=GROQ_API_KEY)
        
        # Call Groq API with JSON mode enabled using GPT-OSS 120B
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are an expert medical data extractor. You must output ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        content = completion.choices[0].message.content
        
        # Convert text to Python dictionary
        data = json.loads(content)
        
        # --- Strict Filtering (Drop hallucinated types) ---
        valid_entities = [
            e for e in data.get("entities", []) 
            if e.get("type") in ALLOWED_ENTITY_TYPES and "name" in e
        ]
        
        valid_relationships = [
            r for r in data.get("relationships", []) 
            if r.get("type") in ALLOWED_RELATIONSHIPS and "source" in r and "target" in r
        ]
        
        return {
            "entities": valid_entities,
            "relationships": valid_relationships
        }

    except Exception as e:
        print(f"[ERROR] Groq Extraction failed: {e}")
        return {"entities": [], "relationships": []}