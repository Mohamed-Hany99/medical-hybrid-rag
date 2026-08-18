import os
import sys
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.hybrid_rag import (
    retrieve_vector_context,
    retrieve_graph_context,
    generate_hybrid_answer
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JUDGE_MODEL = "openai/gpt-oss-120b"
judge_client = Groq(api_key=GROQ_API_KEY)

# ============================================================
# 1. Benchmark Test Dataset (10 Detailed Medical Test Cases)
# ============================================================
TEST_CASES = [
    {
        "id": 1,
        "question": "What is the recommended moderate physical activity per week according to ESC guidance?",
        "expected_facts": [
            "At least 150 minutes per week of moderate aerobic activity",
            "Or 75 minutes per week of vigorous activity",
            "Cuts relative cardiovascular mortality by ~27%"
        ],
        "target_doc": "Cardiovascular Disease Prevention.pdf"
    },
    {
        "id": 2,
        "question": "What is the difference between HDL and LDL cholesterol, and how does HDL help protect arteries?",
        "expected_facts": [
            "HDL is considered good cholesterol, while LDL is bad cholesterol",
            "HDL helps prevent LDL from sticking to arterial walls",
            "Reduces plaque buildup and lowers risk of heart disease and stroke"
        ],
        "target_doc": "Cholesterol.pdf"
    },
    {
        "id": 3,
        "question": "What are the 8 components of Life's Essential 8 for cardiovascular health?",
        "expected_facts": [
            "Diet, physical activity, nicotine exposure, sleep health",
            "Body mass index (BMI), blood lipids, blood glucose, blood pressure"
        ],
        "target_doc": "Lifeâ__s Essential 8.pdf"
    },
    {
        "id": 4,
        "question": "What non-nicotine prescription medications are used to help people quit smoking?",
        "expected_facts": [
            "Bupropion hydrochloride (Zyban, Wellbutrin)",
            "Varenicline",
            "They reduce cravings and block nicotine-related chemical pathways in the brain"
        ],
        "target_doc": "Smoking .pdf"
    },
    {
        "id": 5,
        "question": "What are the core characteristics and fat sources of a Mediterranean-style diet?",
        "expected_facts": [
            "Rich in vegetables, fruits, whole grains, beans, nuts, and seeds",
            "Olive oil is the primary source of healthy unsaturated fat",
            "Emphasizes fish and poultry over red meat"
        ],
        "target_doc": "diet.pdf"
    },
    {
        "id": 6,
        "question": "How does carbon monoxide from smoking harm the cardiovascular system?",
        "expected_facts": [
            "Decreases oxygen carrying capacity in red blood cells",
            "Increases cholesterol deposition in the arterial inner lining",
            "Contributes to arterial hardening and increases heart attack risk"
        ],
        "target_doc": "Smoking .pdf"
    },
    {
        "id": 7,
        "question": "Why was sleep health added as a new metric to cardiovascular health guidelines?",
        "expected_facts": [
            "Inappropriate sleep duration is independently linked to coronary heart disease and all-cause mortality",
            "Affects blood pressure, glucose homeostasis, inflammation, and metabolic syndrome",
            "7 to 8 hours of sleep per night is the optimal range for adults"
        ],
        "target_doc": "Lifeâ__s Essential 8.pdf"
    },
    {
        "id": 8,
        "question": "What are the specific clinical targets for blood pressure and body weight in CVD prevention?",
        "expected_facts": [
            "Blood pressure target < 140/90 mmHg (lower in diabetic patients)",
            "BMI 20-25 kg/m2",
            "Waist circumference < 94 cm for men or < 80 cm for women"
        ],
        "target_doc": "Cardiovascular Disease Prevention.pdf"
    },
    {
        "id": 9,
        "question": "Why does the ketogenic (keto) diet raise concerns for cholesterol management?",
        "expected_facts": [
            "High in saturated fats which can raise LDL (bad) cholesterol",
            "Does not align with heart-healthy dietary guidelines"
        ],
        "target_doc": "Cholesterol.pdf"
    },
    {
        "id": 10,
        "question": "What are the common symptoms of cardiovascular disease and warning signs of heart issues?",
        "expected_facts": [
            "Chest pain, tightness, shortness of breath, palpitations",
            "High cholesterol often has no symptoms and requires blood tests"
        ],
        "target_doc": "CVDS.pdf"
    }
]

# ============================================================
# 2. LLM-as-a-Judge Evaluation Function
# ============================================================
def evaluate_response(question: str, generated_answer: str, expected_facts: List[str], retrieved_sources: List[str]) -> Dict[str, Any]:
    """Uses LLM to quantitatively grade the generation based on standard RAG metrics."""
    
    prompt = f"""You are an unbiased automated evaluator for a Medical RAG pipeline.
Grade the GENERATED ANSWER against the EXPECTED FACTS and RETRIEVED SOURCES.

QUESTION:
{question}

EXPECTED FACTS:
{json.dumps(expected_facts, indent=2)}

RETRIEVED SOURCES:
{json.dumps(retrieved_sources, indent=2)}

GENERATED ANSWER:
{generated_answer}

Provide your evaluation strictly as a valid JSON object matching this schema:
{{
  "faithfulness_score": <int 1-5, where 5 is perfectly grounded without hallucinations>,
  "relevance_score": <int 1-5, where 5 directly and completely answers the prompt>,
  "fact_coverage_score": <int 1-5, based on how many expected facts were included>,
  "evaluation_summary": "<1-2 sentences explaining the grade>"
}}
"""

    try:
        res = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict evaluation judge. Output ONLY JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        return {
            "faithfulness_score": 0,
            "relevance_score": 0,
            "fact_coverage_score": 0,
            "evaluation_summary": f"Evaluation failed: {e}"
        }

# ============================================================
# 3. Main Validation Runner
# ============================================================
def run_rag_validation():
    print("=" * 80)
    print("STARTING HYBRID RAG VALIDATION BENCHMARK (10 TEST CASES)")
    print("=" * 80)

    results = []
    total_cases = len(TEST_CASES)

    for idx, test in enumerate(TEST_CASES, 1):
        q_id = test["id"]
        question = test["question"]
        print(f"\n[TEST {idx}/{total_cases}] Evaluating Query: \"{question}\"")
        
        # 1. Test Retrieval
        vector_chunks = retrieve_vector_context(question, top_k=3)
        graph_facts = retrieve_graph_context(question)
        
        retrieved_docs = list(set([f"{c.get('source_document')} (P.{c.get('source_page')})" for c in vector_chunks]))
        
        # 2. Test Generation
        generated_answer = generate_hybrid_answer(question)
        
        # 3. Judge Evaluation
        eval_result = evaluate_response(question, generated_answer, test["expected_facts"], retrieved_docs)
        
        avg_score = (
            eval_result.get("faithfulness_score", 0) +
            eval_result.get("relevance_score", 0) +
            eval_result.get("fact_coverage_score", 0)
        ) / 3.0
        
        print(f"   -> Faithfulness:  {eval_result.get('faithfulness_score')}/5")
        print(f"   -> Relevance:     {eval_result.get('relevance_score')}/5")
        print(f"   -> Fact Coverage: {eval_result.get('fact_coverage_score')}/5")
        print(f"   -> Overall Grade: {avg_score:.2f}/5.0")
        print(f"   -> Summary: {eval_result.get('evaluation_summary')}")
        
        results.append({
            "id": q_id,
            "question": question,
            "avg_score": avg_score,
            "details": eval_result,
            "retrieved_sources": retrieved_docs
        })
        time.sleep(1)

    # Final Benchmark Report
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK VALIDATION REPORT")
    print("=" * 80)
    
    total_avg = sum(r["avg_score"] for r in results) / len(results)
    print(f"Total Benchmark Quality Score: {total_avg:.2f} / 5.0\n")
    
    for r in results:
        status = "PASSED" if r["avg_score"] >= 4.0 else "NEEDS REVIEW"
        print(f"Test #{r['id']:02d}: [{status}] Score: {r['avg_score']:.2f}/5.0 | Doc Sources: {', '.join(r['retrieved_sources'])}")
    print("=" * 80)

if __name__ == "__main__":
    run_rag_validation()