import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# إضافة مسار المشروع الرئيسي لضمان نجاح استيراد ملفات الـ RAG
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from rag.hybrid_rag import generate_hybrid_answer

load_dotenv()

# ============================================================
# تهيئة تطبيق FastAPI
# ============================================================
app = FastAPI(
    title="Hybrid Medical RAG API",
    description="API for cardiovascular medical knowledge retrieval using Supabase, Neo4j, and Groq.",
    version="1.0.0"
)

# تفعيل CORS للسماح لأي Frontend بالاتصال بالـ API بدون مشاكل
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Data Models (Pydantic)
# ============================================================
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# ============================================================
# Endpoints
# ============================================================
@app.get("/")
def read_root():
    return {"status": "online", "message": "Hybrid Medical RAG API is running."}

@app.post("/api/v1/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    try:
        print(f"\n[API] Received query: {request.question}")
        
        # استدعاء دالة الـ Hybrid RAG وتمرير السؤال
        answer = generate_hybrid_answer(request.question)
        
        return QueryResponse(answer=answer)
    
    except Exception as e:
        print(f"[API ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# Server Runner
# ============================================================
if __name__ == "__main__":
    print("Starting FastAPI server on http://localhost:8000 ...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)