# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
from query.retriever import DocumentRetriever
from query.generator import AnswerGenerator
# pyrefly: ignore [missing-import]
import uvicorn

app = FastAPI(title="Multimodal RAG API", version="1.0")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list

retriever = DocumentRetriever()
generator = AnswerGenerator()

@app.get("/")
def read_root():
    return {"status": "Multimodal RAG API is running"}

@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    try:
        # 1. Retrieve Context
        contexts = retriever.retrieve_context(request.query, top_k=request.top_k)
        
        if not contexts:
            return QueryResponse(
                answer="Maaf, tidak ada informasi yang relevan di dalam database dokumen.",
                sources=[]
            )
            
        # 2. Generate Answer
        answer = generator.generate_answer(request.query, contexts)
        
        # 3. Format sources
        sources = [
            {"source": ctx["metadata"].get("source", "Unknown"), "page": ctx["metadata"].get("page_number", "?")}
            for ctx in contexts
        ]
        
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)