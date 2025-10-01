from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Make sure this import is present
from pydantic import BaseModel
from document_processor import DocumentProcessor
from rag_engine import RAGEngine
import ollama
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Local Llama Chatbot with RAG", version="1.0.0")

# ✅ Re-add the CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # The origin of your React app
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],
)

# Initialize components
doc_processor = DocumentProcessor()
rag_engine = RAGEngine()

class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True  # Frontend can set this to False for simple questions

class ChatResponse(BaseModel):
    reply: str
    used_rag: bool

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Determine if we should use RAG
        use_rag = request.use_rag and rag_engine.vector_store is not None
        
        response = rag_engine.query(request.message, use_rag=use_rag)
        return ChatResponse(reply=response, used_rag=use_rag)
        
    except Exception as e:
        return ChatResponse(reply=f"Error: {str(e)}", used_rag=False)

@app.post("/initialize-rag")
async def initialize_rag():
    """Endpoint to initialize RAG system with documents"""
    try:
        documents = doc_processor.load_and_chunk_documents()
        result = rag_engine.initialize_vector_store(documents)
        return {"status": "success", "message": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/rag-status")
async def rag_status():
    """Check if RAG system is initialized"""
    return {"rag_initialized": rag_engine.vector_store is not None}

@app.get("/models")
async def list_models():
    try:
        models = ollama.list()
        return {"models": models.get('models', [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)