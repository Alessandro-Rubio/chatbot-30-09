from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import asyncio

from file_manager import EnhancedFileManager
from document_processor import AdvancedDocumentProcessor
from rag_engine import HybridRAGEngine

app = FastAPI(title="Advanced RAG Chatbot", version="4.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
file_manager = EnhancedFileManager()
doc_processor = AdvancedDocumentProcessor()
rag_engine = HybridRAGEngine()

# Global state
rag_initialized = False

# Models
class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True

class ChatResponse(BaseModel):
    reply: str
    used_rag: bool
    source: str

class FileUploadResponse(BaseModel):
    status: str
    filename: str
    message: str

class SystemStatusResponse(BaseModel):
    rag_initialized: bool
    total_files: int
    system_status: str

class RAGStatusResponse(BaseModel):
    rag_initialized: bool
    file_count: int

async def initialize_rag_system():
    """Initialize RAG system on startup"""
    global rag_initialized
    try:
        print("🔄 Initializing RAG system...")
        documents = doc_processor.load_and_chunk_documents()
        if documents:
            rag_engine.initialize_vector_store(documents)
            rag_initialized = True
            print("✅ RAG system initialized successfully")
        else:
            print("⚠️ No documents found for RAG initialization")
            rag_initialized = False
    except Exception as e:
        print(f"❌ RAG initialization failed: {str(e)}")
        rag_initialized = False

@app.on_event("startup")
async def startup_event():
    """Auto-initialize RAG system on startup"""
    await initialize_rag_system()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        if request.use_rag and rag_initialized:
            response = rag_engine.query_with_rag(request.message)
            return ChatResponse(reply=response, used_rag=True, source="rag")
        else:
            # Simple mode - implement your simple response logic here
            simple_response = "Modo simple activado. Para usar RAG, asegúrate de tener documentos cargados."
            return ChatResponse(reply=simple_response, used_rag=False, source="simple")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    try:
        # Validate file type
        allowed_types = ['.pdf', '.txt', '.docx', '.doc']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_types:
            raise HTTPException(status_code=400, 
                              detail=f"Tipo de archivo no soportado. Permitidos: {allowed_types}")
        
        # Upload file
        result = await file_manager.upload_file(file)
        
        if result["status"] == "duplicate":
            return FileUploadResponse(
                status="duplicate",
                filename=file.filename,
                message="El archivo ya existe en el sistema"
            )
        
        # Reinitialize RAG with new files
        background_tasks.add_task(initialize_rag_system)
        
        return FileUploadResponse(
            status="success",
            filename=file.filename,
            message="Archivo subido exitosamente. Sistema RAG se está actualizando."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files/{filename}")
async def delete_file(filename: str, background_tasks: BackgroundTasks = None):
    try:
        success = file_manager.delete_file(filename)
        if not success:
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        # Reinitialize RAG after deletion
        background_tasks.add_task(initialize_rag_system)
        
        return {"message": f"Archivo {filename} eliminado exitosamente. Sistema RAG se está actualizando."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
async def list_files():
    """List all files with detailed metadata"""
    try:
        files = file_manager.list_files()
        return {
            "files": files,
            "total_count": file_manager.get_file_count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ ADD THESE MISSING ENDPOINTS THAT YOUR FRONTEND EXPECTS

@app.get("/rag/status")
async def rag_status():
    """RAG status endpoint that frontend expects"""
    return {
        "rag_initialized": rag_initialized,
        "file_count": file_manager.get_file_count()
    }

@app.post("/rag/reinitialize")
async def reinitialize_rag(background_tasks: BackgroundTasks = None):
    """Reinitialize RAG endpoint that frontend expects"""
    background_tasks.add_task(initialize_rag_system)
    return {"message": "Reinicialización del sistema RAG iniciada"}

@app.post("/initialize-rag")
async def initialize_rag_legacy(background_tasks: BackgroundTasks = None):
    """Legacy initialize endpoint for frontend compatibility"""
    background_tasks.add_task(initialize_rag_system)
    return {"message": "Inicialización del sistema RAG iniciada"}

@app.get("/system/status", response_model=SystemStatusResponse)
async def system_status():
    """Get comprehensive system status"""
    return SystemStatusResponse(
        rag_initialized=rag_initialized,
        total_files=file_manager.get_file_count(),
        system_status="healthy" if rag_initialized else "initializing"
    )

@app.post("/system/reinitialize")
async def reinitialize_system(background_tasks: BackgroundTasks = None):
    """Force reinitialize the entire RAG system"""
    background_tasks.add_task(initialize_rag_system)
    return {"message": "Reinicialización del sistema RAG iniciada"}

@app.get("/")
async def root():
    return {
        "message": "Sistema RAG Híbrido API",
        "status": "operational",
        "rag_initialized": rag_initialized,
        "file_count": file_manager.get_file_count()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)