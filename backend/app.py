from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio

from file_manager import EnhancedFileManager
from document_processor import AdvancedDocumentProcessor
from rag_engine import HybridRAGEngine
from auto_rag import AutoRAGManager  # NEW

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
rag_manager = AutoRAGManager()  # NEW: Use the auto-manager

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
    unprocessed_files: int
    system_status: str

@app.on_event("startup")
async def startup_event():
    """Auto-initialize RAG system on startup"""
    background_tasks = BackgroundTasks()
    background_tasks.add_task(rag_manager.initialize_rag_system)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        if request.use_rag and rag_manager.initialized:
            response = rag_manager.rag_engine.query_with_rag(request.message)
            return ChatResponse(reply=response, used_rag=True, source="rag")
        else:
            # Simple mode implementation
            simple_response = f"Modo simple: {request.message}"
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
        
        # Trigger RAG update
        background_tasks.add_task(rag_manager.handle_file_operation, "upload", file.filename)
        
        return FileUploadResponse(
            status="success",
            filename=file.filename,
            message="Archivo subido y RAG system se está actualizando"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files/{filename}")
async def delete_file(filename: str, background_tasks: BackgroundTasks = None):
    try:
        success = file_manager.delete_file(filename)
        if not success:
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        # Trigger RAG rebuild
        background_tasks.add_task(rag_manager.handle_file_operation, "delete", filename)
        
        return {"message": f"Archivo {filename} eliminado. RAG system se está actualizando."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
async def list_files():
    """List all files with detailed metadata"""
    try:
        files = file_manager.list_files()
        return {
            "files": files,
            "total_count": file_manager.get_file_count(),
            "metadata": file_manager.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/status", response_model=SystemStatusResponse)
async def system_status():
    """Get comprehensive system status"""
    status = rag_manager.get_system_status()
    return SystemStatusResponse(
        rag_initialized=status["rag_initialized"],
        total_files=status["total_files"],
        unprocessed_files=status["unprocessed_files"],
        system_status="healthy" if status["rag_initialized"] else "initializing"
    )

@app.post("/system/reinitialize")
async def reinitialize_system(background_tasks: BackgroundTasks = None):
    """Force reinitialize the entire RAG system"""
    background_tasks.add_task(rag_manager.initialize_rag_system)
    return {"message": "Reinicialización del sistema RAG iniciada"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)