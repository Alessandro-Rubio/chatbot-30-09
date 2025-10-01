from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import asyncio

from file_manager import EnhancedFileManager
from document_processor import AdvancedDocumentProcessor
from rag_engine import HybridRAGEngine

app = FastAPI(title="Chatbot RAG Inteligente", version="5.0.0")

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
    use_rag: Optional[bool] = None  # None = automático, True = forzar RAG, False = forzar simple

class ChatResponse(BaseModel):
    reply: str
    used_rag: bool
    source: str
    mode: str  # 'auto', 'forced_rag', 'forced_simple'

class FileUploadResponse(BaseModel):
    status: str
    filename: str
    message: str

class SystemStatusResponse(BaseModel):
    rag_initialized: bool
    total_files: int
    system_status: str

async def initialize_rag_system():
    """Initialize RAG system on startup"""
    global rag_initialized
    try:
        print("🔄 Inicializando sistema RAG...")
        documents = doc_processor.load_and_chunk_documents()
        if documents:
            rag_engine.initialize_vector_store(documents)
            rag_initialized = rag_engine.is_rag_initialized()
            if rag_initialized:
                print("✅ Sistema RAG inicializado correctamente")
            else:
                print("⚠️ Sistema RAG no se pudo inicializar")
        else:
            print("ℹ️ No hay documentos para inicializar RAG")
            rag_initialized = False
    except Exception as e:
        print(f"❌ Error inicializando RAG: {str(e)}")
        rag_initialized = False

@app.on_event("startup")
async def startup_event():
    """Auto-initialize RAG system on startup"""
    await initialize_rag_system()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Lógica de decisión automática
        if request.use_rag is None:
            # Modo automático - el sistema decide
            if rag_initialized:
                response = rag_engine.query_with_rag(request.message)
                # Determinar si se usó RAG basado en la respuesta
                used_rag = rag_engine._should_use_rag(request.message)[0]
                return ChatResponse(
                    reply=response, 
                    used_rag=used_rag, 
                    source="rag" if used_rag else "simple",
                    mode="auto"
                )
            else:
                # No hay RAG disponible, usar modo simple
                response = rag_engine._generate_simple_response(request.message)
                return ChatResponse(
                    reply=response, 
                    used_rag=False, 
                    source="simple",
                    mode="auto"
                )
        
        elif request.use_rag:
            # Modo RAG forzado
            if rag_initialized:
                response = rag_engine.query_with_rag(request.message)
                return ChatResponse(
                    reply=response, 
                    used_rag=True, 
                    source="rag",
                    mode="forced_rag"
                )
            else:
                response = "El sistema RAG no está disponible. No hay documentos cargados o el sistema no se ha inicializado correctamente."
                return ChatResponse(
                    reply=response, 
                    used_rag=False, 
                    source="error",
                    mode="forced_rag"
                )
        
        else:
            # Modo simple forzado
            response = rag_engine._generate_simple_response(request.message)
            return ChatResponse(
                reply=response, 
                used_rag=False, 
                source="simple",
                mode="forced_simple"
            )
            
    except Exception as e:
        error_response = f"Lo siento, ocurrió un error: {str(e)}"
        return ChatResponse(
            reply=error_response, 
            used_rag=False, 
            source="error",
            mode="error"
        )

@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    try:
        # Validar tipo de archivo
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

@app.get("/rag/status")
async def rag_status():
    """RAG status endpoint"""
    return {
        "rag_initialized": rag_initialized,
        "file_count": file_manager.get_file_count()
    }

@app.post("/rag/reinitialize")
async def reinitialize_rag(background_tasks: BackgroundTasks = None):
    """Reinitialize RAG endpoint"""
    background_tasks.add_task(initialize_rag_system)
    return {"message": "Reinicialización del sistema RAG iniciada"}

@app.post("/initialize-rag")
async def initialize_rag_legacy(background_tasks: BackgroundTasks = None):
    """Legacy initialize endpoint"""
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

@app.get("/")
async def root():
    return {
        "message": "Sistema RAG Inteligente API - Modo Automático",
        "status": "operational",
        "rag_initialized": rag_initialized,
        "file_count": file_manager.get_file_count(),
        "modes": {
            "auto": "El sistema decide cuándo usar RAG",
            "forced_rag": "Forzar uso de documentos",
            "forced_simple": "Forzar modo conversacional"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)