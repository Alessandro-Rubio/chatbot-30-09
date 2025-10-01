from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import asyncio
import time

from file_manager import EnhancedFileManager
from document_processor import AdvancedDocumentProcessor
from rag_engine import HybridRAGEngine

# Global state
rag_initialized = False
startup_time = time.time()

# Models - DEBEN DEFINIRSE ANTES de los endpoints
class ChatRequest(BaseModel):
    message: str
    use_rag: Optional[bool] = None  # None = automático

class ChatResponse(BaseModel):
    reply: str
    used_rag: bool
    source: str
    mode: str
    response_time: float

class FileUploadResponse(BaseModel):  # ✅ FALTABA ESTE MODELO
    status: str
    filename: str
    message: str

class SystemStatusResponse(BaseModel):
    rag_initialized: bool
    total_files: int
    system_status: str
    mode: str
    model: str

async def initialize_rag_system():
    """Initialize RAG system on startup - Versión optimizada"""
    global rag_initialized
    try:
        print("🚀 Inicializando sistema RAG optimizado...")
        documents = doc_processor.load_and_chunk_documents()
        
        if documents:
            rag_engine.initialize_vector_store(documents)
            rag_initialized = rag_engine.is_rag_initialized()
            
            if rag_initialized:
                print(f"✅ RAG inicializado con {len(documents)} documentos")
            else:
                print("⚠️ RAG no inicializado, modo simple activado")
        else:
            print("🔶 Modo simple: No hay documentos para RAG")
            rag_initialized = False
            
    except Exception as e:
        print(f"❌ Error en inicialización: {str(e)}")
        print("🔄 Continuando en modo simple...")
        rag_initialized = False

# ✅ REEMPLAZAR EL EVENTO OBSOLETO CON LIFESPAN
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager para reemplazar @app.on_event"""
    # Startup
    await initialize_rag_system()
    yield
    # Shutdown (puedes añadir lógica de limpieza aquí si es necesario)
    print("🔴 Apagando servidor...")

# Inicializar componentes DESPUÉS de los modelos
file_manager = EnhancedFileManager()
doc_processor = AdvancedDocumentProcessor()
rag_engine = HybridRAGEngine()

# ✅ USAR LIFESPAN EN LUGAR DE on_event
app = FastAPI(
    title="Chatbot RAG Optimizado", 
    version="6.0.0",
    lifespan=lifespan  # Esto reemplaza @app.on_event("startup")
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    
    try:
        # Lógica optimizada de decisión
        if request.use_rag is None:
            # Modo automático inteligente
            if rag_initialized:
                response = rag_engine.query_with_rag(request.message)
                used_rag = rag_engine._should_use_rag(request.message)[0]
                mode = "auto_rag" if used_rag else "auto_simple"
            else:
                # Fallback a modo simple si RAG no está disponible
                response = rag_engine._generate_fast_response(request.message)
                used_rag = False
                mode = "simple_fallback"
        
        elif request.use_rag and rag_initialized:
            # RAG forzado (solo si está disponible)
            response = rag_engine.query_with_rag(request.message)
            used_rag = True
            mode = "forced_rag"
        
        elif request.use_rag and not rag_initialized:
            # RAG solicitado pero no disponible
            response = "⚠️ El sistema RAG no está disponible actualmente. Respondiendo en modo simple.\n\n"
            response += rag_engine._generate_fast_response(request.message)
            used_rag = False
            mode = "fallback"
        
        else:
            # Modo simple forzado
            response = rag_engine._generate_fast_response(request.message)
            used_rag = False
            mode = "forced_simple"
        
        response_time = time.time() - start_time
        
        return ChatResponse(
            reply=response,
            used_rag=used_rag,
            source="rag" if used_rag else "simple",
            mode=mode,
            response_time=round(response_time, 2)
        )
        
    except Exception as e:
        response_time = time.time() - start_time
        error_msg = f"⚠️ Error: {str(e)}"
        return ChatResponse(
            reply=error_msg,
            used_rag=False,
            source="error",
            mode="error",
            response_time=round(response_time, 2)
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
        
        # Reinitialize RAG con new files
        background_tasks.add_task(initialize_rag_system)
        
        return FileUploadResponse(
            status="success",
            filename=file.filename,
            message="Archivo subido. Sistema RAG se está actualizando."
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
    rag_stats = rag_engine.get_rag_stats()
    
    return SystemStatusResponse(
        rag_initialized=rag_initialized,
        total_files=file_manager.get_file_count(),
        system_status="healthy" if rag_initialized else "simple_mode",
        mode=rag_stats.get("mode", "simple"),
        model="llama3.1:8b-instruct-q4_K_M"
    )

@app.get("/")
async def root():
    return {
        "message": "🚀 Chatbot RAG Optimizado - Modo Quantizado",
        "status": "operational",
        "rag_initialized": rag_initialized,
        "file_count": file_manager.get_file_count(),
        "model": "llama3.1:8b-instruct-q4_K_M",
        "optimizations": ["quantized_model", "fast_fallback", "auto_rag_detection"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)