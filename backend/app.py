from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
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
    metadata_filters: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    reply: str
    used_rag: bool
    source: str
    mode: str
    response_time: float
    intent_info: Optional[Dict] = None

class FileUploadResponse(BaseModel):
    status: str
    filename: str
    message: str

class SystemStatusResponse(BaseModel):
    rag_initialized: bool
    total_files: int
    system_status: str
    mode: str
    model: str

class SystemInfoResponse(BaseModel):
    rag_initialized: bool
    total_files: int
    system_status: str
    mode: str
    model: str
    supported_formats: Dict
    index_stats: Dict
    system_capabilities: List[str]

async def initialize_rag_system(incremental: bool = False, specific_files: List[str] = None):
    """Initialize RAG system on startup - Versión optimizada"""
    global rag_initialized
    try:
        print("🚀 Inicializando sistema RAG optimizado...")
        
        if incremental and specific_files:
            print(f"🔄 Indexación incremental para {len(specific_files)} archivos...")
            documents, failed_files = doc_processor.load_and_chunk_documents(specific_files)
        else:
            documents, failed_files = doc_processor.load_and_chunk_documents()
        
        if documents:
            rag_engine.initialize_vector_store(documents, incremental=incremental)
            rag_initialized = rag_engine.is_rag_initialized()
            
            if rag_initialized:
                # Actualizar estado de indexación
                if incremental and specific_files:
                    rag_engine.incremental_indexer.mark_files_as_indexed(
                        specific_files, 
                        len(documents),
                        file_manager.metadata["files"]
                    )
                print(f"✅ RAG inicializado con {len(documents)} chunks")
                
                if failed_files:
                    print(f"⚠️ Archivos fallidos: {failed_files}")
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
    await initialize_rag_system(incremental=True)
    
    # Print registered routes for debugging
    print("🛣️  Endpoints registrados:")
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            print(f"  {list(route.methods)} {route.path}")
    
    yield
    # Shutdown (puedes añadir lógica de limpieza aquí si es necesario)
    print("🔴 Apagando servidor...")

# Inicializar componentes DESPUÉS de los modelos
file_manager = EnhancedFileManager()
doc_processor = AdvancedDocumentProcessor()
rag_engine = HybridRAGEngine()

# ✅ USAR LIFESPAN EN LUGAR DE on_event
app = FastAPI(
    title="Chatbot RAG Avanzado", 
    version="7.0.0",
    lifespan=lifespan
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
        # Inicializar variables para evitar el error de variable no asociada
        response = ""
        used_rag = False
        mode = "unknown"
        intent_info = None
        
        # Lógica optimizada de decisión con detección de intención
        if request.use_rag is None:
            # Modo automático inteligente
            if rag_initialized:
                should_use_rag, intent_data = rag_engine.intent_detector.should_use_rag(request.message)
                intent_info = intent_data
                
                if should_use_rag:
                    response = rag_engine.query_with_rag(
                        request.message, 
                        metadata_filters=request.metadata_filters
                    )
                    used_rag = True
                    mode = "auto_rag"
                else:
                    response = rag_engine._generate_fast_response(request.message)
                    used_rag = False
                    mode = "auto_simple"
            else:
                # Fallback a modo simple si RAG no está disponible
                response = rag_engine._generate_fast_response(request.message)
                used_rag = False
                mode = "simple_fallback"
        
        elif request.use_rag and rag_initialized:
            # RAG forzado (solo si está disponible)
            response = rag_engine.query_with_rag(
                request.message, 
                metadata_filters=request.metadata_filters
            )
            used_rag = True
            mode = "forced_rag"
        
        elif request.use_rag and not rag_initialized:
            # RAG solicitado pero no disponible
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
            response_time=round(response_time, 2),
            intent_info=intent_info
        )
        
    except Exception as e:
        response_time = time.time() - start_time
        error_msg = f"⚠️ Error: {str(e)}"
        print(f"❌ Error en chat endpoint: {str(e)}")
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
        allowed_types = ['.pdf', '.txt', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.csv']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Tipo de archivo no soportado. Permitidos: {allowed_types}"
            )
        
        # Upload file
        result = await file_manager.upload_file(file)
        
        if result["status"] == "duplicate":
            return FileUploadResponse(
                status="duplicate",
                filename=file.filename,
                message="El archivo ya existe en el sistema"
            )
        
        # Reinitialize RAG con new files (incremental)
        background_tasks.add_task(initialize_rag_system, True, [file.filename])
        
        return FileUploadResponse(
            status="success",
            filename=file.filename,
            message="Archivo subido. Sistema RAG se está actualizando."
        )
        
    except Exception as e:
        print(f"❌ Error subiendo archivo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files/{filename}")
async def delete_file(filename: str, background_tasks: BackgroundTasks = None):
    try:
        # Security: Validate filename to prevent path traversal
        if '/' in filename or '\\' in filename:
            raise HTTPException(status_code=400, detail="Nombre de archivo inválido")
        
        success = file_manager.delete_file(filename)
        if not success:
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        # Remove from index and reinitialize RAG
        rag_engine.incremental_indexer.remove_file_from_index(filename)
        background_tasks.add_task(initialize_rag_system, True)
        
        return {
            "status": "success",
            "message": f"Archivo {filename} eliminado exitosamente",
            "rag_update": "Sistema RAG se está actualizando"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error eliminando archivo {filename}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno del servidor: {str(e)}"
        )

@app.get("/files")
async def list_files():
    """List all files with detailed metadata"""
    try:
        files = file_manager.list_files()
        return {
            "files": files,
            "total_count": file_manager.get_file_count(),
            "status": "success"
        }
    except Exception as e:
        print(f"❌ Error listando archivos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/status")
async def rag_status():
    """RAG status endpoint que el frontend espera"""
    try:
        return {
            "rag_initialized": rag_initialized,
            "file_count": file_manager.get_file_count(),
            "status": "initialized" if rag_initialized else "not_initialized",
            "system_mode": "hybrid" if rag_initialized else "simple"
        }
    except Exception as e:
        print(f"❌ Error en rag/status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/reinitialize")
async def reinitialize_rag(background_tasks: BackgroundTasks = None):
    """Reinitialize RAG endpoint que el frontend espera"""
    try:
        background_tasks.add_task(initialize_rag_system, False)
        return {
            "status": "success",
            "message": "Reinicialización del sistema RAG iniciada"
        }
    except Exception as e:
        print(f"❌ Error reinicializando RAG: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/initialize-rag")
async def initialize_rag_legacy(background_tasks: BackgroundTasks = None):
    """Legacy initialize endpoint para compatibilidad con frontend"""
    try:
        background_tasks.add_task(initialize_rag_system, False)
        return {
            "status": "success", 
            "message": "Inicialización del sistema RAG iniciada"
        }
    except Exception as e:
        print(f"❌ Error en initialize-rag: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/status", response_model=SystemStatusResponse)
async def system_status():
    """Get comprehensive system status"""
    try:
        rag_stats = rag_engine.get_rag_stats()
        
        return SystemStatusResponse(
            rag_initialized=rag_initialized,
            total_files=file_manager.get_file_count(),
            system_status="healthy" if rag_initialized else "simple_mode",
            mode=rag_stats.get("mode", "simple"),
            model="llama3.1:8b-instruct-q4_K_M"
        )
    except Exception as e:
        print(f"❌ Error en system/status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/info", response_model=SystemInfoResponse)
async def system_info():
    """Obtener información completa del sistema"""
    try:
        system_info_data = rag_engine.get_system_info()
        
        return SystemInfoResponse(
            rag_initialized=rag_initialized,
            total_files=file_manager.get_file_count(),
            system_status="healthy" if rag_initialized else "simple_mode",
            mode=system_info_data.get("mode", "hybrid"),
            model="llama3.1:8b-instruct-q4_K_M",
            supported_formats=doc_processor.get_supported_formats(),
            index_stats=rag_engine.incremental_indexer.get_index_stats(),
            system_capabilities=[
                "incremental_indexing",
                "metadata_search", 
                "advanced_intent_detection",
                "multi_format_support"
            ]
        )
    except Exception as e:
        print(f"❌ Error en system/info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/filters")
async def get_search_filters():
    """Obtener filtros de búsqueda disponibles"""
    try:
        if rag_engine.metadata_search:
            return {
                "status": "success",
                "filters": rag_engine.metadata_search.get_available_filters()
            }
        return {
            "status": "success",
            "filters": {}
        }
    except Exception as e:
        print(f"❌ Error en search/filters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "🚀 Chatbot RAG Avanzado - Sistema Mejorado",
        "status": "operational",
        "rag_initialized": rag_initialized,
        "file_count": file_manager.get_file_count(),
        "model": "llama3.1:8b-instruct-q4_K_M",
        "endpoints_available": [
            "/chat", "/upload", "/files", "/rag/status", 
            "/system/status", "/system/info", "/search/filters"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)