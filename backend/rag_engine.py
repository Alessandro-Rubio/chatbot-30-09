from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
import os
from typing import List, Dict, Tuple

class HybridRAGEngine:
    def __init__(self, persistence_path: str = "./vector_store"):
        self.embedding_function = OllamaEmbeddings(model="nomic-embed-text")
        
        # ✅ MODELOS QUANTIZADOS - ELIGE UNO:
        # Opción 1: Máximo rendimiento (recomendado)
        self.llm = OllamaLLM(
            model="llama3.1:8b-instruct-q4_K_M", 
            temperature=0.1,
            num_ctx=4096,  # Contexto reducido para mayor velocidad
            num_thread=8,  # Optimizar uso de CPU
            num_gpu=1      # Usar GPU si está disponible
        )
        
        self.persistence_path = persistence_path
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        
    def initialize_vector_store(self, documents: List[Document]):
        """Inicializar sistema RAG con documentos - Versión optimizada"""
        if not documents:
            print("⚠️ No hay documentos para RAG. El sistema funcionará en modo simple.")
            self.vector_store = None
            self.bm25_retriever = None
            return
        
        try:
            # Inicializar vector store con timeout reducido
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                persist_directory=self.persistence_path
            )
            
            # Inicializar BM25 retriever solo si hay documentos suficientes
            if len(documents) > 5:
                self.bm25_retriever = BM25Retriever.from_documents(documents)
                self.bm25_retriever.k = 2
                
                # Crear ensemble retriever
                vector_retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": 2}  # Reducir documentos recuperados
                )
                
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[vector_retriever, self.bm25_retriever],
                    weights=[0.7, 0.3]
                )
            else:
                # Para pocos documentos, usar solo vector store
                self.ensemble_retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": 2}
                )
            
            print("✅ Sistema RAG híbrido inicializado correctamente")
            
        except Exception as e:
            print(f"❌ Error inicializando RAG: {str(e)}")
            print("🔄 El sistema funcionará en modo simple sin RAG")
            self.vector_store = None

    def _should_use_rag(self, question: str) -> Tuple[bool, List[Document]]:
        """Determina si la pregunta debe usar RAG - Versión optimizada"""
        if not self.ensemble_retriever:
            return False, []
        
        try:
            # Análisis rápido de la pregunta
            question_lower = question.lower().strip()
            
            # Preguntas generales que NO necesitan RAG
            general_questions = {
                'hola', 'hello', 'hi', 'hey', 'buenos días', 'buenas tardes', 'buenas noches',
                'cómo estás', 'qué tal', 'quién eres', 'cuál es tu nombre', 
                'qué puedes hacer', 'ayuda', 'help', 'gracias', 'thanks', 'bye', 
                'adiós', 'hasta luego', 'ok', 'okay', 'entendido', 'de acuerdo'
            }
            
            # Si es pregunta general, evitar RAG completamente
            if question_lower in general_questions:
                return False, []
            
            # Palabras clave que SÍ necesitan RAG
            document_keywords = {
                'documento', 'archivo', 'pdf', 'texto', 'informe', 'reporte',
                'según', 'según el documento', 'en el archivo', 'en el pdf',
                'procedimiento', 'política', 'protocolo', 'guía', 'manual',
                'especificación', 'requisito', 'norma', 'cláusula', 'artículo',
                'contrato', 'acuerdo', 'documentación'
            }
            
            # Verificar palabras clave rápidamente
            if any(keyword in question_lower for keyword in document_keywords):
                retrieved_docs = self.ensemble_retriever.get_relevant_documents(question)
                return len(retrieved_docs) > 0, retrieved_docs
            
            # Para otras preguntas, búsqueda rápida
            retrieved_docs = self.ensemble_retriever.get_relevant_documents(question)
            return len(retrieved_docs) >= 1, retrieved_docs
            
        except Exception as e:
            print(f"⚠️ Error en detección RAG: {str(e)}")
            return False, []
    
    def query_with_rag(self, question: str, top_k: int = 2) -> str:
        """Consulta RAG optimizada para velocidad"""
        should_use_rag, relevant_docs = self._should_use_rag(question)
        
        if not should_use_rag:
            # Respuesta rápida directa
            return self._generate_fast_response(question)
        
        try:
            # Contexto optimizado (solo 2 documentos máximo)
            context = "\n\n".join([
                f"Fuente: {doc.metadata.get('source_file', 'Documento')}\n"
                f"Contenido: {doc.page_content[:500]}"  # Limitar longitud
                for doc in relevant_docs[:top_k]
            ])
            
            # Prompt optimizado para respuestas rápidas
            prompt = f"""Responde brevemente basándote en este contexto:

{context}

Pregunta: {question}

Respuesta concisa:"""
            
            response = self.llm.invoke(prompt)
            return response
            
        except Exception as e:
            # Fallback a respuesta simple
            return self._generate_fast_response(question)
    
    def _generate_fast_response(self, question: str) -> str:
        """Generar respuesta rápida para preguntas generales"""
        # Prompt optimizado para velocidad
        prompt = f"""Responde de forma breve y directa:

Pregunta: {question}

Respuesta:"""
        
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            return f"Lo siento, ocurrió un error. Por favor intenta nuevamente."

    def is_rag_initialized(self) -> bool:
        """Verificar si el sistema RAG está listo"""
        return self.ensemble_retriever is not None
    
    def get_rag_stats(self) -> Dict:
        """Obtener estadísticas del sistema RAG"""
        if not self.vector_store:
            return {"status": "no_initialized", "mode": "simple"}
        
        try:
            collection = self.vector_store._collection
            return {
                "status": "initialized",
                "document_count": collection.count() if collection else 0,
                "vector_store_path": self.persistence_path,
                "mode": "hybrid"
            }
        except:
            return {"status": "no_initialized", "mode": "simple"}