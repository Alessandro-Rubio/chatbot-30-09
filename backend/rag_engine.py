from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
import os
from typing import List, Dict, Tuple

class HybridRAGEngine:
    def __init__(self, persistence_path: str = "./vector_store"):
        self.embedding_function = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = Ollama(model="llama3.1", temperature=0.1)
        self.persistence_path = persistence_path
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        
    def initialize_vector_store(self, documents: List[Document]):
        """Inicializar sistema RAG con documentos"""
        if not documents:
            print("No hay documentos para procesar")
            self.vector_store = None
            self.bm25_retriever = None
            return
        
        # Inicializar vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=self.persistence_path
        )
        
        # Inicializar BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 2
        
        # Crear ensemble retriever
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 3}
        )
        
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, self.bm25_retriever],
            weights=[0.7, 0.3]
        )
        
        print("Sistema RAG híbrido inicializado correctamente")
    
    def _should_use_rag(self, question: str) -> Tuple[bool, List[Document]]:
        """Determina si la pregunta debe usar RAG basado en relevancia"""
        if not self.ensemble_retriever:
            return False, []
        
        try:
            # Recuperar documentos relevantes
            retrieved_docs = self.ensemble_retriever.get_relevant_documents(question)
            
            if not retrieved_docs:
                return False, []
            
            # Analizar la pregunta para decidir si necesita documentos
            question_lower = question.lower()
            
            # Palabras clave que indican necesidad de documentos específicos
            document_keywords = [
                'documento', 'archivo', 'pdf', 'texto', 'informe', 'reporte',
                'según', 'según el documento', 'en el archivo', 'en el pdf',
                'procedimiento', 'política', 'protocolo', 'guía', 'manual',
                'especificación', 'requisito', 'norma'
            ]
            
            # Preguntas generales que NO necesitan documentos
            general_questions = [
                'hola', 'hello', 'hi', 'cómo estás', 'qué tal', 'quién eres',
                'cuál es tu nombre', 'qué puedes hacer', 'ayuda',
                'gracias', 'thanks', 'bye', 'adiós', 'hasta luego'
            ]
            
            # Verificar si es una pregunta general
            if any(keyword in question_lower for keyword in general_questions):
                return False, []
            
            # Verificar si la pregunta menciona explícitamente documentos
            if any(keyword in question_lower for keyword in document_keywords):
                return True, retrieved_docs
            
            # Si hay documentos muy relevantes (basado en similitud), usar RAG
            # Podemos ajustar este threshold según necesidad
            if len(retrieved_docs) >= 1:  # Si hay al menos un documento relevante
                return True, retrieved_docs
            else:
                return False, []
                
        except Exception as e:
            print(f"Error en detección RAG: {str(e)}")
            return False, []
    
    def query_with_rag(self, question: str, top_k: int = 3) -> str:
        """Consulta usando RAG híbrido"""
        should_use_rag, relevant_docs = self._should_use_rag(question)
        
        if not should_use_rag:
            # Modo simple - responder pregunta general
            return self._generate_simple_response(question)
        
        try:
            # Construir contexto con documentos relevantes
            context = "\n\n".join([
                f"Documento: {doc.metadata.get('source_file', 'Desconocido')}\n"
                f"Contenido: {doc.page_content}"
                for doc in relevant_docs[:top_k]  # Usar solo los top_k más relevantes
            ])
            
            # Prompt mejorado para RAG
            prompt = f"""Basándote en el siguiente contexto de documentos, responde la pregunta de manera precisa y útil.

Contexto:
{context}

Pregunta: {question}

Instrucciones:
1. Responde principalmente con la información proporcionada en el contexto
2. Si el contexto no contiene información suficiente, complementa con tu conocimiento general pero indica la diferencia
3. Sé conciso pero completo
4. Si es relevante, menciona de qué documento proviene la información principal

Respuesta:"""
            
            response = self.llm.invoke(prompt)
            return response
            
        except Exception as e:
            return f"Error en la consulta RAG: {str(e)}"
    
    def _generate_simple_response(self, question: str) -> str:
        """Generar respuesta para preguntas generales"""
        prompt = f"""Eres un asistente útil y amigable. Responde la siguiente pregunta de manera natural y conversacional.

Pregunta: {question}

Responde de forma amigable y útil, como en una conversación normal:"""
        
        return self.llm.invoke(prompt)
    
    def is_rag_initialized(self) -> bool:
        """Verificar si el sistema RAG está listo"""
        return self.ensemble_retriever is not None
    
    def get_rag_stats(self) -> Dict:
        """Obtener estadísticas del sistema RAG"""
        if not self.vector_store:
            return {"status": "no_initialized"}
        
        collection = self.vector_store._collection
        return {
            "status": "initialized",
            "document_count": collection.count() if collection else 0,
            "vector_store_path": self.persistence_path
        }