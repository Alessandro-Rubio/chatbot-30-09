from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
import os
from typing import List, Dict

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
    
    def query_with_rag(self, question: str, top_k: int = 5) -> str:
        """Consulta usando RAG híbrido"""
        if not self.ensemble_retriever:
            return "Sistema RAG no inicializado. Por favor, carga documentos primero."
        
        try:
            # Recuperación híbrida
            retrieved_docs = self.ensemble_retriever.get_relevant_documents(question)
            
            if not retrieved_docs:
                return "No se encontró información relevante en los documentos."
            
            # Construir contexto
            context = "\n\n".join([
                f"Documento: {doc.metadata.get('source_file', 'Desconocido')}\n"
                f"Contenido: {doc.page_content}"
                for doc in retrieved_docs
            ])
            
            # Prompt mejorado
            prompt = f"""Basándote en el siguiente contexto de documentos, responde la pregunta de manera precisa.

Contexto:
{context}

Pregunta: {question}

Instrucciones:
1. Responde solo con la información proporcionada en el contexto
2. Si el contexto no contiene la respuesta, indica claramente que no hay información suficiente
3. Mantén la respuesta concisa y relevante
4. Si es relevante, menciona de qué documento proviene la información

Respuesta:"""
            
            response = self.llm.invoke(prompt)
            return response
            
        except Exception as e:
            return f"Error en la consulta RAG: {str(e)}"
    
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