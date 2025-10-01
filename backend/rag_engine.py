from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import os

class RAGEngine:
    def __init__(self, persistence_path="./vector_store/chroma_db"):
        self.embedding_function = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = Ollama(model="llama3.1")
        self.persistence_path = persistence_path
        
        # Initialize or load vector store
        if os.path.exists(persistence_path):
            self.vector_store = Chroma(
                persist_directory=persistence_path,
                embedding_function=self.embedding_function
            )
        else:
            self.vector_store = None

    def initialize_vector_store(self, documents):
        """Create new vector store from documents"""
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=self.persistence_path
        )
        return "Vector store initialized successfully"

    def query(self, question, use_rag=True, top_k=3):
        """Query the RAG system. use_rag=False for simple questions"""
        if not use_rag or self.vector_store is None:
            # Direct LLM query for simple questions
            return self.llm.invoke(question)
        else:
            # RAG query with retrieved context
            retrieved_docs = self.vector_store.similarity_search(question, k=top_k)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            prompt = f"""Based on the following context, answer the question. If the context doesn't contain the answer, use your general knowledge.

Context:
{context}

Question: {question}

Answer:"""
            
            return self.llm.invoke(prompt)