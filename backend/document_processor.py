import os
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict
import hashlib

class AdvancedDocumentProcessor:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.supported_extensions = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader
        }
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    def _get_loader_for_file(self, file_path: str):
        """Obtener loader apropiado para el tipo de archivo"""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext in self.supported_extensions:
            return self.supported_extensions[ext](file_path)
        else:
            raise ValueError(f"Formato no soportado: {ext}")
    
    def load_and_chunk_documents(self) -> List[Document]:
        """Cargar y dividir todos los documentos con metadatos enriquecidos"""
        all_documents = []
        
        for filename in os.listdir(self.data_dir):
            if filename == "file_metadata.json":
                continue
                
            file_path = os.path.join(self.data_dir, filename)
            
            try:
                loader = self._get_loader_for_file(file_path)
                documents = loader.load()
                
                # Enriquecer metadatos
                for doc in documents:
                    doc.metadata.update({
                        "source_file": filename,
                        "file_type": os.path.splitext(filename)[1],
                        "chunk_hash": hashlib.md5(
                            doc.page_content.encode()
                        ).hexdigest()[:8]
                    })
                
                # Dividir en chunks
                chunks = self.text_splitter.split_documents(documents)
                all_documents.extend(chunks)
                
                print(f"Procesado {filename}: {len(chunks)} chunks")
                
            except Exception as e:
                print(f"Error procesando {filename}: {str(e)}")
                continue
        
        print(f"Total chunks generados: {len(all_documents)}")
        return all_documents
    
    def get_processing_stats(self) -> Dict:
        """Obtener estadísticas del procesamiento"""
        files = [f for f in os.listdir(self.data_dir) 
                if f != "file_metadata.json"]
        return {
            "total_files": len(files),
            "supported_files": files,
            "supported_extensions": list(self.supported_extensions.keys())
        }