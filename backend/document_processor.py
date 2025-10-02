import os
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    UnstructuredFileLoader  # Loader genérico para otros formatos
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Tuple
import hashlib

class AdvancedDocumentProcessor:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        
        # Soporte extendido para formatos de archivo
        self.supported_extensions = {
            # Documentos
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader,
            '.rtf': UnstructuredFileLoader,
            
            # Presentaciones
            '.pptx': UnstructuredPowerPointLoader,
            '.ppt': UnstructuredPowerPointLoader,
            '.odp': UnstructuredFileLoader,
            
            # Hojas de cálculo
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader,
            '.csv': CSVLoader,
            '.ods': UnstructuredFileLoader,
            
            # Otros formatos
            '.html': UnstructuredFileLoader,
            '.htm': UnstructuredFileLoader,
            '.xml': UnstructuredFileLoader,
            '.epub': UnstructuredFileLoader,
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
            loader_class = self.supported_extensions[ext]
            return loader_class(file_path)
        else:
            # Intentar con loader genérico para formatos no soportados explícitamente
            try:
                return UnstructuredFileLoader(file_path)
            except:
                raise ValueError(f"Formato no soportado: {ext}")
    
    def load_and_chunk_documents(self, specific_files: List[str] = None) -> Tuple[List[Document], List[str]]:
        """Cargar y dividir documentos, retornando documentos procesados y archivos fallidos"""
        all_documents = []
        failed_files = []
        
        files_to_process = specific_files if specific_files else [
            f for f in os.listdir(self.data_dir) 
            if f != "file_metadata.json" and not f.startswith('.')
        ]
        
        for filename in files_to_process:
            file_path = os.path.join(self.data_dir, filename)
            
            try:
                loader = self._get_loader_for_file(file_path)
                documents = loader.load()
                
                # Enriquecer metadatos con información extendida
                for doc in documents:
                    doc.metadata.update({
                        "source_file": filename,
                        "file_type": os.path.splitext(filename)[1],
                        "file_name": filename,
                        "file_size": os.path.getsize(file_path),
                        "file_modified": str(os.path.getmtime(file_path)),
                        "chunk_hash": hashlib.md5(doc.page_content.encode()).hexdigest()[:8],
                        "processing_date": str(datetime.now().isoformat())
                    })
                
                # Dividir en chunks
                chunks = self.text_splitter.split_documents(documents)
                all_documents.extend(chunks)
                
                print(f"✅ Procesado {filename}: {len(chunks)} chunks")
                
            except Exception as e:
                print(f"❌ Error procesando {filename}: {str(e)}")
                failed_files.append(filename)
                continue
        
        print(f"📊 Total: {len(all_documents)} chunks generados, {len(failed_files)} archivos fallidos")
        return all_documents, failed_files
    
    def get_supported_formats(self) -> Dict:
        """Obtener lista de formatos soportados"""
        return {
            "documentos": ['.pdf', '.txt', '.docx', '.doc', '.rtf'],
            "presentaciones": ['.pptx', '.ppt', '.odp'],
            "hojas_calculo": ['.xlsx', '.xls', '.csv', '.ods'],
            "otros": ['.html', '.htm', '.xml', '.epub']
        }
    
    def validate_file_format(self, filename: str) -> bool:
        """Validar si un archivo tiene formato soportado"""
        _, ext = os.path.splitext(filename)
        return ext.lower() in self.supported_extensions