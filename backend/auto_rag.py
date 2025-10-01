import os
import asyncio
from typing import Dict, List
from file_manager import EnhancedFileManager
from document_processor import AdvancedDocumentProcessor
from rag_engine import HybridRAGEngine
import time

class AutoRAGManager:
    def __init__(self):
        self.file_manager = EnhancedFileManager()
        self.doc_processor = AdvancedDocumentProcessor()
        self.rag_engine = HybridRAGEngine()
        self.initialized = False
    
    async def initialize_rag_system(self) -> Dict:
        """Initialize RAG system on startup"""
        try:
            print("🔄 Auto-initializing RAG system...")
            
            # Check if vector store exists and is current
            if self._should_rebuild_vector_store():
                documents = self.doc_processor.load_and_chunk_documents()
                if documents:
                    self.rag_engine.initialize_vector_store(documents)
                    # Mark all files as processed
                    for filename in self.file_manager.metadata["files"]:
                        self.file_manager.mark_file_processed(filename)
                    
                    self.initialized = True
                    print("✅ RAG system initialized successfully")
                    return {"status": "initialized", "documents_processed": len(documents)}
                else:
                    print("⚠️ No documents found for RAG initialization")
                    return {"status": "no_documents"}
            else:
                # Load existing vector store
                self.rag_engine.initialize_vector_store([])
                self.initialized = True
                print("✅ Existing RAG system loaded")
                return {"status": "loaded_existing"}
                
        except Exception as e:
            print(f"❌ RAG initialization failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _should_rebuild_vector_store(self) -> bool:
        """Check if vector store needs rebuilding"""
        # Rebuild if no vector store exists
        if not os.path.exists(self.rag_engine.persistence_path):
            return True
        
        # Rebuild if there are unprocessed files
        if self.file_manager.get_unprocessed_files():
            return True
        
        # TODO: Add more sophisticated checks (file modifications, etc.)
        return False
    
    async def handle_file_operation(self, operation_type: str, filename: str = None):
        """Handle file operations and trigger RAG updates"""
        if operation_type == "upload":
            # Mark file for processing
            unprocessed = self.file_manager.get_unprocessed_files()
            if unprocessed:
                print(f"🔄 Processing {len(unprocessed)} new files...")
                await self.initialize_rag_system()
        
        elif operation_type == "delete":
            # Rebuild RAG system after deletion
            print("🔄 Rebuilding RAG system after file deletion...")
            await self.initialize_rag_system()
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            "rag_initialized": self.initialized,
            "total_files": self.file_manager.metadata["stats"]["total_files"],
            "unprocessed_files": len(self.file_manager.get_unprocessed_files()),
            "vector_store_path": self.rag_engine.persistence_path
        }