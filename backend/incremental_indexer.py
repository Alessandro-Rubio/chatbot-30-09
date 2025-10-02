import os
import json
from datetime import datetime
from typing import List, Dict, Set
from langchain.schema import Document

class IncrementalIndexer:
    def __init__(self, index_state_file: str = "./data/index_state.json"):
        self.index_state_file = index_state_file
        self.index_state = self._load_index_state()
    
    def _load_index_state(self) -> Dict:
        """Cargar estado del índice"""
        if os.path.exists(self.index_state_file):
            try:
                with open(self.index_state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            "indexed_files": {},
            "last_update": None,
            "total_chunks": 0
        }
    
    def _save_index_state(self):
        """Guardar estado del índice"""
        os.makedirs(os.path.dirname(self.index_state_file), exist_ok=True)
        with open(self.index_state_file, 'w', encoding='utf-8') as f:
            json.dump(self.index_state, f, indent=2, ensure_ascii=False)
    
    def get_files_to_index(self, current_files: List[Dict]) -> List[Dict]:
        """Obtener archivos que necesitan indexación"""
        files_to_index = []
        
        for file_info in current_files:
            filename = file_info["filename"]
            file_metadata = file_info["metadata"]
            
            # Verificar si el archivo necesita reindexación
            if self._needs_reindexing(filename, file_metadata):
                files_to_index.append(file_info)
        
        return files_to_index
    
    def _needs_reindexing(self, filename: str, current_metadata: Dict) -> bool:
        """Determinar si un archivo necesita reindexación"""
        indexed_file = self.index_state["indexed_files"].get(filename)
        
        if not indexed_file:
            return True  # Archivo no indexado
        
        # Verificar si el hash cambió (archivo modificado)
        if indexed_file.get("hash") != current_metadata.get("hash"):
            return True
        
        # Verificar si ha pasado mucho tiempo desde la última indexación
        last_indexed = indexed_file.get("last_indexed")
        if last_indexed:
            last_indexed_date = datetime.fromisoformat(last_indexed)
            time_diff = datetime.now() - last_indexed_date
            if time_diff.days > 7:  # Reindexar cada 7 días
                return True
        
        return False
    
    def mark_files_as_indexed(self, filenames: List[str], chunks_count: int, file_metadata: Dict):
        """Marcar archivos como indexados"""
        for filename in filenames:
            self.index_state["indexed_files"][filename] = {
                "hash": file_metadata.get("hash"),
                "last_indexed": datetime.now().isoformat(),
                "chunks_count": chunks_count,
                "size": file_metadata.get("size")
            }
        
        self.index_state["last_update"] = datetime.now().isoformat()
        self.index_state["total_chunks"] += chunks_count
        self._save_index_state()
    
    def remove_file_from_index(self, filename: str):
        """Remover archivo del índice"""
        if filename in self.index_state["indexed_files"]:
            chunks_count = self.index_state["indexed_files"][filename].get("chunks_count", 0)
            del self.index_state["indexed_files"][filename]
            self.index_state["total_chunks"] = max(0, self.index_state["total_chunks"] - chunks_count)
            self._save_index_state()
    
    def get_index_stats(self) -> Dict:
        """Obtener estadísticas del índice"""
        return {
            "total_indexed_files": len(self.index_state["indexed_files"]),
            "total_chunks": self.index_state["total_chunks"],
            "last_update": self.index_state["last_update"]
        }