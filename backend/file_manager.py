import os
import shutil
from fastapi import UploadFile
import hashlib
from typing import List, Dict
import json
from datetime import datetime

class EnhancedFileManager:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.metadata_file = os.path.join(data_dir, "file_metadata.json")
        os.makedirs(data_dir, exist_ok=True)
        self.metadata = {"files": {}, "stats": {"total_files": 0}}
        self._load_metadata()
    
    def _load_metadata(self):
        """Cargar metadatos de archivos existentes"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    file_content = f.read().strip()
                    if file_content:
                        loaded_metadata = json.loads(file_content)
                        # Ensure the structure is correct
                        if isinstance(loaded_metadata, dict) and "files" in loaded_metadata:
                            self.metadata = loaded_metadata
                        else:
                            # Handle old format by converting
                            self.metadata = {"files": loaded_metadata, "stats": {"total_files": len(loaded_metadata)}}
                            self._save_metadata()
                    else:
                        self.metadata = {"files": {}, "stats": {"total_files": 0}}
            except (json.JSONDecodeError, Exception) as e:
                print(f"⚠️ Error loading metadata: {e}. Initializing empty metadata")
                self.metadata = {"files": {}, "stats": {"total_files": 0}}
                # Backup corrupted file
                if os.path.exists(self.metadata_file):
                    backup_file = self.metadata_file + ".corrupted"
                    shutil.copy2(self.metadata_file, backup_file)
                    print(f"⚠️ Corrupted metadata backed up as: {backup_file}")
        else:
            self.metadata = {"files": {}, "stats": {"total_files": 0}}
    
    def _save_metadata(self):
        """Guardar metadatos a archivo"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calcular hash único del archivo"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    async def upload_file(self, file: UploadFile) -> Dict:
        """Subir archivo con validación"""
        file_path = os.path.join(self.data_dir, file.filename)
        
        # Guardar archivo
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Calcular hash y metadatos
        file_hash = self._calculate_file_hash(file_path)
        file_size = os.path.getsize(file_path)
        
        # Verificar si el archivo ya existe por hash
        for existing_file, meta in self.metadata["files"].items():
            if meta.get('hash') == file_hash:
                os.remove(file_path)  # Eliminar duplicado
                return {"status": "duplicate", "filename": file.filename}
        
        # Guardar metadatos
        self.metadata["files"][file.filename] = {
            "hash": file_hash,
            "size": file_size,
            "upload_time": datetime.now().isoformat(),
            "file_type": getattr(file, 'content_type', 'unknown')
        }
        self.metadata["stats"]["total_files"] = len(self.metadata["files"])
        self._save_metadata()
        
        return {"status": "success", "filename": file.filename}
    
    def delete_file(self, filename: str) -> bool:
        """Eliminar archivo"""
        file_path = os.path.join(self.data_dir, filename)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            os.remove(file_path)
            if filename in self.metadata["files"]:
                del self.metadata["files"][filename]
                self.metadata["stats"]["total_files"] = len(self.metadata["files"])
                self._save_metadata()
            return True
        return False
    
    def list_files(self) -> List[Dict]:
        """Listar todos los archivos con metadatos"""
        files = []
        for filename in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, filename)
            if filename == "file_metadata.json" or os.path.isdir(file_path):
                continue
            file_info = {
                "filename": filename,
                "metadata": self.metadata["files"].get(filename, {})
            }
            files.append(file_info)
        return files
    
    def get_file_count(self) -> int:
        """Obtener número de archivos"""
        return self.metadata["stats"]["total_files"]