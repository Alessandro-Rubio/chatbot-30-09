from typing import List, Dict, Optional
from datetime import datetime, timedelta
import re

class MetadataSearchEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def search_with_metadata_filters(self, query: str, filters: Dict) -> List[Dict]:
        """Buscar con filtros de metadatos"""
        try:
            # Construir filtros para ChromaDB
            chroma_filters = self._build_chroma_filters(filters)
            
            # Realizar búsqueda con filtros
            results = self.vector_store.similarity_search(
                query, 
                k=filters.get('top_k', 5),
                filter=chroma_filters
            )
            
            return self._format_results(results)
            
        except Exception as e:
            print(f"Error en búsqueda con metadatos: {str(e)}")
            return []
    
    def _build_chroma_filters(self, filters: Dict) -> Dict:
        """Construir filtros compatibles con ChromaDB"""
        chroma_filters = {}
        
        for key, value in filters.items():
            if key == 'file_type' and value:
                chroma_filters['file_type'] = value
            
            elif key == 'source_file' and value:
                chroma_filters['source_file'] = value
            
            elif key == 'date_range' and value:
                # Filtrar por rango de fechas
                start_date, end_date = self._parse_date_range(value)
                if start_date and end_date:
                    chroma_filters['file_modified'] = {
                        '$gte': start_date,
                        '$lte': end_date
                    }
            
            elif key == 'file_size' and value:
                # Filtrar por tamaño de archivo
                min_size, max_size = self._parse_size_range(value)
                if min_size is not None or max_size is not None:
                    size_filter = {}
                    if min_size is not None:
                        size_filter['$gte'] = min_size
                    if max_size is not None:
                        size_filter['$lte'] = max_size
                    chroma_filters['file_size'] = size_filter
        
        return chroma_filters
    
    def _parse_date_range(self, date_range: str) -> tuple:
        """Parsear rango de fechas"""
        try:
            if date_range == 'last_week':
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)
            elif date_range == 'last_month':
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
            elif date_range == 'last_year':
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
            else:
                # Formato personalizado: "2024-01-01:2024-12-31"
                start_str, end_str = date_range.split(':')
                start_date = datetime.fromisoformat(start_str)
                end_date = datetime.fromisoformat(end_str)
            
            return str(start_date.timestamp()), str(end_date.timestamp())
        except:
            return None, None
    
    def _parse_size_range(self, size_range: str) -> tuple:
        """Parsear rango de tamaños"""
        try:
            if ':' in size_range:
                min_str, max_str = size_range.split(':')
                min_size = int(min_str) if min_str else None
                max_size = int(max_str) if max_str else None
            else:
                min_size = int(size_range)
                max_size = None
            
            return min_size, max_size
        except:
            return None, None
    
    def _format_results(self, results) -> List[Dict]:
        """Formatear resultados de búsqueda"""
        formatted_results = []
        
        for doc in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'source_file': doc.metadata.get('source_file', 'Desconocido'),
                'file_type': doc.metadata.get('file_type', 'Desconocido'),
                'score': getattr(doc, 'score', 0.0)
            })
        
        return formatted_results
    
    def get_available_filters(self) -> Dict:
        """Obtener filtros disponibles"""
        return {
            "file_type": "Tipo de archivo (pdf, docx, xlsx, etc.)",
            "source_file": "Nombre específico del archivo",
            "date_range": "Rango de fechas (last_week, last_month, last_year, o formato YYYY-MM-DD:YYYY-MM-DD)",
            "file_size": "Tamaño del archivo en bytes (min:max o solo min)"
        }