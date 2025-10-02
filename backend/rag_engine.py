# rag_engine.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.retrievers import BM25Retriever  # Corrected import
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import re
import json
import hashlib

# --- New Component 1: Advanced Intent Detector ---
class AdvancedIntentDetector:
    def __init__(self):
        self.general_intents = self._load_general_intents()
        self.rag_intents = self._load_rag_intents()
        
    def _load_general_intents(self) -> Dict:
        """Cargar patrones de intención general"""
        return {
            'greeting': {
                'patterns': [
                    r'hola', r'hello', r'hi', r'hey', r'buenos días', r'buenas tardes', 
                    r'buenas noches', r'qué tal', r'cómo estás'
                ],
                'weight': 1.0
            },
            'farewell': {
                'patterns': [
                    r'adiós', r'bye', r'chao', r'hasta luego', r'nos vemos', 
                    r'que tengas buen día', r'gracias', r'thanks'
                ],
                'weight': 1.0
            },
            'identity': {
                'patterns': [
                    r'quién eres', r'cuál es tu nombre', r'qué eres', 
                    r'qué puedes hacer', r'tu función'
                ],
                'weight': 0.9
            },
            'small_talk': {
                'patterns': [
                    r'cómo estás', r'qué opinas', r'cuéntame un chiste',
                    r'qué tiempo hace', r'hablamos', r'conversemos'
                ],
                'weight': 0.8
            },
            'help': {
                'patterns': [
                    r'ayuda', r'help', r'qué puedes hacer', r'funciones',
                    r'cómo usar', r'instrucciones'
                ],
                'weight': 0.9
            }
        }
    
    def _load_rag_intents(self) -> Dict:
        """Cargar patrones de intención que requieren RAG"""
        return {
            'document_query': {
                'patterns': [
                    r'documento', r'archivo', r'pdf', r'informe', r'reporte',
                    r'según.*documento', r'en el.*archivo', r'en el.*pdf',
                    r'contiene.*documento', r'menciona.*archivo'
                ],
                'weight': 0.95
            },
            'specific_content': {
                'patterns': [
                    r'procedimiento', r'política', r'protocolo', r'guía', 
                    r'manual', r'especificación', r'requisito', r'norma',
                    r'cláusula', r'artículo', r'contrato', r'acuerdo'
                ],
                'weight': 0.9
            },
            'data_query': {
                'patterns': [
                    r'datos', r'estadística', r'número', r'cifra', 
                    r'porcentaje', r'gráfico', r'tabla', r'figura',
                    r'resultado', r'métrica', r'indicador'
                ],
                'weight': 0.85
            },
            'technical_query': {
                'patterns': [
                    r'cómo funciona', r'paso a paso', r'instrucciones',
                    r'método', r'técnica', r'proceso', r'flujo',
                    r'diagrama', r'esquema', r'metodología'
                ],
                'weight': 0.8
            },
            'search_query': {
                'patterns': [
                    r'busca', r'encuentra', r'localiza', r'dónde está',
                    r'qué dice sobre', r'información sobre', 
                    r'detalles de', r'explicación de'
                ],
                'weight': 0.75
            }
        }
    
    def detect_intent(self, question: str) -> Tuple[str, float, Dict]:
        """Detectar intención de la pregunta"""
        question_lower = question.lower().strip()
        
        # Calcular scores para cada tipo de intención
        general_score = self._calculate_intent_score(question_lower, self.general_intents)
        rag_score = self._calculate_intent_score(question_lower, self.rag_intents)
        
        # Determinar intención principal
        if general_score > rag_score and general_score > 0.3:
            intent_type = "general"
            confidence = general_score
            intent_details = self._get_intent_details(question_lower, self.general_intents)
        elif rag_score > 0.3:
            intent_type = "rag"
            confidence = rag_score
            intent_details = self._get_intent_details(question_lower, self.rag_intents)
        else:
            intent_type = "unknown"
            confidence = max(general_score, rag_score)
            intent_details = {"category": "unknown", "patterns_found": []}
        
        return intent_type, confidence, intent_details
    
    def _calculate_intent_score(self, question: str, intents: Dict) -> float:
        """Calcular score de intención"""
        total_score = 0.0
        max_possible_score = sum(intent['weight'] for intent in intents.values())
        
        for intent_name, intent_data in intents.items():
            for pattern in intent_data['patterns']:
                if re.search(pattern, question, re.IGNORECASE):
                    total_score += intent_data['weight']
                    break  # Solo contar una vez por intención
        
        return total_score / max_possible_score if max_possible_score > 0 else 0
    
    def _get_intent_details(self, question: str, intents: Dict) -> Dict:
        """Obtener detalles específicos de la intención detectada"""
        found_patterns = []
        detected_category = "unknown"
        
        for intent_name, intent_data in intents.items():
            for pattern in intent_data['patterns']:
                if re.search(pattern, question, re.IGNORECASE):
                    found_patterns.append(pattern)
                    detected_category = intent_name
                    break
        
        return {
            "category": detected_category,
            "patterns_found": found_patterns,
            "question_length": len(question),
            "word_count": len(question.split())
        }
    
    def should_use_rag(self, question: str, min_confidence: float = 0.3) -> Tuple[bool, Dict]:
        """Determinar si se debe usar RAG basado en la intención"""
        intent_type, confidence, details = self.detect_intent(question)
        
        use_rag = intent_type == "rag" and confidence >= min_confidence
        
        return use_rag, {
            "intent_type": intent_type,
            "confidence": confidence,
            "details": details,
            "threshold_used": min_confidence
        }

# --- New Component 2: Incremental Indexer ---
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

# --- New Component 3: Metadata Search Engine ---
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

# --- Main Hybrid RAG Engine with All Integrations ---
class HybridRAGEngine:
    def __init__(self, persistence_path: str = "./vector_store"):
        self.embedding_function = OllamaEmbeddings(model="nomic-embed-text")
        
        # Modelo quantizado para mayor velocidad
        self.llm = Ollama(
            model="llama3.1:8b-instruct-q4_K_M", 
            temperature=0.1,
            num_ctx=4096,
            num_thread=8
        )
        
        self.persistence_path = persistence_path
        
        # Inicializar nuevos componentes
        self.intent_detector = AdvancedIntentDetector()
        self.incremental_indexer = IncrementalIndexer()
        self.metadata_search = None
        
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        
    def initialize_vector_store(self, documents: List[Document], incremental: bool = False):
        """Inicializar sistema RAG con indexación incremental"""
        if not documents:
            print("No hay documentos para procesar")
            return
        
        try:
            if incremental and self.vector_store is not None:
                # Indexación incremental
                print("🔄 Realizando indexación incremental...")
                self.vector_store.add_documents(documents)
                print(f"✅ {len(documents)} documentos añadidos incrementalmente")
            else:
                # Indexación completa
                print("🚀 Realizando indexación completa...")
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedding_function,
                    persist_directory=self.persistence_path
                )
            
            # Inicializar componentes que dependen del vector store
            if self.vector_store:
                self.metadata_search = MetadataSearchEngine(self.vector_store)
                
                # Inicializar BM25 solo para indexación completa
                if not incremental:
                    self.bm25_retriever = BM25Retriever.from_documents(documents)
                    self.bm25_retriever.k = 2
                    
                    vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
                    self.ensemble_retriever = EnsembleRetriever(
                        retrievers=[vector_retriever, self.bm25_retriever],
                        weights=[0.7, 0.3]
                    )
            
            print("✅ Sistema RAG inicializado correctamente")
            
        except Exception as e:
            print(f"❌ Error inicializando RAG: {str(e)}")
    
    def _should_use_rag(self, question: str) -> Tuple[bool, List[Document]]:
        """Nueva detección de intención mejorada"""
        if not self.ensemble_retriever:
            return False, []
        
        try:
            # Usar el detector de intención avanzado
            use_rag, intent_info = self.intent_detector.should_use_rag(question)
            
            print(f"🎯 Intención detectada: {intent_info['intent_type']} "
                  f"(confianza: {intent_info['confidence']:.2f})")
            
            if use_rag:
                retrieved_docs = self.ensemble_retriever.get_relevant_documents(question)
                return len(retrieved_docs) > 0, retrieved_docs
            else:
                return False, []
                
        except Exception as e:
            print(f"⚠️ Error en detección de intención: {str(e)}")
            return False, []
    
    def query_with_rag(self, question: str, top_k: int = 3, metadata_filters: Optional[Dict] = None) -> str:
        """Consulta RAG con soporte para filtros de metadatos"""
        should_use_rag, relevant_docs = self._should_use_rag(question)
        
        if not should_use_rag:
            return self._generate_simple_response(question)
        
        try:
            # Aplicar filtros de metadatos si se especifican
            if metadata_filters and self.metadata_search:
                filtered_results = self.metadata_search.search_with_metadata_filters(question, metadata_filters)
                if filtered_results:
                    # Convertir resultados filtrados a formato Document
                    relevant_docs = [
                        Document(page_content=result['content'], metadata=result['metadata'])
                        for result in filtered_results
                    ]
                    print(f"🔍 Búsqueda con filtros aplicados: {len(relevant_docs)} resultados")
            
            if not relevant_docs:
                return "No se encontró información relevante con los filtros aplicados."
            
            # Construir contexto
            context = "\n\n".join([
                f"📄 Documento: {doc.metadata.get('source_file', 'Desconocido')}\n"
                f"📊 Tipo: {doc.metadata.get('file_type', 'N/A')}\n"
                f"📝 Contenido: {doc.page_content}"
                for doc in relevant_docs[:top_k]
            ])
            
            # Prompt mejorado
            prompt = f"""Basándote en el siguiente contexto de documentos, responde la pregunta de manera precisa.

{context}

Pregunta: {question}

Instrucciones:
1. Responde principalmente con la información del contexto
2. Si el contexto no contiene suficiente información, complementa con conocimiento general
3. Sé conciso pero informativo
4. Menciona las fuentes cuando sea relevante

Respuesta:"""
            
            response = self.llm.invoke(prompt)
            return response
            
        except Exception as e:
            return f"Error en la consulta RAG: {str(e)}"
    
    def _generate_fast_response(self, question: str) -> str:
        """Generar respuesta rápida para preguntas generales"""
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
    
    def get_system_info(self) -> Dict:
        """Obtener información completa del sistema"""
        rag_stats = self.get_rag_stats()
        index_stats = self.incremental_indexer.get_index_stats()
        intent_capabilities = {
            "general_intents": list(self.intent_detector.general_intents.keys()),
            "rag_intents": list(self.intent_detector.rag_intents.keys()),
            "metadata_filters": self.metadata_search.get_available_filters() if self.metadata_search else {}
        }
        
        return {
            **rag_stats,
            **index_stats,
            "intent_detection": intent_capabilities,
            "incremental_indexing": True,
            "metadata_search": self.metadata_search is not None
        }

# Import necesario para MetadataSearchEngine
from datetime import timedelta