import re
from typing import Dict, Tuple, List
from datetime import datetime

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