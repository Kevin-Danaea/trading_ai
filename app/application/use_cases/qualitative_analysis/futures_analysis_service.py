"""
Futures Analysis Service - Servicio de Análisis de Futuros
==========================================================

Servicio especializado para análisis cualitativo de oportunidades de trading en futuros.
Utiliza un prompt específico optimizado para el análisis de apalancamiento, riesgo y 
timing de futuros.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from app.domain.entities.crypto_candidate import CryptoCandidate
from app.domain.entities.qualitative_analysis import QualitativeAnalysis
from app.infrastructure.providers.sentiment_data_provider import SentimentDataProvider
from app.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)


class FuturesAnalysisService:
    """
    Servicio especializado para análisis cualitativo de futuros.
    
    Se enfoca en:
    - Análisis de apalancamiento óptimo
    - Evaluación de riesgo específico para futuros
    - Timing de mercado para posiciones apalancadas
    - Análisis de funding rates y open interest
    """
    
    def __init__(self, sentiment_provider: SentimentDataProvider):
        """
        Inicializa el servicio de análisis de futuros.
        
        Args:
            sentiment_provider: Proveedor de datos de sentimiento
        """
        self.sentiment_provider = sentiment_provider
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa el modelo de Google Gemini para análisis de futuros."""
        try:
            import google.genai as genai
            
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
            self.model_name = 'gemini-2.5-pro'
            logger.info("✅ Cliente Gemini inicializado para análisis de futuros")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Gemini para futuros: {e}")
            self.client = None
    
    def analyze_futures_opportunity(self, candidate: CryptoCandidate) -> Optional[QualitativeAnalysis]:
        """
        Analiza una oportunidad de trading específicamente para futuros.
        
        Args:
            candidate: Candidato de crypto para análisis
            
        Returns:
            Análisis cualitativo específico para futuros o None si falla
        """
        if not self.client:
            logger.error("❌ Cliente Gemini no disponible para análisis de futuros")
            return None
        
        try:
            # Obtener datos de mercado actuales
            market_data = self._get_market_data(candidate.symbol)
            
            # Construir prompt especializado para futuros
            prompt = self._build_futures_prompt(candidate, market_data)
            
            # Generar análisis
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            # Procesar respuesta
            response_text = response.text if hasattr(response, 'text') else str(response)
            analysis = self._process_futures_response(str(response_text), candidate)
            
            if analysis:
                logger.info(f"✅ Análisis de futuros completado para {candidate.symbol}")
                return analysis
            else:
                logger.warning(f"⚠️ No se pudo procesar análisis de futuros para {candidate.symbol}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error en análisis de futuros para {candidate.symbol}: {e}")
            return None
    
    def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Obtiene datos de mercado específicos para futuros.
        
        Args:
            symbol: Símbolo del activo
            
        Returns:
            Datos de mercado relevantes para futuros
        """
        try:
            # Obtener datos básicos de sentimiento
            sentiment_df = self.sentiment_provider.get_sentiment_data(days_back=7)
            sentiment_score = self.sentiment_provider.calculate_sentiment_score(sentiment_df)
            
            # Datos específicos para futuros (simulados por ahora)
            futures_data = {
                'sentiment': 'neutral',
                'sentiment_score': sentiment_score,
                'fear_greed_index': 50,
                'volatility': 'medium',
                'funding_rate': 0.01,  # 1% funding rate
                'open_interest_24h_change': 5.2,  # 5.2% increase
                'volume_24h_change': 12.5,  # 12.5% increase
                'long_short_ratio': 1.8,  # 1.8:1 ratio
                'liquidation_data': {
                    'longs_liquidated_24h': 2500000,  # $2.5M
                    'shorts_liquidated_24h': 1800000   # $1.8M
                }
            }
            
            return futures_data
            
        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo datos de mercado para {symbol}: {e}")
            return {
                'sentiment': 'neutral',
                'sentiment_score': 50.0,
                'fear_greed_index': 50,
                'volatility': 'medium',
                'funding_rate': 0.01,
                'open_interest_24h_change': 0.0,
                'volume_24h_change': 0.0,
                'long_short_ratio': 1.0,
                'liquidation_data': {
                    'longs_liquidated_24h': 0,
                    'shorts_liquidated_24h': 0
                }
            }
    
    def _build_futures_prompt(self, candidate: CryptoCandidate, market_data: Dict[str, Any]) -> str:
        """
        Construye el prompt especializado para análisis de futuros.
        
        Args:
            candidate: Candidato de crypto
            market_data: Datos de mercado
            
        Returns:
            Prompt optimizado para análisis de futuros
        """
        return f"""
ANÁLISIS ESPECIALIZADO DE FUTUROS - {candidate.symbol}
=================================================

Eres un experto en trading de futuros de criptomonedas. Analiza esta oportunidad ESPECÍFICAMENTE para trading con apalancamiento.

DATOS CUANTITATIVOS:
- Símbolo: {candidate.symbol}
- Precio Actual: ${candidate.current_price:.4f}
- Cambio 24h: {candidate.price_change_24h:.1f}%
- Cambio 7d: {candidate.price_change_7d:.1f}%
- Volatilidad 7d: {candidate.volatility_7d:.1f}%
- Volumen 24h: ${candidate.volume_24h:,.0f}
- Score Scanner: {candidate.score:.1f}/100

DATOS DE FUTUROS:
- Funding Rate: {market_data.get('funding_rate', 0):.3f}%
- Open Interest (24h): {market_data.get('open_interest_24h_change', 0):+.1f}%
- Volumen (24h): {market_data.get('volume_24h_change', 0):+.1f}%
- Ratio Long/Short: {market_data.get('long_short_ratio', 1):.1f}
- Liquidaciones Long: ${market_data.get('liquidation_data', {}).get('longs_liquidated_24h', 0):,.0f}
- Liquidaciones Short: ${market_data.get('liquidation_data', {}).get('shorts_liquidated_24h', 0):,.0f}

SENTIMIENTO DE MERCADO:
- Sentimiento General: {market_data.get('sentiment', 'neutral')}
- Sentiment Score: {market_data.get('sentiment_score', 50):.1f}/100
- Fear & Greed Index: {market_data.get('fear_greed_index', 50)}
- Volatilidad: {market_data.get('volatility', 'medium')}

INSTRUCCIONES ESPECÍFICAS PARA FUTUROS:

1. **ANÁLISIS DE APALANCAMIENTO**: Evalúa el apalancamiento óptimo considerando:
   - Volatilidad del activo
   - Liquidaciones recientes
   - Funding rates
   - Experiencia requerida del trader

2. **EVALUACIÓN DE RIESGO**: Clasifica el riesgo específico para futuros:
   - low: Activos estables, bajo apalancamiento (x3-x5)
   - medium: Volatilidad moderada, apalancamiento medio (x5-x10)
   - high: Alta volatilidad, apalancamiento alto (x10-x20)
   - extreme: Condiciones extremas, máximo cuidado

3. **TIMING DE MERCADO**: Determina si ES EL MOMENTO ADECUADO para operar futuros:
   - Considera funding rates
   - Analiza sentiment extremos
   - Evalúa liquidaciones recientes
   - Revisa open interest

4. **DIRECCIONALIDAD**: Basándote en todos los datos, determina:
   - LONG o SHORT
   - Razón técnica y fundamental
   - Nivel de confianza

**RESPONDE EN FORMATO JSON EXACTO:**

```json
{{
  "reasoning": "Tu análisis detallado específico para futuros: funding rates, liquidaciones, volatilidad, timing semanal",
  "market_context": "Contexto del mercado actual específico para futuros y cómo afecta el apalancamiento",
  "strategy_analysis": {{
    "recommended_strategy": "grid/dca/btd",
    "strategy_reasoning": "Por qué esta estrategia es la mejor para futuros con apalancamiento",
    "alternative_strategies": "Por qué las otras estrategias no son tan buenas para futuros",
    "direction": "long/short",
    "direction_reasoning": "Explicación técnica de por qué LONG o SHORT para futuros"
  }},
  "futures_analysis": {{
    "suitable_for_futures": true/false,
    "futures_recommendation": "Recomendación específica para trading de futuros con apalancamiento",
    "optimal_leverage": "x3/x5/x10/x20",
    "futures_risk_level": "low/medium/high/extreme",
    "futures_timing": "¿Operar futuros ESTA SEMANA? Análisis de timing específico"
  }},
  "risk_factors": ["Riesgo 1 específico para futuros", "Riesgo 2 específico para futuros", "Riesgo 3 específico para futuros"],
  "opportunity_factors": ["Fortaleza 1 específica para futuros", "Fortaleza 2 específica para futuros", "Fortaleza 3 específica para futuros"],
  "strategic_notes": "Notas estratégicas específicas para la estrategia recomendada en futuros",
  "confidence_level": "high/medium/low",
  "recommendation": "buy/hold/avoid",
  "execution_notes": "Notas sobre cuándo y cómo ejecutar la estrategia de futuros con apalancamiento"
}}
```

**FORMATO OBLIGATORIO - NO CAMBIES LOS NOMBRES DE CAMPOS:**

- **recommended_strategy**: DEBE ser exactamente "grid", "dca", o "btd" (sin espacios, sin mayúsculas)
- **direction**: DEBE ser exactamente "long" o "short" (sin mayúsculas)
- **confidence_level**: DEBE ser exactamente "high", "medium", o "low" (sin mayúsculas)
- **recommendation**: DEBE ser exactamente "buy", "hold", o "avoid" (sin mayúsculas)
- **suitable_for_futures**: DEBE ser exactamente true o false (boolean)
- **optimal_leverage**: DEBE ser exactamente "x3", "x5", "x10", o "x20"
- **futures_risk_level**: DEBE ser exactamente "low", "medium", "high", o "extreme"

**CRITERIOS DE EVALUACIÓN PARA FUTUROS:**
- ✅ **buy**: Excelente oportunidad para futuros con riesgo controlado y timing adecuado
- ⚠️ **hold**: Oportunidad interesante pero timing incierto o volatilidad alta
- ❌ **avoid**: Demasiado riesgo para apalancamiento o condiciones desfavorables

**IMPORTANTE**: Analiza específicamente para trading de futuros con apalancamiento. Considera funding rates, liquidaciones, volatilidad y timing semanal. Tu análisis determinará si usamos apalancamiento real.
"""
    
    def _process_futures_response(self, response: str, candidate: CryptoCandidate) -> Optional[QualitativeAnalysis]:
        """
        Procesa la respuesta JSON del modelo para análisis de futuros.
        
        Args:
            response: Respuesta del modelo en formato JSON
            candidate: Candidato original
            
        Returns:
            Análisis cualitativo procesado o None si falla
        """
        try:
            # Parsear JSON igual que el servicio principal
            analysis_data = self._parse_gemini_response(response)
            if not analysis_data:
                logger.error("❌ No se pudo parsear la respuesta JSON de futuros")
                return None
            
            # Normalizar datos usando el mismo método que el servicio principal
            normalized_data = self._normalize_gemini_response(analysis_data)
            
            # Extraer campos específicos
            strategy_analysis = normalized_data.get('strategy_analysis', {})
            futures_analysis = normalized_data.get('futures_analysis', {})
            
            # Crear análisis cualitativo
            return QualitativeAnalysis(
                reasoning=normalized_data.get('reasoning', ''),
                market_context=normalized_data.get('market_context', ''),
                risk_factors=normalized_data.get('risk_factors', []),
                opportunity_factors=normalized_data.get('opportunity_factors', []),
                recommended_strategy=strategy_analysis.get('recommended_strategy', 'grid'),
                strategy_reasoning=strategy_analysis.get('strategy_reasoning', ''),
                alternative_strategies_notes=strategy_analysis.get('alternative_strategies', ''),
                direction=strategy_analysis.get('direction', 'long'),
                direction_reasoning=strategy_analysis.get('direction_reasoning', ''),
                suitable_for_futures=futures_analysis.get('suitable_for_futures', False),
                futures_recommendation=futures_analysis.get('futures_recommendation', ''),
                optimal_leverage=futures_analysis.get('optimal_leverage', 'x3'),
                futures_risk_level=futures_analysis.get('futures_risk_level', 'medium'),
                futures_timing=futures_analysis.get('futures_timing', 'no'),
                strategic_notes=normalized_data.get('strategic_notes', ''),
                confidence_level=normalized_data.get('confidence_level', 'medium'),
                recommendation=normalized_data.get('recommendation', 'hold'),
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Error procesando respuesta JSON de futuros: {e}")
            return None
    
    def _parse_gemini_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parsea la respuesta JSON de Gemini.
        
        Args:
            response_text: Texto de respuesta de Gemini
            
        Returns:
            Diccionario parseado o None si falla
        """
        try:
            import json
            
            # Buscar JSON en la respuesta
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            
            logger.warning("⚠️ No se encontró JSON válido en la respuesta de Gemini para futuros")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error parseando JSON de Gemini para futuros: {e}")
            return None
    
    def _normalize_gemini_response(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normaliza la respuesta de Gemini para futuros.
        
        Args:
            analysis_data: Datos de análisis parseados
            
        Returns:
            Datos normalizados
        """
        try:
            # Normalizar campos básicos
            normalized = {
                'reasoning': str(analysis_data.get('reasoning', '')),
                'market_context': str(analysis_data.get('market_context', '')),
                'risk_factors': analysis_data.get('risk_factors', []),
                'opportunity_factors': analysis_data.get('opportunity_factors', []),
                'strategic_notes': str(analysis_data.get('strategic_notes', '')),
                'confidence_level': str(analysis_data.get('confidence_level', 'medium')).lower(),
                'recommendation': str(analysis_data.get('recommendation', 'hold')).lower()
            }
            
            # Normalizar strategy_analysis
            strategy_analysis = analysis_data.get('strategy_analysis', {})
            normalized['strategy_analysis'] = {
                'recommended_strategy': str(strategy_analysis.get('recommended_strategy', 'grid')).lower(),
                'strategy_reasoning': str(strategy_analysis.get('strategy_reasoning', '')),
                'alternative_strategies': str(strategy_analysis.get('alternative_strategies', '')),
                'direction': str(strategy_analysis.get('direction', 'long')).lower(),
                'direction_reasoning': str(strategy_analysis.get('direction_reasoning', ''))
            }
            
            # Normalizar futures_analysis
            futures_analysis = analysis_data.get('futures_analysis', {})
            normalized['futures_analysis'] = {
                'suitable_for_futures': bool(futures_analysis.get('suitable_for_futures', False)),
                'futures_recommendation': str(futures_analysis.get('futures_recommendation', '')),
                'optimal_leverage': str(futures_analysis.get('optimal_leverage', 'x3')).lower(),
                'futures_risk_level': str(futures_analysis.get('futures_risk_level', 'medium')).lower(),
                'futures_timing': str(futures_analysis.get('futures_timing', 'no'))
            }
            
            return normalized
            
        except Exception as e:
            logger.error(f"❌ Error normalizando respuesta de Gemini para futuros: {e}")
            return {}
    
 