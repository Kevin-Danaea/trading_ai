"""
Qualitative Filter Service - Servicio de Filtro Cualitativo
===========================================================

Servicio que aplica análisis cualitativo usando LLM (Gemini) sobre las mejores oportunidades.
El "Jefe de Estrategia" final que aplica sentido común y contexto de mercado.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from google import genai
from dataclasses import dataclass

from app.domain.entities import TradingOpportunity, QualitativeAnalysis
from app.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class QualitativeResult:
    """Resultado del análisis cualitativo de una oportunidad."""
    opportunity: TradingOpportunity
    analysis: QualitativeAnalysis
    confidence_score: float
    strategic_recommendation: str
    risk_assessment: str
    execution_priority: int  # 1-5 (1 = máxima prioridad)


class QualitativeFilterService:
    """
    Servicio que aplica filtro cualitativo usando Gemini AI.
    
    Toma las Top 3-5 oportunidades con mejores métricas cuantitativas
    y las pasa por análisis de "sentido común" estratégico.
    """
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        """
        Inicializa el servicio de filtro cualitativo.
        
        Args:
            gemini_api_key: API key de Gemini (opcional, usa settings si no se proporciona)
        """
        self.api_key = gemini_api_key or settings.GEMINI_API_KEY
        
        if not self.api_key:
            raise ValueError("❌ API Key de Gemini no configurada")
        
        # Configurar cliente de Gemini con la nueva API
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = 'gemini-2.5-pro'
        
        logger.info("🧠 QualitativeFilterService inicializado con Gemini")
    
    def analyze_opportunities(self, opportunities: List[TradingOpportunity]) -> List[QualitativeResult]:
        """
        Analiza cualitativamente las oportunidades usando Gemini.
        
        Args:
            opportunities: Lista de oportunidades a analizar
            
        Returns:
            Lista de resultados cualitativos ordenados por prioridad
        """
        logger.info(f"🔍 Iniciando análisis cualitativo de {len(opportunities)} oportunidades")
        
        results = []
        
        for i, opportunity in enumerate(opportunities, 1):
            logger.info(f"📊 Analizando oportunidad {i}/{len(opportunities)}: {opportunity.candidate.symbol}")
            
            try:
                # Obtener análisis cualitativo de Gemini
                analysis = self._get_gemini_analysis(opportunity)
                
                if analysis:
                    # Calcular métricas adicionales
                    confidence_score = self._calculate_confidence_score(opportunity, analysis)
                    strategic_recommendation = self._generate_strategic_recommendation(opportunity, analysis)
                    risk_assessment = self._assess_risk_level(opportunity, analysis)
                    execution_priority = self._determine_execution_priority(opportunity, analysis, confidence_score)
                    
                    result = QualitativeResult(
                        opportunity=opportunity,
                        analysis=analysis,
                        confidence_score=confidence_score,
                        strategic_recommendation=strategic_recommendation,
                        risk_assessment=risk_assessment,
                        execution_priority=execution_priority
                    )
                    
                    results.append(result)
                    logger.info(f"✅ Análisis completado para {opportunity.candidate.symbol}")
                else:
                    logger.warning(f"⚠️ No se pudo obtener análisis para {opportunity.candidate.symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Error analizando {opportunity.candidate.symbol}: {e}")
                continue
        
        # Ordenar por prioridad de ejecución
        results.sort(key=lambda x: x.execution_priority)
        
        logger.info(f"🎯 Análisis cualitativo completado: {len(results)} oportunidades procesadas")
        return results
    
    def _get_gemini_analysis(self, opportunity: TradingOpportunity) -> Optional[QualitativeAnalysis]:
        """
        Obtiene análisis cualitativo de Gemini para una oportunidad específica.
        
        Args:
            opportunity: Oportunidad a analizar
            
        Returns:
            Análisis cualitativo o None si falla
        """
        try:
            # Construir prompt maestro
            prompt = self._build_strategy_master_prompt(opportunity)
            
            # Enviar a Gemini con la nueva API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            # Parsear respuesta
            analysis_data = self._parse_gemini_response(str(response.text))
            
            if analysis_data:
                # Extraer análisis de estrategia
                strategy_analysis = analysis_data.get('strategy_analysis', {})
                
                return QualitativeAnalysis(
                    reasoning=analysis_data.get('reasoning', ''),
                    market_context=analysis_data.get('market_context', ''),
                    risk_factors=analysis_data.get('risk_factors', []),
                    opportunity_factors=analysis_data.get('opportunity_factors', []),
                    recommended_strategy=strategy_analysis.get('recommended_strategy', 'grid'),
                    strategy_reasoning=strategy_analysis.get('strategy_reasoning', 'Análisis no disponible'),
                    alternative_strategies_notes=strategy_analysis.get('alternative_strategies', 'No especificado'),
                    strategic_notes=analysis_data.get('strategic_notes', ''),
                    confidence_level=analysis_data.get('confidence_level', 'medium'),
                    recommendation=analysis_data.get('recommendation', 'hold'),
                    analysis_timestamp=datetime.now(),
                    execution_notes=analysis_data.get('execution_notes')
                )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo análisis de Gemini: {e}")
            return None
    
    def _build_strategy_master_prompt(self, opportunity: TradingOpportunity) -> str:
        """
        Construye el prompt maestro de jefe de estrategia.
        
        Args:
            opportunity: Oportunidad a analizar
            
        Returns:
            Prompt completo para Gemini
        """
        
        # Extraer datos cuantitativos
        candidate = opportunity.candidate
        all_strategies = opportunity.get_all_strategies_comparison()
        strategy_ranking = opportunity.get_strategy_ranking()
        
        prompt = f"""
# 🎯 ANÁLISIS ESTRATÉGICO DE OPORTUNIDAD DE TRADING

Eres un **Jefe de Estrategia** experto en trading de criptomonedas con 15 años de experiencia en mercados financieros.

## 📊 DATOS CUANTITATIVOS DE LA OPORTUNIDAD

### Criptomoneda: {candidate.symbol}
- **Precio Actual**: ${candidate.current_price:.4f}
- **Volumen 24h**: ${candidate.volume_24h:,.0f}
- **Market Cap Rank**: #{candidate.market_cap_rank}
- **Score Cuantitativo**: {candidate.score:.2f}/100

### ANÁLISIS DE LAS 3 ESTRATEGIAS OPTIMIZADAS

{self._format_all_strategies_comparison(all_strategies)}

### Ranking Cuantitativo de Estrategias
{self._format_strategy_ranking(strategy_ranking)}

### Contexto de Mercado
- **Ranking Global**: Top {candidate.market_cap_rank}/100
- **Volatilidad 7d**: {candidate.volatility_7d*100:.1f}%
- **Cambio 24h**: {candidate.price_change_24h:.2f}%
- **Sentimiento**: {candidate.sentiment_score:.2f}

---

## 🧠 TU MISIÓN: ANÁLISIS CUALITATIVO Y SELECCIÓN DE ESTRATEGIA

Aplica tu experiencia y sentido común para evaluar esta oportunidad. **IMPORTANTE**: Analiza las 3 estrategias y recomienda cuál es la mejor.

### Considera estos aspectos:

1. **Análisis de las 3 Estrategias**: ¿Cuál de las 3 estrategias (Grid, DCA, BTD) es más adecuada para esta criptomoneda?
2. **Contexto de Mercado**: ¿Cómo se ve esta oportunidad en el contexto actual?
3. **Riesgo vs Recompensa**: ¿Los números tienen sentido estratégico?
4. **Timing**: ¿Es buen momento para esta estrategia y criptomoneda?
5. **Factores Cualitativos**: ¿Qué no capturan los números?

### Características de cada estrategia:
- **GRID TRADING**: Ideal para mercados laterales con volatilidad controlada
- **DCA**: Mejor para tendencias alcistas sostenidas o mercados inciertos
- **BTD**: Óptimo para aprovechar dips en activos con fundamentos sólidos

**RESPONDE EN FORMATO JSON:**

```json
{{
  "reasoning": "Tu análisis detallado de por qué esta oportunidad es buena/mala/regular y por qué elegiste esa estrategia",
  "market_context": "Contexto del mercado actual y cómo afecta esta oportunidad",
  "strategy_analysis": {{
    "recommended_strategy": "grid/dca/btd",
    "strategy_reasoning": "Por qué esta estrategia es la mejor para esta criptomoneda",
    "alternative_strategies": "Breve comentario sobre por qué las otras 2 estrategias son menos adecuadas"
  }},
  "risk_factors": ["Factor de riesgo 1", "Factor de riesgo 2", "..."],
  "opportunity_factors": ["Factor positivo 1", "Factor positivo 2", "..."],
  "strategic_notes": "Notas estratégicas específicas para la estrategia recomendada",
  "confidence_level": "high/medium/low",
  "recommendation": "buy/hold/avoid",
  "execution_notes": "Notas sobre cuándo y cómo ejecutar la estrategia recomendada"
}}
```

**CRITERIOS DE EVALUACIÓN:**
- ✅ **BUY**: Oportunidad excelente con riesgo controlado y estrategia clara
- ⚠️ **HOLD**: Oportunidad interesante pero con reservas o timing incierto
- ❌ **AVOID**: Demasiado riesgo o métricas engañosas

**IMPORTANTE**: Debes elegir UNA de las 3 estrategias como recomendada y explicar por qué es mejor que las otras dos. Sé crítico pero constructivo. Tu análisis determinará si invertimos dinero real.
"""
        
        return prompt
    
    def _format_strategy_params(self, params: Dict[str, Any]) -> str:
        """Formatea los parámetros de estrategia para el prompt."""
        formatted = []
        for key, value in params.items():
            if isinstance(value, float):
                formatted.append(f"  - {key}: {value:.4f}")
            else:
                formatted.append(f"  - {key}: {value}")
        return "\n".join(formatted)
    
    def _format_all_strategies_comparison(self, all_strategies: Dict[str, Dict[str, Any]]) -> str:
        """Formatea la comparación de todas las estrategias para el prompt."""
        comparison_text = ""
        
        strategy_names = {
            'grid': 'GRID TRADING',
            'dca': 'DOLLAR COST AVERAGING (DCA)', 
            'btd': 'BUY THE DIP (BTD)'
        }
        
        for strategy_name, metrics in all_strategies.items():
            display_name = strategy_names.get(strategy_name, strategy_name.upper())
            comparison_text += f"""
#### {display_name}
- **ROI**: {metrics['roi_percentage']:.2f}%
- **Sharpe Ratio**: {metrics['sharpe_ratio']:.3f}
- **Max Drawdown**: {metrics['max_drawdown_percentage']:.2f}%
- **Win Rate**: {metrics['win_rate_percentage']:.2f}%
- **Total Trades**: {metrics['total_trades']}
- **Volatilidad**: {metrics['volatility_percentage']:.3f}%
- **Performance Score**: {metrics['performance_score']:.2f}
- **Risk/Return Ratio**: {metrics['risk_return_ratio']:.2f}

**Parámetros Optimizados**:
{self._format_strategy_params(metrics['optimized_params'])}
"""
        
        return comparison_text
    
    def _format_strategy_ranking(self, strategy_ranking: List[Dict[str, Any]]) -> str:
        """Formatea el ranking de estrategias para el prompt."""
        ranking_text = ""
        
        for i, strategy in enumerate(strategy_ranking, 1):
            status = "✅ RECOMENDADA" if strategy['is_recommended'] else ""
            ranking_text += f"""
{i}. **{strategy['strategy_name'].upper()}** {status}
   - Performance Score: {strategy['performance_score']:.2f}
   - ROI: {strategy['roi_percentage']:.2f}%
   - Sharpe: {strategy['sharpe_ratio']:.3f}
   - Drawdown: {strategy['max_drawdown_percentage']:.2f}%
"""
        
        return ranking_text
    
    def _parse_gemini_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parsea la respuesta JSON de Gemini.
        
        Args:
            response_text: Texto de respuesta de Gemini
            
        Returns:
            Diccionario parseado o None si falla
        """
        try:
            # Buscar JSON en la respuesta
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            
            logger.warning("⚠️ No se encontró JSON válido en la respuesta de Gemini")
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error parseando JSON de Gemini: {e}")
            return None
    
    def _calculate_confidence_score(self, opportunity: TradingOpportunity, analysis: QualitativeAnalysis) -> float:
        """
        Calcula un score de confianza combinando métricas cuantitativas y cualitativas.
        
        Args:
            opportunity: Oportunidad analizada
            analysis: Análisis cualitativo
            
        Returns:
            Score de confianza (0-100)
        """
        # Score base cuantitativo
        quant_score = opportunity.candidate.score
        
        # Bonus/penalty por análisis cualitativo
        confidence_bonus = {
            'high': 10,
            'medium': 0,
            'low': -15
        }.get(analysis.confidence_level.lower(), 0)
        
        recommendation_bonus = {
            'buy': 5,
            'hold': 0,
            'avoid': -20
        }.get(analysis.recommendation.lower(), 0)
        
        # Score final
        final_score = min(100, max(0, quant_score + confidence_bonus + recommendation_bonus))
        
        return final_score
    
    def _generate_strategic_recommendation(self, opportunity: TradingOpportunity, analysis: QualitativeAnalysis) -> str:
        """Genera recomendación estratégica final."""
        base_rec = analysis.recommendation.upper()
        
        if base_rec == 'BUY':
            return f"✅ EJECUTAR: {opportunity.candidate.symbol} con estrategia {opportunity.strategy_name.upper()}"
        elif base_rec == 'HOLD':
            return f"⚠️ MONITOREAR: {opportunity.candidate.symbol} - Esperar mejor momento"
        else:
            return f"❌ EVITAR: {opportunity.candidate.symbol} - Riesgo elevado"
    
    def _assess_risk_level(self, opportunity: TradingOpportunity, analysis: QualitativeAnalysis) -> str:
        """Evalúa el nivel de riesgo general."""
        risk_factors_count = len(analysis.risk_factors)
        max_drawdown = opportunity.max_drawdown_percentage
        
        if risk_factors_count >= 3 or max_drawdown > 25:
            return "🔴 ALTO RIESGO"
        elif risk_factors_count >= 2 or max_drawdown > 15:
            return "🟡 RIESGO MODERADO"
        else:
            return "🟢 BAJO RIESGO"
    
    def _determine_execution_priority(self, opportunity: TradingOpportunity, analysis: QualitativeAnalysis, confidence_score: float) -> int:
        """
        Determina prioridad de ejecución (1-5, donde 1 es máxima prioridad).
        
        Args:
            opportunity: Oportunidad analizada
            analysis: Análisis cualitativo
            confidence_score: Score de confianza
            
        Returns:
            Prioridad de ejecución (1-5)
        """
        if analysis.recommendation.lower() == 'avoid':
            return 5  # Mínima prioridad
        
        if confidence_score >= 85 and analysis.confidence_level.lower() == 'high':
            return 1  # Máxima prioridad
        elif confidence_score >= 75 and analysis.confidence_level.lower() in ['high', 'medium']:
            return 2  # Alta prioridad
        elif confidence_score >= 60:
            return 3  # Prioridad media
        else:
            return 4  # Baja prioridad
    
    def get_execution_summary(self, results: List[QualitativeResult]) -> Dict[str, Any]:
        """
        Genera resumen ejecutivo de todas las oportunidades analizadas.
        
        Args:
            results: Lista de resultados cualitativos
            
        Returns:
            Resumen ejecutivo
        """
        if not results:
            return {'error': 'No hay resultados para analizar'}
        
        # Filtrar recomendaciones
        buy_recommendations = [r for r in results if r.analysis.recommendation.lower() == 'buy']
        hold_recommendations = [r for r in results if r.analysis.recommendation.lower() == 'hold']
        avoid_recommendations = [r for r in results if r.analysis.recommendation.lower() == 'avoid']
        
        # Top 3 recomendaciones finales
        top_recommendations = sorted(buy_recommendations, key=lambda x: x.confidence_score, reverse=True)[:3]
        
        # Análisis de estrategias recomendadas por Gemini
        strategy_distribution = {}
        for result in results:
            strategy = result.analysis.recommended_strategy
            strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1
        
        return {
            'total_analyzed': len(results),
            'buy_count': len(buy_recommendations),
            'hold_count': len(hold_recommendations),
            'avoid_count': len(avoid_recommendations),
            'strategy_distribution': strategy_distribution,  # ¿Qué estrategias prefiere Gemini?
            'top_recommendations': [
                {
                    'symbol': r.opportunity.candidate.symbol,
                    'quantitative_strategy': r.opportunity.strategy_name,  # Lo que dice el ranking
                    'gemini_recommended_strategy': r.analysis.recommended_strategy,  # Lo que dice Gemini
                    'strategy_match': r.opportunity.strategy_name == r.analysis.recommended_strategy,  # ¿Coinciden?
                    'confidence_score': r.confidence_score,
                    'execution_priority': r.execution_priority,
                    'strategic_recommendation': r.strategic_recommendation,
                    'risk_assessment': r.risk_assessment,
                    'roi_expected': r.opportunity.roi_percentage,
                    'sharpe_ratio': r.opportunity.sharpe_ratio,
                    'strategy_reasoning': r.analysis.strategy_reasoning
                }
                for r in top_recommendations
            ],
            'analysis_timestamp': datetime.now(),
            'ready_for_execution': len(top_recommendations) > 0,
            'gemini_insights': {
                'strategy_preferences': strategy_distribution,
                'consensus_rate': len([r for r in results if r.opportunity.strategy_name == r.analysis.recommended_strategy]) / len(results) if results else 0
            }
        } 