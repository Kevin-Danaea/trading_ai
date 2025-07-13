"""
Qualitative Analysis Entity - Entidad de Análisis Cualitativo
============================================================

Entidad de dominio que almacena el análisis cualitativo realizado por LLM
sobre una oportunidad de trading.
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class QualitativeAnalysis:
    """
    Representa el análisis cualitativo de una oportunidad de trading realizado por LLM.
    
    Contiene el razonamiento, contexto de mercado, factores de riesgo y oportunidad,
    así como recomendaciones estratégicas generadas por inteligencia artificial.
    """
    
    # Análisis principal
    reasoning: str  # Razonamiento detallado del LLM
    market_context: str  # Contexto del mercado actual
    
    # Factores identificados
    risk_factors: List[str]  # Factores de riesgo identificados
    opportunity_factors: List[str]  # Factores de oportunidad identificados
    
    # Análisis de estrategia
    recommended_strategy: str  # 'grid', 'dca', 'btd'
    strategy_reasoning: str  # Por qué esta estrategia es la mejor
    alternative_strategies_notes: str  # Por qué las otras no son tan buenas
    direction: str  # 'long', 'short'
    direction_reasoning: str  # Por qué LONG o SHORT
    
    # Análisis de futuros
    suitable_for_futures: bool  # Si es apto para futuros
    futures_recommendation: str  # Recomendación para futuros esta semana
    optimal_leverage: str  # 'x3', 'x5', 'x10', 'x20'
    futures_risk_level: str  # 'low', 'medium', 'high', 'extreme'
    futures_timing: str  # Análisis de timing para futuros
    
    # Recomendaciones estratégicas
    strategic_notes: str  # Notas estratégicas específicas
    confidence_level: str  # 'high', 'medium', 'low'
    recommendation: str  # 'buy', 'hold', 'avoid'
    
    # Metadatos
    analysis_timestamp: datetime
    execution_notes: Optional[str] = None  # Notas sobre ejecución
    
    def __post_init__(self):
        """Validaciones post-inicialización."""
        valid_confidence_levels = ['high', 'medium', 'low']
        if self.confidence_level.lower() not in valid_confidence_levels:
            raise ValueError(f"confidence_level debe ser uno de {valid_confidence_levels}")
        
        valid_recommendations = ['buy', 'hold', 'avoid']
        if self.recommendation.lower() not in valid_recommendations:
            raise ValueError(f"recommendation debe ser uno de {valid_recommendations}")
        
        valid_strategies = ['grid', 'dca', 'btd']
        if self.recommended_strategy.lower() not in valid_strategies:
            raise ValueError(f"recommended_strategy debe ser uno de {valid_strategies}")
        
        valid_directions = ['long', 'short']
        if self.direction.lower() not in valid_directions:
            raise ValueError(f"direction debe ser uno de {valid_directions}")
        
        valid_leverages = ['x3', 'x5', 'x10', 'x20']
        if self.optimal_leverage.lower() not in valid_leverages:
            raise ValueError(f"optimal_leverage debe ser uno de {valid_leverages}")
        
        valid_futures_risk_levels = ['low', 'medium', 'high', 'extreme']
        if self.futures_risk_level.lower() not in valid_futures_risk_levels:
            raise ValueError(f"futures_risk_level debe ser uno de {valid_futures_risk_levels}")
    
    @property
    def risk_score(self) -> float:
        """Calcula score de riesgo basado en factores identificados (0-100)."""
        if not self.risk_factors:
            return 0.0
        
        # Más factores de riesgo = mayor score de riesgo
        risk_count = len(self.risk_factors)
        return min(100.0, risk_count * 15.0)  # Máximo 100
    
    @property
    def opportunity_score(self) -> float:
        """Calcula score de oportunidad basado en factores identificados (0-100)."""
        if not self.opportunity_factors:
            return 0.0
        
        # Más factores de oportunidad = mayor score
        opp_count = len(self.opportunity_factors)
        return min(100.0, opp_count * 20.0)  # Máximo 100
    
    @property
    def confidence_score(self) -> float:
        """Convierte confidence_level a score numérico (0-100)."""
        confidence_mapping = {
            'high': 85.0,
            'medium': 60.0,
            'low': 30.0
        }
        return confidence_mapping.get(self.confidence_level.lower(), 60.0)
    
    @property
    def recommendation_score(self) -> float:
        """Convierte recommendation a score numérico (0-100)."""
        recommendation_mapping = {
            'buy': 85.0,
            'hold': 50.0,
            'avoid': 15.0
        }
        return recommendation_mapping.get(self.recommendation.lower(), 50.0)
    
    @property
    def is_positive_analysis(self) -> bool:
        """Determina si el análisis es positivo."""
        return (
            self.recommendation.lower() == 'buy' and
            self.confidence_level.lower() in ['high', 'medium']
        )
    
    @property
    def is_negative_analysis(self) -> bool:
        """Determina si el análisis es negativo."""
        return (
            self.recommendation.lower() == 'avoid' or
            (self.recommendation.lower() == 'hold' and self.confidence_level.lower() == 'low')
        )
    
    def get_summary(self) -> dict:
        """Retorna resumen del análisis para logging/display."""
        return {
            'recommendation': self.recommendation.upper(),
            'confidence_level': self.confidence_level.upper(),
            'risk_factors_count': len(self.risk_factors),
            'opportunity_factors_count': len(self.opportunity_factors),
            'risk_score': f"{self.risk_score:.1f}/100",
            'opportunity_score': f"{self.opportunity_score:.1f}/100",
            'confidence_score': f"{self.confidence_score:.1f}/100",
            'recommendation_score': f"{self.recommendation_score:.1f}/100",
            'is_positive': self.is_positive_analysis,
            'is_negative': self.is_negative_analysis,
            'analysis_timestamp': self.analysis_timestamp.isoformat()
        }
    
    def get_detailed_analysis(self) -> dict:
        """Retorna análisis detallado para revisión profunda."""
        return {
            'reasoning': self.reasoning,
            'market_context': self.market_context,
            'strategy_analysis': {
                'recommended_strategy': self.recommended_strategy,
                'strategy_reasoning': self.strategy_reasoning,
                'alternative_strategies_notes': self.alternative_strategies_notes
            },
            'risk_factors': self.risk_factors,
            'opportunity_factors': self.opportunity_factors,
            'strategic_notes': self.strategic_notes,
            'execution_notes': self.execution_notes,
            'confidence_level': self.confidence_level,
            'recommendation': self.recommendation,
            'scores': {
                'risk_score': self.risk_score,
                'opportunity_score': self.opportunity_score,
                'confidence_score': self.confidence_score,
                'recommendation_score': self.recommendation_score
            },
            'analysis_timestamp': self.analysis_timestamp.isoformat()
        } 