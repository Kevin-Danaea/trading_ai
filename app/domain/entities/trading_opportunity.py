"""
Trading Opportunity Entity - Entidad de Oportunidad de Trading
==============================================================

Entidad de dominio que representa una oportunidad de trading completa,
incluyendo el candidato, la estrategia optimizada y las métricas de rendimiento.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime

from .crypto_candidate import CryptoCandidate


@dataclass
class StrategyResult:
    """
    Resultado de una estrategia específica para backtesting y optimización.
    """
    strategy_name: str  # 'grid', 'dca', 'btd'
    optimized_params: Dict[str, Any]
    
    # Métricas de rendimiento
    roi_percentage: float
    sharpe_ratio: float
    max_drawdown_percentage: float
    win_rate_percentage: float
    total_trades: int
    avg_trade_percentage: float
    volatility_percentage: float
    calmar_ratio: float
    sortino_ratio: float
    exposure_time_percentage: float
    
    # Información del proceso de optimización
    optimization_iterations: int
    optimization_duration_seconds: float
    confidence_level: float  # Confianza en la predicción (0-1)
    
    @property
    def performance_score(self) -> float:
        """Calcula un score de rendimiento combinado."""
        # Fórmula simple: ROI - penalización por drawdown + bonus por Sharpe
        return self.roi_percentage - (self.max_drawdown_percentage * 0.5) + (self.sharpe_ratio * 10)
    
    @property
    def risk_return_ratio(self) -> float:
        """Calcula ratio riesgo-retorno."""
        if self.max_drawdown_percentage <= 0:
            return float('inf')
        return abs(self.roi_percentage / self.max_drawdown_percentage)
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumen de la estrategia."""
        return {
            'strategy': self.strategy_name,
            'roi': f"{self.roi_percentage:.1f}%",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'max_drawdown': f"{self.max_drawdown_percentage:.1f}%",
            'win_rate': f"{self.win_rate_percentage:.1f}%",
            'total_trades': self.total_trades,
            'performance_score': f"{self.performance_score:.1f}",
            'confidence': f"{self.confidence_level:.2f}"
        }


@dataclass
class TradingOpportunity:
    """
    Representa una oportunidad de trading completa lista para decisión.
    
    Combina:
    - Candidato de criptomoneda (del scanner)
    - Estrategia optimizada (del optimizador bayesiano)
    - Métricas de rendimiento (del backtesting)
    - Score de ranking final
    """
    
    # Información del candidato base
    candidate: CryptoCandidate
    
    # Resultados de todas las estrategias analizadas
    strategy_results: Dict[str, StrategyResult]  # {'grid': StrategyResult, 'dca': StrategyResult, 'btd': StrategyResult}
    
    # Estrategia actualmente recomendada por el ranking cuantitativo
    recommended_strategy_name: str  # 'grid', 'dca', 'btd'
    
    # Información del proceso general
    backtest_period_days: int
    
    # Score final y ranking
    final_score: float  # 0-100, combinando múltiples factores
    risk_adjusted_score: float  # Score ajustado por riesgo
    
    # Metadatos
    created_at: datetime
    market_conditions: str  # 'bullish', 'bearish', 'sideways'
    
    def __post_init__(self):
        """Validaciones post-inicialización."""
        if self.final_score < 0 or self.final_score > 100:
            raise ValueError(f"final_score debe estar entre 0-100, recibido: {self.final_score}")
        
        if self.recommended_strategy_name not in self.strategy_results:
            raise ValueError(f"recommended_strategy_name '{self.recommended_strategy_name}' no encontrada en strategy_results")
    
    @property
    def symbol(self) -> str:
        """Retorna el símbolo de la criptomoneda."""
        return self.candidate.symbol
    
    @property
    def recommended_strategy(self) -> StrategyResult:
        """Retorna la estrategia actualmente recomendada."""
        return self.strategy_results[self.recommended_strategy_name]
    
    @property
    def strategy_name(self) -> str:
        """Retorna el nombre de la estrategia recomendada (para compatibilidad)."""
        return self.recommended_strategy_name
    
    @property
    def optimized_params(self) -> Dict[str, Any]:
        """Retorna los parámetros optimizados de la estrategia recomendada."""
        return self.recommended_strategy.optimized_params
    
    @property
    def roi_percentage(self) -> float:
        """ROI de la estrategia recomendada."""
        return self.recommended_strategy.roi_percentage
    
    @property
    def sharpe_ratio(self) -> float:
        """Sharpe ratio de la estrategia recomendada."""
        return self.recommended_strategy.sharpe_ratio
    
    @property
    def max_drawdown_percentage(self) -> float:
        """Max drawdown de la estrategia recomendada."""
        return self.recommended_strategy.max_drawdown_percentage
    
    @property
    def win_rate_percentage(self) -> float:
        """Win rate de la estrategia recomendada."""
        return self.recommended_strategy.win_rate_percentage
    
    @property
    def total_trades(self) -> int:
        """Total trades de la estrategia recomendada."""
        return self.recommended_strategy.total_trades
    
    @property
    def volatility_percentage(self) -> float:
        """Volatilidad de la estrategia recomendada."""
        return self.recommended_strategy.volatility_percentage
    
    @property
    def confidence_level(self) -> float:
        """Nivel de confianza de la estrategia recomendada."""
        return self.recommended_strategy.confidence_level
    
    @property
    def calmar_ratio(self) -> float:
        """Calmar ratio de la estrategia recomendada."""
        return self.recommended_strategy.calmar_ratio
    
    @property
    def sortino_ratio(self) -> float:
        """Sortino ratio de la estrategia recomendada."""
        return self.recommended_strategy.sortino_ratio
    
    @property
    def avg_trade_percentage(self) -> float:
        """Promedio de trade de la estrategia recomendada."""
        return self.recommended_strategy.avg_trade_percentage
    
    @property
    def exposure_time_percentage(self) -> float:
        """Tiempo de exposición de la estrategia recomendada."""
        return self.recommended_strategy.exposure_time_percentage
    
    @property
    def optimization_iterations(self) -> int:
        """Iteraciones de optimización de la estrategia recomendada."""
        return self.recommended_strategy.optimization_iterations
    
    @property
    def optimization_duration_seconds(self) -> float:
        """Duración de optimización de la estrategia recomendada."""
        return self.recommended_strategy.optimization_duration_seconds
    
    @property
    def risk_return_ratio(self) -> float:
        """Calcula ratio riesgo-retorno (ROI / Max Drawdown)."""
        if self.max_drawdown_percentage <= 0:
            return float('inf')
        return abs(self.roi_percentage / self.max_drawdown_percentage)
    
    @property
    def is_low_risk(self) -> bool:
        """Determina si es una oportunidad de bajo riesgo."""
        return (
            self.max_drawdown_percentage < 15.0 and
            self.volatility_percentage < 25.0 and
            self.sharpe_ratio > 1.0
        )
    
    @property
    def is_high_performance(self) -> bool:
        """Determina si es una oportunidad de alto rendimiento."""
        return (
            self.roi_percentage > 20.0 and
            self.sharpe_ratio > 1.5 and
            self.win_rate_percentage > 60.0
        )
    
    @property
    def performance_category(self) -> str:
        """Categoriza la oportunidad según rendimiento y riesgo."""
        if self.is_high_performance and self.is_low_risk:
            return "PREMIUM"  # Alto rendimiento, bajo riesgo
        elif self.is_high_performance:
            return "AGGRESSIVE"  # Alto rendimiento, riesgo moderado/alto
        elif self.is_low_risk:
            return "CONSERVATIVE"  # Bajo riesgo, rendimiento moderado
        else:
            return "BALANCED"  # Equilibrado
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna un resumen de la oportunidad para logging/display."""
        return {
            'symbol': self.symbol,
            'strategy': self.strategy_name,
            'roi': f"{self.roi_percentage:.1f}%",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'max_drawdown': f"{self.max_drawdown_percentage:.1f}%",
            'win_rate': f"{self.win_rate_percentage:.1f}%",
            'final_score': f"{self.final_score:.1f}/100",
            'risk_adjusted_score': f"{self.risk_adjusted_score:.1f}/100",
            'category': self.performance_category,
            'confidence': f"{self.confidence_level:.2f}"
        }
    
    def get_all_strategies_comparison(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna comparación de todas las estrategias para análisis cualitativo.
        
        Returns:
            Diccionario con métricas de todas las estrategias para enviar a Gemini AI
        """
        comparison = {}
        
        for strategy_name, strategy_result in self.strategy_results.items():
            comparison[strategy_name] = {
                'roi_percentage': strategy_result.roi_percentage,
                'sharpe_ratio': strategy_result.sharpe_ratio,
                'max_drawdown_percentage': strategy_result.max_drawdown_percentage,
                'win_rate_percentage': strategy_result.win_rate_percentage,
                'total_trades': strategy_result.total_trades,
                'volatility_percentage': strategy_result.volatility_percentage,
                'calmar_ratio': strategy_result.calmar_ratio,
                'sortino_ratio': strategy_result.sortino_ratio,
                'performance_score': strategy_result.performance_score,
                'risk_return_ratio': strategy_result.risk_return_ratio,
                'optimized_params': strategy_result.optimized_params,
                'confidence_level': strategy_result.confidence_level
            }
        
        return comparison
    
    def get_strategy_ranking(self) -> List[Dict[str, Any]]:
        """
        Retorna ranking de estrategias ordenado por performance score.
        
        Returns:
            Lista de estrategias ordenadas de mejor a peor
        """
        strategy_list = []
        
        for strategy_name, strategy_result in self.strategy_results.items():
            strategy_list.append({
                'strategy_name': strategy_name,
                'performance_score': strategy_result.performance_score,
                'roi_percentage': strategy_result.roi_percentage,
                'sharpe_ratio': strategy_result.sharpe_ratio,
                'max_drawdown_percentage': strategy_result.max_drawdown_percentage,
                'is_recommended': strategy_name == self.recommended_strategy_name
            })
        
        # Ordenar por performance score descendente
        strategy_list.sort(key=lambda x: x['performance_score'], reverse=True)
        
        return strategy_list
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Retorna métricas detalladas para análisis profundo."""
        return {
            # Básicas
            'symbol': self.symbol,
            'strategy': self.strategy_name,
            'optimized_params': self.optimized_params,
            
            # Rendimiento
            'roi_percentage': self.roi_percentage,
            'sharpe_ratio': self.sharpe_ratio,
            'calmar_ratio': self.calmar_ratio,
            'sortino_ratio': self.sortino_ratio,
            
            # Riesgo
            'max_drawdown_percentage': self.max_drawdown_percentage,
            'volatility_percentage': self.volatility_percentage,
            'risk_return_ratio': self.risk_return_ratio,
            
            # Trading
            'win_rate_percentage': self.win_rate_percentage,
            'total_trades': self.total_trades,
            'avg_trade_percentage': self.avg_trade_percentage,
            'exposure_time_percentage': self.exposure_time_percentage,
            
            # Scores
            'final_score': self.final_score,
            'risk_adjusted_score': self.risk_adjusted_score,
            'confidence_level': self.confidence_level,
            
            # Metadatos
            'optimization_iterations': self.optimization_iterations,
            'optimization_duration_seconds': self.optimization_duration_seconds,
            'backtest_period_days': self.backtest_period_days,
            'market_conditions': self.market_conditions,
            'performance_category': self.performance_category,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class RankingResult:
    """
    Resultado del proceso de ranking con las mejores oportunidades.
    """
    
    # Top oportunidades seleccionadas
    top_opportunities: List[TradingOpportunity]
    
    # Todas las oportunidades evaluadas (para referencia)
    all_opportunities: List[TradingOpportunity]
    
    # Metadatos del ranking
    ranking_criteria: Dict[str, float]  # Pesos usados para el ranking
    total_evaluated: int
    selected_count: int
    
    # Estadísticas del ranking
    avg_score: float
    best_score: float
    score_threshold: float
    
    # Distribución por categorías
    category_distribution: Dict[str, int]
    
    # Timestamp
    ranked_at: datetime
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumen del resultado de ranking."""
        return {
            'total_evaluated': self.total_evaluated,
            'selected_count': self.selected_count,
            'best_score': f"{self.best_score:.1f}/100",
            'avg_score': f"{self.avg_score:.1f}/100",
            'score_threshold': f"{self.score_threshold:.1f}/100",
            'top_symbols': [opp.symbol for opp in self.top_opportunities],
            'category_distribution': self.category_distribution,
            'ranked_at': self.ranked_at.isoformat()
        }
    
    def get_top_symbols(self) -> List[str]:
        """Retorna lista de símbolos de las top oportunidades."""
        return [opp.symbol for opp in self.top_opportunities]
    
    def get_best_opportunity(self) -> Optional[TradingOpportunity]:
        """Retorna la mejor oportunidad (score más alto)."""
        return self.top_opportunities[0] if self.top_opportunities else None 