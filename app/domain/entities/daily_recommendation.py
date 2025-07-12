"""
Daily Recommendation Entity - Entidad de Recomendación Diaria
============================================================

Entidad de dominio que representa una recomendación de trading diaria completa,
incluyendo análisis cuantitativo y cualitativo, lista para guardarse en la base
de datos y generar reportes para Telegram.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime
from decimal import Decimal

from .trading_opportunity import TradingOpportunity
from .qualitative_analysis import QualitativeAnalysis


@dataclass
class RecomendacionDiaria:
    """
    Representa una recomendación de trading diaria completa con análisis 
    cuantitativo y cualitativo.
    
    Esta entidad combina:
    - Oportunidad de trading (análisis cuantitativo)
    - Análisis cualitativo de Gemini AI
    - Información consolidada para BD y reportes
    """
    
    # === INFORMACIÓN BÁSICA ===
    fecha: datetime
    simbolo: str  # BTC/USDT
    
    # === ESTRATEGIA RECOMENDADA ===
    estrategia_recomendada: str  # 'grid', 'dca', 'btd'
    parametros_optimizados: Dict[str, Any]
    
    # === ANÁLISIS CUANTITATIVO ===
    roi_porcentaje: float
    sharpe_ratio: float
    max_drawdown_porcentaje: float
    win_rate_porcentaje: float
    total_trades: int
    score_final: float  # 0-100
    score_ajustado_riesgo: float  # 0-100
    
    # === ANÁLISIS CUALITATIVO (GEMINI AI) ===
    estrategia_gemini: str  # Estrategia recomendada por Gemini
    razon_gemini: str  # Razón de la recomendación
    fortalezas_gemini: str  # Fortalezas identificadas
    riesgos_gemini: str  # Riesgos identificados
    condiciones_mercado_gemini: str  # Análisis de condiciones de mercado
    score_confianza_gemini: float  # 0-100
    
    # === CONSENSO CUANTITATIVO vs CUALITATIVO ===
    consenso_estrategia: bool  # True si coinciden las estrategias
    diferencia_scores: Optional[float]  # Diferencia entre scores si aplica
    
    # === INFORMACIÓN ADICIONAL ===
    categoria_rendimiento: str  # PREMIUM, AGGRESSIVE, CONSERVATIVE, BALANCED
    condiciones_mercado: str  # bullish, bearish, sideways
    periodo_backtesting_dias: int
    
    # === METADATOS ===
    creado_en: datetime
    version_pipeline: str  # Para tracking de versiones
    
    def __post_init__(self):
        """Validaciones post-inicialización."""
        if not (0 <= self.score_final <= 100):
            raise ValueError(f"score_final debe estar entre 0-100, recibido: {self.score_final}")
        
        if not (0 <= self.score_confianza_gemini <= 100):
            raise ValueError(f"score_confianza_gemini debe estar entre 0-100, recibido: {self.score_confianza_gemini}")
        
        if self.estrategia_recomendada not in ['grid', 'dca', 'btd']:
            raise ValueError(f"estrategia_recomendada debe ser 'grid', 'dca' o 'btd', recibido: {self.estrategia_recomendada}")
    
    @classmethod
    def from_trading_opportunity_and_analysis(
        cls,
        opportunity: TradingOpportunity,
        qualitative_analysis: QualitativeAnalysis,
        version_pipeline: str = "1.0"
    ) -> 'RecomendacionDiaria':
        """
        Crea una RecomendacionDiaria a partir de una TradingOpportunity y QualitativeAnalysis.
        
        Args:
            opportunity: Oportunidad de trading con análisis cuantitativo
            qualitative_analysis: Análisis cualitativo de Gemini AI
            version_pipeline: Versión del pipeline para tracking
            
        Returns:
            RecomendacionDiaria completamente formada
        """
        return cls(
            # Información básica
            fecha=datetime.now(),
            simbolo=opportunity.symbol,
            
            # Estrategia recomendada
            estrategia_recomendada=opportunity.recommended_strategy_name,
            parametros_optimizados=opportunity.optimized_params,
            
            # Análisis cuantitativo
            roi_porcentaje=opportunity.roi_percentage,
            sharpe_ratio=opportunity.sharpe_ratio,
            max_drawdown_porcentaje=opportunity.max_drawdown_percentage,
            win_rate_porcentaje=opportunity.win_rate_percentage,
            total_trades=opportunity.total_trades,
            score_final=opportunity.final_score,
            score_ajustado_riesgo=opportunity.risk_adjusted_score,
            
            # Análisis cualitativo
            estrategia_gemini=qualitative_analysis.recommended_strategy,
            razon_gemini=qualitative_analysis.strategy_reasoning,
            fortalezas_gemini='; '.join(qualitative_analysis.opportunity_factors),
            riesgos_gemini='; '.join(qualitative_analysis.risk_factors),
            condiciones_mercado_gemini=qualitative_analysis.market_context,
            score_confianza_gemini=qualitative_analysis.confidence_score,
            
            # Consenso
            consenso_estrategia=(opportunity.recommended_strategy_name == qualitative_analysis.recommended_strategy),
            diferencia_scores=abs(opportunity.final_score - qualitative_analysis.confidence_score),
            
            # Información adicional
            categoria_rendimiento=opportunity.performance_category,
            condiciones_mercado=opportunity.market_conditions,
            periodo_backtesting_dias=opportunity.backtest_period_days,
            
            # Metadatos
            creado_en=datetime.now(),
            version_pipeline=version_pipeline
        )
    
    @property
    def es_consenso_positivo(self) -> bool:
        """Determina si hay consenso positivo entre análisis cuantitativo y cualitativo."""
        return (
            self.consenso_estrategia and
            self.score_final >= 70 and
            self.score_confianza_gemini >= 70
        )
    
    @property
    def nivel_riesgo(self) -> str:
        """Determina el nivel de riesgo basado en métricas."""
        if self.max_drawdown_porcentaje < 10:
            return "BAJO"
        elif self.max_drawdown_porcentaje < 20:
            return "MEDIO"
        else:
            return "ALTO"
    
    @property
    def recomendacion_final(self) -> str:
        """Genera recomendación final basada en consenso y métricas."""
        if self.es_consenso_positivo:
            return "FUERTE_COMPRA"
        elif self.consenso_estrategia and self.score_final >= 60:
            return "COMPRA"
        elif self.score_final >= 50:
            return "NEUTRAL_POSITIVO"
        else:
            return "NEUTRAL"
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumen completo de la recomendación."""
        return {
            'fecha': self.fecha.strftime('%Y-%m-%d'),
            'simbolo': self.simbolo,
            'estrategia_recomendada': self.estrategia_recomendada,
            'estrategia_gemini': self.estrategia_gemini,
            'consenso_estrategia': self.consenso_estrategia,
            'roi_porcentaje': f"{self.roi_porcentaje:.1f}%",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'max_drawdown_porcentaje': f"{self.max_drawdown_porcentaje:.1f}%",
            'win_rate_porcentaje': f"{self.win_rate_porcentaje:.1f}%",
            'score_final': f"{self.score_final:.1f}/100",
            'score_confianza_gemini': f"{self.score_confianza_gemini:.1f}/100",
            'categoria_rendimiento': self.categoria_rendimiento,
            'nivel_riesgo': self.nivel_riesgo,
            'recomendacion_final': self.recomendacion_final
        }
    
    def get_telegram_data(self) -> Dict[str, Any]:
        """Retorna datos formateados específicamente para Telegram."""
        return {
            'simbolo': self.simbolo,
            'estrategia_final': self.estrategia_gemini if self.consenso_estrategia else f"{self.estrategia_recomendada} (Quant) vs {self.estrategia_gemini} (AI)",
            'consenso': "✅ Consenso" if self.consenso_estrategia else "⚠️ Divergencia",
            'recomendacion': self.recomendacion_final,
            'roi': f"{self.roi_porcentaje:.1f}%",
            'drawdown': f"{self.max_drawdown_porcentaje:.1f}%",
            'win_rate': f"{self.win_rate_porcentaje:.1f}%",
            'nivel_riesgo': self.nivel_riesgo,
            'categoria': self.categoria_rendimiento,
            'razon_gemini': self.razon_gemini,
            'fortalezas': self.fortalezas_gemini,
            'riesgos': self.riesgos_gemini,
            'parametros': self.parametros_optimizados
        }
    
    def get_database_record(self) -> Dict[str, Any]:
        """Retorna diccionario para insertar en la base de datos."""
        return {
            'fecha': self.fecha,
            'simbolo': self.simbolo,
            'estrategia_recomendada': self.estrategia_recomendada,
            'estrategia_gemini': self.estrategia_gemini,
            'parametros_optimizados': self.parametros_optimizados,  # JSON
            'roi_porcentaje': Decimal(str(self.roi_porcentaje)),
            'sharpe_ratio': Decimal(str(self.sharpe_ratio)),
            'max_drawdown_porcentaje': Decimal(str(self.max_drawdown_porcentaje)),
            'win_rate_porcentaje': Decimal(str(self.win_rate_porcentaje)),
            'total_trades': self.total_trades,
            'score_final': Decimal(str(self.score_final)),
            'score_ajustado_riesgo': Decimal(str(self.score_ajustado_riesgo)),
            'razon_gemini': self.razon_gemini,
            'fortalezas_gemini': self.fortalezas_gemini,
            'riesgos_gemini': self.riesgos_gemini,
            'condiciones_mercado_gemini': self.condiciones_mercado_gemini,
            'score_confianza_gemini': Decimal(str(self.score_confianza_gemini)),
            'consenso_estrategia': self.consenso_estrategia,
            'diferencia_scores': Decimal(str(self.diferencia_scores)) if self.diferencia_scores else None,
            'categoria_rendimiento': self.categoria_rendimiento,
            'condiciones_mercado': self.condiciones_mercado,
            'periodo_backtesting_dias': self.periodo_backtesting_dias,
            'creado_en': self.creado_en,
            'version_pipeline': self.version_pipeline
        } 