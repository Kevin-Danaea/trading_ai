"""
Domain Entities - Entidades del Dominio
======================================

Módulo que expone todas las entidades del dominio del sistema de trading.
Las entidades representan objetos de negocio con identidad.
"""

from .crypto_candidate import CryptoCandidate
from .optimization_result import OptimizationResult
from .trading_opportunity import TradingOpportunity, RankingResult, StrategyResult
from .qualitative_analysis import QualitativeAnalysis
from .daily_recommendation import RecomendacionDiaria

# Exportar todas las entidades disponibles
__all__ = [
    'CryptoCandidate',
    'OptimizationResult',
    'TradingOpportunity',
    'RankingResult',
    'StrategyResult',
    'QualitativeAnalysis',
    'RecomendacionDiaria'
]
