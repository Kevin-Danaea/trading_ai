"""
Domain Entities - Entidades del Dominio
======================================

Módulo que expone todas las entidades del dominio del sistema de trading.
Las entidades representan objetos de negocio con identidad.
"""

from .crypto_candidate import CryptoCandidate
from .optimization_result import OptimizationResult

# Exportar todas las entidades disponibles
__all__ = [
    'CryptoCandidate',
    'OptimizationResult'
]
