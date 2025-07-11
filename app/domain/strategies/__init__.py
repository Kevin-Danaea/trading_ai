"""
Domain Strategies - Estrategias de Trading del Dominio
=====================================================

Módulo que expone todas las estrategias de trading disponibles.
Estas estrategias son lógica de dominio pura.
"""

from .grid_trading import GridTradingStrategy
from .dca import DCAStrategy
from .btd import BTDStrategy

# Exportar todas las estrategias disponibles
__all__ = [
    'GridTradingStrategy',
    'DCAStrategy', 
    'BTDStrategy'
]

# Diccionario para facilitar el acceso dinámico
AVAILABLE_STRATEGIES = {
    'grid': GridTradingStrategy,
    'dca': DCAStrategy,
    'btd': BTDStrategy
}
