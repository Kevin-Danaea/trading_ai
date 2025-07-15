"""
Domain Strategies - Estrategias de Trading del Dominio
=====================================================

Módulo que expone todas las estrategias de trading disponibles.
Estas estrategias son lógica de dominio pura.
"""

# No importar estrategias aquí para evitar ciclos de importación

# Diccionario para facilitar el acceso dinámico
AVAILABLE_STRATEGIES = {
    'grid': None,  # Se debe importar dinámicamente
    'dca': None,
    'btd': None,
    'futures_grid': None
}
