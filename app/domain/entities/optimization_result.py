"""
Optimization Result Entity - Entidad Resultado de Optimización
============================================================

Entidad del dominio que representa el resultado de una optimización
de estrategia para una criptomoneda específica.

Esta es una entidad pura del dominio - sin dependencias externas.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class OptimizationResult:
    """
    Entidad de dominio para representar el resultado de optimización.
    
    Contiene toda la información sobre la optimización bayesiana
    realizada para una estrategia y símbolo específicos.
    """
    symbol: str
    strategy: str
    best_params: Dict[str, Any]
    best_value: float
    optimization_time: float
    trials_completed: int
    study_stats: Dict[str, Any]
    
    def is_successful(self) -> bool:
        """Determina si la optimización fue exitosa."""
        return (self.best_value > 0 and 
                self.trials_completed > 10 and
                bool(self.best_params))
    
    def get_roi(self) -> float:
        """Obtiene el ROI de los stats si está disponible."""
        return self.study_stats.get('roi', 0.0)
    
    def get_sharpe_ratio(self) -> float:
        """Obtiene el Sharpe Ratio de los stats si está disponible."""
        return self.study_stats.get('sharpe_ratio', 0.0)
    
    def get_max_drawdown(self) -> float:
        """Obtiene el Max Drawdown de los stats si está disponible."""
        return self.study_stats.get('max_drawdown', 0.0)
    
    def get_win_rate(self) -> float:
        """Obtiene el Win Rate de los stats si está disponible."""
        return self.study_stats.get('win_rate', 0.0)
    
    def is_high_performance(self, min_roi: float = 10.0, min_sharpe: float = 0.5) -> bool:
        """Determina si el resultado tiene alto rendimiento."""
        return (self.get_roi() >= min_roi and 
                self.get_sharpe_ratio() >= min_sharpe and
                self.get_max_drawdown() <= 20.0)
    
    def get_efficiency_score(self) -> float:
        """Calcula una puntuación de eficiencia de la optimización."""
        if self.optimization_time <= 0:
            return 0.0
        
        # Score basado en valor óptimo encontrado vs tiempo invertido
        time_hours = self.optimization_time / 3600
        return self.best_value / max(time_hours, 0.1)  # Evitar división por 0
    
    def __str__(self) -> str:
        """Representación string del resultado."""
        return f"OptimizationResult({self.symbol}, {self.strategy}, value={self.best_value:.2f})"
    
    def __repr__(self) -> str:
        """Representación detallada del resultado."""
        return (f"OptimizationResult(symbol='{self.symbol}', "
                f"strategy='{self.strategy}', "
                f"best_value={self.best_value:.2f}, "
                f"trials={self.trials_completed})") 