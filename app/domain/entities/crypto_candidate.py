"""
Crypto Candidate Entity - Entidad Candidato Cripto
=================================================

Entidad del dominio que representa un candidato de criptomoneda
analizado por el sistema de trading.

Esta es una entidad pura del dominio - sin dependencias externas.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class CryptoCandidate:
    """
    Entidad de dominio para representar un candidato cripto analizado.
    
    Contiene toda la información necesaria para evaluar si una criptomoneda
    es un buen candidato para trading basándose en análisis técnico y fundamental.
    """
    symbol: str
    market_cap_rank: int
    current_price: float
    volatility_24h: float
    volatility_7d: float
    adx: float
    sentiment_score: float
    sentiment_ma7: float
    volume_24h: float
    volume_change_24h: float
    price_change_24h: float
    price_change_7d: float
    score: float  # Puntuación final calculada (0-100)
    reasons: List[str]  # Razones por las que es buen candidato
    
    def is_high_score(self, threshold: float = 70.0) -> bool:
        """Determina si el candidato tiene una puntuación alta."""
        return self.score >= threshold
    
    def is_suitable_for_grid(self) -> bool:
        """Determina si es adecuado para estrategia Grid Trading."""
        # Grid funciona bien en mercados laterales con volatilidad moderada
        return (self.adx < 30 and  # Mercado lateral
                0.02 <= self.volatility_7d <= 0.08)  # Volatilidad moderada
    
    def is_suitable_for_dca(self) -> bool:
        """Determina si es adecuado para estrategia DCA."""
        # DCA funciona bien en tendencias alcistas con sentimiento positivo
        return (self.sentiment_score > 0 and  # Sentimiento positivo
                self.price_change_7d > -5)  # No en caída fuerte
    
    def is_suitable_for_btd(self) -> bool:
        """Determina si es adecuado para estrategia Buy The Dip."""
        # BTD funciona bien en dips con potencial de recuperación
        return (self.price_change_24h < -2 and  # En dip reciente
                self.volume_24h > 5_000_000)  # Con volumen suficiente
    
    def get_recommended_strategies(self) -> List[str]:
        """Obtiene las estrategias recomendadas para este candidato."""
        strategies = []
        
        if self.is_suitable_for_grid():
            strategies.append('grid')
        
        if self.is_suitable_for_dca():
            strategies.append('dca')
            
        if self.is_suitable_for_btd():
            strategies.append('btd')
        
        return strategies if strategies else ['grid']  # Grid como fallback
    
    def __str__(self) -> str:
        """Representación string del candidato."""
        return f"CryptoCandidate({self.symbol}, score={self.score:.1f})"
    
    def __repr__(self) -> str:
        """Representación detallada del candidato."""
        return (f"CryptoCandidate(symbol='{self.symbol}', "
                f"score={self.score:.1f}, "
                f"price=${self.current_price:.4f}, "
                f"vol_7d={self.volatility_7d*100:.1f}%)") 