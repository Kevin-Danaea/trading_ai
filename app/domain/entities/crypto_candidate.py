"""
Crypto Candidate Entity - Entidad Candidato de Criptomoneda
===========================================================

Entidad que representa un candidato de criptomoneda con sus métricas
y datos relevantes para el análisis de trading.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime

@dataclass
class CryptoCandidate:
    """
    Candidato de criptomoneda con métricas completas.
    
    Representa una criptomoneda que ha pasado el filtro inicial
    y contiene todas las métricas necesarias para el análisis.
    """
    
    # Identificación básica
    symbol: str
    market_cap_rank: int
    current_price: float
    
    # Métricas de volatilidad y riesgo (compatibilidad con scanner)
    volatility_24h: float
    volatility_7d: float
    adx: float
    
    # Métricas de sentimiento (compatibilidad con scanner)
    sentiment_score: float
    sentiment_ma7: float
    
    # Métricas de precio y volumen
    volume_24h: float
    volume_change_24h: float
    price_change_24h: float
    price_change_7d: float
    
    # Score final del scanner
    score: float
    reasons: List[str]
    
    # Campos adicionales para análisis avanzado
    name: Optional[str] = None
    market_cap: Optional[float] = None
    volatility_30d: Optional[float] = None
    beta: Optional[float] = None
    social_volume: Optional[float] = None
    developer_activity: Optional[float] = None
    rsi_14: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_position: Optional[float] = None
    
    # Datos para futuros
    leverage_available: Optional[float] = None
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    open_interest_change_24h: Optional[float] = None
    futures_volume_24h: Optional[float] = None
    
    # Timestamp
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Inicialización post-creación."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        # Asignar valores por defecto para campos opcionales
        if self.name is None:
            self.name = self.symbol.replace('/USDT', '').replace('USDT', '')
        
        if self.market_cap is None:
            self.market_cap = 1000000000.0  # Valor por defecto
        
        if self.volatility_30d is None:
            self.volatility_30d = self.volatility_7d * 1.2  # Estimación
        
        if self.beta is None:
            self.beta = 1.0 + (self.volatility_7d - 0.15) * 2  # Estimación basada en volatilidad
        
        if self.social_volume is None:
            self.social_volume = max(100, self.volume_24h / 1000000)  # Estimación
        
        if self.developer_activity is None:
            self.developer_activity = 50.0  # Valor neutro
        
        if self.rsi_14 is None:
            self.rsi_14 = 50.0  # Valor neutro
        
        if self.macd_signal is None:
            self.macd_signal = 0.0  # Valor neutro
        
        if self.bollinger_position is None:
            self.bollinger_position = 0.5  # Valor neutro
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte el candidato a diccionario.
        
        Returns:
            Diccionario con todos los datos del candidato
        """
        return {
            'symbol': self.symbol,
            'name': self.name,
            'current_price': self.current_price,
            'market_cap': self.market_cap,
            'market_cap_rank': self.market_cap_rank,
            'price_change_24h': self.price_change_24h,
            'price_change_7d': self.price_change_7d,
            'volume_24h': self.volume_24h,
            'volume_change_24h': self.volume_change_24h,
            'volatility_7d': self.volatility_7d,
            'volatility_30d': self.volatility_30d,
            'beta': self.beta,
            'sentiment_score': self.sentiment_score,
            'social_volume': self.social_volume,
            'developer_activity': self.developer_activity,
            'rsi_14': self.rsi_14,
            'macd_signal': self.macd_signal,
            'bollinger_position': self.bollinger_position,
            'score': self.score,
            'leverage_available': self.leverage_available,
            'funding_rate': self.funding_rate,
            'open_interest': self.open_interest,
            'open_interest_change_24h': self.open_interest_change_24h,
            'futures_volume_24h': self.futures_volume_24h,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
    
    def get_risk_level(self) -> str:
        """
        Determina el nivel de riesgo del candidato.
        
        Returns:
            Nivel de riesgo: 'BAJO', 'MEDIO', 'ALTO'
        """
        # Usar valores por defecto si son None
        volatility_7d = self.volatility_7d if self.volatility_7d is not None else 0.20
        beta = self.beta if self.beta is not None else 1.0
        
        if volatility_7d < 0.15 and beta < 1.2:
            return 'BAJO'
        elif volatility_7d < 0.25 and beta < 1.8:
            return 'MEDIO'
        else:
            return 'ALTO'
    
    def is_suitable_for_futures(self) -> bool:
        """
        Determina si el candidato es adecuado para trading de futuros.
        
        Returns:
            True si es adecuado para futuros, False en caso contrario
        """
        # Verificar que tengan datos de futuros
        if not all([self.leverage_available, self.funding_rate is not None, 
                   self.open_interest, self.futures_volume_24h]):
            return False
        
        # Usar valores seguros para comparaciones
        futures_volume = self.futures_volume_24h or 0
        open_interest_change = self.open_interest_change_24h or 0
        funding_rate = self.funding_rate or 0
        market_cap = self.market_cap or 0
        
        # Criterios de solidez para futuros
        criteria = [
            # Volumen de futuros suficiente (al menos $10M)
            futures_volume >= 10_000_000,
            
            # Open Interest estable o creciente
            open_interest_change >= -10,
            
            # Funding rate no extremo (entre -0.5% y 0.5%)
            abs(funding_rate) <= 0.005,
            
            # Volatilidad controlada pero suficiente para futuros
            0.10 <= self.volatility_7d <= 0.40,
            
            # Market cap suficiente para liquidez
            market_cap >= 100_000_000,
            
            # Ranking no muy bajo
            self.market_cap_rank <= 200
        ]
        
        return sum(criteria) >= 4  # Al menos 4 de 6 criterios
    
    def get_futures_risk_level(self) -> str:
        """
        Determina el nivel de riesgo específico para futuros.
        
        Returns:
            Nivel de riesgo para futuros: 'BAJO', 'MEDIO', 'ALTO'
        """
        if not self.is_suitable_for_futures():
            return 'ALTO'
        
        # Factores de riesgo específicos para futuros
        high_risk_factors = [
            self.volatility_7d > 0.30,
            self.funding_rate is not None and abs(self.funding_rate) > 0.003,
            self.open_interest_change_24h is not None and self.open_interest_change_24h < -20,
            self.futures_volume_24h is not None and self.futures_volume_24h < 50_000_000
        ]
        
        risk_score = sum(high_risk_factors)
        
        if risk_score == 0:
            return 'BAJO'
        elif risk_score <= 1:
            return 'MEDIO'
        else:
            return 'ALTO'
    
    def get_optimal_leverage_for_futures(self) -> float:
        """
        Calcula el leverage óptimo recomendado para futuros.
        
        Returns:
            Leverage recomendado (1.0 - leverage_available)
        """
        if not self.is_suitable_for_futures():
            return 1.0
        
        # Base leverage según volatilidad
        if self.volatility_7d <= 0.15:
            base_leverage = 5.0
        elif self.volatility_7d <= 0.25:
            base_leverage = 3.0
        else:
            base_leverage = 2.0
        
        # Ajustar por funding rate
        if self.funding_rate is not None and abs(self.funding_rate) > 0.002:
            base_leverage *= 0.8
        
        # Ajustar por open interest
        if self.open_interest_change_24h is not None and self.open_interest_change_24h < -10:
            base_leverage *= 0.7
        
        # Limitar al leverage disponible
        return min(base_leverage, self.leverage_available or 1.0)
    
    def __str__(self) -> str:
        """Representación string del candidato."""
        return f"CryptoCandidate({self.symbol}, score={self.score:.2f}, price=${self.current_price:.4f})"
    
    def __repr__(self) -> str:
        """Representación para debugging."""
        return self.__str__() 