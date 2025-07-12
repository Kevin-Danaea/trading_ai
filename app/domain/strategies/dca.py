"""
DCA Strategy - Estrategia de Dollar Cost Averaging
=================================================

Estrategia de Dollar Cost Averaging (DCA) modernizada.
Compra de forma periódica cuando detecta tendencias alcistas
y dips, vendiendo con objetivo de ganancia fijo.

Esta es lógica de dominio pura - sin dependencias externas.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from backtesting import Strategy
import logging

logger = logging.getLogger(__name__)


class DCAStrategy(Strategy):
    """
    Estrategia de Dollar Cost Averaging (DCA) modernizada.
    
    Compra de forma periódica cuando detecta tendencias alcistas
    y dips, vendiendo con objetivo de ganancia fijo.
    
    Parámetros:
    - intervalo_compra: Días entre compras (1-7)
    - monto_compra: Monto por compra como % del capital (0.1-1.0)
    - objetivo_ganancia: % de ganancia objetivo (0.05-0.30)
    - dip_threshold: % de caída para detectar dip (0.02-0.15)
    - tendencia_alcista_dias: Días para confirmar tendencia (3-14)
    - stop_loss: % de pérdida máxima (0.10-0.40)
    """
    
    # Parámetros optimizables
    intervalo_compra = 3
    monto_compra = 0.2  # 20% del capital por compra
    objetivo_ganancia = 0.15  # 15% de ganancia
    dip_threshold = 0.05  # 5% de caída
    tendencia_alcista_dias = 7
    stop_loss = 0.20  # 20% stop loss
    
    def init(self):
        """Inicializa la estrategia DCA."""
        
        # Validar datos antes de calcular indicadores
        if len(self.data.df) < 21:
            raise ValueError("Datos insuficientes para DCA (mínimo 21 períodos)")
        
        # Calcular medias móviles para tendencia con manejo de NaN
        try:
            close_series = pd.Series(self.data.Close)
            self.sma_short = self.I(lambda x: pd.Series(x).rolling(7, min_periods=1).mean(), self.data.Close)
            self.sma_long = self.I(lambda x: pd.Series(x).rolling(21, min_periods=1).mean(), self.data.Close)
        except Exception as e:
            logger.warning(f"Error calculando SMAs: {e}")
            # Valores por defecto
            self.sma_short = pd.Series(self.data.Close, index=self.data.df.index)
            self.sma_long = pd.Series(self.data.Close, index=self.data.df.index)
        
        # Variables de estado
        self.last_buy_day = -self.intervalo_compra  # Permitir compra inmediata
        self.entry_prices = []  # Precios de entrada para cálculo de ganancia
        
        logger.info(f"🔧 DCAStrategy inicializada: {self.intervalo_compra}d interval, "
                   f"{self.objetivo_ganancia*100:.0f}% target")
    
    def is_bullish_trend(self) -> bool:
        """Detecta si estamos en tendencia alcista."""
        try:
            if len(self.data) < self.tendencia_alcista_dias:
                return False
            
            # Validar que los indicadores no son NaN
            if (pd.isna(float(self.sma_short[-1])) or pd.isna(float(self.sma_long[-1])) or 
                pd.isna(float(self.data.Close[-1])) or pd.isna(float(self.data.Close[-self.tendencia_alcista_dias]))):
                return False
            
            # Tendencia: SMA corta > SMA larga y precio creciendo
            sma_condition = self.sma_short[-1] > self.sma_long[-1]
            price_condition = self.data.Close[-1] > self.data.Close[-self.tendencia_alcista_dias]
            
            return sma_condition and price_condition
        
        except Exception:
            return False
    
    def is_dip_opportunity(self) -> bool:
        """Detecta oportunidades de compra en dips."""
        try:
            if len(self.data) < 5:
                return False
            
            # Validar datos
            recent_prices = self.data.Close[-5:]
            if any(pd.isna(float(price)) for price in recent_prices):
                return False
            
            # Buscar caída reciente desde máximo local
            recent_high = max(recent_prices)
            current_price = self.data.Close[-1]
            
            if recent_high <= 0:
                return False
            
            dip_size = (recent_high - current_price) / recent_high
            return dip_size >= self.dip_threshold
        
        except Exception:
            return False
    
    def should_buy(self, current_day: int) -> bool:
        """Determina si debe comprar."""
        try:
            # Verificar intervalo de compra
            if current_day - self.last_buy_day < self.intervalo_compra:
                return False
            
            # Verificar condiciones de mercado
            bullish = self.is_bullish_trend()
            dip = self.is_dip_opportunity()
            
            return bullish and dip
        
        except Exception:
            return False
    
    def should_sell(self) -> bool:
        """Determina si debe vender."""
        try:
            if not self.position or not self.entry_prices:
                return False
            
            current_price = self.data.Close[-1]
            avg_entry = np.mean(self.entry_prices)
            
            # Calcular ganancia/pérdida
            pnl_pct = (current_price - avg_entry) / avg_entry
            
            # Vender si alcanzamos objetivo o stop loss
            take_profit = pnl_pct >= self.objetivo_ganancia
            stop_loss_hit = pnl_pct <= -self.stop_loss
            
            return take_profit or stop_loss_hit
        
        except Exception:
            return False
    
    def next(self):
        """Lógica principal de DCA."""
        try:
            current_day = len(self.data) - 1
            current_price = self.data.Close[-1]
            
            # Lógica de compra
            if self.should_buy(current_day):
                # Asegurar que el tamaño sea válido para el broker
                buy_size = max(0.1, self.monto_compra)  # Mínimo 10% del capital
                self.buy(size=buy_size)
                self.entry_prices.append(current_price)
                self.last_buy_day = current_day
                logger.debug(f"DCA compra: ${current_price:.4f}")
            
            # Lógica de venta
            elif self.should_sell():
                self.sell(size=self.position.size)
                self.entry_prices.clear()  # Reset para nuevo ciclo
                logger.debug(f"DCA venta: ${current_price:.4f}")
        
        except Exception as e:
            logger.error(f"Error en DCA next(): {e}") 