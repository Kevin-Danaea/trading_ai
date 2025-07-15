"""
BTD Strategy - Estrategia de Buy The Dip
=======================================

Estrategia Buy The Dip (BTD) modernizada.
Estrategia que busca aprovechar caídas temporales del mercado
comprando en dips y vendiendo en recuperaciones.

Esta es lógica de dominio pura - sin dependencias externas.
"""

from typing import Optional
import pandas as pd
import numpy as np
from backtesting import Strategy
import logging

logger = logging.getLogger(__name__)


class BTDStrategy(Strategy):
    """
    Estrategia Buy The Dip (BTD) modernizada.
    
    Estrategia que busca aprovechar caídas temporales del mercado
    comprando en dips y vendiendo en recuperaciones.
    
    Parámetros:
    - intervalo_venta: Días entre operaciones (1-7)
    - monto_venta: Monto por operación como % del capital (0.1-1.0)
    - objetivo_ganancia: % de ganancia objetivo (0.05-0.25)
    - rip_threshold: % de subida para detectar entrada (0.02-0.12)
    - tendencia_bajista_dias: Días para confirmar dip (3-14)
    - stop_loss: % de pérdida máxima (0.10-0.35)
    """
    
    # Parámetros optimizables - Valores originales restrictivos
    intervalo_venta = 2
    monto_venta = 0.25  # 25% del capital por operación
    objetivo_ganancia = 0.12  # 12% de ganancia
    rip_threshold = 0.04  # 4% de subida
    tendencia_bajista_dias = 5
    stop_loss = 0.15  # 15% stop loss
    
    def init(self):
        """Inicializa la estrategia BTD."""
        
        # Validar datos antes de calcular indicadores
        if len(self.data.df) < 14:
            raise ValueError("Datos insuficientes para BTD (mínimo 14 períodos)")
        
        # Calcular indicadores con manejo de NaN
        try:
            self.sma_trend = self.I(lambda x: pd.Series(x).rolling(14, min_periods=1).mean(), self.data.Close)
            self.rsi = self.I(self.calculate_rsi, self.data.Close)
        except Exception as e:
            logger.warning(f"Error calculando indicadores BTD: {e}")
            # Valores por defecto
            self.sma_trend = pd.Series(self.data.Close, index=self.data.df.index)
            self.rsi = pd.Series(50.0, index=self.data.df.index)
        
        # Variables de estado
        self.last_operation_day = -self.intervalo_venta
        self.entry_price = None
        
        logger.info(f"🔧 BTDStrategy inicializada: {self.intervalo_venta}d interval, "
                   f"{self.objetivo_ganancia*100:.0f}% target")
    
    def get_parameters(self) -> dict:
        """
        Retorna los parámetros actuales de la estrategia.
        
        Returns:
            dict: Diccionario con los parámetros de la estrategia.
        """
        return {
            'intervalo_venta': self.intervalo_venta,
            'monto_venta': self.monto_venta,
            'objetivo_ganancia': self.objetivo_ganancia,
            'rip_threshold': self.rip_threshold,
            'tendencia_bajista_dias': self.tendencia_bajista_dias,
            'stop_loss': self.stop_loss
        }
    
    def validate_parameters(self, parameters: dict) -> bool:
        """
        Valida un conjunto de parámetros para la estrategia.
        
        Args:
            parameters (dict): Parámetros a validar.
            
        Returns:
            bool: True si los parámetros son válidos.
        """
        try:
            # Validaciones básicas
            if parameters.get('intervalo_venta', self.intervalo_venta) < 1:
                raise ValueError("intervalo_venta debe ser >= 1")
            if parameters.get('monto_venta', self.monto_venta) <= 0 or parameters.get('monto_venta', self.monto_venta) > 1:
                raise ValueError("monto_venta debe estar entre 0 y 1")
            if parameters.get('objetivo_ganancia', self.objetivo_ganancia) <= 0:
                raise ValueError("objetivo_ganancia debe ser > 0")
            if parameters.get('stop_loss', self.stop_loss) <= 0:
                raise ValueError("stop_loss debe ser > 0")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validando parámetros BTD: {e}")
            raise
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula RSI para detectar sobrecompra/sobreventa."""
        try:
            # Validar datos
            if len(prices) < period:
                return pd.Series(50.0, index=prices.index)
            
            # Convertir a pandas Series si no lo es
            if not isinstance(prices, pd.Series):
                prices = pd.Series(prices)
            
            # Limpiar NaN
            prices = prices.ffill().bfill()
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            
            # Evitar división por cero
            loss_safe = loss.copy()
            loss_safe[loss_safe == 0] = 1e-10
            rs = gain / loss_safe
            rsi = 100 - (100 / (1 + rs))
            
            # Asegurar que retornamos pandas Series y manejar NaN
            if isinstance(rsi, pd.Series):
                return rsi.fillna(50)
            else:
                return pd.Series(rsi, index=prices.index).fillna(50)
        
        except Exception:
            return pd.Series(50, index=prices.index)
    
    def is_dip_confirmed(self) -> bool:
        """Confirma si estamos en un dip válido."""
        try:
            if len(self.data) < self.tendencia_bajista_dias:
                return False
            
            # Validar que los indicadores no son NaN
            if (pd.isna(float(self.data.Close[-1])) or pd.isna(float(self.data.Close[-self.tendencia_bajista_dias])) or
                pd.isna(float(self.rsi[-1]))):
                return False
            
            # Verificar que hay tendencia bajista reciente
            price_trend = (self.data.Close[-1] < 
                          self.data.Close[-self.tendencia_bajista_dias])
            
            # Verificar RSI de sobreventa
            rsi_oversold = self.rsi[-1] < 30
            
            return price_trend and rsi_oversold
        
        except Exception:
            return False
    
    def is_recovery_signal(self) -> bool:
        """Detecta señales de recuperación después del dip."""
        try:
            if len(self.data) < 3:
                return False
            
            # Validar datos
            recent_prices = self.data.Close[-3:]
            if any(pd.isna(float(price)) for price in recent_prices):
                return False
            
            # Buscar recuperación desde mínimo reciente
            recent_low = min(recent_prices)
            current_price = self.data.Close[-1]
            
            if recent_low <= 0:
                return False
            
            recovery_size = (current_price - recent_low) / recent_low
            return recovery_size >= self.rip_threshold
        
        except Exception:
            return False
    
    def should_buy(self, current_day: int) -> bool:
        """Determina si debe comprar en el dip."""
        try:
            # Verificar intervalo
            if current_day - self.last_operation_day < self.intervalo_venta:
                return False
            
            # Verificar condiciones de mercado
            dip_confirmed = self.is_dip_confirmed()
            recovery_signal = self.is_recovery_signal()
            
            return dip_confirmed and recovery_signal and not self.position
        
        except Exception:
            return False
    
    def should_sell(self) -> bool:
        """Determina si debe vender."""
        try:
            if not self.position or self.entry_price is None:
                return False
            
            current_price = self.data.Close[-1]
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            # Vender en objetivo o stop loss
            take_profit = pnl_pct >= self.objetivo_ganancia
            stop_loss_hit = pnl_pct <= -self.stop_loss
            
            return take_profit or stop_loss_hit
        
        except Exception:
            return False
    
    def next(self):
        """Lógica principal de BTD."""
        try:
            current_day = len(self.data) - 1
            current_price = self.data.Close[-1]
            
            # Lógica de compra
            if self.should_buy(current_day):
                # Asegurar que el tamaño sea válido para el broker
                buy_size = max(0.1, self.monto_venta)  # Mínimo 10% del capital
                self.buy(size=buy_size)
                self.entry_price = current_price
                self.last_operation_day = current_day
                logger.debug(f"BTD compra: ${current_price:.4f}")
            
            # Lógica de venta
            elif self.should_sell():
                self.sell(size=self.position.size)
                self.entry_price = None
                logger.debug(f"BTD venta: ${current_price:.4f}")
        
        except Exception as e:
            logger.error(f"Error en BTD next(): {e}") 