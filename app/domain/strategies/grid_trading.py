"""
Grid Trading Strategy - Estrategia de Rejilla
============================================

Estrategia de Grid Trading modernizada usando backtesting.py.
Implementa un sistema de rejilla (grid) que coloca órdenes de compra y venta
en niveles predefinidos alrededor del precio actual.

Esta es lógica de dominio pura - sin dependencias externas.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from backtesting import Strategy
import logging

logger = logging.getLogger(__name__)


class GridTradingStrategy(Strategy):
    """
    Estrategia de Grid Trading modernizada usando backtesting.py.
    
    Implementa un sistema de rejilla (grid) que coloca órdenes de compra y venta
    en niveles predefinidos alrededor del precio actual.
    
    Parámetros:
    - levels: Número de niveles en el grid (3-8)
    - range_percent: Porcentaje de rango del grid (2.0-15.0)
    - umbral_adx: Filtro ADX para mercados laterales (15.0-40.0)
    - umbral_volatilidad: Filtro de volatilidad mínima (0.01-0.05)
    - umbral_sentimiento: Filtro de sentimiento (-0.3-0.3)
    """
    
    # Parámetros optimizables
    levels = 4
    range_percent = 8.0
    umbral_adx = 25.0
    umbral_volatilidad = 0.02
    umbral_sentimiento = 0.0
    
    def init(self):
        """Inicializa la estrategia y calcula indicadores necesarios."""
        
        # Validar que tenemos las columnas necesarias
        required_columns = ['Open', 'High', 'Low', 'Close']
        for col in required_columns:
            if col not in self.data.df.columns:
                raise ValueError(f"Columna faltante: {col}")
        
        # Calcular indicadores técnicos si no están presentes
        self.setup_indicators()
        
        # Variables de estado del grid
        self.grid_initialized = False
        self.buy_levels = []
        self.sell_levels = []
        self.grid_center = None
        self.active_orders = {}
        
        # Tamaño de posición fijo (porcentaje del capital)
        self.position_size = 1.0 / self.levels  # Dividir capital entre niveles
        
        logger.info(f"🔧 GridStrategy inicializada: {self.levels} niveles, ±{self.range_percent}%")
    
    def setup_indicators(self):
        """Calcula indicadores técnicos necesarios."""
        try:
            # ADX (Average Directional Index)
            if 'adx' not in self.data.df.columns:
                self.adx = self.calculate_adx()
            else:
                self.adx = self.data.df['adx']
            
            # Volatilidad
            if 'volatility' not in self.data.df.columns:
                self.volatility = self.calculate_volatility()
            else:
                self.volatility = self.data.df['volatility']
            
            # Sentimiento (si está disponible)
            if 'sentiment_ma7' in self.data.df.columns:
                self.sentiment = self.data.df['sentiment_ma7']
            else:
                # Crear sentimiento neutral si no está disponible
                self.sentiment = pd.Series(0.0, index=self.data.df.index)
        
        except Exception as e:
            logger.warning(f"Error calculando indicadores: {e}")
            # Valores por defecto en caso de error
            self.adx = pd.Series(25.0, index=self.data.df.index)
            self.volatility = pd.Series(0.03, index=self.data.df.index)
            self.sentiment = pd.Series(0.0, index=self.data.df.index)
    
    def calculate_adx(self, period: int = 14) -> pd.Series:
        """Calcula ADX simplificado usando operaciones numpy básicas."""
        try:
            # Usar arrays numpy directamente
            high = np.array(self.data.High)
            low = np.array(self.data.Low) 
            close = np.array(self.data.Close)
            
            # Cálculo simplificado de ADX
            if len(high) < period:
                return pd.Series(25.0, index=self.data.df.index)
            
            # True Range básico
            tr = high - low
            
            # Media móvil simple del TR como proxy de ADX
            adx_values = np.convolve(tr, np.ones(period)/period, mode='same')
            
            # Normalizar entre 0-100
            adx_values = np.clip(adx_values * 100 / np.max(adx_values), 0, 100)
            
            return pd.Series(adx_values, index=self.data.df.index)
        
        except Exception:
            return pd.Series(25.0, index=self.data.df.index)
    
    def calculate_volatility(self, period: int = 20) -> pd.Series:
        """Calcula volatilidad usando operaciones numpy básicas."""
        try:
            close = np.array(self.data.Close)
            
            if len(close) < 2:
                return pd.Series(0.03, index=self.data.df.index)
            
            # Calcular retornos
            returns = np.diff(close) / close[:-1]
            
            # Volatilidad con ventana deslizante
            vol_values = []
            for i in range(len(close)):
                if i < period:
                    vol_values.append(0.03)  # Valor por defecto
                else:
                    window_returns = returns[i-period:i]
                    vol = np.std(window_returns) if len(window_returns) > 0 else 0.03
                    vol_values.append(vol)
            
            return pd.Series(vol_values, index=self.data.df.index)
        
        except Exception:
            return pd.Series(0.03, index=self.data.df.index)
    
    def should_operate(self, i: int) -> bool:
        """Determina si debe operar basándose en filtros técnicos."""
        try:
            # Verificar que tenemos datos suficientes
            if i < 20:  # Necesitamos al menos 20 períodos para indicadores
                return False
            
            current_adx = self.adx.iloc[i] if i < len(self.adx) else 25.0
            current_vol = self.volatility.iloc[i] if i < len(self.volatility) else 0.03
            current_sent = self.sentiment.iloc[i] if i < len(self.sentiment) else 0.0
            
            # Aplicar filtros
            adx_ok = current_adx < self.umbral_adx  # Mercado lateral
            vol_ok = current_vol > self.umbral_volatilidad  # Suficiente volatilidad
            sent_ok = current_sent > self.umbral_sentimiento  # Sentimiento positivo
            
            return adx_ok and vol_ok and sent_ok
        
        except Exception:
            return True  # Por defecto operar si hay problemas
    
    def initialize_grid(self, current_price: float):
        """Inicializa el grid alrededor del precio actual."""
        try:
            range_value = current_price * (self.range_percent / 100)
            min_price = current_price - (range_value / 2)
            max_price = current_price + (range_value / 2)
            
            # Calcular niveles de compra (por debajo del precio)
            buy_levels_count = self.levels // 2
            if buy_levels_count > 0:
                buy_step = (current_price - min_price) / buy_levels_count
                self.buy_levels = [current_price - buy_step * (i + 1) 
                                 for i in range(buy_levels_count)]
            
            # Calcular niveles de venta (por encima del precio)
            sell_levels_count = self.levels - buy_levels_count
            if sell_levels_count > 0:
                sell_step = (max_price - current_price) / sell_levels_count
                self.sell_levels = [current_price + sell_step * (i + 1) 
                                  for i in range(sell_levels_count)]
            
            self.grid_center = current_price
            self.grid_initialized = True
            
            logger.debug(f"Grid inicializado: centro={current_price:.4f}, "
                        f"compra={len(self.buy_levels)}, venta={len(self.sell_levels)}")
        
        except Exception as e:
            logger.error(f"Error inicializando grid: {e}")
    
    def next(self):
        """Lógica principal ejecutada en cada barra."""
        try:
            current_price = self.data.Close[-1]
            current_index = len(self.data) - 1
            
            # Verificar filtros técnicos
            if not self.should_operate(current_index):
                return
            
            # Inicializar grid si es necesario
            if not self.grid_initialized:
                self.initialize_grid(current_price)
                return
            
            # Lógica de grid trading
            self.execute_grid_logic(current_price)
        
        except Exception as e:
            logger.error(f"Error en next(): {e}")
    
    def execute_grid_logic(self, current_price: float):
        """Ejecuta la lógica del grid trading."""
        try:
            # Verificar niveles de compra
            for level in self.buy_levels:
                if (current_price <= level and 
                    not self.position and 
                    level not in self.active_orders):
                    
                    # Comprar en nivel de soporte
                    self.buy(size=self.position_size)
                    self.active_orders[level] = 'buy'
                    logger.debug(f"Compra en nivel {level:.4f}")
                    break
            
            # Verificar niveles de venta
            for level in self.sell_levels:
                if (current_price >= level and 
                    self.position and 
                    level not in self.active_orders):
                    
                    # Vender en nivel de resistencia
                    self.sell(size=self.position.size)
                    self.active_orders[level] = 'sell'
                    logger.debug(f"Venta en nivel {level:.4f}")
                    break
        
        except Exception as e:
            logger.error(f"Error ejecutando grid: {e}") 