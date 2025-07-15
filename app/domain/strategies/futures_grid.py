"""
Futures Grid Trading Strategy - Estrategia de Rejilla para Futuros
================================================================

Estrategia de Grid Trading específica para futuros con apalancamiento.
Es idéntica a la estrategia de spot pero con parámetros adicionales para futuros.

Características específicas de futuros:
- Apalancamiento configurable
- Cálculo de precio de liquidación
- Lógica de liquidación automática
- Gestión de margen
- Funding rate

Esta es lógica de dominio pura - sin dependencias externas.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from backtesting import Strategy
import logging

logger = logging.getLogger(__name__)


class FuturesGridStrategy(Strategy):
    """
    Estrategia de Grid Trading para futuros con apalancamiento.
    
    Es idéntica a la estrategia de spot pero con parámetros adicionales para futuros:
    - leverage: Apalancamiento (1-20)
    - maintenance_margin_rate: Tasa de margen de mantenimiento (0.005-0.05)
    
    Parámetros base (igual que spot):
    - levels: Número de niveles en el grid (3-8)
    - range_percent: Porcentaje de rango del grid (2.0-15.0)
    - umbral_adx: Filtro ADX para mercados laterales (15.0-40.0)
    - umbral_volatilidad: Filtro de volatilidad mínima (0.01-0.05)
    - umbral_sentimiento: Filtro de sentimiento (-0.3-0.3)
    """
    
    # Parámetros base (igual que spot)
    levels = 4
    range_percent = 8.0
    umbral_adx = 25.0
    umbral_volatilidad = 0.02
    umbral_sentimiento = 0.0
    
    # Parámetros específicos de futuros
    leverage = 10
    maintenance_margin_rate = 0.01  # 1% margen de mantenimiento
    
    def init(self):
        """Inicializa la estrategia y calcula indicadores necesarios."""
        
        # Validar que tenemos las columnas necesarias
        required_columns = ['Open', 'High', 'Low', 'Close']
        for col in required_columns:
            if col not in self.data.df.columns:
                raise ValueError(f"Columna faltante: {col}")
        
        # Calcular indicadores técnicos si no están presentes
        self.setup_indicators()
        
        # Variables de estado del grid (igual que spot)
        self.grid_initialized = False
        self.buy_levels = []
        self.sell_levels = []
        self.grid_center = None
        self.active_orders = {}
        
        # Tamaño de posición fijo (igual que spot)
        # Asegurar que el tamaño sea al menos 0.15 (15%) para evitar errores del broker
        self.position_size = max(0.15, 1.0 / self.levels)  # Mínimo 15% del capital
        
        # Variables específicas de futuros
        self.entry_price = 0
        self.position_side = None  # 'long' o 'short'
        self.liquidation_price = 0
        self.margin_used = 0
        self.unrealized_pnl = 0
        
        # Estadísticas de liquidación
        self.liquidation_count = 0
        self.was_liquidated = False
        self.liquidation_history = []
        
        # Funding rate tracking
        self.total_funding_paid = 0
        self.funding_payments = []
        self.last_funding_time = None
        
        logger.info(f"🚀 FuturesGridStrategy inicializada: {self.levels} niveles, ±{self.range_percent}%, {self.leverage}x")
    
    def get_parameters(self) -> dict:
        """
        Retorna los parámetros actuales de la estrategia.
        
        Returns:
            dict: Diccionario con los parámetros de la estrategia.
        """
        return {
            'levels': self.levels,
            'range_percent': self.range_percent,
            'umbral_adx': self.umbral_adx,
            'umbral_volatilidad': self.umbral_volatilidad,
            'umbral_sentimiento': self.umbral_sentimiento,
            'leverage': self.leverage,
            'maintenance_margin_rate': self.maintenance_margin_rate
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
            if parameters.get('levels', self.levels) < 3 or parameters.get('levels', self.levels) > 8:
                raise ValueError("levels debe estar entre 3 y 8")
            if parameters.get('range_percent', self.range_percent) < 2.0 or parameters.get('range_percent', self.range_percent) > 15.0:
                raise ValueError("range_percent debe estar entre 2.0 y 15.0")
            if parameters.get('leverage', self.leverage) < 1 or parameters.get('leverage', self.leverage) > 20:
                raise ValueError("leverage debe estar entre 1 y 20")
            if parameters.get('maintenance_margin_rate', self.maintenance_margin_rate) < 0.005 or parameters.get('maintenance_margin_rate', self.maintenance_margin_rate) > 0.05:
                raise ValueError("maintenance_margin_rate debe estar entre 0.005 y 0.05")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validando parámetros FuturesGrid: {e}")
            raise
    
    def setup_indicators(self):
        """Calcula indicadores técnicos necesarios (igual que spot)."""
        try:
            # Validar que tenemos datos suficientes
            if len(self.data.df) < 20:
                raise ValueError("Datos insuficientes para calcular indicadores")
            
            # ADX (Average Directional Index)
            if 'adx' not in self.data.df.columns:
                self.adx = self.calculate_adx()
            else:
                self.adx = self.data.df['adx'].fillna(25.0)
            
            # Volatilidad
            if 'volatility' not in self.data.df.columns:
                self.volatility = self.calculate_volatility()
            else:
                self.volatility = self.data.df['volatility'].fillna(0.03)
            
            # Sentimiento (si está disponible)
            if 'sentiment_ma7' in self.data.df.columns:
                self.sentiment = self.data.df['sentiment_ma7'].fillna(0.0)
            elif 'sentiment_score' in self.data.df.columns:
                self.sentiment = self.data.df['sentiment_score'].fillna(0.0)
            else:
                # Crear sentimiento neutral si no está disponible
                self.sentiment = pd.Series(0.0, index=self.data.df.index)
            
            # Validar que no hay NaN en los indicadores
            self.adx = self.adx.fillna(25.0)
            self.volatility = self.volatility.fillna(0.03)
            self.sentiment = self.sentiment.fillna(0.0)
        
        except Exception as e:
            logger.warning(f"Error calculando indicadores: {e}")
            # Valores por defecto en caso de error
            self.adx = pd.Series(25.0, index=self.data.df.index)
            self.volatility = pd.Series(0.03, index=self.data.df.index)
            self.sentiment = pd.Series(0.0, index=self.data.df.index)
    
    def calculate_adx(self, period: int = 14) -> pd.Series:
        """Calcula ADX simplificado usando operaciones numpy básicas (igual que spot)."""
        try:
            # Usar arrays numpy directamente
            high = np.array(self.data.High)
            low = np.array(self.data.Low) 
            close = np.array(self.data.Close)
            
            # Validar datos
            if len(high) < period or np.any(np.isnan(high)) or np.any(np.isnan(low)):
                return pd.Series(25.0, index=self.data.df.index)
            
            # True Range básico
            tr = high - low
            
            # Media móvil simple del TR como proxy de ADX
            adx_values = np.convolve(tr, np.ones(period)/period, mode='same')
            
            # Normalizar entre 0-100 y manejar NaN
            if np.max(adx_values) > 0:
                adx_values = np.clip(adx_values * 100 / np.max(adx_values), 0, 100)
            else:
                adx_values = np.full_like(adx_values, 25.0)
            
            # Rellenar NaN
            adx_values = np.nan_to_num(adx_values, nan=25.0)
            
            return pd.Series(adx_values, index=self.data.df.index)
        
        except Exception:
            return pd.Series(25.0, index=self.data.df.index)
    
    def calculate_volatility(self, period: int = 20) -> pd.Series:
        """Calcula volatilidad usando operaciones numpy básicas (igual que spot)."""
        try:
            close = np.array(self.data.Close)
            
            # Validar datos
            if len(close) < 2 or np.any(np.isnan(close)):
                return pd.Series(0.03, index=self.data.df.index)
            
            # Calcular retornos
            returns = np.diff(close) / close[:-1]
            returns = np.nan_to_num(returns, nan=0.0)  # Manejar división por cero
            
            # Volatilidad con ventana deslizante
            vol_values = []
            for i in range(len(close)):
                if i < period:
                    vol_values.append(0.03)  # Valor por defecto
                else:
                    window_returns = returns[i-period:i]
                    if len(window_returns) > 0:
                        vol = np.std(window_returns)
                        vol = 0.03 if np.isnan(vol) or vol == 0 else vol
                    else:
                        vol = 0.03
                    vol_values.append(vol)
            
            return pd.Series(vol_values, index=self.data.df.index)
        
        except Exception:
            return pd.Series(0.03, index=self.data.df.index)
    
    def should_operate(self, i: int) -> bool:
        """Determina si debe operar basándose en filtros técnicos (igual que spot)."""
        try:
            # Verificar que tenemos datos suficientes
            if i < 20:  # Necesitamos al menos 20 períodos para indicadores
                return False
            
            # No operar si ya fue liquidado
            if self.was_liquidated:
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
        """Inicializa el grid alrededor del precio actual (igual que spot)."""
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
        """Lógica principal ejecutada en cada barra (igual que spot + futuros)."""
        try:
            current_price = self.data.Close[-1]
            current_high = self.data.High[-1]
            current_low = self.data.Low[-1]
            current_index = len(self.data) - 1
            
            # Aplicar funding rate si es momento de funding (solo futuros)
            self.apply_funding_rate(current_index)
            
            # Verificar liquidación antes de operar (solo futuros)
            if self.check_liquidation(current_high, current_low):
                self.execute_liquidation()
                return
            
            # Verificar filtros técnicos (igual que spot)
            if not self.should_operate(current_index):
                return
            
            # Inicializar grid si es necesario (igual que spot)
            if not self.grid_initialized:
                self.initialize_grid(current_price)
                return
            
            # Lógica de grid trading (igual que spot)
            self.execute_grid_logic(current_price)
        
        except Exception as e:
            logger.error(f"Error en next(): {e}")
    
    def execute_grid_logic(self, current_price: float):
        """Ejecuta la lógica del grid trading (igual que spot)."""
        try:
            # Verificar niveles de compra
            for level in self.buy_levels:
                if (current_price <= level and 
                    not self.position and 
                    level not in self.active_orders):
                    
                    # Comprar en nivel de soporte
                    # Verificar que hay suficiente capital disponible
                    if self.equity > 0 and self.position_size > 0:
                        self.buy(size=self.position_size)
                        self.active_orders[level] = 'buy'
                        
                        # Configurar variables de futuros
                        self.position_side = 'long'
                        self.entry_price = current_price
                        self.margin_used = self.equity * self.position_size / self.leverage
                        self.liquidation_price = self.calculate_liquidation_price(
                            current_price, 'long', self.margin_used
                        )
                        
                        logger.info(f"🟢 Compra LONG en {level:.4f} (precio: {current_price:.4f})")
                        break
            
            # Verificar niveles de venta
            for level in self.sell_levels:
                if (current_price >= level and 
                    self.position and 
                    level not in self.active_orders):
                    
                    # Vender en nivel de resistencia
                    # Usar todo el tamaño de la posición disponible
                    if self.position.size > 0:
                        self.sell(size=self.position.size)
                        
                        # Resetear variables de futuros
                        self.liquidation_price = None
                        self.position_side = None
                        self.entry_price = None
                        self.margin_used = 0.0
                        
                    self.active_orders[level] = 'sell'
                    logger.info(f"🔴 Venta (cierre LONG) en {level:.4f} (precio: {current_price:.4f})")
                    break
        
        except Exception as e:
            logger.error(f"Error ejecutando grid: {e}")
    
    # Métodos específicos de futuros (nuevos)
    
    def calculate_liquidation_price(self, entry_price: float, position_side: str, 
                                  margin_used: float) -> float:
        """
        Calcula el precio de liquidación para una posición.
        
        Args:
            entry_price: Precio de entrada
            position_side: 'long' o 'short'
            margin_used: Margen utilizado
            
        Returns:
            Precio de liquidación
        """
        try:
            # Fórmula simplificada de liquidación
            # Liquidación = Entry ± (Margin / (Position_Size * Leverage))
            
            position_value = margin_used * self.leverage
            position_size = position_value / entry_price
            
            # Distancia al precio de liquidación
            liquidation_distance = margin_used / (position_size * self.leverage)
            
            if position_side == 'long':
                # Para posición larga, liquidación por debajo del precio de entrada
                liquidation_price = entry_price - liquidation_distance
            else:
                # Para posición corta, liquidación por encima del precio de entrada
                liquidation_price = entry_price + liquidation_distance
            
            return max(liquidation_price, 0.0001)  # Evitar precios negativos
        
        except Exception as e:
            logger.error(f"Error calculando precio de liquidación: {e}")
            return entry_price * 0.5 if position_side == 'long' else entry_price * 2.0
    
    def check_liquidation(self, current_high: float, current_low: float) -> bool:
        """
        Verifica si la posición debe ser liquidada.
        
        Args:
            current_high: Precio máximo de la vela actual
            current_low: Precio mínimo de la vela actual
            
        Returns:
            True si debe liquidarse
        """
        try:
            if not self.position or self.liquidation_price is None:
                return False
            
            # Verificar liquidación según el lado de la posición
            if self.position_side == 'long':
                # Posición larga se liquida si el precio LOW toca el precio de liquidación
                if current_low <= self.liquidation_price:
                    return True
            elif self.position_side == 'short':
                # Posición corta se liquida si el precio HIGH toca el precio de liquidación
                if current_high >= self.liquidation_price:
                    return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error verificando liquidación: {e}")
            return False
    
    def execute_liquidation(self):
        """Ejecuta la liquidación de la posición."""
        try:
            if self.position:
                # Cerrar posición con pérdida del 100% del margen
                self.position.close()
                
                # Registrar liquidación
                self.was_liquidated = True
                self.liquidation_count += 1
                
                # Guardar en historial
                liquidation_record = {
                    'timestamp': len(self.data) - 1,
                    'entry_price': self.entry_price,
                    'liquidation_price': self.liquidation_price,
                    'position_side': self.position_side,
                    'margin_lost': self.margin_used
                }
                self.liquidation_history.append(liquidation_record)
                
                logger.warning(f"💥 LIQUIDACIÓN: {self.position_side} en {self.liquidation_price:.4f}, "
                             f"margen perdido: ${self.margin_used:.2f}")
                
                # Resetear variables
                self.liquidation_price = None
                self.position_side = None
                self.entry_price = None
                self.margin_used = 0.0
        
        except Exception as e:
            logger.error(f"Error ejecutando liquidación: {e}")
    
    def apply_funding_rate(self, current_index: int):
        """
        Aplica el funding rate si es momento de funding.
        
        Args:
            current_index: Índice actual en los datos
        """
        try:
            # Verificar si hay datos de funding rate
            if not hasattr(self.data, 'funding_rate') or not hasattr(self.data, 'funding_time'):
                return
            
            # Verificar si es momento de funding
            if not self.data.funding_time[current_index]:
                return
            
            # Verificar si ya aplicamos funding en este período
            if self.last_funding_time == current_index:
                return
                
            # Solo aplicar funding si tenemos posición abierta
            if self.position_size == 0:
                return
            
            # Obtener funding rate actual
            funding_rate = self.data.funding_rate[current_index]
            current_price = self.data.Close[current_index]
            
            # Calcular costo de funding
            # Funding cost = position_value * funding_rate
            position_value = abs(self.position_size) * current_price
            funding_cost = position_value * funding_rate
            
            # Si estamos en long, pagamos funding positivo
            # Si estamos en short, pagamos funding negativo (recibimos si funding es positivo)
            if self.position_side == 'long':
                actual_funding_cost = funding_cost
            else:  # short
                actual_funding_cost = -funding_cost
            
            # Aplicar el costo de funding
            self.total_funding_paid += actual_funding_cost
            
            # Registrar el pago
            self.funding_payments.append({
                'timestamp': current_index,
                'funding_rate': funding_rate,
                'position_value': position_value,
                'funding_cost': actual_funding_cost,
                'position_side': self.position_side
            })
            
            # Actualizar último momento de funding
            self.last_funding_time = current_index
            
            logger.debug(f"💰 Funding aplicado: {actual_funding_cost:.2f} USDT "
                        f"(rate: {funding_rate*100:.3f}%, side: {self.position_side})")
            
        except Exception as e:
            logger.error(f"Error aplicando funding rate: {e}")
    
    def get_liquidation_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de liquidación y funding."""
        return {
            'liquidation_count': self.liquidation_count,
            'was_liquidated': self.was_liquidated,
            'liquidation_history': self.liquidation_history,
            'total_funding_paid': self.total_funding_paid,
            'funding_payments_count': len(self.funding_payments),
            'avg_funding_rate': np.mean([p['funding_rate'] for p in self.funding_payments]) if self.funding_payments else 0,
            'funding_payments': self.funding_payments
        } 