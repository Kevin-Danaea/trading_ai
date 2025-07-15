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

# Importar validador de parámetros
from app.infrastructure.services.parameter_validator_service import ParameterValidatorService, ParameterValidationError
# Importar manejador de errores
from app.infrastructure.services.error_handler_service import ErrorHandlerService, ErrorSeverity, ErrorType
# Importar límites de seguridad
from app.infrastructure.services.safety_limits_service import SafetyLimitsService, SafetyViolation

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
    
    # Parámetros optimizables - Valores originales restrictivos
    levels = 4
    range_percent = 8.0
    umbral_adx = 25.0
    umbral_volatilidad = 0.02
    umbral_sentimiento = 0.0
    
    def init(self):
        """Inicializa la estrategia y calcula indicadores necesarios."""
        
        # Inicializar manejador de errores mejorado - NUNCA maneja errores silenciosamente
        self.error_handler = ErrorHandlerService("GridTradingStrategy", {
            'force_logging': True,
            'log_all_errors': True,
            'alert_critical_errors': True,
            'max_errors_per_strategy': 5
        })
        
        # Inicializar límites de seguridad
        self.safety_limits = SafetyLimitsService("GridTradingStrategy", {'trading_type': 'spot'})
        
        # Validar parámetros antes de cualquier lógica
        self._validar_parametros()
        
        # Validar que tenemos las columnas necesarias
        required_columns = ['Open', 'High', 'Low', 'Close']
        for col in required_columns:
            if col not in self.data.df.columns:
                error_msg = f"Columna faltante: {col}"
                self.error_handler.handle_error(
                    ValueError(error_msg), 
                    "Validación de columnas", 
                    ErrorSeverity.CRITICAL, 
                    ErrorType.DATA_ERROR
                )
                raise ValueError(error_msg)
        
        # Calcular indicadores técnicos si no están presentes
        self.setup_indicators()
        
        # Variables de estado del grid
        self.grid_initialized = False
        self.buy_levels = []
        self.sell_levels = []
        self.grid_center = None
        self.active_orders = {}
        
        # Tamaño de posición fijo (porcentaje del capital)
        # Asegurar que el tamaño sea al menos 0.15 (15%) para evitar errores del broker
        self.position_size = max(0.15, 1.0 / self.levels)  # Mínimo 15% del capital
        
        logger.info(f"🔧 GridStrategy inicializada: {self.levels} niveles, ±{self.range_percent}%")
    
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
            'umbral_sentimiento': self.umbral_sentimiento
        }
    
    def validate_parameters(self, parameters: dict) -> bool:
        """
        Valida un conjunto de parámetros para la estrategia.
        
        Args:
            parameters (dict): Parámetros a validar.
            
        Returns:
            bool: True si los parámetros son válidos.
            
        Raises:
            ParameterValidationError: Si los parámetros son inválidos.
        """
        try:
            validator = ParameterValidatorService()
            
            # Convertir parámetros al formato esperado por el validador
            grid_params = {
                'grid_levels': parameters.get('levels', self.levels),
                'grid_spacing': parameters.get('range_percent', self.range_percent) / 100,
                'take_profit': 0.12,
                'stop_loss': 0.10,
                'max_positions': parameters.get('levels', self.levels),
                'rebalance_threshold': 0.1
            }
            
            # Validar parámetros
            validator.validar_parametros(grid_params, 'grid', 'spot')
            return True
            
        except ParameterValidationError as e:
            self.error_handler.handle_error(
                e,
                "Validación de parámetros externos",
                ErrorSeverity.MEDIUM,
                ErrorType.PARAMETER_ERROR
            )
            raise
    
    def _validar_parametros(self):
        """Valida los parámetros de la estrategia usando ParameterValidatorService."""
        try:
            validator = ParameterValidatorService()
            
            # Convertir parámetros de clase a diccionario
            params = {
                'grid_levels': self.levels,
                'grid_spacing': self.range_percent / 100,  # Convertir a decimal
                'take_profit': 0.12,  # 12% - mayor que grid_spacing (8%)
                'stop_loss': 0.10,    # 10% - mayor que grid_spacing (8%)
                'max_positions': self.levels,
                'rebalance_threshold': 0.1  # Valor por defecto
            }
            
            # Validar parámetros para spot trading
            params_validados = validator.validar_parametros(params, 'grid', 'spot')
            
            # Actualizar parámetros con valores validados
            self.levels = int(params_validados['grid_levels'])
            self.range_percent = params_validados['grid_spacing'] * 100  # Convertir de vuelta a porcentaje
            
            logger.info("✅ Parámetros de GridTradingStrategy validados correctamente")
            
        except ParameterValidationError as e:
            self.error_handler.handle_error(
                e, 
                "Validación de parámetros", 
                ErrorSeverity.CRITICAL, 
                ErrorType.PARAMETER_ERROR
            )
            raise
        except Exception as e:
            self.error_handler.handle_error(
                e, 
                "Error inesperado en validación de parámetros", 
                ErrorSeverity.CRITICAL, 
                ErrorType.SYSTEM_ERROR
            )
            raise
    
    def setup_indicators(self):
        """Calcula indicadores técnicos necesarios."""
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
            self.error_handler.handle_error(
                e, 
                "Cálculo de indicadores", 
                ErrorSeverity.HIGH, 
                ErrorType.CALCULATION_ERROR
            )
            # Valores por defecto en caso de error
            self.adx = pd.Series(25.0, index=self.data.df.index)
            self.volatility = pd.Series(0.03, index=self.data.df.index)
            self.sentiment = pd.Series(0.0, index=self.data.df.index)
    
    def calculate_adx(self, period: int = 14) -> pd.Series:
        """Calcula ADX simplificado usando operaciones numpy básicas."""
        return self.error_handler.safe_execute(
            self._calculate_adx_impl,
            period,
            context="Cálculo de ADX",
            default_return=pd.Series(25.0, index=self.data.df.index),
            severity=ErrorSeverity.MEDIUM,
            error_type=ErrorType.CALCULATION_ERROR
        )
    
    def _calculate_adx_impl(self, period: int = 14) -> pd.Series:
        """Implementación del cálculo de ADX."""
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
        """Calcula volatilidad usando operaciones numpy básicas."""
        return self.error_handler.safe_execute(
            self._calculate_volatility_impl,
            period,
            context="Cálculo de volatilidad",
            default_return=pd.Series(0.03, index=self.data.df.index),
            severity=ErrorSeverity.MEDIUM,
            error_type=ErrorType.CALCULATION_ERROR
        )
    
    def _calculate_volatility_impl(self, period: int = 20) -> pd.Series:
        """Implementación del cálculo de volatilidad."""
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
    
    def _check_safety_limits(self, operation_type: str, size: float, price: float):
        """
        Verifica límites de seguridad antes de una operación.
        
        Args:
            operation_type (str): 'buy' o 'sell'.
            size (float): Tamaño de la operación.
            price (float): Precio de la operación.
        """
        try:
            current_capital = self.equity
            current_equity = self.equity
            
            # Calcular exposición adicional
            additional_exposure = size * price if operation_type == 'buy' else 0
            
            # Verificar todos los límites
            self.safety_limits.check_all_limits(
                new_position_size=size,
                additional_exposure=additional_exposure,
                current_capital=current_capital,
                current_equity=current_equity
            )
            
            logger.debug(f"✅ Límites de seguridad verificados para {operation_type}")
            
        except SafetyViolation as e:
            self.error_handler.handle_error(
                e,
                f"Verificación de límites de seguridad para {operation_type}",
                ErrorSeverity.HIGH,
                ErrorType.TRADING_ERROR
            )
            raise
    
    def execute_grid_logic(self, current_price: float):
        """Ejecuta la lógica del grid con verificación de límites de seguridad."""
        try:
            # Verificar límites de seguridad antes de operar
            self._check_safety_limits('grid_operation', self.position_size, current_price)
            
            # Lógica original del grid
            if not self.grid_initialized:
                self.initialize_grid(current_price)
            
            # Verificar señales de compra
            for buy_level in self.buy_levels:
                if current_price <= buy_level and not self.position:
                    # Verificar límites antes de comprar
                    self._check_safety_limits('buy', self.position_size, current_price)
                    self.buy(size=self.position_size)
                    self.safety_limits.record_trade('buy', self.position_size, current_price)
                    logger.debug(f"Grid compra en ${current_price:.4f}")
                    break
            
            # Verificar señales de venta
            if self.position:
                for sell_level in self.sell_levels:
                    if current_price >= sell_level:
                        # Verificar límites antes de vender
                        self._check_safety_limits('sell', self.position.size, current_price)
                        self.sell(size=self.position.size)
                        # Calcular PnL aproximado (puede no estar disponible en backtesting.py)
                        pnl = 0.0  # Valor por defecto
                        self.safety_limits.record_trade('sell', self.position.size, current_price, pnl)
                        logger.debug(f"Grid vende en ${current_price:.4f}")
                        break
        
        except SafetyViolation as e:
            logger.warning(f"🚨 Operación cancelada por límites de seguridad: {e}")
        except Exception as e:
            self.error_handler.handle_error(
                e,
                "Lógica del grid",
                ErrorSeverity.HIGH,
                ErrorType.TRADING_ERROR
            ) 