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

# Importar validador de parámetros
from app.infrastructure.services.parameter_validator_service import ParameterValidatorService, ParameterValidationError
# Importar manejador de errores
from app.infrastructure.services.error_handler_service import ErrorHandlerService, ErrorSeverity, ErrorType
# Importar límites de seguridad
from app.infrastructure.services.safety_limits_service import SafetyLimitsService, SafetyViolation

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
        
        # Inicializar manejador de errores mejorado - NUNCA maneja errores silenciosamente
        self.error_handler = ErrorHandlerService("DCAStrategy", {
            'force_logging': True,
            'log_all_errors': True,
            'alert_critical_errors': True,
            'max_errors_per_strategy': 5
        })
        
        # Inicializar límites de seguridad
        self.safety_limits = SafetyLimitsService("DCAStrategy", {'trading_type': 'spot'})
        
        # Validar parámetros antes de cualquier lógica
        self._validar_parametros()
        
        # Validar datos antes de calcular indicadores
        if len(self.data.df) < 21:
            error_msg = "Datos insuficientes para DCA (mínimo 21 períodos)"
            self.error_handler.handle_error(
                ValueError(error_msg), 
                "Validación de datos", 
                ErrorSeverity.CRITICAL, 
                ErrorType.DATA_ERROR
            )
            raise ValueError(error_msg)
        
        # Calcular medias móviles para tendencia con manejo de NaN
        try:
            close_series = pd.Series(self.data.Close)
            self.sma_short = self.I(lambda x: pd.Series(x).rolling(7, min_periods=1).mean(), self.data.Close)
            self.sma_long = self.I(lambda x: pd.Series(x).rolling(21, min_periods=1).mean(), self.data.Close)
        except Exception as e:
            self.error_handler.handle_error(
                e, 
                "Cálculo de medias móviles", 
                ErrorSeverity.HIGH, 
                ErrorType.CALCULATION_ERROR
            )
            # Valores por defecto
            self.sma_short = pd.Series(self.data.Close, index=self.data.df.index)
            self.sma_long = pd.Series(self.data.Close, index=self.data.df.index)
        
        # Variables de estado
        self.last_buy_day = -self.intervalo_compra  # Permitir compra inmediata
        self.entry_prices = []  # Precios de entrada para cálculo de ganancia
        
        logger.info(f"🔧 DCAStrategy inicializada: {self.intervalo_compra}d interval, "
                   f"{self.objetivo_ganancia*100:.0f}% target")
    
    def get_parameters(self) -> dict:
        """
        Retorna los parámetros actuales de la estrategia.
        
        Returns:
            dict: Diccionario con los parámetros de la estrategia.
        """
        return {
            'intervalo_compra': self.intervalo_compra,
            'monto_compra': self.monto_compra,
            'objetivo_ganancia': self.objetivo_ganancia,
            'dip_threshold': self.dip_threshold,
            'tendencia_alcista_dias': self.tendencia_alcista_dias,
            'stop_loss': self.stop_loss
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
            dca_params = {
                'initial_investment': 1000,
                'dca_amount': parameters.get('monto_compra', self.monto_compra) * 1000,
                'dca_frequency': parameters.get('intervalo_compra', self.intervalo_compra),
                'take_profit': parameters.get('objetivo_ganancia', self.objetivo_ganancia),
                'stop_loss': parameters.get('stop_loss', self.stop_loss),
                'max_dca_cycles': 50
            }
            
            # Validar parámetros
            validator.validar_parametros(dca_params, 'dca', 'spot')
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
                'initial_investment': 1000,  # Valor por defecto
                'dca_amount': self.monto_compra * 1000,  # Convertir a monto fijo
                'dca_frequency': self.intervalo_compra,
                'take_profit': self.objetivo_ganancia,
                'stop_loss': self.stop_loss,
                'max_dca_cycles': 50  # Valor por defecto
            }
            
            # Validar parámetros para spot trading
            params_validados = validator.validar_parametros(params, 'dca', 'spot')
            
            # Actualizar parámetros con valores validados
            self.intervalo_compra = int(params_validados['dca_frequency'])
            self.monto_compra = params_validados['dca_amount'] / 1000  # Convertir de vuelta a porcentaje
            self.objetivo_ganancia = params_validados['take_profit']
            self.stop_loss = params_validados['stop_loss']
            
            logger.info("✅ Parámetros de DCAStrategy validados correctamente")
            
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
    
    def is_bullish_trend(self) -> bool:
        """Detecta si estamos en tendencia alcista."""
        return self.error_handler.safe_execute(
            self._is_bullish_trend_impl,
            context="Detección de tendencia alcista",
            default_return=False,
            severity=ErrorSeverity.MEDIUM,
            error_type=ErrorType.CALCULATION_ERROR
        )
    
    def _is_bullish_trend_impl(self) -> bool:
        """Implementación de detección de tendencia alcista."""
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
        return self.error_handler.safe_execute(
            self._is_dip_opportunity_impl,
            context="Detección de oportunidad de dip",
            default_return=False,
            severity=ErrorSeverity.MEDIUM,
            error_type=ErrorType.CALCULATION_ERROR
        )
    
    def _is_dip_opportunity_impl(self) -> bool:
        """Implementación de detección de oportunidad de dip."""
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
        return self.error_handler.safe_execute(
            self._should_buy_impl,
            current_day,
            context="Evaluación de compra",
            default_return=False,
            severity=ErrorSeverity.MEDIUM,
            error_type=ErrorType.TRADING_ERROR
        )
    
    def _should_buy_impl(self, current_day: int) -> bool:
        """Implementación de evaluación de compra."""
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
        return self.error_handler.safe_execute(
            self._should_sell_impl,
            context="Evaluación de venta",
            default_return=False,
            severity=ErrorSeverity.MEDIUM,
            error_type=ErrorType.TRADING_ERROR
        )
    
    def _should_sell_impl(self) -> bool:
        """Implementación de evaluación de venta."""
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
                self._check_safety_limits('buy', buy_size, current_price)
                self.buy(size=buy_size)
                self.entry_prices.append(current_price)
                self.last_buy_day = current_day
                logger.debug(f"DCA compra: ${current_price:.4f}")
            
            # Lógica de venta
            elif self.should_sell():
                self._check_safety_limits('sell', self.position.size, current_price)
                self.sell(size=self.position.size)
                self.entry_prices.clear()  # Reset para nuevo ciclo
                logger.debug(f"DCA venta: ${current_price:.4f}")
        
        except Exception as e:
            self.error_handler.handle_error(
                e, 
                "Lógica principal de DCA", 
                ErrorSeverity.HIGH, 
                ErrorType.TRADING_ERROR
            ) 