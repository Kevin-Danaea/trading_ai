"""
SafetyLimitsService - Servicio de Límites de Seguridad
=====================================================

Implementa límites de seguridad para operaciones de trading:
- Límites de posición máxima
- Límites de drawdown máximo
- Límites de exposición total
- Límites de frecuencia de operaciones
- Detección y alertas de violaciones
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd

class LimitType(Enum):
    """Tipos de límites de seguridad."""
    POSITION_SIZE = "position_size"
    DRAWDOWN = "drawdown"
    EXPOSURE = "exposure"
    FREQUENCY = "frequency"
    LEVERAGE = "leverage"

class SafetyViolation(Exception):
    """Excepción lanzada cuando se viola un límite de seguridad."""
    def __init__(self, limit_type: LimitType, current_value: float, limit_value: float, message: str = ""):
        super().__init__(message or f"Violación de límite {limit_type.value}: {current_value} > {limit_value}")
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value

class SafetyLimitsService:
    """
    Servicio para gestionar límites de seguridad en operaciones de trading.
    
    Args:
        strategy_name (str): Nombre de la estrategia.
        config (dict, opcional): Configuración de límites personalizada.
    
    Example:
        >>> safety = SafetyLimitsService("GridStrategy")
        >>> safety.check_position_limit(current_position, new_position_size)
    """
    
    DEFAULT_LIMITS = {
        'spot': {
            'max_position_size_pct': 0.25,  # 25% del capital por posición
            'max_total_exposure_pct': 0.8,  # 80% del capital total
            'max_drawdown_pct': 0.15,       # 15% drawdown máximo
            'max_trades_per_day': 50,       # 50 operaciones por día
            'min_time_between_trades': 300, # 5 minutos entre operaciones
            'max_consecutive_losses': 5     # 5 pérdidas consecutivas
        },
        'futures': {
            'max_position_size_pct': 0.15,  # 15% del capital por posición (más conservador)
            'max_total_exposure_pct': 0.6,  # 60% del capital total
            'max_drawdown_pct': 0.10,       # 10% drawdown máximo (más conservador)
            'max_trades_per_day': 30,       # 30 operaciones por día
            'min_time_between_trades': 600, # 10 minutos entre operaciones
            'max_consecutive_losses': 3,    # 3 pérdidas consecutivas
            'max_leverage': 10              # Máximo 10x leverage
        }
    }
    
    def __init__(self, strategy_name: str, config: Optional[Dict[str, Any]] = None):
        self.strategy_name = strategy_name
        self.config = config or {}
        self.logger = logging.getLogger(f"safety.{strategy_name}")
        
        # Estado de la estrategia
        self.current_position_size = 0.0
        self.total_exposure = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.trades_today = 0
        self.last_trade_time = None
        self.consecutive_losses = 0
        self.trade_history = []
        
        # Obtener límites según tipo de trading
        self.trading_type = self.config.get('trading_type', 'spot')
        self.limits = self.DEFAULT_LIMITS[self.trading_type].copy()
        
        # Actualizar con configuración personalizada
        if 'custom_limits' in self.config:
            self.limits.update(self.config['custom_limits'])
        
        self.logger.info(f"🔒 Límites de seguridad configurados para {strategy_name} ({self.trading_type})")
    
    def check_position_limit(self, 
                           new_position_size: float, 
                           current_capital: float) -> bool:
        """
        Verifica si una nueva posición viola el límite de tamaño.
        
        Args:
            new_position_size (float): Tamaño de la nueva posición.
            current_capital (float): Capital actual.
            
        Returns:
            bool: True si está dentro del límite.
            
        Raises:
            SafetyViolation: Si se viola el límite.
        """
        position_pct = new_position_size / current_capital
        max_position_pct = self.limits['max_position_size_pct']
        
        if position_pct > max_position_pct:
            violation = SafetyViolation(
                LimitType.POSITION_SIZE,
                position_pct,
                max_position_pct,
                f"Posición {position_pct:.2%} excede límite {max_position_pct:.2%}"
            )
            self._log_violation(violation)
            raise violation
        
        return True
    
    def check_exposure_limit(self, 
                           additional_exposure: float, 
                           current_capital: float) -> bool:
        """
        Verifica si la exposición total viola el límite.
        
        Args:
            additional_exposure (float): Exposición adicional.
            current_capital (float): Capital actual.
            
        Returns:
            bool: True si está dentro del límite.
            
        Raises:
            SafetyViolation: Si se viola el límite.
        """
        new_total_exposure = self.total_exposure + additional_exposure
        exposure_pct = new_total_exposure / current_capital
        max_exposure_pct = self.limits['max_total_exposure_pct']
        
        if exposure_pct > max_exposure_pct:
            violation = SafetyViolation(
                LimitType.EXPOSURE,
                exposure_pct,
                max_exposure_pct,
                f"Exposición {exposure_pct:.2%} excede límite {max_exposure_pct:.2%}"
            )
            self._log_violation(violation)
            raise violation
        
        return True
    
    def check_drawdown_limit(self, current_equity: float) -> bool:
        """
        Verifica si el drawdown actual viola el límite.
        
        Args:
            current_equity (float): Equity actual.
            
        Returns:
            bool: True si está dentro del límite.
            
        Raises:
            SafetyViolation: Si se viola el límite.
        """
        # Actualizar peak equity si es necesario
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        self.current_equity = current_equity
        
        if self.peak_equity <= 0:
            return True
        
        drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity
        max_drawdown_pct = self.limits['max_drawdown_pct']
        
        if drawdown_pct > max_drawdown_pct:
            violation = SafetyViolation(
                LimitType.DRAWDOWN,
                drawdown_pct,
                max_drawdown_pct,
                f"Drawdown {drawdown_pct:.2%} excede límite {max_drawdown_pct:.2%}"
            )
            self._log_violation(violation)
            raise violation
        
        return True
    
    def check_frequency_limit(self) -> bool:
        """
        Verifica si la frecuencia de operaciones viola el límite.
        
        Returns:
            bool: True si está dentro del límite.
            
        Raises:
            SafetyViolation: Si se viola el límite.
        """
        now = datetime.now()
        
        # Verificar límite diario
        if self.trades_today >= self.limits['max_trades_per_day']:
            violation = SafetyViolation(
                LimitType.FREQUENCY,
                self.trades_today,
                self.limits['max_trades_per_day'],
                f"Operaciones diarias {self.trades_today} exceden límite {self.limits['max_trades_per_day']}"
            )
            self._log_violation(violation)
            raise violation
        
        # Verificar tiempo mínimo entre operaciones
        if self.last_trade_time:
            time_since_last = (now - self.last_trade_time).total_seconds()
            min_time = self.limits['min_time_between_trades']
            
            if time_since_last < min_time:
                violation = SafetyViolation(
                    LimitType.FREQUENCY,
                    time_since_last,
                    min_time,
                    f"Tiempo entre operaciones {time_since_last}s es menor que {min_time}s"
                )
                self._log_violation(violation)
                raise violation
        
        return True
    
    def check_leverage_limit(self, leverage: float) -> bool:
        """
        Verifica si el leverage viola el límite (solo para futuros).
        
        Args:
            leverage (float): Leverage a verificar.
            
        Returns:
            bool: True si está dentro del límite.
            
        Raises:
            SafetyViolation: Si se viola el límite.
        """
        if self.trading_type != 'futures':
            return True
        
        max_leverage = self.limits.get('max_leverage', 10)
        
        if leverage > max_leverage:
            violation = SafetyViolation(
                LimitType.LEVERAGE,
                leverage,
                max_leverage,
                f"Leverage {leverage}x excede límite {max_leverage}x"
            )
            self._log_violation(violation)
            raise violation
        
        return True
    
    def check_consecutive_losses_limit(self) -> bool:
        """
        Verifica si las pérdidas consecutivas violan el límite.
        
        Returns:
            bool: True si está dentro del límite.
            
        Raises:
            SafetyViolation: Si se viola el límite.
        """
        max_consecutive_losses = self.limits['max_consecutive_losses']
        
        if self.consecutive_losses >= max_consecutive_losses:
            violation = SafetyViolation(
                LimitType.FREQUENCY,
                self.consecutive_losses,
                max_consecutive_losses,
                f"Pérdidas consecutivas {self.consecutive_losses} exceden límite {max_consecutive_losses}"
            )
            self._log_violation(violation)
            raise violation
        
        return True
    
    def check_all_limits(self, 
                        new_position_size: float = 0.0,
                        additional_exposure: float = 0.0,
                        current_capital: float = 100000.0,
                        current_equity: float = 100000.0,
                        leverage: float = 1.0) -> bool:
        """
        Verifica todos los límites de seguridad.
        
        Args:
            new_position_size (float): Tamaño de nueva posición.
            additional_exposure (float): Exposición adicional.
            current_capital (float): Capital actual.
            current_equity (float): Equity actual.
            leverage (float): Leverage (solo para futuros).
            
        Returns:
            bool: True si todos los límites están dentro de rango.
            
        Raises:
            SafetyViolation: Si se viola cualquier límite.
        """
        try:
            # Verificar límites en orden de importancia
            self.check_drawdown_limit(current_equity)
            self.check_exposure_limit(additional_exposure, current_capital)
            self.check_position_limit(new_position_size, current_capital)
            self.check_leverage_limit(leverage)
            self.check_frequency_limit()
            self.check_consecutive_losses_limit()
            
            return True
            
        except SafetyViolation as e:
            # Re-lanzar la violación
            raise
    
    def record_trade(self, 
                    trade_type: str, 
                    size: float, 
                    price: float, 
                    pnl: float = 0.0):
        """
        Registra una operación para tracking de límites.
        
        Args:
            trade_type (str): 'buy' o 'sell'.
            size (float): Tamaño de la operación.
            price (float): Precio de la operación.
            pnl (float): P&L de la operación (si aplica).
        """
        now = datetime.now()
        
        # Actualizar contadores
        self.trades_today += 1
        self.last_trade_time = now
        
        # Actualizar pérdidas consecutivas
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Actualizar exposición
        if trade_type == 'buy':
            self.total_exposure += size * price
        else:  # sell
            self.total_exposure = max(0, self.total_exposure - size * price)
        
        # Registrar en historial
        trade_record = {
            'timestamp': now,
            'type': trade_type,
            'size': size,
            'price': price,
            'pnl': pnl,
            'total_exposure': self.total_exposure,
            'consecutive_losses': self.consecutive_losses
        }
        self.trade_history.append(trade_record)
        
        self.logger.debug(f"Operación registrada: {trade_type} {size} @ {price}")
    
    def reset_daily_limits(self):
        """Resetea límites diarios (llamar al inicio de cada día)."""
        self.trades_today = 0
        self.logger.info(f"🔄 Límites diarios reseteados para {self.strategy_name}")
    
    def get_safety_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen del estado de seguridad."""
        current_drawdown = 0.0
        if self.peak_equity > 0:
            current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        
        return {
            'strategy_name': self.strategy_name,
            'trading_type': self.trading_type,
            'current_position_size': self.current_position_size,
            'total_exposure': self.total_exposure,
            'current_drawdown': current_drawdown,
            'trades_today': self.trades_today,
            'consecutive_losses': self.consecutive_losses,
            'peak_equity': self.peak_equity,
            'current_equity': self.current_equity,
            'limits': self.limits.copy()
        }
    
    def _log_violation(self, violation: SafetyViolation):
        """Loggea una violación de límite de seguridad."""
        self.logger.warning(f"🚨 VIOLACIÓN DE SEGURIDAD en {self.strategy_name}: {violation}")
        self.logger.warning(f"   Tipo: {violation.limit_type.value}")
        self.logger.warning(f"   Valor actual: {violation.current_value}")
        self.logger.warning(f"   Límite: {violation.limit_value}")

# Ejemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Crear servicio de límites de seguridad
    safety = SafetyLimitsService("TestStrategy", {'trading_type': 'spot'})
    
    # Ejemplo 1: Verificar límites válidos
    try:
        safety.check_all_limits(
            new_position_size=10000,  # 10% del capital
            additional_exposure=20000,  # 20% del capital
            current_capital=100000,
            current_equity=95000
        )
        print("✅ Todos los límites están dentro de rango")
    except SafetyViolation as e:
        print(f"❌ Violación de límite: {e}")
    
    # Ejemplo 2: Violación de límite de posición
    try:
        safety.check_position_limit(50000, 100000)  # 50% del capital
        print("✅ Límite de posición OK")
    except SafetyViolation as e:
        print(f"❌ Violación de posición: {e}")
    
    # Ejemplo 3: Registrar operaciones
    safety.record_trade('buy', 1000, 100.0)
    safety.record_trade('sell', 500, 105.0, pnl=2500)
    
    # Ejemplo 4: Resumen de seguridad
    summary = safety.get_safety_summary()
    print(f"\n📊 Resumen de seguridad: {summary}") 