"""
ParameterValidatorService - Servicio de Validación de Parámetros
==============================================================

Valida parámetros de estrategias de trading para asegurar rangos seguros y coherencia.
Permite configuración por estrategia y tipo de activo (spot/futuros).
"""

from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

class ParameterValidationError(Exception):
    """Excepción lanzada cuando la validación de parámetros falla."""
    pass

class StrategyType(Enum):
    """Tipos de estrategias disponibles."""
    GRID = "grid"
    DCA = "dca"
    BTD = "btd"
    FUTURES_GRID = "futures_grid"

class ParameterValidatorService:
    """
    Servicio para validar parámetros de estrategias de trading.
    
    Args:
        config (dict, opcional): Configuración personalizada de validación.
    
    Example:
        >>> validator = ParameterValidatorService()
        >>> validator.validar_parametros_grid(params, tipo='spot')
    """
    
    # Rangos seguros por estrategia y tipo
    PARAMETER_RANGES = {
        StrategyType.GRID.value: {
            'spot': {
                'grid_levels': {'min': 3, 'max': 50, 'type': 'int'},
                'grid_spacing': {'min': 0.001, 'max': 0.1, 'type': 'float'},
                'investment_per_grid': {'min': 10, 'max': 10000, 'type': 'float'},
                'take_profit': {'min': 0.005, 'max': 0.5, 'type': 'float'},
                'stop_loss': {'min': 0.005, 'max': 0.3, 'type': 'float'},
                'max_positions': {'min': 1, 'max': 20, 'type': 'int'},
                'rebalance_threshold': {'min': 0.01, 'max': 0.2, 'type': 'float'}
            },
            'futures': {
                'grid_levels': {'min': 3, 'max': 30, 'type': 'int'},
                'grid_spacing': {'min': 0.002, 'max': 0.05, 'type': 'float'},
                'investment_per_grid': {'min': 50, 'max': 5000, 'type': 'float'},
                'leverage': {'min': 1, 'max': 20, 'type': 'int'},
                'take_profit': {'min': 0.01, 'max': 0.3, 'type': 'float'},
                'stop_loss': {'min': 0.01, 'max': 0.2, 'type': 'float'},
                'max_positions': {'min': 1, 'max': 15, 'type': 'int'},
                'liquidation_buffer': {'min': 0.1, 'max': 0.5, 'type': 'float'}
            }
        },
        StrategyType.DCA.value: {
            'spot': {
                'initial_investment': {'min': 100, 'max': 100000, 'type': 'float'},
                'dca_amount': {'min': 10, 'max': 10000, 'type': 'float'},
                'dca_frequency': {'min': 1, 'max': 30, 'type': 'int'},
                'take_profit': {'min': 0.05, 'max': 1.0, 'type': 'float'},
                'stop_loss': {'min': 0.05, 'max': 0.5, 'type': 'float'},
                'max_dca_cycles': {'min': 1, 'max': 100, 'type': 'int'}
            },
            'futures': {
                'initial_investment': {'min': 200, 'max': 50000, 'type': 'float'},
                'dca_amount': {'min': 50, 'max': 5000, 'type': 'float'},
                'leverage': {'min': 1, 'max': 10, 'type': 'int'},
                'dca_frequency': {'min': 1, 'max': 20, 'type': 'int'},
                'take_profit': {'min': 0.1, 'max': 0.5, 'type': 'float'},
                'stop_loss': {'min': 0.1, 'max': 0.3, 'type': 'float'},
                'max_dca_cycles': {'min': 1, 'max': 50, 'type': 'int'}
            }
        },
        StrategyType.BTD.value: {
            'spot': {
                'buy_threshold': {'min': 0.02, 'max': 0.5, 'type': 'float'},
                'sell_threshold': {'min': 0.02, 'max': 0.5, 'type': 'float'},
                'position_size': {'min': 10, 'max': 10000, 'type': 'float'},
                'max_positions': {'min': 1, 'max': 10, 'type': 'int'},
                'take_profit': {'min': 0.05, 'max': 1.0, 'type': 'float'},
                'stop_loss': {'min': 0.05, 'max': 0.5, 'type': 'float'}
            },
            'futures': {
                'buy_threshold': {'min': 0.03, 'max': 0.3, 'type': 'float'},
                'sell_threshold': {'min': 0.03, 'max': 0.3, 'type': 'float'},
                'position_size': {'min': 50, 'max': 5000, 'type': 'float'},
                'leverage': {'min': 1, 'max': 10, 'type': 'int'},
                'max_positions': {'min': 1, 'max': 8, 'type': 'int'},
                'take_profit': {'min': 0.1, 'max': 0.5, 'type': 'float'},
                'stop_loss': {'min': 0.1, 'max': 0.3, 'type': 'float'}
            }
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    def validar_parametros(self, 
                          params: Dict[str, Any], 
                          strategy: str, 
                          tipo: str = 'spot') -> Dict[str, Any]:
        """
        Valida parámetros para una estrategia específica.
        
        Args:
            params (dict): Parámetros a validar.
            strategy (str): Tipo de estrategia ('grid', 'dca', 'btd', 'futures_grid').
            tipo (str): 'spot' o 'futuros'.
            
        Returns:
            dict: Parámetros validados.
            
        Raises:
            ParameterValidationError: Si la validación falla.
        """
        errores = []
        
        # Normalizar nombre de estrategia
        if strategy == 'futures_grid':
            strategy = 'grid'
        
        # Verificar que la estrategia existe
        if strategy not in self.PARAMETER_RANGES:
            errores.append(f"Estrategia no válida: {strategy}")
            raise ParameterValidationError("\n".join(errores))
        
        # Verificar que el tipo existe para la estrategia
        if tipo not in self.PARAMETER_RANGES[strategy]:
            errores.append(f"Tipo '{tipo}' no válido para estrategia '{strategy}'")
            raise ParameterValidationError("\n".join(errores))
        
        # Obtener rangos para esta estrategia y tipo
        ranges = self.PARAMETER_RANGES[strategy][tipo]
        
        # Validar cada parámetro
        for param_name, param_value in params.items():
            if param_name in ranges:
                range_config = ranges[param_name]
                error = self._validar_parametro_individual(
                    param_name, param_value, range_config, tipo
                )
                if error:
                    errores.append(error)
        
        # Validar coherencia entre parámetros
        coherencia_errors = self._validar_coherencia_parametros(params, strategy, tipo)
        errores.extend(coherencia_errors)
        
        # Validar parámetros específicos del contexto
        contexto_errors = self._validar_contexto_parametros(params, strategy, tipo)
        errores.extend(contexto_errors)
        
        if errores:
            raise ParameterValidationError("\n".join(errores))
        
        return params.copy()
    
    def _validar_parametro_individual(self, 
                                    param_name: str, 
                                    param_value: Any, 
                                    range_config: Dict[str, Any],
                                    tipo: str) -> Optional[str]:
        """Valida un parámetro individual."""
        try:
            # Verificar tipo
            expected_type = range_config['type']
            if expected_type == 'int':
                param_value = int(param_value)
            elif expected_type == 'float':
                param_value = float(param_value)
            
            # Verificar rango
            min_val = range_config['min']
            max_val = range_config['max']
            
            if param_value < min_val or param_value > max_val:
                return f"Parámetro '{param_name}' fuera de rango: {param_value} (debe estar entre {min_val} y {max_val})"
            
            return None
            
        except (ValueError, TypeError):
            return f"Parámetro '{param_name}' tiene tipo inválido: {type(param_value)} (esperado: {expected_type})"
    
    def _validar_coherencia_parametros(self, 
                                     params: Dict[str, Any], 
                                     strategy: str, 
                                     tipo: str) -> List[str]:
        """Valida coherencia entre parámetros relacionados."""
        errores = []
        
        if strategy == 'grid':
            # En GRID, take_profit debe ser mayor que grid_spacing
            if 'take_profit' in params and 'grid_spacing' in params:
                if params['take_profit'] <= params['grid_spacing']:
                    errores.append("take_profit debe ser mayor que grid_spacing")
            
            # En GRID, stop_loss debe ser mayor que grid_spacing
            if 'stop_loss' in params and 'grid_spacing' in params:
                if params['stop_loss'] <= params['grid_spacing']:
                    errores.append("stop_loss debe ser mayor que grid_spacing")
        
        elif strategy == 'dca':
            # En DCA, take_profit debe ser mayor que stop_loss
            if 'take_profit' in params and 'stop_loss' in params:
                if params['take_profit'] <= params['stop_loss']:
                    errores.append("take_profit debe ser mayor que stop_loss")
            
            # En DCA, dca_amount debe ser menor que initial_investment
            if 'dca_amount' in params and 'initial_investment' in params:
                if params['dca_amount'] >= params['initial_investment']:
                    errores.append("dca_amount debe ser menor que initial_investment")
        
        elif strategy == 'btd':
            # En BTD, buy_threshold y sell_threshold deben ser positivos
            if 'buy_threshold' in params and 'sell_threshold' in params:
                if params['buy_threshold'] <= 0 or params['sell_threshold'] <= 0:
                    errores.append("buy_threshold y sell_threshold deben ser positivos")
        
        return errores
    
    def _validar_contexto_parametros(self, 
                                   params: Dict[str, Any], 
                                   strategy: str, 
                                   tipo: str) -> List[str]:
        """Valida parámetros específicos del contexto (spot vs futuros)."""
        errores = []
        
        if tipo == 'futuros':
            # En futuros, leverage es obligatorio
            if 'leverage' not in params:
                errores.append("Parámetro 'leverage' es obligatorio para futuros")
            
            # En futuros, liquidation_buffer es recomendado
            if 'liquidation_buffer' not in params and strategy == 'grid':
                errores.append("Parámetro 'liquidation_buffer' es recomendado para GRID en futuros")
        
        elif tipo == 'spot':
            # En spot, leverage no debe estar presente
            if 'leverage' in params:
                errores.append("Parámetro 'leverage' no es válido para spot trading")
            
            # En spot, liquidation_buffer no debe estar presente
            if 'liquidation_buffer' in params:
                errores.append("Parámetro 'liquidation_buffer' no es válido para spot trading")
        
        return errores
    
    def obtener_rangos_recomendados(self, strategy: str, tipo: str = 'spot') -> Dict[str, Any]:
        """
        Obtiene los rangos recomendados para una estrategia.
        
        Args:
            strategy (str): Tipo de estrategia.
            tipo (str): 'spot' o 'futuros'.
            
        Returns:
            dict: Rangos recomendados.
        """
        if strategy not in self.PARAMETER_RANGES:
            raise ValueError(f"Estrategia no válida: {strategy}")
        
        if tipo not in self.PARAMETER_RANGES[strategy]:
            raise ValueError(f"Tipo '{tipo}' no válido para estrategia '{strategy}'")
        
        return self.PARAMETER_RANGES[strategy][tipo].copy()

# Ejemplo de uso
if __name__ == "__main__":
    validator = ParameterValidatorService()
    
    # Ejemplo 1: Parámetros válidos para GRID spot
    params_grid_spot = {
        'grid_levels': 10,
        'grid_spacing': 0.02,
        'investment_per_grid': 100,
        'take_profit': 0.05,
        'stop_loss': 0.03,
        'max_positions': 5
    }
    
    try:
        params_validados = validator.validar_parametros(params_grid_spot, 'grid', 'spot')
        print("✅ Parámetros GRID spot válidos:")
        print(params_validados)
    except ParameterValidationError as e:
        print("❌ Error de validación:", e)
    
    # Ejemplo 2: Parámetros inválidos (take_profit menor que grid_spacing)
    params_invalidos = {
        'grid_levels': 10,
        'grid_spacing': 0.05,
        'investment_per_grid': 100,
        'take_profit': 0.02,  # Menor que grid_spacing
        'stop_loss': 0.03,
        'max_positions': 5
    }
    
    try:
        params_validados = validator.validar_parametros(params_invalidos, 'grid', 'spot')
        print("✅ Parámetros válidos")
    except ParameterValidationError as e:
        print("❌ Error de validación esperado:")
        print(e)
    
    # Ejemplo 3: Rangos recomendados
    rangos = validator.obtener_rangos_recomendados('grid', 'spot')
    print("\n📊 Rangos recomendados para GRID spot:")
    for param, config in rangos.items():
        print(f"  {param}: {config['min']} - {config['max']} ({config['type']})") 