"""
Backtesting Service - Servicio de Backtesting Moderno
====================================================

Servicio de infraestructura que usa la librería backtesting.py profesional.
Migrado desde scripts/modern_strategies.py para usar backtesting.py
en lugar de simuladores manuales caseros.

Contiene:
- run_modern_backtest(): Función principal que usa backtesting.py
- BacktestingService: Servicio que coordina las estrategias modernas
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Type
from datetime import datetime
import logging
from backtesting import Backtest, Strategy

# Importar estrategias modernas del dominio
from app.domain.strategies import GridTradingStrategy, DCAStrategy, BTDStrategy

logger = logging.getLogger(__name__)


def run_modern_backtest(df: pd.DataFrame, 
                       strategy_class: Type[Strategy],
                       strategy_params: Dict[str, Any],
                       commission: float = 0.001,
                       cash: float = 10000) -> Dict[str, Any]:
    """
    Ejecuta backtesting moderno usando la librería backtesting.py.
    
    Esta función reemplaza los simuladores manuales por una implementación
    profesional y estándar de la industria.
    
    Args:
        df: DataFrame con datos OHLC (Open, High, Low, Close)
        strategy_class: Clase de estrategia (GridTradingStrategy, etc.)
        strategy_params: Parámetros de la estrategia
        commission: Comisión por transacción (default 0.1%)
        cash: Capital inicial (default $10,000)
        
    Returns:
        Diccionario con resultados del backtesting
    """
    try:
        logger.info(f"🚀 Ejecutando backtesting moderno con {strategy_class.__name__}")
        
        # Validar DataFrame
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columnas faltantes en DataFrame: {missing_columns}")
        
        # Preparar datos - backtesting.py espera columnas con mayúsculas
        df_bt = df.copy()
        df_bt.columns = [col.capitalize() for col in df_bt.columns]
        
        # Asegurar que tenemos las columnas correctas
        if 'Open' not in df_bt.columns:
            df_bt['Open'] = df_bt.get('open', df_bt['Close'].shift(1).fillna(df_bt['Close']))
        if 'High' not in df_bt.columns:
            df_bt['High'] = df_bt.get('high', df_bt['Close'])
        if 'Low' not in df_bt.columns:
            df_bt['Low'] = df_bt.get('low', df_bt['Close'])
        if 'Volume' not in df_bt.columns and 'volume' in df.columns:
            df_bt['Volume'] = df['volume']
        
        # Eliminar valores nulos
        df_bt = df_bt.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        if len(df_bt) < 30:
            raise ValueError(f"Datos insuficientes después de limpieza: {len(df_bt)} registros")
        
        # Configurar estrategia con parámetros
        class ConfiguredStrategy(strategy_class):
            pass
        
        # Aplicar parámetros dinámicamente
        for param_name, param_value in strategy_params.items():
            if hasattr(ConfiguredStrategy, param_name):
                setattr(ConfiguredStrategy, param_name, param_value)
        
        # Crear y ejecutar backtest
        bt = Backtest(
            data=df_bt,
            strategy=ConfiguredStrategy,
            cash=cash,
            commission=commission,
            exclusive_orders=True,  # Solo una orden a la vez
            trade_on_close=True     # Ejecutar al cierre de la barra
        )
        
        # Ejecutar backtesting
        logger.info(f"📊 Ejecutando backtest: {len(df_bt)} períodos, capital=${cash:,.0f}")
        stats = bt.run()
        
        # Extraer métricas principales
        results = {
            # Métricas principales de rendimiento
            'Return [%]': float(stats.get('Return [%]', 0.0)),
            'Buy & Hold Return [%]': float(stats.get('Buy & Hold Return [%]', 0.0)),
            'Max. Drawdown [%]': float(stats.get('Max. Drawdown [%]', 0.0)),
            'Volatility [%]': float(stats.get('Volatility [%]', 0.0)),
            
            # Métricas de trading
            '# Trades': int(stats.get('# Trades', 0)),
            'Win Rate [%]': float(stats.get('Win Rate [%]', 0.0)),
            'Best Trade [%]': float(stats.get('Best Trade [%]', 0.0)),
            'Worst Trade [%]': float(stats.get('Worst Trade [%]', 0.0)),
            'Avg. Trade [%]': float(stats.get('Avg. Trade [%]', 0.0)),
            
            # Métricas de riesgo-retorno
            'Sharpe Ratio': float(stats.get('Sharpe Ratio', 0.0)),
            'Calmar Ratio': float(stats.get('Calmar Ratio', 0.0)),
            'Sortino Ratio': float(stats.get('Sortino Ratio', 0.0)),
            
            # Información adicional
            'Start': str(stats.get('Start', '')),
            'End': str(stats.get('End', '')),
            'Duration': str(stats.get('Duration', '')),
            'Exposure Time [%]': float(stats.get('Exposure Time [%]', 0.0)),
            
            # Valores absolutos
            'Equity Final [$]': float(stats.get('Equity Final [$]', cash)),
            'Return [USD]': float(stats.get('Equity Final [$]', cash)) - cash,
            
            # Configuración utilizada
            'strategy': strategy_class.__name__,
            'initial_cash': cash,
            'commission': commission,
            'parameters': strategy_params
        }
        
        # Calcular métricas adicionales
        total_return = results['Return [%]']
        max_drawdown = abs(results['Max. Drawdown [%]'])
        
        # Calificación de performance (0-100)
        performance_score = max(0, min(100, 
            50 +  # Base score
            (total_return * 0.5) +  # +0.5 por cada 1% de retorno
            (results['Win Rate [%]'] * 0.2) -  # +0.2 por cada 1% de win rate
            (max_drawdown * 0.3)  # -0.3 por cada 1% de drawdown
        ))
        
        results['performance_score'] = round(performance_score, 1)
        
        logger.info(f"✅ Backtesting completado:")
        logger.info(f"   💰 Retorno: {total_return:.2f}%")
        logger.info(f"   📊 Trades: {results['# Trades']}")
        logger.info(f"   🎯 Win Rate: {results['Win Rate [%]']:.1f}%")
        logger.info(f"   📉 Max Drawdown: {max_drawdown:.2f}%")
        logger.info(f"   ⭐ Score: {performance_score:.1f}/100")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error en backtesting moderno: {e}")
        return {
            'error': str(e),
            'Return [%]': 0.0,
            'Max. Drawdown [%]': 100.0,
            '# Trades': 0,
            'Win Rate [%]': 0.0,
            'Sharpe Ratio': 0.0,
            'performance_score': 0.0,
            'strategy': strategy_class.__name__ if strategy_class else 'Unknown',
            'parameters': strategy_params
        }


class BacktestingService:
    """
    Servicio principal de backtesting que coordina las estrategias modernas.
    
    Usa la librería backtesting.py profesional en lugar de simuladores caseros.
    """
    
    def __init__(self):
        """Inicializa el servicio de backtesting moderno."""
        logger.info("🧪 BacktestingService moderno inicializado (con backtesting.py)")
    
    def run_grid_simulation(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta simulación de Grid Trading usando backtesting.py.
        
        Args:
            df: DataFrame con datos OHLC
            config: Configuración de la estrategia
            
        Returns:
            Resultados del backtesting
        """
        try:
            # Preparar parámetros para GridTradingStrategy
            strategy_params = {
                'levels': config.get('levels', 4),
                'range_percent': config.get('range_percent', 8.0),
                'umbral_adx': config.get('umbral_adx', 25.0),
                'umbral_volatilidad': config.get('umbral_volatilidad', 0.02),
                'umbral_sentimiento': config.get('umbral_sentimiento', 0.0)
            }
            
            cash = config.get('initial_capital', 10000.0)
            commission = config.get('commission', 0.001)
            
            return run_modern_backtest(
                df=df,
                strategy_class=GridTradingStrategy,
                strategy_params=strategy_params,
                commission=commission,
                cash=cash
            )
            
        except Exception as e:
            logger.error(f"❌ Error en simulación Grid: {e}")
            return {'error': str(e), 'strategy': 'Grid Trading'}
    
    def run_dca_simulation(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta simulación de DCA usando backtesting.py.
        
        Args:
            df: DataFrame con datos OHLC
            config: Configuración de la estrategia
            
        Returns:
            Resultados del backtesting
        """
        try:
            # Preparar parámetros para DCAStrategy
            strategy_params = {
                'intervalo_compra': config.get('intervalo_compra', 3),
                'monto_compra': config.get('monto_compra', 0.2),
                'objetivo_ganancia': config.get('objetivo_ganancia', 0.15),
                'dip_threshold': config.get('dip_threshold', 0.05),
                'tendencia_alcista_dias': config.get('tendencia_alcista_dias', 7),
                'stop_loss': config.get('stop_loss', 0.20)
            }
            
            cash = config.get('initial_capital', 10000.0)
            commission = config.get('commission', 0.001)
            
            return run_modern_backtest(
                df=df,
                strategy_class=DCAStrategy,
                strategy_params=strategy_params,
                commission=commission,
                cash=cash
            )
            
        except Exception as e:
            logger.error(f"❌ Error en simulación DCA: {e}")
            return {'error': str(e), 'strategy': 'DCA'}
    
    def run_btd_simulation(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta simulación de Buy The Dip usando backtesting.py.
        
        Args:
            df: DataFrame con datos OHLC
            config: Configuración de la estrategia
            
        Returns:
            Resultados del backtesting
        """
        try:
            # Preparar parámetros para BTDStrategy
            strategy_params = {
                'intervalo_venta': config.get('intervalo_venta', 2),
                'monto_venta': config.get('monto_venta', 0.25),
                'objetivo_ganancia': config.get('objetivo_ganancia', 0.12),
                'rip_threshold': config.get('rip_threshold', 0.04),
                'tendencia_bajista_dias': config.get('tendencia_bajista_dias', 5),
                'stop_loss': config.get('stop_loss', 0.15)
            }
            
            cash = config.get('initial_capital', 10000.0)
            commission = config.get('commission', 0.001)
            
            return run_modern_backtest(
                df=df,
                strategy_class=BTDStrategy,
                strategy_params=strategy_params,
                commission=commission,
                cash=cash
            )
            
        except Exception as e:
            logger.error(f"❌ Error en simulación BTD: {e}")
            return {'error': str(e), 'strategy': 'BTD'}
    
    def run_strategy_backtest(self, 
                            df: pd.DataFrame, 
                            strategy_name: str, 
                            params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Método genérico para ejecutar cualquier estrategia.
        
        Args:
            df: DataFrame con datos OHLC
            strategy_name: Nombre de la estrategia ('grid', 'dca', 'btd')
            params: Parámetros de configuración
            
        Returns:
            Resultados del backtesting
        """
        try:
            if strategy_name.lower() == 'grid':
                return self.run_grid_simulation(df, params)
            elif strategy_name.lower() == 'dca':
                return self.run_dca_simulation(df, params)
            elif strategy_name.lower() == 'btd':
                return self.run_btd_simulation(df, params)
            else:
                raise ValueError(f"Estrategia no soportada: {strategy_name}")
                
        except Exception as e:
            logger.error(f"❌ Error ejecutando estrategia {strategy_name}: {e}")
            return {'error': str(e), 'strategy': strategy_name}
    
    def compare_strategies(self, 
                          df: pd.DataFrame, 
                          strategies_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compara múltiples estrategias en los mismos datos.
        
        Args:
            df: DataFrame con datos OHLC
            strategies_config: Configuración de múltiples estrategias
            
        Returns:
            Comparación de resultados
        """
        logger.info(f"📊 Comparando {len(strategies_config)} estrategias...")
        
        results = {}
        
        for strategy_name, config in strategies_config.items():
            logger.info(f"🔄 Ejecutando {strategy_name}...")
            result = self.run_strategy_backtest(df, strategy_name, config)
            results[strategy_name] = result
        
        # Encontrar la mejor estrategia
        best_strategy = None
        best_score = -float('inf')
        
        for strategy_name, result in results.items():
            if 'error' not in result:
                score = result.get('performance_score', 0)
                if score > best_score:
                    best_score = score
                    best_strategy = strategy_name
        
        return {
            'results': results,
            'best_strategy': best_strategy,
            'best_score': best_score,
            'comparison_summary': {
                name: {
                    'return': result.get('Return [%]', 0),
                    'max_drawdown': result.get('Max. Drawdown [%]', 0),
                    'sharpe_ratio': result.get('Sharpe Ratio', 0),
                    'trades': result.get('# Trades', 0),
                    'score': result.get('performance_score', 0)
                }
                for name, result in results.items()
                if 'error' not in result
            }
        } 