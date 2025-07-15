"""
Futures Backtesting Service - Servicio de Backtesting para Futuros
================================================================

Servicio especializado para backtesting de estrategias de futuros con:
- Cálculo de liquidación
- Métricas específicas de futuros
- Objetivo: maximizar rendimiento sin liquidaciones

Extiende el servicio de backtesting regular con funcionalidades específicas
para trading de futuros con apalancamiento.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Type
from datetime import datetime
import logging
from backtesting import Backtest, Strategy

# Importar estrategia de futuros
from app.domain.strategies.futures_grid import FuturesGridStrategy
from app.infrastructure.services.data_validator_service import DataValidatorService, DataValidationError

logger = logging.getLogger(__name__)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convierte un valor a float de forma segura."""
    if value is None:
        return default
    try:
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            return float(value)
        else:
            return float(value)
    except (ValueError, TypeError, AttributeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Convierte un valor a int de forma segura."""
    if value is None:
        return default
    try:
        if isinstance(value, (int, float)):
            return int(value)
        elif isinstance(value, str):
            return int(float(value))
        else:
            return int(value)
    except (ValueError, TypeError, AttributeError):
        return default


def run_futures_backtest(df: pd.DataFrame, 
                        strategy_class: Type[Strategy],
                        strategy_params: Dict[str, Any],
                        commission: float = 0.001,
                        cash: float = 10000) -> Dict[str, Any]:
    """
    Ejecuta backtesting específico para futuros con métricas de liquidación.
    
    Args:
        df: DataFrame con datos OHLC
        strategy_class: Clase de estrategia de futuros
        strategy_params: Parámetros de la estrategia
        commission: Comisión por transacción
        cash: Capital inicial
        
    Returns:
        Diccionario con resultados incluyendo métricas de liquidación
    """
    try:
        logger.info(f"🚀 Ejecutando backtesting de futuros con {strategy_class.__name__}")
        
        # Preparar datos - copiar DataFrame original
        df_bt = df.copy()

        # Mapeo robusto de columnas OHLC y volume a minúsculas (antes de cualquier cambio)
        ohlc_keys = ['open', 'high', 'low', 'close', 'volume']
        col_map = {}
        for col in df_bt.columns:
            if col.lower() in ohlc_keys:
                col_map[col] = col.lower()
        df_bt = df_bt.rename(columns=col_map)
        
        # Validar datos DESPUÉS del mapeo de columnas
        validator = DataValidatorService()
        try:
            df_bt = validator.validar_dataframe(df_bt, tipo='futuros')
        except DataValidationError as e:
            logger.error(f"❌ Error de validación de datos: {e}")
            raise

        # Verificar columnas OHLC básicas
        required_columns_lower = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns_lower if col not in df_bt.columns]
        if missing_columns:
            raise ValueError(f"Columnas faltantes en DataFrame después de mapeo robusto: {missing_columns}. Columnas disponibles: {list(df_bt.columns)}")

        # Convertir a formato requerido por backtesting.py
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close'
        }
        df_bt = df_bt.rename(columns=column_mapping)

        # Manejar columna Volume si existe
        if 'volume' in df_bt.columns:
            df_bt = df_bt.rename(columns={'volume': 'Volume'})

        # Asegurar columna sentiment
        if 'sentiment_score' in df_bt.columns and 'sentiment' not in df_bt.columns:
            df_bt['sentiment'] = df_bt['sentiment_score']
        elif 'sentiment' not in df_bt.columns:
            df_bt['sentiment'] = 0.0

        # Limpiar datos
        df_bt = df_bt.dropna(subset=['Open', 'High', 'Low', 'Close'])

        if len(df_bt) < 30:
            raise ValueError(f"Datos insuficientes: {len(df_bt)} registros")

        # Validar precios
        df_bt = df_bt[(df_bt['High'] >= df_bt['Low']) & (df_bt['Close'] > 0)].copy()

        if len(df_bt) < 30:
            raise ValueError(f"Datos insuficientes después de validación: {len(df_bt)} registros")

        # Limpiar valores NaN de forma segura
        df_bt = df_bt.fillna(method='ffill').fillna(method='bfill')
        df_bt = df_bt.fillna(0.0)

        # Asegurar que df_bt sea DataFrame
        if not isinstance(df_bt, pd.DataFrame):
            df_bt = pd.DataFrame(df_bt)

        # Configurar estrategia con parámetros
        class ConfiguredFuturesStrategy(strategy_class):
            pass

        # Aplicar parámetros dinámicamente
        for param_name, param_value in strategy_params.items():
            if hasattr(ConfiguredFuturesStrategy, param_name):
                setattr(ConfiguredFuturesStrategy, param_name, param_value)

        # Verificar datos antes de crear Backtest
        logger.info(f"🔍 Verificando datos antes de backtest:")
        logger.info(f"   Columnas finales: {list(df_bt.columns)}")
        logger.info(f"   Shape: {df_bt.shape}")
        logger.info(f"   Primeras filas:")
        logger.info(f"   {df_bt.head(3)}")

        # Crear y ejecutar backtest
        try:
            bt = Backtest(
                data=df_bt,
                strategy=ConfiguredFuturesStrategy,
                cash=cash,
                commission=commission,
                exclusive_orders=True,
                trade_on_close=True
            )

            # Ejecutar backtesting
            logger.info(f"📊 Ejecutando backtest de futuros: {len(df_bt)} períodos, "
                       f"capital=${cash:,.0f}, leverage={strategy_params.get('leverage', 1)}x")

            stats = bt.run()

        except Exception as e:
            logger.error(f"❌ Error en Backtest.run(): {str(e)}")
            logger.error(f"   Tipo de error: {type(e).__name__}")
            logger.error(f"   Datos de entrada:")
            logger.error(f"   - Columnas: {list(df_bt.columns)}")
            logger.error(f"   - Shape: {df_bt.shape}")
            logger.error(f"   - Tipos: {df_bt.dtypes.to_dict()}")
            raise
        
        # Acceder a estadísticas de liquidación de la estrategia
        liquidation_stats = {}
        try:
            # Obtener la instancia de la estrategia ejecutada (no la clase)
            strategy_instance = stats._strategy
            if hasattr(strategy_instance, 'get_liquidation_stats'):
                liquidation_stats = strategy_instance.get_liquidation_stats()
        except Exception as e:
            logger.warning(f"No se pudieron obtener estadísticas de liquidación: {e}")
        
        # Extraer métricas principales
        results = {
            # Métricas principales de rendimiento
            'Return [%]': safe_float(stats.get('Return [%]')),
            'Buy & Hold Return [%]': safe_float(stats.get('Buy & Hold Return [%]')),
            'Max Drawdown [%]': safe_float(stats.get('Max. Drawdown [%]', stats.get('Max Drawdown [%]', 0))),
            'Volatility [%]': safe_float(stats.get('Volatility [%]')),
            'Sharpe Ratio': safe_float(stats.get('Sharpe Ratio')),
            'Sortino Ratio': safe_float(stats.get('Sortino Ratio')),
            'Calmar Ratio': safe_float(stats.get('Calmar Ratio')),
            
            # Métricas de trading
            'Win Rate [%]': safe_float(stats.get('Win Rate [%]')),
            'Best Trade [%]': safe_float(stats.get('Best Trade [%]')),
            'Worst Trade [%]': safe_float(stats.get('Worst Trade [%]')),
            'Avg Trade [%]': safe_float(stats.get('Avg. Trade [%]')),
            'Max Trade Duration': safe_int(stats.get('Max. Trade Duration')),
            'Avg Trade Duration': safe_int(stats.get('Avg. Trade Duration')),
            'Profit Factor': safe_float(stats.get('Profit Factor')),
            'Expectancy [%]': safe_float(stats.get('Expectancy [%]')),
            'SQN': safe_float(stats.get('SQN')),
            
            # Métricas específicas de futuros
            'Liquidation Count': liquidation_stats.get('liquidation_count', 0),
            'Was Liquidated': liquidation_stats.get('was_liquidated', False),
            'Liquidation History': liquidation_stats.get('liquidation_history', []),
            
            # Métricas de funding rate
            'Total Funding Paid': liquidation_stats.get('total_funding_paid', 0),
            'Funding Payments Count': liquidation_stats.get('funding_payments_count', 0),
            'Average Funding Rate': liquidation_stats.get('avg_funding_rate', 0),
            'Funding Payments': liquidation_stats.get('funding_payments', []),
            
            # Métricas de equity
            'Start': safe_float(stats.get('Start')),
            'End': safe_float(stats.get('End')),
            'Duration': safe_int(stats.get('Duration')),
            'Exposure Time [%]': safe_float(stats.get('Exposure Time [%]')),
            'Equity Final [$]': safe_float(stats.get('Equity Final [$]')),
            'Equity Peak [$]': safe_float(stats.get('Equity Peak [$]')),
            'Buy & Hold Final [$]': safe_float(stats.get('Buy & Hold Final [$]')),
            
            # Contadores de trades
            'Total Trades': safe_int(stats.get('# Trades')),
            'Total Closed Trades': safe_int(stats.get('# Trades')),
            'Total Open Trades': 0,
            'Total Won Trades': safe_int(stats.get('# Trades')) * safe_float(stats.get('Win Rate [%]', 0)) / 100,
            'Total Lost Trades': safe_int(stats.get('# Trades')) * (100 - safe_float(stats.get('Win Rate [%]', 0))) / 100,
            '# Trades': safe_int(stats.get('# Trades')),
            
            # Configuración de la estrategia
            'Strategy': strategy_class.__name__,
            'Parameters': strategy_params,
            'Commission': commission,
            'Initial Cash': cash,
            'Leverage': strategy_params.get('leverage', 1)
        }
        
        # Calcular métricas adicionales de funding
        if liquidation_stats.get('funding_payments'):
            funding_payments = liquidation_stats['funding_payments']
            total_funding_cost = sum(abs(p['funding_cost']) for p in funding_payments)
            results['Total Funding Cost'] = total_funding_cost
            results['Funding Cost as % of Return'] = (total_funding_cost / max(abs(results['Return [%]']), 0.01)) * 100
        else:
            results['Total Funding Cost'] = 0
            results['Funding Cost as % of Return'] = 0
        
        # Calcular métricas adicionales específicas de futuros
        total_return = results['Return [%]']
        max_drawdown = abs(results['Max Drawdown [%]'])
        liquidation_count = results['Liquidation Count']
        was_liquidated = results['Was Liquidated']
        
        # Penalización por liquidación
        liquidation_penalty = liquidation_count * 50  # -50 puntos por liquidación
        
        # Calificación de performance específica para futuros
        # Objetivo: maximizar rendimiento SIN liquidaciones
        if was_liquidated:
            # Si fue liquidado, score muy bajo
            performance_score = max(0, 10 - liquidation_penalty)
        else:
            # Si no fue liquidado, score normal con bonus
            performance_score = max(0, min(100, 
                60 +  # Base score más alto por no liquidar
                (total_return * 0.6) +  # Mayor peso al retorno
                (results['Win Rate [%]'] * 0.15) -  # Menor peso al win rate
                (max_drawdown * 0.2) +  # Menor penalización por drawdown
                10  # Bonus por no liquidar
            ))
        
        results['performance_score'] = round(performance_score, 1)
        
        # Métrica especial: Rendimiento ajustado por liquidación
        if was_liquidated:
            results['liquidation_adjusted_return'] = -100.0  # Pérdida total si liquidado
        else:
            results['liquidation_adjusted_return'] = total_return
        
        # Ratio de eficiencia de apalancamiento
        leverage_used = strategy_params.get('leverage', 1)
        if leverage_used > 1:
            results['leverage_efficiency'] = total_return / leverage_used
        else:
            results['leverage_efficiency'] = total_return
        
        logger.info(f"✅ Backtesting de futuros completado:")
        logger.info(f"   💰 Retorno: {total_return:.2f}%")
        logger.info(f"   📊 Trades: {results.get('# Trades', 0)}")
        logger.info(f"   🎯 Win Rate: {results.get('Win Rate [%]', 0):.1f}%")
        logger.info(f"   📉 Max Drawdown: {max_drawdown:.2f}%")
        logger.info(f"   💥 Liquidaciones: {liquidation_count}")
        logger.info(f"   🚨 Fue Liquidado: {was_liquidated}")
        logger.info(f"   ⚡ Leverage: {leverage_used}x")
        logger.info(f"   ⭐ Score: {performance_score:.1f}/100")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error en backtesting de futuros: {e}")
        return {
            'error': str(e),
            'Return [%]': 0.0,
            'Max Drawdown [%]': 100.0,
            '# Trades': 0,
            'Win Rate [%]': 0.0,
            'Sharpe Ratio': 0.0,
            'Liquidation Count': 999,  # Indicador de error
            'Was Liquidated': True,
            'performance_score': 0.0,
            'liquidation_adjusted_return': -100.0,
            'leverage_efficiency': 0.0,
            'strategy': strategy_class.__name__ if strategy_class else 'Unknown',
            'parameters': strategy_params
        }


class FuturesBacktestingService:
    """
    Servicio de backtesting especializado para futuros.
    
    Proporciona métodos específicos para backtesting de estrategias de futuros
    con métricas de liquidación y gestión de apalancamiento.
    """
    
    def __init__(self):
        """Inicializa el servicio de backtesting de futuros."""
        logger.info("🔧 Inicializando FuturesBacktestingService")
    
    def run_futures_grid_simulation(self, df: pd.DataFrame, 
                                   config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta simulación de Grid Trading para futuros.
        
        Args:
            df: DataFrame con datos OHLC
            config: Configuración de la estrategia
            
        Returns:
            Resultados del backtesting
        """
        try:
            logger.info("🔄 Ejecutando simulación de Grid Trading para futuros")
            
            # Parámetros por defecto para futuros
            default_params = {
                'levels': 4,
                'range_percent': 8.0,
                'leverage': 10,
                'umbral_adx': 25.0,
                'umbral_volatilidad': 0.02,
                'umbral_sentimiento': 0.0,
                'maintenance_margin_rate': 0.01
            }
            
            # Combinar con configuración proporcionada
            params = {**default_params, **config}
            
            # Ejecutar backtesting
            results = run_futures_backtest(
                df=df,
                strategy_class=FuturesGridStrategy,
                strategy_params=params,
                commission=0.001,
                cash=10000
            )
            
            # Agregar información específica de la estrategia
            results['strategy_type'] = 'FuturesGrid'
            results['market_type'] = 'futures'
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en simulación de Grid Trading para futuros: {e}")
            return {
                'error': str(e),
                'strategy_type': 'FuturesGrid',
                'market_type': 'futures',
                'Return [%]': 0.0,
                'Liquidation Count': 999,
                'Was Liquidated': True,
                'performance_score': 0.0
            }
    
    def run_strategy_backtest(self, df: pd.DataFrame, 
                            strategy_name: str, 
                            params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta backtesting para una estrategia específica de futuros.
        
        Args:
            df: DataFrame con datos OHLC
            strategy_name: Nombre de la estrategia
            params: Parámetros de la estrategia
            
        Returns:
            Resultados del backtesting
        """
        try:
            strategy_map = {
                'futures_grid': self.run_futures_grid_simulation,
                'FuturesGrid': self.run_futures_grid_simulation,
                'FUTURES_GRID': self.run_futures_grid_simulation
            }
            
            if strategy_name.lower() not in [k.lower() for k in strategy_map.keys()]:
                raise ValueError(f"Estrategia de futuros no soportada: {strategy_name}")
            
            # Encontrar la función correcta
            strategy_func = None
            for key, func in strategy_map.items():
                if key.lower() == strategy_name.lower():
                    strategy_func = func
                    break
            
            if strategy_func is None:
                raise ValueError(f"No se encontró la función para estrategia: {strategy_name}")
            
            return strategy_func(df, params)
            
        except Exception as e:
            logger.error(f"❌ Error ejecutando estrategia de futuros {strategy_name}: {e}")
            return {
                'error': str(e),
                'strategy': strategy_name,
                'market_type': 'futures',
                'Return [%]': 0.0,
                'Liquidation Count': 999,
                'Was Liquidated': True,
                'performance_score': 0.0
            }
    
    def evaluate_no_liquidation_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evalúa el rendimiento con el objetivo de no liquidación.
        
        Args:
            results: Resultados del backtesting
            
        Returns:
            Resultados con métricas adicionales de evaluación
        """
        try:
            was_liquidated = results.get('Was Liquidated', False)
            total_return = results.get('Return [%]', 0.0)
            liquidation_count = results.get('Liquidation Count', 0)
            
            # Criterio principal: NO liquidación
            if was_liquidated:
                results['meets_criteria'] = False
                results['criteria_score'] = 0.0
                results['recommendation'] = 'RECHAZAR - Fue liquidado'
            else:
                results['meets_criteria'] = True
                # Score basado en rendimiento sin liquidación
                criteria_score = min(100, max(0, 50 + total_return * 2))
                results['criteria_score'] = criteria_score
                
                if total_return > 20:
                    results['recommendation'] = 'EXCELENTE - Alto rendimiento sin liquidación'
                elif total_return > 10:
                    results['recommendation'] = 'BUENO - Rendimiento positivo sin liquidación'
                elif total_return > 0:
                    results['recommendation'] = 'ACEPTABLE - Rendimiento moderado sin liquidación'
                else:
                    results['recommendation'] = 'REVISAR - Sin liquidación pero rendimiento negativo'
            
            # Métricas adicionales
            results['liquidation_risk_score'] = liquidation_count * 10  # Penalización por liquidaciones
            results['safety_score'] = 100 - results['liquidation_risk_score']
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error evaluando rendimiento sin liquidación: {e}")
            results['meets_criteria'] = False
            results['criteria_score'] = 0.0
            results['recommendation'] = f'ERROR - {str(e)}'
            return results 