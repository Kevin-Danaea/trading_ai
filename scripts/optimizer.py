#!/usr/bin/env python3
"""
Bayesian Optimizer - Optimizador Inteligente con Optuna
======================================================

El "Master Chef" del sistema que encuentra la receta perfecta para cada moneda.
Usa optimización bayesiana con Optuna para encontrar los parámetros óptimos
de las estrategias GRID, DCA y BTD en solo 100-200 iteraciones inteligentes.

Características:
- Optimización bayesiana con Optuna (vs fuerza bruta)
- Espacios de búsqueda inteligentes por estrategia
- Integración con backtesting_engine.py existente
- Carga optimizada de datos históricos (6-12 meses)
- Múltiples objetivos: ROI, Sharpe Ratio, Max Drawdown
"""

import os
import sys
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import logging
import time
from dataclasses import dataclass

# Agregar el directorio padre al path para importar módulos del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.modern_strategies import (
    GridTradingStrategy, DCAStrategy, BTDStrategy, run_modern_backtest
)
from scripts.data_collector import fetch_and_prepare_data_optimized
from shared.config.settings import settings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suprimir logs de Optuna para reducir ruido
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptimizationResult:
    """Resultado de la optimización para una moneda específica."""
    symbol: str
    strategy: str
    best_params: Dict[str, Any]
    best_value: float
    optimization_time: float
    trials_completed: int
    study_stats: Dict[str, Any]


class BayesianOptimizer:
    """
    Optimizador bayesiano que encuentra la configuración óptima para cada estrategia.
    
    Usa Optuna con Tree-structured Parzen Estimator (TPE) para búsqueda inteligente
    de hiperparámetros en lugar de búsqueda exhaustiva.
    """
    
    def __init__(self, optimization_window_months: int = 9):
        """
        Inicializa el optimizador bayesiano.
        
        Args:
            optimization_window_months: Ventana de datos históricos para optimización (meses)
        """
        self.optimization_window_months = optimization_window_months
        
        # Configuración de Optuna
        self.sampler = TPESampler(seed=42, n_startup_trials=20)
        self.pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=30)
        
        logger.info("🧠 BayesianOptimizer inicializado")
        logger.info(f"📅 Ventana de optimización: {optimization_window_months} meses")
        logger.info(f"🔬 Sampler: TPE, Pruner: Median")
    
    def load_optimization_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Carga datos históricos optimizados para la ventana de tiempo especificada.
        
        Args:
            symbol: Símbolo de la criptomoneda (ej: 'BTC/USDT')
            
        Returns:
            DataFrame con datos históricos o None si hay error
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.optimization_window_months * 30)
            
            logger.info(f"📊 Cargando datos para {symbol}: {start_date.date()} a {end_date.date()}")
            
            # Cargar datos OHLCV + indicadores técnicos + sentimiento
            df = fetch_and_prepare_data_optimized(
                pair=symbol,
                timeframe='1d',
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or len(df) < 50:  # Mínimo 50 días para optimización válida
                logger.warning(f"⚠️ Datos insuficientes para {symbol}: {len(df) if df is not None else 0} registros")
                return None
            
            logger.info(f"✅ Datos cargados para {symbol}: {len(df)} registros")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error cargando datos para {symbol}: {e}")
            return None
    
    def _grid_objective(self, trial: optuna.Trial, df: pd.DataFrame, symbol: str) -> float:
        """
        Función objetivo para optimización de Grid Trading usando backtesting.py.
        
        Args:
            trial: Trial de Optuna para sugerir parámetros
            df: DataFrame con datos históricos
            symbol: Símbolo de la moneda
            
        Returns:
            Valor objetivo a maximizar (ROI ajustado)
        """
        try:
            # Espacios de búsqueda inteligentes para Grid Trading
            params = {
                'levels': trial.suggest_int('levels', 3, 8),  # 3-8 niveles
                'range_percent': trial.suggest_float('range_percent', 2.0, 15.0),  # 2%-15% rango
                
                # Filtros de indicadores técnicos
                'umbral_adx': trial.suggest_float('umbral_adx', 15.0, 40.0),
                'umbral_volatilidad': trial.suggest_float('umbral_volatilidad', 0.01, 0.05),
                'umbral_sentimiento': trial.suggest_float('umbral_sentimiento', -0.3, 0.3),
            }
            
            # Ejecutar backtesting moderno
            results = run_modern_backtest(
                df=df,
                strategy_class=GridTradingStrategy,
                strategy_params=params,
                commission=0.001,
                cash=10000
            )
            
            # Función objetivo compuesta usando métricas modernas
            roi = results.get('Return [%]', 0)
            max_drawdown = abs(results.get('Max. Drawdown [%]', 100))
            sharpe_ratio = results.get('Sharpe Ratio', 0)
            
            # Penalizar drawdowns altos y premiar Sharpe alto
            objective_value = roi - (max_drawdown * 0.5) + (sharpe_ratio * 10)
            
            # Informar métricas adicionales para análisis
            trial.set_user_attr('roi', roi)
            trial.set_user_attr('max_drawdown', max_drawdown)
            trial.set_user_attr('sharpe_ratio', sharpe_ratio)
            trial.set_user_attr('total_trades', results.get('# Trades', 0))
            trial.set_user_attr('win_rate', results.get('Win Rate [%]', 0))
            trial.set_user_attr('calmar_ratio', results.get('Calmar Ratio', 0))
            
            return objective_value
            
        except Exception as e:
            logger.warning(f"Error en trial Grid {trial.number}: {e}")
            return -1000.0  # Penalizar errores
    
    def _dca_objective(self, trial: optuna.Trial, df: pd.DataFrame, symbol: str) -> float:
        """
        Función objetivo para optimización de DCA usando backtesting.py.
        
        Args:
            trial: Trial de Optuna
            df: DataFrame con datos históricos
            symbol: Símbolo de la moneda
            
        Returns:
            Valor objetivo a maximizar
        """
        try:
            # Espacios de búsqueda para DCA modernos
            params = {
                'intervalo_compra': trial.suggest_int('intervalo_compra', 1, 7),  # 1-7 días
                'monto_compra': trial.suggest_float('monto_compra', 0.1, 1.0),  # 10%-100% del capital
                'objetivo_ganancia': trial.suggest_float('objetivo_ganancia', 0.05, 0.30),  # 5%-30%
                'dip_threshold': trial.suggest_float('dip_threshold', 0.02, 0.15),  # 2%-15% dip
                'tendencia_alcista_dias': trial.suggest_int('tendencia_alcista_dias', 3, 14),  # 3-14 días
                'stop_loss': trial.suggest_float('stop_loss', 0.10, 0.40),  # 10%-40%
            }
            
            # Ejecutar backtesting moderno
            results = run_modern_backtest(
                df=df,
                strategy_class=DCAStrategy,
                strategy_params=params,
                commission=0.001,
                cash=10000
            )
            
            # Función objetivo
            roi = results.get('Return [%]', 0)
            max_drawdown = abs(results.get('Max. Drawdown [%]', 100))
            sharpe_ratio = results.get('Sharpe Ratio', 0)
            
            objective_value = roi - (max_drawdown * 0.3) + (sharpe_ratio * 15)
            
            trial.set_user_attr('roi', roi)
            trial.set_user_attr('max_drawdown', max_drawdown)
            trial.set_user_attr('sharpe_ratio', sharpe_ratio)
            trial.set_user_attr('total_trades', results.get('# Trades', 0))
            trial.set_user_attr('win_rate', results.get('Win Rate [%]', 0))
            
            return objective_value
            
        except Exception as e:
            logger.warning(f"Error en trial DCA {trial.number}: {e}")
            return -1000.0
    
    def _btd_objective(self, trial: optuna.Trial, df: pd.DataFrame, symbol: str) -> float:
        """
        Función objetivo para optimización de BTD usando backtesting.py.
        
        Args:
            trial: Trial de Optuna
            df: DataFrame con datos históricos
            symbol: Símbolo de la moneda
            
        Returns:
            Valor objetivo a maximizar
        """
        try:
            # Espacios de búsqueda para BTD modernos
            params = {
                'intervalo_venta': trial.suggest_int('intervalo_venta', 1, 7),  # 1-7 días
                'monto_venta': trial.suggest_float('monto_venta', 0.1, 1.0),  # 10%-100% del capital
                'objetivo_ganancia': trial.suggest_float('objetivo_ganancia', 0.05, 0.25),  # 5%-25%
                'rip_threshold': trial.suggest_float('rip_threshold', 0.02, 0.12),  # 2%-12% rip
                'tendencia_bajista_dias': trial.suggest_int('tendencia_bajista_dias', 3, 14),  # 3-14 días
                'stop_loss': trial.suggest_float('stop_loss', 0.10, 0.35),  # 10%-35%
            }
            
            # Ejecutar backtesting moderno
            results = run_modern_backtest(
                df=df,
                strategy_class=BTDStrategy,
                strategy_params=params,
                commission=0.001,
                cash=10000
            )
            
            # Función objetivo
            roi = results.get('Return [%]', 0)
            max_drawdown = abs(results.get('Max. Drawdown [%]', 100))
            sharpe_ratio = results.get('Sharpe Ratio', 0)
            
            objective_value = roi - (max_drawdown * 0.4) + (sharpe_ratio * 12)
            
            trial.set_user_attr('roi', roi)
            trial.set_user_attr('max_drawdown', max_drawdown)
            trial.set_user_attr('sharpe_ratio', sharpe_ratio)
            trial.set_user_attr('total_trades', results.get('# Trades', 0))
            trial.set_user_attr('win_rate', results.get('Win Rate [%]', 0))
            
            return objective_value
            
        except Exception as e:
            logger.warning(f"Error en trial BTD {trial.number}: {e}")
            return -1000.0
    
    def optimize_symbol(self, 
                       symbol: str, 
                       strategy: str = 'grid',
                       n_trials: int = 150,
                       timeout_minutes: int = 30) -> Optional[OptimizationResult]:
        """
        Optimiza una moneda específica con una estrategia dada.
        
        Args:
            symbol: Símbolo de la moneda (ej: 'BTC/USDT')
            strategy: Estrategia a optimizar ('grid', 'dca', 'btd')
            n_trials: Número máximo de trials (iteraciones)
            timeout_minutes: Timeout en minutos
            
        Returns:
            Resultado de la optimización o None si hay error
        """
        start_time = time.time()
        
        logger.info(f"🔧 Optimizando {symbol} con estrategia {strategy.upper()}")
        logger.info(f"🎯 Max trials: {n_trials}, Timeout: {timeout_minutes}min")
        
        try:
            # Cargar datos
            df = self.load_optimization_data(symbol)
            if df is None:
                return None
            
            # Seleccionar función objetivo
            objective_functions = {
                'grid': self._grid_objective,
                'dca': self._dca_objective,
                'btd': self._btd_objective
            }
            
            if strategy not in objective_functions:
                logger.error(f"❌ Estrategia no soportada: {strategy}")
                return None
            
            objective_func = objective_functions[strategy]
            
            # Crear estudio de Optuna
            study_name = f"{symbol.replace('/', '_')}_{strategy}_{int(time.time())}"
            study = optuna.create_study(
                direction='maximize',
                sampler=self.sampler,
                pruner=self.pruner,
                study_name=study_name
            )
            
            # Función objetivo con datos pre-cargados
            def objective(trial):
                return objective_func(trial, df, symbol)
            
            # Ejecutar optimización
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout_minutes * 60,
                show_progress_bar=False
            )
            
            optimization_time = time.time() - start_time
            
            # Extraer resultados
            best_trial = study.best_trial
            
            result = OptimizationResult(
                symbol=symbol,
                strategy=strategy,
                best_params=best_trial.params,
                best_value=best_trial.value if best_trial.value is not None else -1000.0,
                optimization_time=optimization_time,
                trials_completed=len(study.trials),
                study_stats={
                    'best_roi': best_trial.user_attrs.get('roi', 0),
                    'best_max_drawdown': best_trial.user_attrs.get('max_drawdown', 0),
                    'best_sharpe_ratio': best_trial.user_attrs.get('sharpe_ratio', 0),
                    'total_trades': best_trial.user_attrs.get('total_trades', 0),
                    'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                    'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                    'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
                }
            )
            
            logger.info(f"✅ Optimización completada para {symbol}")
            logger.info(f"🏆 Mejor valor objetivo: {best_trial.value:.2f}")
            logger.info(f"📊 ROI: {result.study_stats['best_roi']:.2f}%")
            logger.info(f"📉 Max Drawdown: {result.study_stats['best_max_drawdown']:.2f}%")
            logger.info(f"📈 Sharpe Ratio: {result.study_stats['best_sharpe_ratio']:.3f}")
            logger.info(f"⏱️ Tiempo: {optimization_time:.1f}s, Trials: {result.trials_completed}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error optimizando {symbol}: {e}")
            return None
    
    def optimize_portfolio(self, 
                          symbols: List[str],
                          strategies: Optional[List[str]] = None,
                          n_trials_per_symbol: int = 150) -> Dict[str, List[OptimizationResult]]:
        """
        Optimiza un portafolio completo de monedas.
        
        Args:
            symbols: Lista de símbolos a optimizar
            strategies: Lista de estrategias por símbolo (o None para todas)
            n_trials_per_symbol: Trials por símbolo
            
        Returns:
            Diccionario con resultados de optimización por símbolo
        """
        if strategies is None:
            strategies = ['grid', 'dca', 'btd'] * len(symbols)
        
        results = {}
        total_symbols = len(symbols)
        
        logger.info(f"🚀 INICIANDO OPTIMIZACIÓN DE PORTAFOLIO")
        logger.info(f"📊 {total_symbols} símbolos, {n_trials_per_symbol} trials c/u")
        logger.info(f"⏱️ Tiempo estimado: {total_symbols * 5:.0f} minutos")
        logger.info("=" * 60)
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"🔧 [{i}/{total_symbols}] Optimizando {symbol}...")
            
            symbol_results = []
            
            # Optimizar cada estrategia para esta moneda
            for strategy in ['grid']:  # Empezar solo con Grid por eficiencia
                result = self.optimize_symbol(
                    symbol=symbol,
                    strategy=strategy,
                    n_trials=n_trials_per_symbol,
                    timeout_minutes=5  # 5 min por moneda
                )
                
                if result:
                    symbol_results.append(result)
            
            if symbol_results:
                results[symbol] = symbol_results
                logger.info(f"✅ {symbol} completado: {len(symbol_results)} estrategias optimizadas")
            else:
                logger.warning(f"⚠️ {symbol} sin resultados válidos")
            
            logger.info("")
        
        logger.info(f"🏆 OPTIMIZACIÓN DE PORTAFOLIO COMPLETADA")
        logger.info(f"✅ {len(results)} símbolos optimizados exitosamente")
        
        return results
    
    def get_best_configuration(self, optimization_results: Dict[str, List[OptimizationResult]]) -> Dict[str, Dict[str, Any]]:
        """
        Extrae la mejor configuración para cada símbolo.
        
        Args:
            optimization_results: Resultados de optimización del portafolio
            
        Returns:
            Diccionario con mejores configuraciones por símbolo
        """
        best_configs = {}
        
        for symbol, results in optimization_results.items():
            if not results:
                continue
            
            # Encontrar el mejor resultado (mayor valor objetivo)
            best_result = max(results, key=lambda x: x.best_value)
            
            best_configs[symbol] = {
                'strategy': best_result.strategy,
                'params': best_result.best_params,
                'metrics': {
                    'objective_value': best_result.best_value,
                    'roi': best_result.study_stats.get('best_roi', 0),
                    'max_drawdown': best_result.study_stats.get('best_max_drawdown', 0),
                    'sharpe_ratio': best_result.study_stats.get('best_sharpe_ratio', 0)
                },
                'optimization_stats': {
                    'trials_completed': best_result.trials_completed,
                    'optimization_time': best_result.optimization_time
                }
            }
        
        return best_configs


def main():
    """Función principal para demostración del optimizador."""
    print("🧠 BAYESIAN OPTIMIZER - Master Chef de Trading")
    print("=" * 60)
    
    # Ejemplo de optimización
    optimizer = BayesianOptimizer(optimization_window_months=6)
    
    # Símbolos de prueba
    test_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    # Optimizar portafolio
    results = optimizer.optimize_portfolio(
        symbols=test_symbols,
        n_trials_per_symbol=50  # Reducido para demo
    )
    
    # Mostrar mejores configuraciones
    best_configs = optimizer.get_best_configuration(results)
    
    print("\n🏆 MEJORES CONFIGURACIONES ENCONTRADAS:")
    print("=" * 60)
    
    for symbol, config in best_configs.items():
        print(f"\n📊 {symbol} - {config['strategy'].upper()}")
        print(f"   ROI: {config['metrics']['roi']:.2f}%")
        print(f"   Max Drawdown: {config['metrics']['max_drawdown']:.2f}%")
        print(f"   Sharpe Ratio: {config['metrics']['sharpe_ratio']:.3f}")
        print(f"   Trials: {config['optimization_stats']['trials_completed']}")
        print(f"   Tiempo: {config['optimization_stats']['optimization_time']:.1f}s")
    
    print("\n🎉 Optimización completada!")


if __name__ == "__main__":
    main() 