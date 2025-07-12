"""
Bayesian Optimizer Service - Servicio de Optimización Bayesiana
===============================================================

Caso de uso para la optimización inteligente de estrategias de trading.
El "Master Chef" del sistema que encuentra la receta perfecta para cada moneda.

Usa optimización bayesiana con Optuna para encontrar los parámetros óptimos
en solo 100-200 iteraciones inteligentes vs 1,500 tradicionales.

## Arquitectura
Este servicio NO instancia las estrategias directamente. En su lugar:
- Usa BacktestingService como capa de abstracción
- BacktestingService se encarga de instanciar las clases de estrategia
- Esto mantiene la separación de responsabilidades y facilita el testing
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import logging
import time

from app.domain.entities import OptimizationResult

# Nota: Las estrategias se usan a través del BacktestingService como capa de abstracción
# No necesitamos importar las clases de estrategia directamente aquí
# ya que el BacktestingService se encarga de instanciarlas y ejecutarlas

logger = logging.getLogger(__name__)

# Suprimir logs de Optuna para reducir ruido
optuna.logging.set_verbosity(optuna.logging.WARNING)


class BayesianOptimizerService:
    """
    Servicio de optimización bayesiana que encuentra la configuración óptima para cada estrategia.
    
    Usa Optuna con Tree-structured Parzen Estimator (TPE) para búsqueda inteligente
    de hiperparámetros en lugar de búsqueda exhaustiva.
    """
    
    def __init__(self, 
                 backtesting_service,
                 data_service,
                 optimization_window_months: int = 9):
        """
        Inicializa el servicio de optimización bayesiana.
        
        Args:
            backtesting_service: Servicio de backtesting para ejecutar pruebas
            data_service: Servicio de datos para cargar información histórica
            optimization_window_months: Ventana de datos históricos para optimización (meses)
        """
        self.backtesting_service = backtesting_service
        self.data_service = data_service
        self.optimization_window_months = optimization_window_months
        
        # Configuración de Optuna
        self.sampler = TPESampler(seed=42, n_startup_trials=20)
        self.pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=30)
        
        logger.info("🧠 BayesianOptimizerService inicializado")
        logger.info(f"📅 Ventana de optimización: {optimization_window_months} meses")
        logger.info(f"🔬 Sampler: TPE, Pruner: Median")
    
    def optimize_symbol(self, 
                       symbol: str, 
                       strategy: str = 'grid',
                       n_trials: int = 50,
                       timeout_minutes: int = 30) -> Optional[OptimizationResult]:
        """
        Optimiza una estrategia específica para un símbolo.
        
        Args:
            symbol: Símbolo de la criptomoneda (ej: 'BTC/USDT')
            strategy: Estrategia a optimizar ('grid', 'dca', 'btd')
            n_trials: Número máximo de trials
            timeout_minutes: Timeout en minutos
            
        Returns:
            OptimizationResult con el mejor resultado encontrado
        """
        logger.info(f"🎯 Optimizando {strategy.upper()} para {symbol}")
        start_time = time.time()
        
        try:
            # 1. Cargar datos de optimización
            df = self._load_optimization_data(symbol)
            if df is None or len(df) < 50:
                logger.warning(f"⚠️ Datos insuficientes para {symbol}")
                return None
            
            # 2. Crear estudio de Optuna
            study_name = f"{symbol}_{strategy}_{int(time.time())}"
            study = optuna.create_study(
                direction='maximize',
                sampler=self.sampler,
                pruner=self.pruner,
                study_name=study_name
            )
            
            # 3. Seleccionar función objetivo según estrategia
            objective_func = self._get_objective_function(strategy, df, symbol)
            
            # 4. Ejecutar optimización
            study.optimize(
                objective_func,
                n_trials=n_trials,
                timeout=timeout_minutes * 60,
                catch=(Exception,)
            )
            
            # 5. Crear resultado
            optimization_time = time.time() - start_time
            
            if study.best_trial is None:
                logger.error(f"❌ No se encontraron trials válidos para {symbol}")
                return None
            
            # Extraer estadísticas del mejor trial
            best_trial = study.best_trial
            study_stats = {
                'roi': best_trial.user_attrs.get('roi', 0.0),
                'max_drawdown': best_trial.user_attrs.get('max_drawdown', 0.0),
                'sharpe_ratio': best_trial.user_attrs.get('sharpe_ratio', 0.0),
                'total_trades': best_trial.user_attrs.get('total_trades', 0),
                'win_rate': best_trial.user_attrs.get('win_rate', 0.0),
                'calmar_ratio': best_trial.user_attrs.get('calmar_ratio', 0.0),
                'trials_total': len(study.trials),
                'trials_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            }
            
            result = OptimizationResult(
                symbol=symbol,
                strategy=strategy,
                best_params=best_trial.params,
                best_value=best_trial.value or 0.0,
                optimization_time=optimization_time,
                trials_completed=len(study.trials),
                study_stats=study_stats
            )
            
            logger.info(f"✅ Optimización completada para {symbol}")
            logger.info(f"   🎯 Mejor valor: {result.best_value:.2f}")
            logger.info(f"   📊 ROI: {study_stats['roi']:.2f}%")
            logger.info(f"   ⏱️  Tiempo: {optimization_time:.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error optimizando {symbol}: {e}")
            return None
    
    def optimize_portfolio(self, 
                          symbols: List[str],
                          strategies: Optional[List[str]] = None,
                          n_trials_per_symbol: int = 150) -> Dict[str, List[OptimizationResult]]:
        """
        Optimiza un portafolio completo de símbolos.
        
        Args:
            symbols: Lista de símbolos a optimizar
            strategies: Lista de estrategias (default: todas)
            n_trials_per_symbol: Trials por símbolo
            
        Returns:
            Diccionario con resultados por símbolo
        """
        if strategies is None:
            strategies = ['grid', 'dca', 'btd']
        
        logger.info(f"🚀 Iniciando optimización de portafolio")
        logger.info(f"📊 Símbolos: {len(symbols)}")
        logger.info(f"🎯 Estrategias: {strategies}")
        logger.info(f"🔬 Trials por símbolo: {n_trials_per_symbol}")
        
        results = {}
        total_symbols = len(symbols)
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"\n🔍 Optimizando {symbol} ({i}/{total_symbols})")
            
            symbol_results = []
            for strategy in strategies:
                result = self.optimize_symbol(
                    symbol=symbol,
                    strategy=strategy,
                    n_trials=n_trials_per_symbol,
                    timeout_minutes=30
                )
                
                if result:
                    symbol_results.append(result)
            
            results[symbol] = symbol_results
            
            # Log progreso
            completed_symbols = i
            progress = (completed_symbols / total_symbols) * 100
            logger.info(f"📈 Progreso general: {progress:.1f}% ({completed_symbols}/{total_symbols})")
        
        logger.info(f"\n🎉 Optimización de portafolio completada")
        self._log_portfolio_summary(results)
        
        return results
    
    def get_best_configuration(self, optimization_results: Dict[str, List[OptimizationResult]]) -> Dict[str, Dict[str, Any]]:
        """
        Extrae la mejor configuración para cada símbolo.
        
        Args:
            optimization_results: Resultados de optimización por símbolo
            
        Returns:
            Diccionario con mejores configuraciones por símbolo
        """
        best_configs = {}
        
        for symbol, results in optimization_results.items():
            if not results:
                continue
            
            # Encontrar la mejor estrategia para este símbolo
            best_result = max(results, key=lambda r: r.best_value)
            
            best_configs[symbol] = {
                'strategy': best_result.strategy,
                'params': best_result.best_params,
                'metrics': {
                    'roi': best_result.get_roi(),
                    'sharpe_ratio': best_result.get_sharpe_ratio(),
                    'max_drawdown': best_result.get_max_drawdown(),
                    'win_rate': best_result.get_win_rate(),
                    'objective_value': best_result.best_value
                },
                'optimization_stats': {
                    'trials_completed': best_result.trials_completed,
                    'optimization_time': best_result.optimization_time,
                    'efficiency_score': best_result.get_efficiency_score()
                }
            }
        
        return best_configs
    
    def _load_optimization_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Carga datos históricos optimizados para la ventana de tiempo especificada.
        
        Args:
            symbol: Símbolo de la criptomoneda
            
        Returns:
            DataFrame con datos históricos o None si hay error
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.optimization_window_months * 30)
            
            logger.debug(f"📊 Cargando datos para {symbol}: {start_date.date()} a {end_date.date()}")
            
            # Cargar datos usando el servicio de datos
            df = self.data_service.fetch_and_prepare_data_optimized(
                pair=symbol,
                timeframe='1d',
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or len(df) < 50:
                logger.warning(f"⚠️ Datos insuficientes para {symbol}: {len(df) if df is not None else 0}")
                return None
            
            logger.debug(f"✅ Datos cargados para {symbol}: {len(df)} registros")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error cargando datos para {symbol}: {e}")
            return None
    
    def _get_objective_function(self, strategy: str, df: pd.DataFrame, symbol: str) -> Callable:
        """
        Obtiene la función objetivo apropiada para la estrategia.
        
        Args:
            strategy: Estrategia ('grid', 'dca', 'btd')
            df: DataFrame con datos históricos
            symbol: Símbolo
            
        Returns:
            Función objetivo para Optuna
        """
        if strategy == 'grid':
            return lambda trial: self._grid_objective(trial, df, symbol)
        elif strategy == 'dca':
            return lambda trial: self._dca_objective(trial, df, symbol)
        elif strategy == 'btd':
            return lambda trial: self._btd_objective(trial, df, symbol)
        else:
            raise ValueError(f"Estrategia no soportada: {strategy}")
    
    def _grid_objective(self, trial: optuna.Trial, df: pd.DataFrame, symbol: str) -> float:
        """Función objetivo para optimización de Grid Trading."""
        try:
            # Espacios de búsqueda inteligentes para Grid Trading
            params = {
                'levels': trial.suggest_int('levels', 5, 180),  # Corregido: 5-180 como permite Bitsgap
                'range_percent': trial.suggest_float('range_percent', 1.0, 20.0),  # Más flexible: 1-20%
                'umbral_adx': trial.suggest_float('umbral_adx', 10.0, 50.0),  # Más flexible: 10-50
                'umbral_volatilidad': trial.suggest_float('umbral_volatilidad', 0.005, 0.08),  # Más flexible: 0.5-8%
                'umbral_sentimiento': trial.suggest_float('umbral_sentimiento', -0.5, 0.5),  # Más flexible: -0.5 a 0.5
            }
            
            # Ejecutar backtesting moderno
            config = {
                **params,
                'initial_capital': 10000.0,
                'commission': 0.001
            }
            results = self.backtesting_service.run_grid_simulation(df, config)
            
            # Verificar si hay error
            if 'error' in results:
                logger.warning(f"Error en trial Grid {trial.number}: {results['error']}")
                return -1000.0
            
            # Función objetivo compuesta
            roi = results.get('Return [%]', 0)
            max_drawdown = abs(results.get('Max. Drawdown [%]', 100))
            sharpe_ratio = results.get('Sharpe Ratio', 0)
            
            objective_value = roi - (max_drawdown * 0.5) + (sharpe_ratio * 10)
            
            # Informar métricas adicionales
            trial.set_user_attr('roi', roi)
            trial.set_user_attr('max_drawdown', max_drawdown)
            trial.set_user_attr('sharpe_ratio', sharpe_ratio)
            trial.set_user_attr('total_trades', results.get('# Trades', 0))
            trial.set_user_attr('win_rate', results.get('Win Rate [%]', 0))
            trial.set_user_attr('calmar_ratio', results.get('Calmar Ratio', 0))
            
            return objective_value
            
        except Exception as e:
            logger.warning(f"Error en trial Grid {trial.number}: {e}")
            return -1000.0
    
    def _dca_objective(self, trial: optuna.Trial, df: pd.DataFrame, symbol: str) -> float:
        """Función objetivo para optimización de DCA."""
        try:
            params = {
                # El % que debe caer el precio desde un máximo reciente para disparar una compra
                'dip_threshold': trial.suggest_float('dip_threshold', 0.02, 0.15), # Probar caídas del 2% al 15%
                
                # El % de ganancia sobre el precio promedio para vender toda la posición
                'take_profit': trial.suggest_float('take_profit', 0.05, 0.30), # Probar objetivos del 5% al 30%
                
                # El % de stop loss sobre el precio promedio para proteger el capital
                'stop_loss': trial.suggest_float('stop_loss', 0.10, 0.40), # Probar stops del 10% al 40%

                # Filtros de Sentimiento y Tendencia Macro (ADX se podría usar aquí también)
                'umbral_sentimiento': trial.suggest_float('umbral_sentimiento', -0.2, 0.2),
            }
            
            # Ejecutar backtesting DCA moderno
            config = {
                **params,
                'initial_capital': 10000.0,
                'commission': 0.001
            }
            results = self.backtesting_service.run_dca_simulation(df, config)
            
            # Verificar si hay error
            if 'error' in results:
                logger.warning(f"Error en trial DCA {trial.number}: {results['error']}")
                return -1000.0
            
            # Métricas y objetivo
            roi = results.get('Return [%]', 0)
            max_drawdown = abs(results.get('Max. Drawdown [%]', 100))
            sharpe_ratio = results.get('Sharpe Ratio', 0)
            
            objective_value = roi - (max_drawdown * 0.3) + (sharpe_ratio * 15)
            
            # Informar métricas
            trial.set_user_attr('roi', roi)
            trial.set_user_attr('max_drawdown', max_drawdown)
            trial.set_user_attr('sharpe_ratio', sharpe_ratio)
            trial.set_user_attr('total_trades', results.get('# Trades', 0))
            trial.set_user_attr('win_rate', results.get('Win Rate [%]', 0))
            trial.set_user_attr('calmar_ratio', results.get('Calmar Ratio', 0))
            
            return objective_value
            
        except Exception as e:
            logger.warning(f"Error en trial DCA {trial.number}: {e}")
            return -1000.0
    
    def _btd_objective(self, trial: optuna.Trial, df: pd.DataFrame, symbol: str) -> float:
        """Función objetivo para optimización de BTD."""
        try:
            params = {
                # El % que debe subir el precio desde un mínimo reciente para disparar una venta
                'rip_threshold': trial.suggest_float('rip_threshold', 0.02, 0.15), # Probar subidas del 2% al 15%
                
                # El % de ganancia (en la moneda base) para recomprar toda la posición
                'take_profit': trial.suggest_float('take_profit', 0.03, 0.25), # Probar objetivos del 3% al 25%
                
                # El % de stop loss si el precio sigue subiendo en nuestra contra
                'stop_loss': trial.suggest_float('stop_loss', 0.10, 0.35), # Probar stops del 10% al 35%

                # Filtros de Sentimiento y Tendencia Macro
                'umbral_sentimiento': trial.suggest_float('umbral_sentimiento', -0.2, 0.2),
            }
            
            # Ejecutar backtesting BTD moderno
            config = {
                **params,
                'initial_capital': 10000.0,
                'commission': 0.001
            }
            results = self.backtesting_service.run_btd_simulation(df, config)
            
            # Verificar si hay error
            if 'error' in results:
                logger.warning(f"Error en trial BTD {trial.number}: {results['error']}")
                return -1000.0
            
            # Métricas y objetivo
            roi = results.get('Return [%]', 0)
            max_drawdown = abs(results.get('Max. Drawdown [%]', 100))
            sharpe_ratio = results.get('Sharpe Ratio', 0)
            
            objective_value = roi - (max_drawdown * 0.4) + (sharpe_ratio * 12)
            
            # Informar métricas
            trial.set_user_attr('roi', roi)
            trial.set_user_attr('max_drawdown', max_drawdown)
            trial.set_user_attr('sharpe_ratio', sharpe_ratio)
            trial.set_user_attr('total_trades', results.get('# Trades', 0))
            trial.set_user_attr('win_rate', results.get('Win Rate [%]', 0))
            trial.set_user_attr('calmar_ratio', results.get('Calmar Ratio', 0))
            
            return objective_value
            
        except Exception as e:
            logger.warning(f"Error en trial BTD {trial.number}: {e}")
            return -1000.0
    
    def _log_portfolio_summary(self, results: Dict[str, List[OptimizationResult]]):
        """Log del resumen de optimización del portafolio."""
        logger.info("\n📊 RESUMEN DE OPTIMIZACIÓN DEL PORTAFOLIO:")
        logger.info("=" * 60)
        
        total_optimizations = sum(len(r) for r in results.values())
        successful_optimizations = sum(
            len([opt for opt in r if opt.is_successful()]) 
            for r in results.values()
        )
        
        logger.info(f"🎯 Total optimizaciones: {total_optimizations}")
        logger.info(f"✅ Exitosas: {successful_optimizations}")
        logger.info(f"📈 Tasa de éxito: {successful_optimizations/total_optimizations*100:.1f}%")
        
        # Top 5 mejores resultados
        all_results = [opt for r in results.values() for opt in r]
        top_results = sorted(all_results, key=lambda x: x.best_value, reverse=True)[:5]
        
        logger.info(f"\n🏆 TOP 5 MEJORES RESULTADOS:")
        for i, result in enumerate(top_results, 1):
            logger.info(f"{i}. {result.symbol} ({result.strategy.upper()}) - "
                       f"Valor: {result.best_value:.2f}, ROI: {result.get_roi():.2f}%") 