"""
Futures Optimization Service - Servicio de Optimización para Futuros
===================================================================

Servicio especializado para optimización de estrategias de futuros con:
- Objetivo: maximizar rendimiento SIN liquidaciones
- Optimización bayesiana con restricciones de liquidación
- Métricas específicas de futuros

Utiliza Optuna para optimización bayesiana con objetivo específico
de evitar liquidaciones mientras maximiza rendimiento.
"""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

from app.infrastructure.services.futures_backtesting_service import FuturesBacktestingService

logger = logging.getLogger(__name__)


class FuturesOptimizationService:
    """
    Servicio de optimización para estrategias de futuros.
    
    Optimiza parámetros de estrategias de futuros con el objetivo específico
    de maximizar rendimiento sin haber sido liquidado nunca.
    """
    
    def __init__(self):
        """Inicializa el servicio de optimización de futuros."""
        self.backtesting_service = FuturesBacktestingService()
        logger.info("🔧 Inicializando FuturesOptimizationService")
    
    def optimize_futures_grid_strategy(self, 
                                     df: pd.DataFrame,
                                     symbol: str,
                                     n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimiza parámetros de la estrategia FuturesGrid.
        
        Args:
            df: DataFrame con datos OHLC
            symbol: Símbolo de la criptomoneda
            n_trials: Número de trials para optimización
            
        Returns:
            Resultados de optimización con mejores parámetros
        """
        try:
            logger.info(f"🎯 Optimizando FuturesGrid para {symbol} ({n_trials} trials)")
            
            # Configurar estudio de optimización
            study = optuna.create_study(
                direction='maximize',
                study_name=f'futures_grid_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            
            # Función objetivo
            def objective(trial):
                return self._futures_grid_objective(trial, df, symbol)
            
            # Ejecutar optimización
            study.optimize(objective, n_trials=n_trials, timeout=300)  # 5 min timeout
            
            # Obtener mejores parámetros
            best_params = study.best_params
            best_value = study.best_value
            
            # Ejecutar backtesting final con mejores parámetros
            final_results = self.backtesting_service.run_futures_grid_simulation(df, best_params)
            
            # Evaluar con criterios de no liquidación
            final_results = self.backtesting_service.evaluate_no_liquidation_performance(final_results)
            
            logger.info(f"✅ Optimización completada para {symbol}:")
            logger.info(f"   🎯 Mejor score: {best_value:.2f}")
            logger.info(f"   💰 Retorno: {final_results.get('Return [%]', 0):.2f}%")
            logger.info(f"   🚨 Liquidado: {final_results.get('Was Liquidated', False)}")
            logger.info(f"   ⚡ Leverage: {best_params.get('leverage', 1)}x")
            
            return {
                'symbol': symbol,
                'strategy': 'FuturesGrid',
                'best_params': best_params,
                'best_score': best_value,
                'optimization_trials': n_trials,
                'final_results': final_results,
                'study_summary': {
                    'n_trials': len(study.trials),
                    'best_trial': study.best_trial.number,
                    'optimization_time': str(datetime.now())
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error optimizando FuturesGrid para {symbol}: {e}")
            return {
                'symbol': symbol,
                'strategy': 'FuturesGrid',
                'error': str(e),
                'best_params': {},
                'best_score': 0.0,
                'final_results': {
                    'Return [%]': 0.0,
                    'Was Liquidated': True,
                    'performance_score': 0.0
                }
            }
    
    def _futures_grid_objective(self, trial, df: pd.DataFrame, symbol: str) -> float:
        """
        Función objetivo para optimización de FuturesGrid.
        
        Args:
            trial: Trial de Optuna
            df: DataFrame con datos
            symbol: Símbolo
            
        Returns:
            Score de optimización (negativo si liquidado)
        """
        try:
            # Definir espacio de búsqueda
            params = {
                'levels': trial.suggest_int('levels', 3, 100),
                'range_percent': trial.suggest_float('range_percent', 3.0, 15.0),
                'leverage': trial.suggest_int('leverage', 2, 20),
                'umbral_adx': trial.suggest_float('umbral_adx', 15.0, 40.0),
                'umbral_volatilidad': trial.suggest_float('umbral_volatilidad', 0.01, 0.05),
                'umbral_sentimiento': trial.suggest_float('umbral_sentimiento', -0.3, 0.3),
                'maintenance_margin_rate': trial.suggest_float('maintenance_margin_rate', 0.005, 0.05)
            }
            
            # Ejecutar backtesting
            results = self.backtesting_service.run_futures_grid_simulation(df, params)
            
            # Objetivo específico para futuros: NO liquidación
            was_liquidated = results.get('Was Liquidated', False)
            total_return = results.get('Return [%]', 0.0)
            liquidation_count = results.get('Liquidation Count', 0)
            
            # Si fue liquidado, score muy negativo
            if was_liquidated:
                return -1000.0 - (liquidation_count * 100)
            
            # Si no fue liquidado, score basado en rendimiento
            # Usar rendimiento ajustado por liquidación como score principal
            score = results.get('liquidation_adjusted_return', total_return)
            
            # Bonus por alta eficiencia de apalancamiento
            leverage_efficiency = results.get('leverage_efficiency', 0.0)
            if leverage_efficiency > 2.0:  # Más de 2% por cada 1x de leverage
                score += 10
            
            # Bonus por buen win rate
            win_rate = results.get('Win Rate [%]', 0.0)
            if win_rate > 60:
                score += 5
            
            # Penalización por alto drawdown
            max_drawdown = abs(results.get('Max. Drawdown [%]', 0.0))
            if max_drawdown > 30:
                score -= 10
            
            return score
            
        except Exception as e:
            logger.error(f"Error en función objetivo: {e}")
            return -1000.0
    
    def optimize_multiple_symbols(self, 
                                data_dict: Dict[str, pd.DataFrame],
                                n_trials_per_symbol: int = 50) -> List[Dict[str, Any]]:
        """
        Optimiza múltiples símbolos en paralelo.
        
        Args:
            data_dict: Diccionario con datos por símbolo
            n_trials_per_symbol: Trials por símbolo
            
        Returns:
            Lista de resultados de optimización
        """
        try:
            logger.info(f"🎯 Optimizando {len(data_dict)} símbolos para futuros")
            
            results = []
            
            for symbol, df in data_dict.items():
                try:
                    logger.info(f"🔄 Optimizando {symbol}...")
                    
                    # Optimizar símbolo individual
                    symbol_result = self.optimize_futures_grid_strategy(
                        df=df,
                        symbol=symbol,
                        n_trials=n_trials_per_symbol
                    )
                    
                    results.append(symbol_result)
                    
                    # Log resultado
                    final_results = symbol_result.get('final_results', {})
                    was_liquidated = final_results.get('Was Liquidated', True)
                    total_return = final_results.get('Return [%]', 0.0)
                    
                    logger.info(f"   ✅ {symbol}: {total_return:.2f}%, "
                               f"Liquidado: {was_liquidated}")
                    
                except Exception as e:
                    logger.error(f"❌ Error optimizando {symbol}: {e}")
                    results.append({
                        'symbol': symbol,
                        'strategy': 'FuturesGrid',
                        'error': str(e),
                        'best_params': {},
                        'best_score': 0.0,
                        'final_results': {
                            'Return [%]': 0.0,
                            'Was Liquidated': True,
                            'performance_score': 0.0
                        }
                    })
            
            # Filtrar y ordenar resultados
            valid_results = [r for r in results if 'error' not in r]
            
            # Ordenar por rendimiento sin liquidación
            valid_results.sort(
                key=lambda x: (
                    not x['final_results'].get('Was Liquidated', True),  # No liquidados primero
                    x['final_results'].get('liquidation_adjusted_return', 0.0)  # Luego por rendimiento
                ),
                reverse=True
            )
            
            logger.info(f"🏆 Optimización múltiple completada:")
            logger.info(f"   📊 Símbolos procesados: {len(results)}")
            logger.info(f"   ✅ Válidos: {len(valid_results)}")
            logger.info(f"   🚫 Sin liquidar: {len([r for r in valid_results if not r['final_results'].get('Was Liquidated', True)])}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en optimización múltiple: {e}")
            return []
    
    def get_optimization_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Genera resumen de optimización.
        
        Args:
            results: Lista de resultados de optimización
            
        Returns:
            Resumen con estadísticas
        """
        try:
            if not results:
                return {'error': 'No hay resultados para resumir'}
            
            valid_results = [r for r in results if 'error' not in r]
            
            # Estadísticas básicas
            total_symbols = len(results)
            valid_symbols = len(valid_results)
            
            if not valid_results:
                return {
                    'total_symbols': total_symbols,
                    'valid_symbols': 0,
                    'success_rate': 0.0,
                    'no_liquidation_count': 0,
                    'avg_return': 0.0,
                    'best_performers': []
                }
            
            # Extraer métricas
            returns = [r['final_results'].get('Return [%]', 0.0) for r in valid_results]
            liquidations = [r['final_results'].get('Was Liquidated', True) for r in valid_results]
            
            # Contar no liquidados
            no_liquidation_count = sum(1 for liq in liquidations if not liq)
            
            # Mejores performers (no liquidados con mejor rendimiento)
            best_performers = [
                r for r in valid_results 
                if not r['final_results'].get('Was Liquidated', True)
            ][:5]  # Top 5
            
            summary = {
                'total_symbols': total_symbols,
                'valid_symbols': valid_symbols,
                'success_rate': (valid_symbols / total_symbols) * 100,
                'no_liquidation_count': no_liquidation_count,
                'no_liquidation_rate': (no_liquidation_count / valid_symbols) * 100,
                'avg_return': np.mean(returns),
                'max_return': max(returns) if returns else 0.0,
                'min_return': min(returns) if returns else 0.0,
                'best_performers': [
                    {
                        'symbol': r['symbol'],
                        'return': r['final_results'].get('Return [%]', 0.0),
                        'leverage': r['best_params'].get('leverage', 1),
                        'score': r['final_results'].get('performance_score', 0.0)
                    }
                    for r in best_performers
                ],
                'optimization_timestamp': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generando resumen: {e}")
            return {'error': str(e)}
    
    def filter_profitable_no_liquidation(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filtra resultados que sean rentables y sin liquidación.
        
        Args:
            results: Lista de resultados de optimización
            
        Returns:
            Lista filtrada de resultados
        """
        try:
            filtered = []
            
            for result in results:
                if 'error' in result:
                    continue
                
                final_results = result.get('final_results', {})
                was_liquidated = final_results.get('Was Liquidated', True)
                total_return = final_results.get('Return [%]', 0.0)
                
                # Filtrar: no liquidado Y rentable
                if not was_liquidated and total_return > 0:
                    filtered.append(result)
            
            # Ordenar por rendimiento descendente
            filtered.sort(
                key=lambda x: x['final_results'].get('Return [%]', 0.0),
                reverse=True
            )
            
            logger.info(f"🎯 Filtrado: {len(filtered)} resultados rentables sin liquidación")
            
            return filtered
            
        except Exception as e:
            logger.error(f"❌ Error filtrando resultados: {e}")
            return [] 