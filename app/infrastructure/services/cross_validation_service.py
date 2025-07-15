"""
CrossValidationService - Validación Cruzada Temporal y Out-of-Sample
===================================================================

Permite evaluar la robustez de estrategias usando validación cruzada temporal (TimeSeriesSplit)
y validación out-of-sample (holdout), reportando métricas agregadas y dispersión.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Type
from datetime import datetime
import logging
from app.infrastructure.services.backtesting_service import run_modern_backtest

logger = logging.getLogger("cross_validation")

class CrossValidationService:
    """
    Servicio para validación cruzada temporal y robustez out-of-sample.
    """
    def __init__(self, n_splits: int = 5, test_size: float = 0.2, random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def time_series_split(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Realiza splits tipo TimeSeriesSplit (sin shuffling).
        Returns lista de (train_idx, test_idx) para cada split.
        """
        n_samples = len(df)
        test_len = int(n_samples * self.test_size)
        train_len = n_samples - test_len
        splits = []
        step = int((train_len) / self.n_splits)
        for i in range(self.n_splits):
            train_end = step * (i + 1)
            if train_end + test_len > n_samples:
                break
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, train_end + test_len)
            splits.append((train_idx, test_idx))
        return splits

    def cross_validate(self,
                      df: pd.DataFrame,
                      strategy_class: Type,
                      strategy_params: Dict[str, Any],
                      commission: float = 0.001,
                      cash: float = 10000.0) -> Dict[str, Any]:
        """
        Ejecuta validación cruzada temporal y reporta métricas agregadas.
        """
        splits = self.time_series_split(df)
        results = []
        for i, (train_idx, test_idx) in enumerate(splits):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            logger.info(f"🧪 Split {i+1}/{len(splits)}: Train={len(train_df)}, Test={len(test_df)}")
            # Entrenamiento (opcional, aquí solo para logging)
            # Backtest en test set
            test_result = run_modern_backtest(
                df=test_df,
                strategy_class=strategy_class,
                strategy_params=strategy_params,
                commission=commission,
                cash=cash
            )
            test_result['split'] = i+1
            test_result['train_range'] = (train_df.index[0], train_df.index[-1])
            test_result['test_range'] = (test_df.index[0], test_df.index[-1])
            results.append(test_result)
        # Agregar métricas agregadas
        agg = self.aggregate_metrics(results)
        return {
            'splits': results,
            'aggregate': agg
        }

    def aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcula métricas agregadas (media, std, min, max) para los splits.
        """
        metrics = ['Return [%]', 'Max. Drawdown [%]', 'Win Rate [%]', 'Sharpe Ratio', 'performance_score']
        agg = {}
        for m in metrics:
            values = [r.get(m, np.nan) for r in results if m in r]
            values = [v for v in values if not (v is None or np.isnan(v))]
            if values:
                agg[m] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        return agg

    def holdout_validation(self,
                          df: pd.DataFrame,
                          strategy_class: Type,
                          strategy_params: Dict[str, Any],
                          holdout_size: float = 0.2,
                          commission: float = 0.001,
                          cash: float = 10000.0) -> Dict[str, Any]:
        """
        Ejecuta validación out-of-sample (holdout final).
        """
        n = len(df)
        split = int(n * (1 - holdout_size))
        train_df = df.iloc[:split].copy()
        test_df = df.iloc[split:].copy()
        logger.info(f"🧪 Holdout: Train={len(train_df)}, Test={len(test_df)}")
        # Backtest en test set
        test_result = run_modern_backtest(
            df=test_df,
            strategy_class=strategy_class,
            strategy_params=strategy_params,
            commission=commission,
            cash=cash
        )
        test_result['holdout'] = True
        test_result['train_range'] = (train_df.index[0], train_df.index[-1])
        test_result['test_range'] = (test_df.index[0], test_df.index[-1])
        return test_result 