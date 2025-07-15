"""
OutlierDetectionService - Servicio de Detección y Manejo de Outliers
===================================================================

Detecta y maneja outliers de forma consistente usando métodos estadísticos estándar.
Permite configuración por tipo de activo y volatilidad.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

class OutlierMethod(Enum):
    """Métodos de detección de outliers."""
    ZSCORE = "zscore"
    IQR = "iqr"
    MODIFIED_ZSCORE = "modified_zscore"

class OutlierAction(Enum):
    """Acciones para manejar outliers."""
    REMOVE = "remove"
    REPLACE_MEDIAN = "replace_median"
    REPLACE_MEAN = "replace_mean"
    WINSORIZE = "winsorize"

class OutlierDetectionService:
    """
    Servicio para detectar y manejar outliers de forma consistente.
    
    Args:
        config (dict, opcional): Configuración personalizada de detección.
    
    Example:
        >>> detector = OutlierDetectionService()
        >>> outliers_mask = detector.detect_outliers(df['close'], method='zscore')
        >>> df_clean = detector.handle_outliers(df, outliers_mask, action='replace_median')
    """
    
    DEFAULT_CONFIG = {
        'zscore_threshold': 3.0,
        'iqr_multiplier': 1.5,
        'modified_zscore_threshold': 3.5,
        'min_data_points': 10,
        'volatility_adjustment': True
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self.DEFAULT_CONFIG.copy()
    
    def detect_outliers(self, 
                       data: pd.Series, 
                       method: str = 'zscore',
                       column_name: str = 'unknown') -> pd.Series:
        """
        Detecta outliers en una serie de datos.
        
        Args:
            data (pd.Series): Serie de datos a analizar.
            method (str): Método de detección ('zscore', 'iqr', 'modified_zscore').
            column_name (str): Nombre de la columna para logging.
            
        Returns:
            pd.Series: Máscara booleana donde True indica outliers.
        """
        if len(data) < self.config['min_data_points']:
            return pd.Series([False] * len(data), index=data.index)
        
        if method == OutlierMethod.ZSCORE.value:
            return self._detect_zscore_outliers(data, column_name)
        elif method == OutlierMethod.IQR.value:
            return self._detect_iqr_outliers(data, column_name)
        elif method == OutlierMethod.MODIFIED_ZSCORE.value:
            return self._detect_modified_zscore_outliers(data, column_name)
        else:
            raise ValueError(f"Método de detección no válido: {method}")
    
    def _detect_zscore_outliers(self, data: pd.Series, column_name: str) -> pd.Series:
        """Detecta outliers usando z-score."""
        z_scores = np.abs((data - data.mean()) / data.std())
        threshold = self.config['zscore_threshold']
        
        # Ajustar threshold por volatilidad si está habilitado
        if self.config['volatility_adjustment']:
            volatility = data.std() / data.mean()
            if volatility > 0.1:  # Alta volatilidad
                threshold *= 1.5
            elif volatility < 0.01:  # Baja volatilidad
                threshold *= 0.8
        
        outliers = z_scores > threshold
        return outliers
    
    def _detect_iqr_outliers(self, data: pd.Series, column_name: str) -> pd.Series:
        """Detecta outliers usando método IQR."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - (self.config['iqr_multiplier'] * IQR)
        upper_bound = Q3 + (self.config['iqr_multiplier'] * IQR)
        
        outliers = (data < lower_bound) | (data > upper_bound)
        return outliers
    
    def _detect_modified_zscore_outliers(self, data: pd.Series, column_name: str) -> pd.Series:
        """Detecta outliers usando z-score modificado (robusto a outliers)."""
        median = data.median()
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            return pd.Series([False] * len(data), index=data.index)
        
        modified_z_scores = 0.6745 * (data - median) / mad
        threshold = self.config['modified_zscore_threshold']
        
        outliers = np.abs(modified_z_scores) > threshold
        return pd.Series(outliers, index=data.index)
    
    def handle_outliers(self, 
                       df: pd.DataFrame, 
                       outliers_mask: pd.Series,
                       action: str = 'replace_median',
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Maneja outliers en un DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame con datos.
            outliers_mask (pd.Series): Máscara de outliers.
            action (str): Acción a realizar ('remove', 'replace_median', 'replace_mean', 'winsorize').
            columns (list, opcional): Columnas específicas a procesar.
            
        Returns:
            pd.DataFrame: DataFrame con outliers manejados.
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df.columns.tolist()
        
        if action == OutlierAction.REMOVE.value:
            return df_clean[~outliers_mask]
        
        elif action == OutlierAction.REPLACE_MEDIAN.value:
            for col in columns:
                if col in df_clean.columns:
                    median_val = df_clean[col].median()
                    df_clean.loc[outliers_mask, col] = median_val
        
        elif action == OutlierAction.REPLACE_MEAN.value:
            for col in columns:
                if col in df_clean.columns:
                    mean_val = df_clean[col].mean()
                    df_clean.loc[outliers_mask, col] = mean_val
        
        elif action == OutlierAction.WINSORIZE.value:
            for col in columns:
                if col in df_clean.columns:
                    df_clean[col] = self._winsorize_column(df_clean[col])
        
        return df_clean
    
    def _winsorize_column(self, series: pd.Series, limits: Tuple[float, float] = (0.05, 0.05)) -> pd.Series:
        """Aplica winsorización a una columna."""
        lower_percentile = series.quantile(limits[0])
        upper_percentile = series.quantile(1 - limits[1])
        
        series_winsorized = series.copy()
        series_winsorized[series < lower_percentile] = lower_percentile
        series_winsorized[series > upper_percentile] = upper_percentile
        
        return series_winsorized
    
    def get_outlier_statistics(self, 
                              df: pd.DataFrame, 
                              columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Obtiene estadísticas de outliers en un DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame a analizar.
            columns (list, opcional): Columnas específicas a analizar.
            
        Returns:
            dict: Estadísticas de outliers por columna.
        """
        if columns is None:
            columns = ['open', 'high', 'low', 'close']
        
        stats = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            data = df[col]
            outliers_zscore = self.detect_outliers(data, 'zscore', col)
            outliers_iqr = self.detect_outliers(data, 'iqr', col)
            
            stats[col] = {
                'total_points': len(data),
                'outliers_zscore': outliers_zscore.sum(),
                'outliers_iqr': outliers_iqr.sum(),
                'outliers_zscore_pct': (outliers_zscore.sum() / len(data)) * 100,
                'outliers_iqr_pct': (outliers_iqr.sum() / len(data)) * 100,
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max()
            }
        
        return stats

# Ejemplo de uso
if __name__ == "__main__":
    import pandas as pd
    
    # Crear datos de ejemplo con outliers
    np.random.seed(42)
    n_points = 100
    
    # Datos normales
    normal_data = np.random.normal(100, 10, n_points)
    
    # Agregar algunos outliers
    normal_data[0] = 200  # Outlier alto
    normal_data[1] = 50   # Outlier bajo
    
    df = pd.DataFrame({
        'close': normal_data,
        'volume': np.random.uniform(1000, 10000, n_points)
    })
    
    # Crear detector
    detector = OutlierDetectionService()
    
    # Detectar outliers
    outliers_mask = detector.detect_outliers(df['close'], 'zscore', 'close')
    print(f"Outliers detectados: {outliers_mask.sum()}")
    
    # Manejar outliers
    df_clean = detector.handle_outliers(df, outliers_mask, 'replace_median')
    print("DataFrame original:")
    print(df.head())
    print("\nDataFrame limpio:")
    print(df_clean.head())
    
    # Estadísticas
    stats = detector.get_outlier_statistics(df)
    print("\nEstadísticas de outliers:")
    for col, stat in stats.items():
        print(f"{col}: {stat['outliers_zscore']} outliers ({stat['outliers_zscore_pct']:.2f}%)") 