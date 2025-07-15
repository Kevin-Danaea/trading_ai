"""
DataValidatorService - Servicio de Validación Robusta de Datos
============================================================

Valida DataFrames de datos de mercado para asegurar integridad antes de backtesting u optimización.
Permite configuración de thresholds y reglas por tipo de activo (spot/futuros).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from app.infrastructure.services.outlier_detection_service import OutlierDetectionService

class DataValidationError(Exception):
    """Excepción lanzada cuando la validación de datos falla."""
    pass

class DataValidatorService:
    """
    Servicio para validar la integridad y consistencia de datos OHLCV.

    Args:
        thresholds (dict, opcional): Diccionario de thresholds personalizados.

    Example:
        >>> validator = DataValidatorService()
        >>> df_validado = validator.validar_dataframe(df, tipo='spot')
    """
    
    DEFAULT_THRESHOLDS = {
        'min_rows': 50,
        'max_null_pct': 0.01,  # 1% nulos permitido
        'max_outlier_pct': 0.01,  # 1% outliers permitido
        'ohlc_columns': ['open', 'high', 'low', 'close'],
        'volume_column': 'volume',
    }
    
    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
        self.outlier_detector = OutlierDetectionService()
    
    def validar_dataframe(self, df: pd.DataFrame, tipo: str = 'spot') -> pd.DataFrame:
        """
        Valida un DataFrame de datos de mercado.

        Args:
            df (pd.DataFrame): DataFrame con columnas OHLCV.
            tipo (str): 'spot' o 'futuros'.
        Returns:
            pd.DataFrame: DataFrame limpio y validado.
        Raises:
            DataValidationError: Si la validación falla.
        """
        errores = []
        
        # 1. Checar filas suficientes
        if len(df) < self.thresholds['min_rows']:
            errores.append(f"Datos insuficientes: {len(df)} filas (mínimo {self.thresholds['min_rows']})")
        
        # 2. Detectar formato de columnas OHLC automáticamente
        ohlc_columns = self._detect_ohlc_columns(df)
        if not ohlc_columns:
            errores.append("No se encontraron columnas OHLC válidas")
            raise DataValidationError("\n".join(errores))
        
        # 3. Checar nulos
        null_pct = df.isnull().mean()
        if isinstance(null_pct, pd.Series):
            for col in ohlc_columns:
                if col in null_pct.index and null_pct[col] > self.thresholds['max_null_pct']:
                    errores.append(f"Columna {col} tiene {null_pct[col]*100:.2f}% nulos (máx {self.thresholds['max_null_pct']*100:.2f}%)")
        
        # 4. Checar precios negativos o cero
        for col in ohlc_columns:
            if (df[col] <= 0).any():
                errores.append(f"Columna {col} tiene valores <= 0")
        
        # 5. Consistencia OHLC usando las columnas detectadas
        high_col = ohlc_columns.get('high')
        low_col = ohlc_columns.get('low')
        open_col = ohlc_columns.get('open')
        close_col = ohlc_columns.get('close')
        
        if high_col and low_col and (df[high_col] < df[low_col]).any():
            errores.append("Hay filas donde high < low")
        if open_col and high_col and (df[open_col] > df[high_col]).any():
            errores.append("Hay filas donde open > high")
        if close_col and high_col and (df[close_col] > df[high_col]).any():
            errores.append("Hay filas donde close > high")
        if open_col and low_col and (df[open_col] < df[low_col]).any():
            errores.append("Hay filas donde open < low")
        if close_col and low_col and (df[close_col] < df[low_col]).any():
            errores.append("Hay filas donde close < low")
        
        # 6. Detectar outliers usando el servicio especializado
        outlier_stats = self.outlier_detector.get_outlier_statistics(df, list(ohlc_columns.values()))
        for col, stats in outlier_stats.items():
            if stats['outliers_zscore_pct'] > self.thresholds['max_outlier_pct'] * 100:
                errores.append(f"Demasiados outliers en {col}: {stats['outliers_zscore_pct']:.2f}% (máx {self.thresholds['max_outlier_pct']*100:.2f}%)")
        
        # 7. Validaciones específicas para futuros
        if tipo == 'futuros':
            if 'leverage' in df.columns and (df['leverage'] < 1).any():
                errores.append("Columna leverage tiene valores < 1 en futuros")
        
        if errores:
            raise DataValidationError("\n".join(errores))
        
        # Limpieza básica: rellenar nulos en volumen y sentimiento
        volume_col = self._detect_volume_column(df)
        if volume_col:
            df[volume_col] = df[volume_col].fillna(0)
        if 'sentiment_score' in df.columns:
            df['sentiment_score'] = df['sentiment_score'].fillna(0)
        
        return df.copy()
    
    def _detect_ohlc_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detecta automáticamente las columnas OHLC en cualquier formato.
        
        Args:
            df: DataFrame con datos
            
        Returns:
            Dict con mapeo de tipo de columna a nombre real
        """
        detected_columns = {}
        available_columns = [col.lower() for col in df.columns]
        
        # Buscar columnas OHLC en cualquier formato
        ohlc_patterns = {
            'open': ['open', 'o'],
            'high': ['high', 'h'],
            'low': ['low', 'l'],
            'close': ['close', 'c']
        }
        
        for ohlc_type, patterns in ohlc_patterns.items():
            for pattern in patterns:
                if pattern in available_columns:
                    # Encontrar el nombre real de la columna (preservar mayúsculas/minúsculas)
                    for real_col in df.columns:
                        if real_col.lower() == pattern:
                            detected_columns[ohlc_type] = real_col
                            break
                    break
        
        return detected_columns
    
    def _detect_volume_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detecta automáticamente la columna de volumen.
        
        Args:
            df: DataFrame con datos
            
        Returns:
            Nombre de la columna de volumen o None
        """
        available_columns = [col.lower() for col in df.columns]
        volume_patterns = ['volume', 'vol', 'v']
        
        for pattern in volume_patterns:
            if pattern in available_columns:
                for real_col in df.columns:
                    if real_col.lower() == pattern:
                        return real_col
        
        return None

# Ejemplo de uso
if __name__ == "__main__":
    import pandas as pd
    # Crear DataFrame de ejemplo
    data = {
        'open': [1, 2, 3, 4, 5],
        'high': [1.1, 2.2, 3.3, 4.4, 5.5],
        'low': [0.9, 1.8, 2.7, 3.6, 4.5],
        'close': [1, 2, 3, 4, 5],
        'volume': [100, 200, 300, 400, 500],
        'sentiment_score': [0, 0, 0, 0, 0]
    }
    df = pd.DataFrame(data)
    validator = DataValidatorService()
    try:
        df_valid = validator.validar_dataframe(df, tipo='spot')
        print("✅ DataFrame validado correctamente:")
        print(df_valid)
    except DataValidationError as e:
        print("❌ Error de validación:", e) 