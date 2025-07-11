#!/usr/bin/env python3
# type: ignore
"""
Data Collector para Backtesting
===============================

Script para recolectar y preparar el dataset maestro para backtesting.
Integra datos de precios OHLCV de Binance con datos de sentimiento de BigQuery.
"""

import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import time
import pandas as pd
import ccxt
from dotenv import load_dotenv

# Agregar el directorio padre al path para importar módulos del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config.settings import settings

# Cargar variables de entorno
load_dotenv()

# Configuración de BigQuery
PROJECT_ID: str = settings.GOOGLE_CLOUD_PROJECT_ID
DATASET: str = 'oraculo_data'
TABLE: str = 'noticias_historicas'

# Configuración de Binance usando settings
BINANCE_CONFIG: Dict[str, Any] = {
    'apiKey': settings.BINANCE_API_KEY,
    'secret': settings.BINANCE_API_SECRET,
    'sandbox': False,  # Cambiar a True para testing
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot'
    }
}

# Límites de rate limiting
RATE_LIMIT_DELAY: float = 0.1  # 100ms entre requests
MAX_RETRIES: int = 3
RETRY_DELAY: int = 5


def fetch_historical_klines(
    pair: str, 
    timeframe: str, 
    start_date: datetime, 
    end_date: datetime,
    limit: int = 1000
) -> pd.DataFrame:
    """
    Descarga datos históricos OHLCV de Binance usando ccxt.
    
    Args:
        pair: Par de trading (ej: 'ETH/USDT')
        timeframe: Intervalo de tiempo (ej: '1d', '4h', '1h')
        start_date: Fecha de inicio
        end_date: Fecha de fin
        limit: Número máximo de velas por request
        
    Returns:
        DataFrame con datos OHLCV limpios
        
    Raises:
        Exception: Si hay error en la descarga
    """
    print(f"📊 Descargando datos OHLCV para {pair} ({timeframe}) desde {start_date} hasta {end_date}")
    
    try:
        # Inicializar exchange
        exchange = ccxt.binance(BINANCE_CONFIG)  # type: ignore
        
        # Convertir fechas a timestamps
        since = exchange.parse8601(start_date.isoformat())
        end_timestamp = exchange.parse8601(end_date.isoformat())
        
        all_klines: List[List[Any]] = []
        current_since = since
        
        while current_since < end_timestamp:
            try:
                print(f"   Descargando desde {datetime.fromtimestamp(current_since/1000)}")
                
                # Obtener datos
                klines = exchange.fetch_ohlcv(
                    symbol=pair,
                    timeframe=timeframe,
                    since=current_since,
                    limit=limit
                )
                
                if not klines:
                    print("   No hay más datos disponibles")
                    break
                
                all_klines.extend(klines)
                
                # Actualizar timestamp para siguiente request
                current_since = klines[-1][0] + 1  # type: ignore
                
                # Rate limiting
                time.sleep(RATE_LIMIT_DELAY)
                
            except ccxt.RateLimitExceeded:
                print(f"   Rate limit alcanzado, esperando {RETRY_DELAY} segundos...")
                time.sleep(RETRY_DELAY)
                continue
                
            except Exception as e:
                print(f"   Error en request: {e}")
                time.sleep(RETRY_DELAY)
                continue
        
        if not all_klines:
            raise Exception("No se pudieron obtener datos de Binance")
        
        # Convertir a DataFrame
        df = pd.DataFrame(
            all_klines,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convertir timestamp a datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('datetime').drop('timestamp', axis=1)
        
        # Filtrar por rango de fechas
        df = df[(df.index >= start_date) & (df.index <= end_date)]  # type: ignore
        
        # Convertir tipos de datos
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"✅ Datos OHLCV descargados: {len(df)} registros")
        print(f"📅 Período final: {df.index.min()} - {df.index.max()}")
        return df
        
    except Exception as e:
        print(f"❌ Error descargando datos OHLCV: {e}")
        raise


def fetch_sentiment_data(
    start_date: datetime, 
    end_date: datetime
) -> pd.DataFrame:
    """
    Descarga datos de sentimiento desde BigQuery.
    
    Args:
        start_date: Fecha de inicio
        end_date: Fecha de fin
        
    Returns:
        DataFrame con datos de sentimiento
        
    Raises:
        Exception: Si hay error en la descarga
    """
    print(f"📰 Descargando datos de sentimiento desde {start_date} hasta {end_date}")
    
    try:
        # Importar pandas_gbq aquí para evitar problemas de importación
        from pandas_gbq import read_gbq
        
        query = f"""
        SELECT 
            published_at,
            sentiment_score,
            primary_emotion,
            news_category,
            source,
            headline
        FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
        WHERE DATE(published_at) BETWEEN '{start_date.date()}' AND '{end_date.date()}'
        ORDER BY published_at
        """
        
        df: pd.DataFrame = read_gbq(query, project_id=PROJECT_ID)  # type: ignore
        
        if df.empty:
            print("⚠️ No se encontraron datos de sentimiento para el período especificado")
            # Crear DataFrame vacío con las columnas correctas
            df = pd.DataFrame(columns=[
                'published_at', 'sentiment_score', 'primary_emotion', 
                'news_category', 'source', 'headline'
            ])  # type: ignore
        else:
            # Convertir published_at a datetime
            df['published_at'] = pd.to_datetime(df['published_at'])
            df = df.set_index('published_at')
            
            # Convertir sentiment_score a numérico
            df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
            
            print(f"✅ Datos de sentimiento descargados: {len(df)} registros")
        
        return df  # type: ignore
        
    except ImportError:
        print("❌ Error: pandas-gbq no está instalado. Ejecuta: pip install pandas-gbq google-cloud-bigquery")
        raise
    except Exception as e:
        print(f"❌ Error descargando datos de sentimiento: {e}")
        raise


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula indicadores técnicos básicos para el DataFrame de precios.
    
    Args:
        df: DataFrame con datos OHLCV
        
    Returns:
        DataFrame con indicadores técnicos agregados
    """
    print("📈 Calculando indicadores técnicos...")
    
    # Copiar DataFrame para no modificar el original
    df_indicators = df.copy()
    
    # RSI (Relative Strength Index)
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df_indicators['rsi'] = calculate_rsi(df_indicators['close'])
    
    # Bollinger Bands
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df_indicators['close'])
    df_indicators['bb_upper'] = bb_upper
    df_indicators['bb_middle'] = bb_middle
    df_indicators['bb_lower'] = bb_lower
    df_indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle  # Ancho de Bollinger normalizado
    
    # ADX (Average Directional Index) - Versión simplificada
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        # Cálculo simplificado del ADX
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Dirección del movimiento
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    df_indicators['adx'] = calculate_adx(df_indicators['high'], df_indicators['low'], df_indicators['close'])
    
    # Volatilidad (desviación estándar de los retornos)
    returns = df_indicators['close'].pct_change()
    df_indicators['volatility'] = returns.rolling(window=20).std()
    
    print("✅ Indicadores técnicos calculados")
    return df_indicators


def fetch_and_prepare_data(
    pair: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Función principal que recolecta y prepara el dataset maestro para backtesting.
    
    Args:
        pair: Par de trading (ej: 'ETH/USDT')
        timeframe: Intervalo de tiempo (ej: '1d', '4h', '1h')
        start_date: Fecha de inicio
        end_date: Fecha de fin
        
    Returns:
        DataFrame maestro con datos de precios, indicadores técnicos y sentimiento
        
    Raises:
        Exception: Si hay error en la preparación de datos
    """
    print(f"🚀 Iniciando recolección de datos para {pair}")
    print(f"📅 Período: {start_date} - {end_date}")
    print(f"⏱️ Timeframe: {timeframe}")
    
    try:
        # 1. Obtener datos de precios OHLCV
        df_prices: pd.DataFrame = fetch_historical_klines(pair, timeframe, start_date, end_date)
        
        # 2. Calcular indicadores técnicos
        df_with_indicators: pd.DataFrame = calculate_technical_indicators(df_prices)
        
        # 3. Obtener datos de sentimiento
        df_sentiment: pd.DataFrame = fetch_sentiment_data(start_date, end_date)
        
        # 4. Fusión de datos usando merge_asof
        print("🔗 Fusionando datos de precios y sentimiento...")
        
        if not df_sentiment.empty:
            # Normalizar tipos de datos para evitar conflictos de zona horaria
            df_prices_normalized = df_with_indicators.reset_index()
            df_sentiment_normalized = df_sentiment.reset_index()
            
            # Convertir datetime a timezone-naive (sin zona horaria) para ambos DataFrames
            if df_prices_normalized['datetime'].dt.tz is not None:
                df_prices_normalized['datetime'] = df_prices_normalized['datetime'].dt.tz_localize(None)
            
            if df_sentiment_normalized['published_at'].dt.tz is not None:
                df_sentiment_normalized['published_at'] = df_sentiment_normalized['published_at'].dt.tz_localize(None)
            
            # Usar merge_asof para unir por tiempo más cercano
            df_master: pd.DataFrame = pd.merge_asof(
                df_prices_normalized,
                df_sentiment_normalized,
                left_on='datetime',
                right_on='published_at',
                direction='backward',  # Buscar el evento más reciente hacia atrás
                tolerance=pd.Timedelta('1D')  # Tolerancia de 1 día
            )
            
            # Limpiar valores faltantes en columnas de sentimiento
            sentiment_columns: List[str] = ['sentiment_score', 'primary_emotion', 'news_category', 'source', 'headline']
            for col in sentiment_columns:
                if col in df_master.columns:
                    df_master[col] = df_master[col].fillna(method='ffill')
                    # Para las primeras filas que no tienen valores previos
                    df_master[col] = df_master[col].fillna(method='bfill')
        else:
            # Si no hay datos de sentimiento, usar solo los datos de precios
            print("⚠️ No hay datos de sentimiento, usando solo datos de precios")
            df_master = df_with_indicators.reset_index()
            
            # Agregar columnas de sentimiento vacías
            df_master['sentiment_score'] = 0.0
            df_master['primary_emotion'] = 'Neutral'
            df_master['news_category'] = 'Mercado/Trading'
            df_master['source'] = 'none'
            df_master['headline'] = ''
        
        # 5. Limpieza final
        print("🧹 Limpiando datos...")
        
        # Eliminar filas con valores NaN en columnas críticas
        critical_columns: List[str] = ['open', 'high', 'low', 'close', 'volume']
        df_master = df_master.dropna(subset=critical_columns)
        
        # Rellenar valores NaN en indicadores técnicos
        indicator_columns: List[str] = ['rsi', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'adx', 'volatility']
        for col in indicator_columns:
            if col in df_master.columns:
                df_master[col] = df_master[col].fillna(method='ffill').fillna(method='bfill')
        
        # Calcular media móvil de sentimiento de 7 días
        if 'sentiment_score' in df_master.columns:
            df_master['sentiment_ma7'] = df_master['sentiment_score'].rolling(window=7).mean()
            # Rellenar valores NaN de la media móvil
            df_master['sentiment_ma7'] = df_master['sentiment_ma7'].fillna(method='ffill').fillna(method='bfill')
        else:
            # Si no hay datos de sentimiento, crear columna con valores neutros
            df_master['sentiment_ma7'] = 0.0
        
        # Establecer datetime como índice
        df_master = df_master.set_index('datetime')
        
        # Ordenar por fecha
        df_master = df_master.sort_index()
        
        print(f"✅ Dataset maestro preparado: {len(df_master)} registros")
        print(f"📊 Columnas disponibles: {list(df_master.columns)}")
        
        return df_master
        
    except Exception as e:
        print(f"❌ Error preparando dataset maestro: {e}")
        raise


def fetch_and_prepare_data_optimized(
    pair: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    sentiment_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Función optimizada que prepara el dataset maestro reutilizando datos de sentimiento.
    
    Args:
        pair: Par de trading (ej: 'ETH/USDT')
        timeframe: Intervalo de tiempo (ej: '1d', '4h', '1h')
        start_date: Fecha de inicio
        end_date: Fecha de fin
        sentiment_data: DataFrame de sentimientos precargado (opcional)
        
    Returns:
        DataFrame maestro con datos de precios, indicadores técnicos y sentimiento
        
    Raises:
        Exception: Si hay error en la preparación de datos
    """
    print(f"🚀 Preparando datos para {pair} (optimizado)")
    
    try:
        # 1. Obtener datos de precios OHLCV
        df_prices: pd.DataFrame = fetch_historical_klines(pair, timeframe, start_date, end_date)
        
        # 2. Calcular indicadores técnicos
        df_with_indicators: pd.DataFrame = calculate_technical_indicators(df_prices)
        
        # 3. Usar datos de sentimiento precargados si están disponibles
        df_sentiment = sentiment_data if sentiment_data is not None else fetch_sentiment_data(start_date, end_date)
        
        # 4. Fusión de datos usando merge_asof
        print("🔗 Fusionando datos de precios y sentimiento...")
        
        if not df_sentiment.empty:
            # Normalizar tipos de datos para evitar conflictos de zona horaria
            df_prices_normalized = df_with_indicators.reset_index()
            df_sentiment_normalized = df_sentiment.reset_index()
            
            # Convertir datetime a timezone-naive (sin zona horaria) para ambos DataFrames
            if df_prices_normalized['datetime'].dt.tz is not None:
                df_prices_normalized['datetime'] = df_prices_normalized['datetime'].dt.tz_localize(None)
            
            if df_sentiment_normalized['published_at'].dt.tz is not None:
                df_sentiment_normalized['published_at'] = df_sentiment_normalized['published_at'].dt.tz_localize(None)
            
            # Usar merge_asof para unir por tiempo más cercano
            df_master: pd.DataFrame = pd.merge_asof(
                df_prices_normalized,
                df_sentiment_normalized,
                left_on='datetime',
                right_on='published_at',
                direction='backward',  # Buscar el evento más reciente hacia atrás
                tolerance=pd.Timedelta('1D')  # Tolerancia de 1 día
            )
            
            # Limpiar valores faltantes en columnas de sentimiento
            sentiment_columns: List[str] = ['sentiment_score', 'primary_emotion', 'news_category', 'source', 'headline']
            for col in sentiment_columns:
                if col in df_master.columns:
                    df_master[col] = df_master[col].fillna(method='ffill')
                    # Para las primeras filas que no tienen valores previos
                    df_master[col] = df_master[col].fillna(method='bfill')
        else:
            # Si no hay datos de sentimiento, usar solo los datos de precios
            print("⚠️ No hay datos de sentimiento, usando solo datos de precios")
            df_master = df_with_indicators.reset_index()
            
            # Agregar columnas de sentimiento vacías
            df_master['sentiment_score'] = 0.0
            df_master['primary_emotion'] = 'Neutral'
            df_master['news_category'] = 'Mercado/Trading'
            df_master['source'] = 'none'
            df_master['headline'] = ''
        
        # 5. Limpieza final
        print("🧹 Limpiando datos...")
        
        # Eliminar filas con valores NaN en columnas críticas
        critical_columns: List[str] = ['open', 'high', 'low', 'close', 'volume']
        df_master = df_master.dropna(subset=critical_columns)
        
        # Rellenar valores NaN en indicadores técnicos
        indicator_columns: List[str] = ['rsi', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'adx', 'volatility']
        for col in indicator_columns:
            if col in df_master.columns:
                df_master[col] = df_master[col].fillna(method='ffill').fillna(method='bfill')
        
        # Calcular media móvil de sentimiento de 7 días
        if 'sentiment_score' in df_master.columns:
            df_master['sentiment_ma7'] = df_master['sentiment_score'].rolling(window=7).mean()
            # Rellenar valores NaN de la media móvil
            df_master['sentiment_ma7'] = df_master['sentiment_ma7'].fillna(method='ffill').fillna(method='bfill')
        else:
            # Si no hay datos de sentimiento, crear columna con valores neutros
            df_master['sentiment_ma7'] = 0.0
        
        # Establecer datetime como índice
        df_master = df_master.set_index('datetime')
        
        # Ordenar por fecha
        df_master = df_master.sort_index()
        
        print(f"✅ Dataset maestro preparado: {len(df_master)} registros")
        
        return df_master
        
    except Exception as e:
        print(f"❌ Error preparando dataset maestro: {e}")
        raise


def save_dataset(df: pd.DataFrame, filename: Optional[str] = None) -> str:
    """
    Guarda el dataset maestro en formato CSV.
    
    Args:
        df: DataFrame a guardar
        filename: Nombre del archivo (opcional)
        
    Returns:
        Ruta del archivo guardado
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dataset_maestro_{timestamp}.csv"
    
    filepath = os.path.join(os.getcwd(), filename)
    df.to_csv(filepath)
    print(f"💾 Dataset guardado en: {filepath}")
    return filepath


def main() -> None:
    """
    Función principal para testing del módulo.
    """
    # Configuración de ejemplo
    pair = "ETH/USDT"
    timeframe = "1d"
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    try:
        # Obtener dataset maestro
        df_master: pd.DataFrame = fetch_and_prepare_data(pair, timeframe, start_date, end_date)
        
        # Guardar dataset
        save_dataset(df_master)
        
        # Mostrar resumen
        print("\n📋 RESUMEN DEL DATASET:")
        print(f"Registros: {len(df_master)}")
        print(f"Período: {df_master.index.min()} - {df_master.index.max()}")
        print(f"Columnas: {list(df_master.columns)}")
        
        # Mostrar primeras filas
        print("\n🔍 PRIMERAS FILAS:")
        print(df_master.head())
        
    except Exception as e:
        print(f"❌ Error en ejecución: {e}")


if __name__ == "__main__":
    main() 