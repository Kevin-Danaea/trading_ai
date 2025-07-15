"""
Market Data Provider - Proveedor de Datos de Mercado
===================================================

Proveedor de infraestructura para obtener datos de mercado desde Binance.
Migrado desde scripts/data_collector.py con toda la funcionalidad original.
"""

import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import time
import pandas as pd
import ccxt
import numpy as np
import logging

from app.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)


class MarketDataProvider:
    """
    Proveedor de datos de mercado que maneja la conexión con Binance
    y proporciona datos OHLCV + indicadores técnicos.
    """
    
    def __init__(self):
        """Inicializa el proveedor con configuración de Binance."""
        # Configuración de Binance
        self.binance_config = {
            'apiKey': settings.BINANCE_API_KEY,
            'secret': settings.BINANCE_API_SECRET,
            'sandbox': False,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        }
        
        # Límites de rate limiting
        self.rate_limit_delay = 0.1  # 100ms entre requests
        self.max_retries = 3
        self.retry_delay = 5
        
        # Inicializar exchange
        self.exchange = ccxt.binance(self.binance_config)  # type: ignore[arg-type]
        
        logger.info("📊 MarketDataProvider inicializado con Binance")
    
    def get_top_symbols(self, top_n: int = 150, min_volume_usdt: float = 2_000_000) -> List[str]:
        """
        Obtiene la lista de Top N criptomonedas por market cap desde Binance.
        
        Args:
            top_n: Número de símbolos a retornar
            min_volume_usdt: Volumen mínimo en USDT
            
        Returns:
            Lista de símbolos en formato 'BTC/USDT'
        """
        logger.info(f"📈 Obteniendo Top {top_n} criptomonedas...")
        
        try:
            # Obtener todos los tickers de mercado
            tickers = self.exchange.fetch_tickers()
            
            # Filtrar solo pares con USDT
            usdt_pairs = {
                symbol: ticker for symbol, ticker in tickers.items() 
                if symbol.endswith('/USDT') and float(ticker['quoteVolume'] or 0) > min_volume_usdt
            }
            
            # Ordenar por volumen (proxy de market cap y liquidez)
            sorted_pairs = sorted(
                usdt_pairs.items(),
                key=lambda x: float(x[1]['quoteVolume'] or 0),
                reverse=True
            )
            
            # Tomar los primeros N
            top_symbols = [symbol for symbol, _ in sorted_pairs[:top_n]]
            
            logger.info(f"✅ Top {len(top_symbols)} monedas obtenidas")
            logger.info(f"🥇 Top 5: {top_symbols[:5]}")
            
            return top_symbols
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo top criptomonedas: {e}")
            # Fallback a lista predefinida
            return [
                'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'MATIC/USDT',
                'LINK/USDT', 'DOT/USDT', 'ADA/USDT', 'DOGE/USDT', 'XRP/USDT'
            ]
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene datos de mercado completos para un símbolo.
        
        Args:
            symbol: Símbolo a analizar (ej: 'BTC/USDT')
            
        Returns:
            Diccionario con datos de mercado o None si hay error
        """
        try:
            # Obtener ticker actual
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Obtener datos históricos para cálculos
            df_24h = self.fetch_historical_klines(symbol, '1h', 24)
            df_7d = self.fetch_historical_klines(symbol, '4h', 42)  # 7 días * 6 períodos de 4h
            
            if df_24h is None or df_7d is None or len(df_24h) < 20 or len(df_7d) < 30:
                logger.warning(f"Datos insuficientes para {symbol}")
                return None
            
            # Extraer precios para cálculos
            prices_24h = df_24h['close'].tolist()
            prices_7d = df_7d['close'].tolist()
            highs = df_7d['high'].tolist()
            lows = df_7d['low'].tolist()
            closes = df_7d['close'].tolist()
            
            market_data = {
                'current_price': float(ticker['last'] or 0),
                'volume_24h': float(ticker['quoteVolume'] or 0),
                'price_change_24h': float(ticker['percentage'] or 0),
                'price_change_7d': self._calculate_price_change_7d(prices_7d),
                'volume_change_24h': 0.0,  # Simplificado por ahora
                'prices_24h': prices_24h,
                'prices_7d': prices_7d,
                'highs': highs,
                'lows': lows,
                'closes': closes
            }
            
            return market_data
            
        except Exception as e:
            logger.warning(f"Error obteniendo datos de mercado para {symbol}: {e}")
            return None
    
    def fetch_historical_klines(self, 
                               pair: str, 
                               timeframe: str, 
                               limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Descarga datos históricos OHLCV de Binance.
        
        Args:
            pair: Par de trading (ej: 'ETH/USDT')
            timeframe: Intervalo de tiempo (ej: '1d', '4h', '1h')
            limit: Número de velas a obtener
            
        Returns:
            DataFrame con datos OHLCV o None si hay error
        """
        try:
            klines = self.exchange.fetch_ohlcv(
                symbol=pair,
                timeframe=timeframe,
                limit=limit
            )
            
            if not klines:
                return None
            
            # Convertir a DataFrame
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = pd.DataFrame(klines, columns=pd.Index(columns))
            
            # Convertir timestamp a datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('datetime').drop('timestamp', axis=1)
            
            # Convertir tipos de datos
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.warning(f"Error obteniendo datos históricos para {pair}: {e}")
            return None
    
    def fetch_and_prepare_data_optimized(self,
                                       pair: str,
                                       timeframe: str,
                                       start_date: datetime,
                                       end_date: datetime,
                                       sentiment_data: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """
        Versión optimizada que carga datos OHLCV + indicadores técnicos + sentimiento.
        Migrado desde scripts/data_collector.py
        
        Args:
            pair: Par de trading (ej: 'ETH/USDT')
            timeframe: Intervalo de tiempo
            start_date: Fecha de inicio
            end_date: Fecha de fin
            sentiment_data: Datos de sentimiento pre-cargados (opcional)
            
        Returns:
            DataFrame completo con OHLCV + indicadores + sentimiento
        """
        try:
            # Calcular límite basado en el rango de fechas
            days_range = (end_date - start_date).days
            if timeframe == '1d':
                limit = min(days_range + 10, 1000)
            elif timeframe == '4h':
                limit = min((days_range * 6) + 50, 1000)
            else:
                limit = min((days_range * 24) + 100, 1000)
            
            # Obtener datos OHLCV
            df = self.fetch_historical_klines(pair, timeframe, limit)
            if df is None:
                return None
            
            # Filtrar por rango de fechas
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            if len(df) < 30:  # Mínimo para cálculos
                logger.warning(f"Datos insuficientes para {pair}: {len(df)} registros")
                return None
            
            # Calcular indicadores técnicos
            assert isinstance(df, pd.DataFrame)
            df = self._calculate_technical_indicators(df)
            
            # Agregar datos de sentimiento si están disponibles
            if sentiment_data is not None and not sentiment_data.empty:
                df = self._merge_sentiment_data(df, sentiment_data)
            else:
                # Valores por defecto para sentimiento
                df['sentiment_score'] = 0.0
                df['sentiment_ma7'] = 0.0
                df['primary_emotion'] = 'neutral'
            
            logger.info(f"✅ Datos preparados para {pair}: {len(df)} registros con indicadores")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error preparando datos para {pair}: {e}")
            return None
    
    def _calculate_price_change_7d(self, prices_7d: List[float]) -> float:
        """Calcula el cambio de precio en 7 días."""
        if len(prices_7d) < 2:
            return 0.0
        
        start_price = prices_7d[0]
        end_price = prices_7d[-1]
        
        if start_price > 0:
            return ((end_price - start_price) / start_price) * 100
        return 0.0
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula indicadores técnicos: RSI, Bollinger Bands, ADX, Volatilidad.
        Migrado desde scripts/data_collector.py
        """
        try:
            # Type assertions para linter
            close_series = pd.Series(df['close'])
            high_series = pd.Series(df['high'])
            low_series = pd.Series(df['low'])
            
            # RSI
            df['rsi'] = self._calculate_rsi(close_series, period=14)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close_series)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            
            # ADX
            df['adx'] = self._calculate_adx(high_series, low_series, close_series)
            
            # Volatilidad (rolling 7 días)
            df['volatility'] = df['close'].pct_change().rolling(window=7).std()
            
            # Rellenar valores nulos
            df = df.bfill().fillna(0)
            
            return df
            
        except Exception as e:
            logger.warning(f"Error calculando indicadores técnicos: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula el RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi_result = 100 - (100 / (1 + rs))
        return pd.Series(rsi_result, index=prices.index)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                 std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcula las Bandas de Bollinger."""
        rolling_mean = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        upper = rolling_mean + (rolling_std * std_dev)
        lower = rolling_mean - (rolling_std * std_dev)
        return pd.Series(upper), pd.Series(rolling_mean), pd.Series(lower)
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                      period: int = 14) -> pd.Series:
        """Calcula el ADX (simplificado)."""
        try:
            # True Range
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement
            dm_plus = (high - high.shift()).where((high - high.shift()) > (low.shift() - low), 0)
            dm_minus = (low.shift() - low).where((low.shift() - low) > (high - high.shift()), 0)
            
            # Smoothed TR and DM
            tr_smooth = tr.rolling(window=period).mean()
            dm_plus_smooth = dm_plus.rolling(window=period).mean()
            dm_minus_smooth = dm_minus.rolling(window=period).mean()
            
            # Directional Indicators
            di_plus = (dm_plus_smooth / tr_smooth) * 100
            di_minus = (dm_minus_smooth / tr_smooth) * 100
            
            # ADX
            dx = ((di_plus - di_minus).abs() / (di_plus + di_minus)) * 100
            adx = dx.rolling(window=period).mean()
            
            return adx.fillna(50)  # Valor neutral para NaN
            
        except Exception as e:
            logger.warning(f"Error calculando ADX: {e}")
            return pd.Series([50] * len(high), index=high.index)
    
    def _merge_sentiment_data(self, df: pd.DataFrame, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Fusiona datos de sentimiento con datos de precios."""
        try:
            # Resample sentiment data to match price data frequency
            sentiment_resampled = sentiment_data.resample('D').agg({
                'sentiment_score': 'mean',
                'primary_emotion': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'neutral'
            })
            
            # Merge con datos de precios
            df = df.join(sentiment_resampled, how='left')
            
            # Forward fill para llenar vacíos
            df['sentiment_score'] = df['sentiment_score'].fillna(method='ffill').fillna(0.0)
            df['primary_emotion'] = df['primary_emotion'].fillna('neutral')
            
            # Calcular media móvil de 7 días del sentimiento
            df['sentiment_ma7'] = df['sentiment_score'].rolling(window=7).mean()
            
            return df
            
        except Exception as e:
            logger.warning(f"Error fusionando datos de sentimiento: {e}")
            # Agregar columnas por defecto
            df['sentiment_score'] = 0.0
            df['sentiment_ma7'] = 0.0
            df['primary_emotion'] = 'neutral'
            return df 

    def fetch_optimization_historical_data(self, symbol: str, months: int = 12) -> Optional[pd.DataFrame]:
        """
        Obtiene datos históricos para optimización de estrategias (spot y futuros).
        
        Args:
            symbol: Símbolo (ej: 'BTC/USDT')
            months: Número de meses de datos históricos (default: 12)
            
        Returns:
            DataFrame con datos OHLCV de los últimos N meses
        """
        try:
            from datetime import datetime, timedelta
            import time
            
            # Calcular fechas (12 meses atrás)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months * 30)  # Aproximadamente 12 meses
            
            logger.info(f"📊 Obteniendo datos históricos para optimización de {symbol}")
            logger.info(f"   📅 Período: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
            logger.info(f"   🎯 Objetivo: Optimizar para trading de hoy y próximos 7 días")
            
            # Usar timeframe de 1 día para 12 meses de datos
            timeframe = '1d'
            
            # Obtener datos históricos en chunks (Binance limita a 1000 por llamada)
            all_data = []
            current_end = end_date
            max_limit = 1000  # Límite máximo de Binance
            max_iterations = 50  # Máximo número de iteraciones para evitar bucles infinitos
            iteration_count = 0
            
            while current_end > start_date and iteration_count < max_iterations:
                iteration_count += 1
                try:
                    # Calcular cuántos días necesitamos en este chunk
                    days_needed = (current_end - start_date).days
                    chunk_limit = min(days_needed, max_limit)
                    
                    if chunk_limit <= 0:
                        break
                    
                    logger.info(f"   📥 Obteniendo chunk {iteration_count}: {chunk_limit} días hasta {current_end.strftime('%Y-%m-%d')}")
                    
                    # Obtener datos para este chunk con timeout
                    try:
                        import threading
                        import queue
                        
                        result_queue = queue.Queue()
                        
                        def fetch_data():
                            try:
                                klines_data = self.exchange.fetch_ohlcv(
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    limit=chunk_limit,
                                    since=int(current_end.timestamp() * 1000) - (chunk_limit * 24 * 60 * 60 * 1000)
                                )
                                result_queue.put(('success', klines_data))
                            except Exception as e:
                                result_queue.put(('error', e))
                        
                        # Ejecutar en thread separado con timeout
                        thread = threading.Thread(target=fetch_data)
                        thread.daemon = True
                        thread.start()
                        thread.join(timeout=30)  # 30 segundos timeout
                        
                        if thread.is_alive():
                            logger.warning(f"   ⚠️ Timeout en chunk hasta {current_end.strftime('%Y-%m-%d')}")
                            # Intentar con un chunk más pequeño
                            if chunk_limit > 100:
                                chunk_limit = chunk_limit // 2
                                # No hacer continue aquí, continuar con el procesamiento
                            else:
                                logger.error(f"   ❌ Chunk demasiado pequeño, saltando a siguiente fecha")
                                # Avanzar la fecha para evitar bucle infinito
                                current_end = current_end - timedelta(days=chunk_limit)
                                if current_end <= start_date:
                                    break
                                continue
                        
                        # Obtener resultado
                        if not result_queue.empty():
                            status, result = result_queue.get()
                            if status == 'success':
                                klines = result
                            else:
                                raise result
                        else:
                            raise Exception("No se obtuvo respuesta de la API")
                        
                    except Exception as e:
                        logger.warning(f"   ⚠️ Error en chunk hasta {current_end.strftime('%Y-%m-%d')}: {e}")
                        # Intentar con un chunk más pequeño
                        if chunk_limit > 100:
                            chunk_limit = chunk_limit // 2
                            # No hacer continue aquí, continuar con el procesamiento
                        else:
                            logger.error(f"   ❌ Chunk demasiado pequeño, saltando a siguiente fecha")
                            # Avanzar la fecha para evitar bucle infinito
                            current_end = current_end - timedelta(days=chunk_limit)
                            if current_end <= start_date:
                                break
                            continue
                    
                    if not klines:
                        logger.warning(f"   ⚠️ No se obtuvieron datos para chunk hasta {current_end.strftime('%Y-%m-%d')}")
                        break
                    
                    all_data.extend(klines)
                    
                    # Actualizar fecha para el siguiente chunk
                    if klines:
                        # El primer elemento es el más antiguo
                        oldest_timestamp = klines[0][0]
                        current_end = datetime.fromtimestamp(oldest_timestamp / 1000)
                    else:
                        break
                    
                    # Rate limiting
                    time.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.error(f"   ❌ Error obteniendo chunk: {e}")
                    break
            
            if not all_data:
                logger.error(f"❌ No se pudieron obtener datos históricos para {symbol}")
                return None
            
            # Convertir a DataFrame
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = pd.DataFrame(all_data, columns=pd.Index(columns))
            
            # Convertir timestamp a datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('datetime').drop('timestamp', axis=1)
            
            # Convertir tipos de datos
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Renombrar columnas para compatibilidad con backtesting
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Remover duplicados y ordenar por fecha
            df = df.drop_duplicates().sort_index()
            
            # Filtrar por rango de fechas exacto
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            if len(df) < 180:  # Mínimo 6 meses de datos
                logger.warning(f"⚠️ Datos insuficientes para {symbol}: {len(df)} registros")
                return None
            
            # Asegurar que df sea DataFrame
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            
            # Agregar columnas específicas para futuros (si es necesario)
            df = self._add_optimization_data(df, symbol)
            
            logger.info(f"✅ Datos históricos obtenidos: {len(df)} registros ({len(df)/30:.1f} meses)")
            
            # Verificar que el índice sea datetime antes de usar strftime
            try:
                min_date = df.index.min()
                max_date = df.index.max()
                if isinstance(min_date, pd.Timestamp) and isinstance(max_date, pd.Timestamp):
                    logger.info(f"   📅 Rango real: {min_date.strftime('%Y-%m-%d')} a {max_date.strftime('%Y-%m-%d')}")
                else:
                    logger.info(f"   📅 Rango: {min_date} a {max_date}")
            except Exception:
                logger.info(f"   📅 Rango: {len(df)} registros")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos históricos para optimización de {symbol}: {e}")
            return None
    
    def _add_optimization_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Agrega datos específicos para optimización (spot y futuros).
        
        Args:
            df: DataFrame con datos OHLCV
            symbol: Símbolo
            
        Returns:
            DataFrame con datos adicionales para optimización
        """
        try:
            # Simular funding rate histórico (solo para futuros, pero aplicable a ambos)
            # En la realidad esto vendría de la API, pero para backtesting simulamos
            np.random.seed(42)  # Para reproducibilidad
            funding_rates = np.random.normal(0.01, 0.15, len(df))  # Media 0.01%, std 0.15%
            funding_rates = np.clip(funding_rates, -0.5, 0.5)  # Limitar entre -0.5% y 0.5%
            
            df['funding_rate'] = funding_rates / 100  # Convertir a decimal
            
            # Simular open interest (para análisis adicional)
            base_oi = df['Volume'].rolling(window=30, min_periods=1).mean()
            df['open_interest'] = base_oi * np.random.uniform(0.8, 1.2, len(df))
            
            # Marcar cada día como momento de funding (simplificación)
            df['funding_time'] = True  # En timeframe 1d, cada día tiene funding
            
            # Agregar indicadores técnicos para optimización
            df = self._add_technical_indicators_for_optimization(df)
            
            logger.info(f"📈 Datos de optimización agregados:")
            logger.info(f"   💰 Funding rate promedio: {df['funding_rate'].mean()*100:.3f}%")
            logger.info(f"   📊 Open interest promedio: {df['open_interest'].mean():,.0f}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error agregando datos de optimización: {e}")
            return df
    
    def _add_technical_indicators_for_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega indicadores técnicos específicos para optimización.
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con indicadores técnicos
        """
        try:
            # Type assertions para linter
            close_series = pd.Series(df['Close'])
            high_series = pd.Series(df['High'])
            low_series = pd.Series(df['Low'])
            volume_series = pd.Series(df['Volume'])
            
            # RSI
            df['RSI'] = self._calculate_rsi(close_series, period=14)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close_series, period=20, std_dev=2)
            df['BB_Upper'] = bb_upper
            df['BB_Middle'] = bb_middle
            df['BB_Lower'] = bb_lower
            df['BB_Width'] = (bb_upper - bb_lower) / bb_middle  # Ancho de las bandas
            
            # ADX
            df['ADX'] = self._calculate_adx(high_series, low_series, close_series, period=14)
            
            # Volatilidad
            df['Volatility'] = close_series.pct_change().rolling(window=20).std()
            
            # Volumen promedio
            df['Volume_MA'] = volume_series.rolling(window=20).mean()
            df['Volume_Ratio'] = volume_series / df['Volume_MA']
            
            # Momentum
            df['Momentum'] = close_series / close_series.shift(10) - 1
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculando indicadores técnicos: {e}")
            return df 