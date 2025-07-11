#!/usr/bin/env python3
"""
Crypto Scanner - Escáner Inteligente de Oportunidades
====================================================

Sistema de scanner que analiza el Top 100 de criptomonedas para identificar 
los mejores candidatos para trading basándose en:
- Volatilidad
- ADX (mercados laterales)
- Sentimiento (BigQuery histórico + DATABASE_URL reciente)

El scanner actúa como "nuestros ojos" pre-filtrando oportunidades antes del backtesting.
"""

import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ccxt
import time
import logging
from dataclasses import dataclass
import json

# Agregar el directorio padre al path para importar módulos del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config.settings import settings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CryptoCandidate:
    """
    Clase de datos para representar un candidato cripto analizado.
    """
    symbol: str
    market_cap_rank: int
    current_price: float
    volatility_24h: float
    volatility_7d: float
    adx: float
    sentiment_score: float
    sentiment_ma7: float
    volume_24h: float
    volume_change_24h: float
    price_change_24h: float
    price_change_7d: float
    score: float  # Puntuación final calculada
    reasons: List[str]  # Razones por las que es buen candidato


class CryptoScanner:
    """
    Scanner inteligente de criptomonedas para identificar oportunidades de trading.
    
    Analiza el Top 100 de monedas usando:
    1. Datos de mercado en tiempo real (Binance)
    2. Cálculo de indicadores técnicos (ADX, volatilidad)
    3. Análisis de sentimiento (BigQuery histórico + DATABASE_URL reciente)
    4. Sistema de puntuación para rankear candidatos
    """
    
    def __init__(self):
        """
        Inicializa el scanner con configuración optimizada para producción 24/7.
        
        Configuración fija:
        - Analiza Top 150 monedas por volumen
        - Volumen mínimo: $2M USDT para mayor liquidez
        - Siempre devuelve exactamente 10 mejores candidatos
        """
        # Configuración optimizada para producción
        self.top_n = 150  # Analizar más monedas para mejor selección
        self.min_volume_usdt = 2_000_000  # $2M mínimo para liquidez garantizada
        self.target_candidates = 10  # Siempre devolver exactamente 10
        
        # Configurar exchange
        self.exchange = ccxt.binance({
            'apiKey': settings.BINANCE_API_KEY,
            'secret': settings.BINANCE_API_SECRET,
            'sandbox': False,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        # Cache para datos de mercado
        self.market_data_cache: Dict[str, Any] = {}
        self.sentiment_cache: Optional[pd.DataFrame] = None
        
        logger.info(f"🔍 CryptoScanner inicializado para producción 24/7")
        logger.info(f"📊 Analizará Top {self.top_n} monedas")
        logger.info(f"💰 Volumen mínimo: ${self.min_volume_usdt:,.0f} USDT")
        logger.info(f"🎯 Objetivo: {self.target_candidates} candidatos siempre")
    
    def get_top_cryptocurrencies(self) -> List[str]:
        """
        Obtiene la lista de Top N criptomonedas por market cap desde Binance.
        
        Returns:
            Lista de símbolos en formato 'BTC/USDT'
        """
        logger.info(f"📈 Obteniendo Top {self.top_n} criptomonedas...")
        
        try:
            # Obtener todos los tickers de mercado
            tickers = self.exchange.fetch_tickers()
            
            # Filtrar solo pares con USDT
            usdt_pairs = {
                symbol: ticker for symbol, ticker in tickers.items() 
                if symbol.endswith('/USDT') and float(ticker['quoteVolume'] or 0) > self.min_volume_usdt
            }
            
            # Ordenar por volumen (proxy de market cap y liquidez)
            sorted_pairs = sorted(
                usdt_pairs.items(),
                key=lambda x: float(x[1]['quoteVolume'] or 0),
                reverse=True
            )
            
            # Tomar los primeros N
            top_symbols = [symbol for symbol, _ in sorted_pairs[:self.top_n]]
            
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
    
    def calculate_volatility(self, prices: List[float]) -> float:
        """
        Calcula la volatilidad (desviación estándar de los retornos).
        
        Args:
            prices: Lista de precios
            
        Returns:
            Volatilidad como decimal (ej: 0.05 = 5%)
        """
        if len(prices) < 2:
            return 0.0
        
        # Calcular retornos logarítmicos
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                return_val = np.log(prices[i] / prices[i-1])
                returns.append(return_val)
        
        if not returns:
            return 0.0
        
        # Desviación estándar de los retornos
        volatility = np.std(returns)
        return float(volatility)
    
    def calculate_adx(self, highs: List[float], lows: List[float], 
                     closes: List[float], period: int = 14) -> float:
        """
        Calcula el ADX (Average Directional Index).
        
        Args:
            highs: Lista de precios máximos
            lows: Lista de precios mínimos  
            closes: Lista de precios de cierre
            period: Período para el cálculo (default 14)
            
        Returns:
            Valor ADX (0-100, donde <20 = mercado lateral)
        """
        if len(closes) < period + 1:
            return 50.0  # Valor neutral si no hay suficientes datos
        
        try:
            # Convertir a arrays de numpy
            high = np.array(highs)
            low = np.array(lows)
            close = np.array(closes)
            
            # Calcular True Range
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Calcular Directional Movement
            dm_plus = np.maximum(high[1:] - high[:-1], 0)
            dm_minus = np.maximum(low[:-1] - low[1:], 0)
            
            # Suavizar con media móvil
            atr = np.convolve(tr, np.ones(period)/period, mode='valid')
            dm_plus_smooth = np.convolve(dm_plus, np.ones(period)/period, mode='valid')
            dm_minus_smooth = np.convolve(dm_minus, np.ones(period)/period, mode='valid')
            
            # Calcular DI+ y DI-
            di_plus = 100 * dm_plus_smooth / atr
            di_minus = 100 * dm_minus_smooth / atr
            
            # Calcular DX
            dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
            
            # ADX es la media móvil del DX
            if len(dx) >= period:
                adx = np.mean(dx[-period:])
                return float(adx)
            else:
                return 50.0
                
        except Exception as e:
            logger.warning(f"Error calculando ADX: {e}")
            return 50.0  # Valor neutral en caso de error
    
    def get_historical_data(self, symbol: str, timeframe: str = '1d', 
                           limit: int = 30) -> Optional[pd.DataFrame]:
        """
        Obtiene datos históricos para un símbolo.
        
        Args:
            symbol: Símbolo (ej: 'BTC/USDT')
            timeframe: Marco temporal ('1d', '4h', '1h')
            limit: Número de velas a obtener
            
        Returns:
            DataFrame con datos OHLCV o None si hay error
        """
        try:
            # Rate limiting
            time.sleep(0.1)
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv)
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('datetime').drop('timestamp', axis=1)
            
            return df
            
        except Exception as e:
            logger.warning(f"Error obteniendo datos para {symbol}: {e}")
            return None
    
    def get_sentiment_data(self, days_back: int = 30) -> pd.DataFrame:
        """
        Obtiene datos de sentimiento desde BigQuery (histórico) y DATABASE_URL (reciente).
        
        Args:
            days_back: Días hacia atrás para buscar datos
            
        Returns:
            DataFrame con datos de sentimiento
        """
        if self.sentiment_cache is not None:
            return self.sentiment_cache
        
        logger.info("📰 Obteniendo datos de sentimiento...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        try:
            # Primero intentar BigQuery para datos históricos
            df_sentiment = self._fetch_bigquery_sentiment(start_date, end_date)
            
            # Luego intentar DATABASE_URL para datos más recientes
            df_recent = self._fetch_recent_sentiment(start_date, end_date)
            
            # Combinar ambos datasets
            if not df_sentiment.empty and not df_recent.empty:
                # Concatenar y eliminar duplicados
                df_combined = pd.concat([df_sentiment, df_recent])
                df_combined = df_combined.drop_duplicates(subset=['published_at'], keep='last')
                df_combined = df_combined.sort_values('published_at')
            elif not df_sentiment.empty:
                df_combined = df_sentiment
            elif not df_recent.empty:
                df_combined = df_recent
            else:
                # Sin datos de sentimiento
                df_combined = pd.DataFrame()
            
            self.sentiment_cache = df_combined
            
            if not df_combined.empty:
                logger.info(f"✅ Datos de sentimiento cargados: {len(df_combined)} registros")
            else:
                logger.warning("⚠️ No se encontraron datos de sentimiento")
            
            return df_combined
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo sentimiento: {e}")
            return pd.DataFrame()
    
    def _fetch_bigquery_sentiment(self, start_date: datetime, 
                                 end_date: datetime) -> pd.DataFrame:
        """Obtiene sentimiento desde BigQuery (datos históricos)."""
        try:
            from pandas_gbq import read_gbq
            
            project_id = settings.GOOGLE_CLOUD_PROJECT_ID
            if not project_id or project_id == 'tu-proyecto-id':
                return pd.DataFrame()
            
            query = f"""
            SELECT 
                published_at,
                sentiment_score,
                primary_emotion,
                news_category,
                source,
                headline
            FROM `{project_id}.oraculo_data.noticias_historicas`
            WHERE DATE(published_at) BETWEEN '{start_date.date()}' AND '{end_date.date()}'
            ORDER BY published_at DESC
            LIMIT 10000
            """
            
            df = read_gbq(query, project_id=project_id)
            
            if df is not None and not df.empty:
                df['published_at'] = pd.to_datetime(df['published_at'])
                df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
                logger.info(f"📊 BigQuery sentimiento: {len(df)} registros")
                return pd.DataFrame(df)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"⚠️ BigQuery no disponible: {e}")
            return pd.DataFrame()
    
    def _fetch_recent_sentiment(self, start_date: datetime, 
                               end_date: datetime) -> pd.DataFrame:
        """Obtiene sentimiento reciente desde DATABASE_URL."""
        try:
            import sqlalchemy
            
            database_url = settings.DATABASE_URL
            if not database_url:
                return pd.DataFrame()
            
            engine = sqlalchemy.create_engine(database_url)
            
            query = """
            SELECT 
                published_at,
                sentiment_score,
                primary_emotion,
                news_category,
                source,
                headline
            FROM noticias_historicas
            WHERE published_at >= %s AND published_at <= %s
            ORDER BY published_at DESC
            LIMIT 5000
            """
            
            df = pd.read_sql(query, engine, params=[start_date, end_date])
            
            if not df.empty:
                df['published_at'] = pd.to_datetime(df['published_at'])
                df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
                logger.info(f"📊 DATABASE_URL sentimiento: {len(df)} registros")
            
            return df
            
        except Exception as e:
            logger.warning(f"⚠️ DATABASE_URL no disponible: {e}")
            return pd.DataFrame()
    
    def calculate_sentiment_score(self, sentiment_data: pd.DataFrame) -> float:
        """
        Calcula una puntuación de sentimiento agregada.
        
        Args:
            sentiment_data: DataFrame con datos de sentimiento
            
        Returns:
            Puntuación de sentimiento (-1.0 a 1.0)
        """
        if sentiment_data.empty:
            return 0.0  # Neutral si no hay datos
        
        try:
            # Usar solo los últimos 7 días
            recent_sentiment = sentiment_data.tail(50)  # Últimas 50 noticias
            
            if recent_sentiment.empty:
                return 0.0
            
            # Calcular media ponderada (más peso a noticias recientes)
            weights = np.linspace(0.5, 1.0, len(recent_sentiment))
            sentiment_scores = recent_sentiment['sentiment_score'].fillna(0)
            
            if len(sentiment_scores) > 0:
                weighted_mean = np.average(sentiment_scores, weights=weights)
                return float(np.clip(weighted_mean, -1.0, 1.0))
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error calculando sentimiento: {e}")
            return 0.0
    
    def analyze_cryptocurrency(self, symbol: str) -> Optional[CryptoCandidate]:
        """
        Analiza una criptomoneda específica y retorna un candidato.
        
        Args:
            symbol: Símbolo a analizar (ej: 'BTC/USDT')
            
        Returns:
            CryptoCandidate con el análisis o None si hay error
        """
        try:
            # 1. Obtener datos históricos
            df_daily = self.get_historical_data(symbol, '1d', 30)
            df_4h = self.get_historical_data(symbol, '4h', 50)
            
            if df_daily is None or len(df_daily) < 14:
                return None
            
            # 2. Calcular indicadores
            closes = df_daily['close'].tolist()
            highs = df_daily['high'].tolist()
            lows = df_daily['low'].tolist()
            
            # Volatilidades
            volatility_24h = self.calculate_volatility(closes[-2:])  # Últimas 24h
            volatility_7d = self.calculate_volatility(closes[-7:])   # Últimos 7 días
            
            # ADX
            adx = self.calculate_adx(highs, lows, closes)
            
            # 3. Datos de mercado actuales
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = float(ticker['last'] or 0)
            volume_24h = float(ticker['quoteVolume'] or 0)
            price_change_24h = float(ticker['percentage'] or 0)
            
            # Cambio de precio 7 días
            price_change_7d = 0.0
            if len(closes) >= 7:
                price_7d_ago = closes[-7]
                price_change_7d = ((current_price - price_7d_ago) / price_7d_ago) * 100
            
            # 4. Análisis de sentimiento
            sentiment_data = self.get_sentiment_data()
            sentiment_score = self.calculate_sentiment_score(sentiment_data)
            sentiment_ma7 = sentiment_score  # Simplificado
            
            # 5. Calcular puntuación final
            score, reasons = self._calculate_candidate_score(
                volatility_7d, adx, sentiment_score, volume_24h, 
                price_change_24h, price_change_7d
            )
            
            # 6. Crear candidato
            candidate = CryptoCandidate(
                symbol=symbol,
                market_cap_rank=0,  # Se asignará después
                current_price=current_price,
                volatility_24h=volatility_24h,
                volatility_7d=volatility_7d,
                adx=adx,
                sentiment_score=sentiment_score,
                sentiment_ma7=sentiment_ma7,
                volume_24h=volume_24h,
                volume_change_24h=0.0,  # Simplificado
                price_change_24h=price_change_24h,
                price_change_7d=price_change_7d,
                score=score,
                reasons=reasons
            )
            
            return candidate
            
        except Exception as e:
            logger.warning(f"Error analizando {symbol}: {e}")
            return None
    
    def _calculate_candidate_score(self, volatility_7d: float, adx: float, 
                                  sentiment_score: float, volume_24h: float,
                                  price_change_24h: float, price_change_7d: float) -> Tuple[float, List[str]]:
        """
        Calcula la puntuación final de un candidato basándose en múltiples factores.
        
        Returns:
            Tuple[score, reasons] donde score es 0-100 y reasons son las justificaciones
        """
        score = 0.0
        reasons = []
        
        # Factor 1: Volatilidad (0-30 puntos) - Más permisivo
        # Queremos volatilidad entre 1%-8% diario para trading
        if 0.01 <= volatility_7d <= 0.08:
            # Puntuación máxima para rango óptimo 2%-5%
            if 0.02 <= volatility_7d <= 0.05:
                score += 30
                reasons.append(f"Volatilidad óptima: {volatility_7d*100:.1f}%")
            else:
                score += 25  # Buena volatilidad fuera del rango óptimo
                reasons.append(f"Buena volatilidad: {volatility_7d*100:.1f}%")
        elif volatility_7d > 0.08:
            score += 20  # Muy volátil pero aún tradeable
            reasons.append(f"Alta volatilidad: {volatility_7d*100:.1f}%")
        elif volatility_7d > 0.005:
            score += 15  # Volatilidad baja pero aceptable
            reasons.append(f"Volatilidad moderada: {volatility_7d*100:.1f}%")
        else:
            score += 5  # Muy baja volatilidad
            reasons.append(f"Baja volatilidad: {volatility_7d*100:.1f}%")
        
        # Factor 2: ADX - Cualquier mercado es bueno (0-25 puntos)
        # Más permisivo - tanto laterales como tendenciales son buenos
        if adx < 25:
            score += 25
            reasons.append(f"Mercado lateral ideal (ADX: {adx:.1f})")
        elif adx < 35:
            score += 20
            reasons.append(f"Mercado con tendencia suave (ADX: {adx:.1f})")
        elif adx < 50:
            score += 15
            reasons.append(f"Mercado trending (ADX: {adx:.1f})")
        else:
            score += 10
            reasons.append(f"Tendencia fuerte (ADX: {adx:.1f})")
        
        # Factor 3: Sentimiento (0-20 puntos) - Más permisivo
        # Cualquier sentimiento puede ser bueno
        if sentiment_score > 0.1:
            score += 20
            reasons.append(f"Sentimiento positivo: {sentiment_score:.2f}")
        elif sentiment_score > -0.1:
            score += 18  # Neutral también es bueno
            reasons.append(f"Sentimiento neutral: {sentiment_score:.2f}")
        elif sentiment_score > -0.3:
            score += 15  # Negativo moderado puede ser oportunidad
            reasons.append(f"Sentimiento negativo moderado: {sentiment_score:.2f}")
        else:
            score += 12  # Muy negativo puede ser oportunidad de compra
            reasons.append(f"Sentimiento muy negativo: {sentiment_score:.2f}")
        
        # Factor 4: Volumen (0-15 puntos) - Ajustado a nuestro mínimo
        # Con nuestro mínimo de $2M, cualquier moneda debería tener puntos
        if volume_24h > 50_000_000:  # >$50M
            score += 15
            reasons.append(f"Volumen excelente: ${volume_24h/1_000_000:.0f}M")
        elif volume_24h > 20_000_000:  # >$20M
            score += 13
            reasons.append(f"Volumen muy bueno: ${volume_24h/1_000_000:.0f}M")
        elif volume_24h > 5_000_000:  # >$5M
            score += 11
            reasons.append(f"Volumen bueno: ${volume_24h/1_000_000:.0f}M")
        else:  # >$2M (nuestro mínimo)
            score += 8
            reasons.append(f"Volumen adecuado: ${volume_24h/1_000_000:.0f}M")
        
        # Factor 5: Momentum reciente (0-10 puntos) - Más permisivo
        # Cualquier momentum puede ser oportunidad
        if -3 <= price_change_24h <= 3:
            score += 10
            reasons.append(f"Momentum estable: {price_change_24h:+.1f}%")
        elif -8 <= price_change_24h <= 8:
            score += 9
            reasons.append(f"Momentum moderado: {price_change_24h:+.1f}%")
        elif -15 <= price_change_24h <= 15:
            score += 7
            reasons.append(f"Momentum activo: {price_change_24h:+.1f}%")
        else:
            score += 5
            reasons.append(f"Momentum extremo: {price_change_24h:+.1f}%")
        
        return min(score, 100.0), reasons
    
    def scan_market(self) -> List[CryptoCandidate]:
        """
        Escanea el mercado y retorna exactamente 10 mejores candidatos para trading.
        
        Optimizado para producción 24/7:
        - Analiza Top 150 monedas por volumen
        - Aplica filtros de liquidez y volatilidad
        - Garantiza siempre 10 candidatos de salida
        
        Returns:
            Lista de exactamente 10 mejores candidatos ordenados por score
        """
        logger.info("🔍 Iniciando escaneo del mercado para producción 24/7...")
        
        # 1. Obtener top criptomonedas
        top_symbols = self.get_top_cryptocurrencies()
        
        # 2. Analizar cada una
        candidates = []
        for i, symbol in enumerate(top_symbols, 1):
            logger.info(f"📊 Analizando {symbol} ({i}/{len(top_symbols)})...")
            
            candidate = self.analyze_cryptocurrency(symbol)
            if candidate:
                candidate.market_cap_rank = i
                candidates.append(candidate)
                logger.info(f"   ✅ Score: {candidate.score:.1f}/100")
            else:
                logger.warning(f"   ❌ No se pudo analizar {symbol}")
            
            # Rate limiting
            time.sleep(0.2)
        
        # 3. Ordenar por score y garantizar exactamente 10 candidatos
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # Garantizar exactamente 10 candidatos con fallback inteligente
        if len(candidates) >= self.target_candidates:
            best_candidates = candidates[:self.target_candidates]
        else:
            # Fallback: relajar filtros para obtener más candidatos
            logger.warning(f"⚠️ Solo {len(candidates)} candidatos iniciales, aplicando fallback...")
            additional_candidates = self._get_additional_candidates(
                top_symbols, len(candidates)
            )
            all_candidates = candidates + additional_candidates
            all_candidates.sort(key=lambda x: x.score, reverse=True)
            
            if len(all_candidates) >= self.target_candidates:
                best_candidates = all_candidates[:self.target_candidates]
                logger.info(f"✅ Fallback exitoso: {len(best_candidates)} candidatos finales")
            else:
                best_candidates = all_candidates
                logger.warning(f"⚠️ Fallback parcial: {len(best_candidates)} candidatos disponibles")
        
        logger.info(f"\n🏆 MEJORES {len(best_candidates)} CANDIDATOS PARA TRADING:")
        logger.info("=" * 60)
        for i, candidate in enumerate(best_candidates, 1):
            logger.info(f"{i}. {candidate.symbol} - Score: {candidate.score:.1f}/100")
            logger.info(f"   💰 Precio: ${candidate.current_price:.4f}")
            logger.info(f"   📊 Vol 7d: {candidate.volatility_7d*100:.1f}%")
            logger.info(f"   📈 ADX: {candidate.adx:.1f}")
            logger.info(f"   😊 Sentimiento: {candidate.sentiment_score:.2f}")
            logger.info(f"   💵 Volumen: ${candidate.volume_24h/1_000_000:.0f}M")
            logger.info(f"   🎯 Razones: {', '.join(candidate.reasons[:2])}")
            logger.info("")
        
        return best_candidates
    
    def _get_additional_candidates(self, top_symbols: List[str], 
                                  current_count: int) -> List[CryptoCandidate]:
        """
        Obtiene candidatos adicionales con filtros más relajados.
        
        Args:
            top_symbols: Lista de símbolos ya analizados
            current_count: Número actual de candidatos
            
        Returns:
            Lista de candidatos adicionales
        """
        additional_candidates = []
        needed = self.target_candidates - current_count
        
        logger.info(f"🔍 Buscando {needed} candidatos adicionales con filtros relajados...")
        
        # Tomar más símbolos de la lista si están disponibles
        remaining_symbols = top_symbols[len(top_symbols)//2:]  # Segunda mitad
        
        for symbol in remaining_symbols[:needed * 2]:  # Analizar el doble para tener opciones
            try:
                candidate = self.analyze_cryptocurrency(symbol)
                if candidate and candidate.score > 30:  # Score mínimo más bajo
                    additional_candidates.append(candidate)
                    if len(additional_candidates) >= needed:
                        break
            except Exception as e:
                logger.debug(f"Error en fallback para {symbol}: {e}")
                continue
        
        logger.info(f"✅ Encontrados {len(additional_candidates)} candidatos adicionales")
        return additional_candidates
    
    def export_candidates_to_json(self, candidates: List[CryptoCandidate], 
                                 filename: Optional[str] = None) -> str:
        """
        Exporta los candidatos a un archivo JSON.
        
        Args:
            candidates: Lista de candidatos
            filename: Nombre del archivo (opcional)
            
        Returns:
            Ruta del archivo generado
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scanner_results_{timestamp}.json"
        
        # Convertir candidatos a diccionarios
        candidates_data = []
        for candidate in candidates:
            data = {
                'symbol': candidate.symbol,
                'market_cap_rank': candidate.market_cap_rank,
                'current_price': candidate.current_price,
                'volatility_24h': candidate.volatility_24h,
                'volatility_7d': candidate.volatility_7d,
                'adx': candidate.adx,
                'sentiment_score': candidate.sentiment_score,
                'sentiment_ma7': candidate.sentiment_ma7,
                'volume_24h': candidate.volume_24h,
                'volume_change_24h': candidate.volume_change_24h,
                'price_change_24h': candidate.price_change_24h,
                'price_change_7d': candidate.price_change_7d,
                'score': candidate.score,
                'reasons': candidate.reasons
            }
            candidates_data.append(data)
        
        # Guardar archivo
        export_data = {
            'scan_timestamp': datetime.now().isoformat(),
            'total_candidates': len(candidates),
            'candidates': candidates_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Candidatos exportados a: {filename}")
        return filename


def main():
    """
    Función principal de demostración del scanner.
    """
    print("🔍 CRYPTO SCANNER - Buscador de Oportunidades")
    print("=" * 60)
    
    # Crear scanner con configuración de producción
    scanner = CryptoScanner()
    
    # Escanear mercado
    candidates = scanner.scan_market()
    
    # Exportar resultados
    if candidates:
        filename = scanner.export_candidates_to_json(candidates)
        print(f"\n💾 Resultados guardados en: {filename}")
        
        # Mostrar símbolos para usar con backtesting
        symbols = [c.symbol for c in candidates]
        print(f"\n🎯 MONEDAS PARA BACKTESTING:")
        print(f"python scripts/find_optimal_parameters.py --monedas {' '.join(symbols)}")
    
    print("\n🎉 Escaneo completado!")


if __name__ == "__main__":
    main() 