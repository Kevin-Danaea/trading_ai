"""
Crypto Scanner Service - Servicio de Escaneo de Criptomonedas
============================================================

Caso de uso para el escaneo inteligente de oportunidades de trading.
Actúa como "nuestros ojos" pre-filtrando oportunidades antes del backtesting.

Este servicio coordina entre el dominio y la infraestructura.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

from app.domain.entities import CryptoCandidate

logger = logging.getLogger(__name__)


class CryptoScannerService:
    """
    Servicio de aplicación para el escaneo inteligente de criptomonedas.
    
    Analiza el Top N de monedas usando:
    1. Datos de mercado en tiempo real
    2. Cálculo de indicadores técnicos (ADX, volatilidad)
    3. Análisis de sentimiento
    4. Sistema de puntuación para rankear candidatos
    
    Este servicio depende de proveedores de infraestructura que se inyectan.
    """
    
    def __init__(self, 
                 market_data_provider,
                 sentiment_data_provider,
                 top_n: int = 150,
                 min_volume_usdt: float = 2_000_000,
                 target_candidates: int = 10):
        """
        Inicializa el servicio de scanner.
        
        Args:
            market_data_provider: Proveedor de datos de mercado (infraestructura)
            sentiment_data_provider: Proveedor de datos de sentimiento (infraestructura)
            top_n: Número de monedas a analizar
            min_volume_usdt: Volumen mínimo en USDT
            target_candidates: Número objetivo de candidatos
        """
        self.market_data_provider = market_data_provider
        self.sentiment_data_provider = sentiment_data_provider
        
        # Configuración optimizada para producción
        self.top_n = top_n
        self.min_volume_usdt = min_volume_usdt
        self.target_candidates = target_candidates
        
        # Cache interno
        self.market_data_cache: Dict[str, Any] = {}
        self.sentiment_cache: Optional[pd.DataFrame] = None
        
        logger.info(f"🔍 CryptoScannerService inicializado para producción 24/7")
        logger.info(f"📊 Analizará Top {self.top_n} monedas")
        logger.info(f"💰 Volumen mínimo: ${self.min_volume_usdt:,.0f} USDT")
        logger.info(f"🎯 Objetivo: {self.target_candidates} candidatos siempre")
    
    def scan_market(self) -> List[CryptoCandidate]:
        """
        Escanea el mercado y retorna exactamente N mejores candidatos para trading.
        
        Proceso:
        1. Obtener top criptomonedas
        2. Analizar cada una (indicadores + sentimiento)
        3. Calcular puntuación
        4. Rankear y seleccionar mejores
        5. Garantizar número objetivo de candidatos
        
        Returns:
            Lista de mejores candidatos ordenados por score
        """
        logger.info("🔍 Iniciando escaneo del mercado para producción 24/7...")
        
        # 1. Obtener top criptomonedas
        top_symbols = self._get_top_cryptocurrencies()
        
        # 2. Analizar cada una
        candidates = []
        for i, symbol in enumerate(top_symbols, 1):
            logger.info(f"📊 Analizando {symbol} ({i}/{len(top_symbols)})...")
            
            candidate = self._analyze_cryptocurrency(symbol)
            if candidate:
                candidate.market_cap_rank = i
                candidates.append(candidate)
                logger.info(f"   ✅ Score: {candidate.score:.1f}/100")
            else:
                logger.warning(f"   ❌ No se pudo analizar {symbol}")
        
        # 3. Ordenar por score y garantizar exactamente N candidatos
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # Garantizar exactamente target_candidates con fallback inteligente
        best_candidates = self._ensure_target_candidates(candidates, top_symbols)
        
        self._log_final_results(best_candidates)
        
        return best_candidates
    
    def _get_top_cryptocurrencies(self) -> List[str]:
        """
        Obtiene la lista de Top N criptomonedas usando el proveedor de datos.
        
        Returns:
            Lista de símbolos en formato 'BTC/USDT'
        """
        logger.info(f"📈 Obteniendo Top {self.top_n} criptomonedas...")
        
        try:
            return self.market_data_provider.get_top_symbols(
                top_n=self.top_n,
                min_volume_usdt=self.min_volume_usdt
            )
        except Exception as e:
            logger.error(f"❌ Error obteniendo top criptomonedas: {e}")
            # Fallback a lista predefinida
            return [
                'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'MATIC/USDT',
                'LINK/USDT', 'DOT/USDT', 'ADA/USDT', 'DOGE/USDT', 'XRP/USDT'
            ]
    
    def _analyze_cryptocurrency(self, symbol: str) -> Optional[CryptoCandidate]:
        """
        Analiza una criptomoneda específica y retorna un candidato.
        
        Args:
            symbol: Símbolo a analizar (ej: 'BTC/USDT')
            
        Returns:
            CryptoCandidate con el análisis o None si hay error
        """
        try:
            # 1. Obtener datos de mercado
            market_data = self.market_data_provider.get_market_data(symbol)
            if not market_data:
                return None
            
            # 2. Calcular indicadores técnicos
            volatility_24h = self._calculate_volatility(market_data['prices_24h'])
            volatility_7d = self._calculate_volatility(market_data['prices_7d'])
            adx = self._calculate_adx(
                market_data['highs'], 
                market_data['lows'], 
                market_data['closes']
            )
            
            # 3. Análisis de sentimiento
            sentiment_score = self._get_sentiment_score()
            sentiment_ma7 = sentiment_score  # Simplificado
            
            # 4. Calcular puntuación final
            score, reasons = self._calculate_candidate_score(
                volatility_7d, adx, sentiment_score, 
                market_data['volume_24h'],
                market_data['price_change_24h'], 
                market_data['price_change_7d']
            )
            
            # 5. Crear candidato
            candidate = CryptoCandidate(
                symbol=symbol,
                market_cap_rank=0,  # Se asignará después
                current_price=market_data['current_price'],
                volatility_24h=volatility_24h,
                volatility_7d=volatility_7d,
                adx=adx,
                sentiment_score=sentiment_score,
                sentiment_ma7=sentiment_ma7,
                volume_24h=market_data['volume_24h'],
                volume_change_24h=market_data.get('volume_change_24h', 0.0),
                price_change_24h=market_data['price_change_24h'],
                price_change_7d=market_data['price_change_7d'],
                score=score,
                reasons=reasons
            )
            
            return candidate
            
        except Exception as e:
            logger.warning(f"Error analizando {symbol}: {e}")
            return None
    
    def _calculate_volatility(self, prices: List[float]) -> float:
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
    
    def _calculate_adx(self, highs: List[float], lows: List[float], 
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
            
            # Calcular True Range simplificado
            tr = high - low
            
            # Media móvil simple como proxy de ADX
            if len(tr) >= period:
                adx = np.mean(tr[-period:]) * 100 / np.mean(close[-period:])
                return float(np.clip(adx, 0, 100))
            else:
                return 50.0
                
        except Exception as e:
            logger.warning(f"Error calculando ADX: {e}")
            return 50.0  # Valor neutral en caso de error
    
    def _get_sentiment_score(self) -> float:
        """
        Obtiene la puntuación de sentimiento agregada.
        
        Returns:
            Puntuación de sentimiento (-1.0 a 1.0)
        """
        try:
            if self.sentiment_cache is None:
                # Usar el método correcto del proveedor
                sentiment_data = self.sentiment_data_provider.get_sentiment_data(days_back=30)
                self.sentiment_cache = sentiment_data
            
            if self.sentiment_cache is None or self.sentiment_cache.empty:
                return 0.0  # Neutral si no hay datos
            
            # Usar el método del proveedor para calcular el score
            # El proveedor retorna 0-100, pero necesitamos -1 a 1
            provider_score = self.sentiment_data_provider.calculate_sentiment_score(self.sentiment_cache)
            
            # Convertir de rango [0, 100] a [-1, 1]
            # donde 50 es neutral (0), 0 es muy negativo (-1) y 100 es muy positivo (1)
            normalized_score = (provider_score - 50) / 50
            
            return float(np.clip(normalized_score, -1.0, 1.0))
                
        except Exception as e:
            logger.warning(f"Error calculando sentimiento: {e}")
            return 0.0
    
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
        if 0.01 <= volatility_7d <= 0.08:
            if 0.02 <= volatility_7d <= 0.05:
                score += 30
                reasons.append(f"Volatilidad óptima: {volatility_7d*100:.1f}%")
            else:
                score += 25
                reasons.append(f"Buena volatilidad: {volatility_7d*100:.1f}%")
        elif volatility_7d > 0.08:
            score += 20
            reasons.append(f"Alta volatilidad: {volatility_7d*100:.1f}%")
        elif volatility_7d > 0.005:
            score += 15
            reasons.append(f"Volatilidad moderada: {volatility_7d*100:.1f}%")
        else:
            score += 5
            reasons.append(f"Baja volatilidad: {volatility_7d*100:.1f}%")
        
        # Factor 2: ADX - Cualquier mercado es bueno (0-25 puntos)
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
        if sentiment_score > 0.1:
            score += 20
            reasons.append(f"Sentimiento positivo: {sentiment_score:.2f}")
        elif sentiment_score > -0.1:
            score += 18
            reasons.append(f"Sentimiento neutral: {sentiment_score:.2f}")
        elif sentiment_score > -0.3:
            score += 15
            reasons.append(f"Sentimiento negativo moderado: {sentiment_score:.2f}")
        else:
            score += 12
            reasons.append(f"Sentimiento muy negativo: {sentiment_score:.2f}")
        
        # Factor 4: Volumen (0-15 puntos)
        if volume_24h > 50_000_000:
            score += 15
            reasons.append(f"Volumen excelente: ${volume_24h/1_000_000:.0f}M")
        elif volume_24h > 20_000_000:
            score += 13
            reasons.append(f"Volumen muy bueno: ${volume_24h/1_000_000:.0f}M")
        elif volume_24h > 5_000_000:
            score += 11
            reasons.append(f"Volumen bueno: ${volume_24h/1_000_000:.0f}M")
        else:
            score += 8
            reasons.append(f"Volumen adecuado: ${volume_24h/1_000_000:.0f}M")
        
        # Factor 5: Momentum reciente (0-10 puntos)
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
    
    def _ensure_target_candidates(self, candidates: List[CryptoCandidate], 
                                 top_symbols: List[str]) -> List[CryptoCandidate]:
        """
        Garantiza exactamente target_candidates candidatos con fallback inteligente.
        
        Args:
            candidates: Lista de candidatos inicial
            top_symbols: Lista de símbolos top originales
            
        Returns:
            Lista de exactamente target_candidates candidatos
        """
        if len(candidates) >= self.target_candidates:
            return candidates[:self.target_candidates]
        
        # Fallback: relajar filtros para obtener más candidatos
        logger.warning(f"⚠️ Solo {len(candidates)} candidatos iniciales, aplicando fallback...")
        
        # Análisis más relajado de símbolos restantes
        remaining_symbols = [s for s in top_symbols 
                           if s not in [c.symbol for c in candidates]]
        
        additional_candidates = []
        for symbol in remaining_symbols[:self.target_candidates * 2]:
            try:
                candidate = self._analyze_cryptocurrency(symbol)
                if candidate and candidate.score > 30:  # Score mínimo más bajo
                    additional_candidates.append(candidate)
                    if len(candidates) + len(additional_candidates) >= self.target_candidates:
                        break
            except Exception:
                continue
        
        all_candidates = candidates + additional_candidates
        all_candidates.sort(key=lambda x: x.score, reverse=True)
        
        final_candidates = all_candidates[:self.target_candidates]
        logger.info(f"✅ Fallback completado: {len(final_candidates)} candidatos finales")
        
        return final_candidates
    
    def _log_final_results(self, candidates: List[CryptoCandidate]):
        """Log de los resultados finales del escaneo."""
        logger.info(f"\n🏆 MEJORES {len(candidates)} CANDIDATOS PARA TRADING:")
        logger.info("=" * 60)
        for i, candidate in enumerate(candidates, 1):
            logger.info(f"{i}. {candidate.symbol} - Score: {candidate.score:.1f}/100")
            logger.info(f"   💰 Precio: ${candidate.current_price:.4f}")
            logger.info(f"   📊 Vol 7d: {candidate.volatility_7d*100:.1f}%")
            logger.info(f"   📈 ADX: {candidate.adx:.1f}")
            logger.info(f"   😊 Sentimiento: {candidate.sentiment_score:.2f}")
            logger.info(f"   💵 Volumen: ${candidate.volume_24h/1_000_000:.0f}M")
            logger.info(f"   🎯 Razones: {', '.join(candidate.reasons[:2])}")
            logger.info("") 