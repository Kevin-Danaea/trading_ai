"""
Sentiment Data Provider - Proveedor de Datos de Sentimiento
===========================================================

Proveedor de infraestructura para obtener datos de sentimiento desde BigQuery y DATABASE_URL.
Migrado desde scripts/data_collector.py con toda la funcionalidad original.
"""

import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import logging
import numpy as np
import pytz

from app.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)


class SentimentDataProvider:
    """
    Proveedor de datos de sentimiento que maneja BigQuery (histórico) y DATABASE_URL (reciente).
    """
    
    def __init__(self):
        """Inicializa el proveedor con configuración de BigQuery."""
        # Configuración de BigQuery
        self.project_id = settings.GOOGLE_CLOUD_PROJECT_ID
        self.dataset = 'oraculo_data'
        self.table = 'noticias_historicas'
        
        # Cache para evitar consultas repetidas
        self.sentiment_cache: Optional[pd.DataFrame] = None
        self.cache_date: Optional[datetime] = None
        
        logger.info("📰 SentimentDataProvider inicializado")
    
    def get_sentiment_data(self, days_back: int = 30) -> pd.DataFrame:
        """
        Obtiene datos de sentimiento combinando BigQuery (histórico) y DATABASE_URL (reciente).
        
        Args:
            days_back: Días hacia atrás para obtener datos
            
        Returns:
            DataFrame con datos de sentimiento
        """
        logger.info(f"📰 Obteniendo datos de sentimiento de los últimos {days_back} días...")
        
        # Verificar cache
        if self._is_cache_valid():
            logger.info("✅ Usando cache de sentimiento")
            return self.sentiment_cache
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Punto de división entre BigQuery y DATABASE_URL (últimos 7 días en DATABASE_URL)
            split_date = end_date - timedelta(days=7)
            
            sentiment_data_list = []
            
            # 1. Datos históricos desde BigQuery (más de 7 días atrás)
            if start_date < split_date:
                logger.info("🏛️ Obteniendo datos históricos desde BigQuery...")
                bigquery_data = self._fetch_bigquery_sentiment(start_date, split_date)
                if not bigquery_data.empty:
                    sentiment_data_list.append(bigquery_data)
                    logger.info(f"   ✅ BigQuery: {len(bigquery_data)} registros")
            
            # 2. Datos recientes desde DATABASE_URL (últimos 7 días)
            logger.info("🔄 Obteniendo datos recientes desde DATABASE_URL...")
            recent_data = self._fetch_recent_sentiment(split_date, end_date)
            if not recent_data.empty:
                sentiment_data_list.append(recent_data)
                logger.info(f"   ✅ DATABASE_URL: {len(recent_data)} registros")
            
            # 3. Combinar datos
            if sentiment_data_list:
                combined_data = pd.concat(sentiment_data_list, ignore_index=True)
                combined_data = combined_data.sort_values('published_at')
                combined_data = combined_data.drop_duplicates(subset=['published_at', 'headline'], keep='last')
                
                # Actualizar cache
                self.sentiment_cache = combined_data
                self.cache_date = datetime.now()
                
                logger.info(f"✅ Datos de sentimiento combinados: {len(combined_data)} registros")
                return combined_data
            else:
                logger.warning("⚠️ No se encontraron datos de sentimiento")
                return self._create_empty_sentiment_dataframe()
                
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos de sentimiento: {e}")
            return self._create_empty_sentiment_dataframe()
    
    def calculate_sentiment_score(self, sentiment_data: pd.DataFrame) -> float:
        """
        Calcula un score agregado de sentimiento de 0-100.
        
        Args:
            sentiment_data: DataFrame con datos de sentimiento
            
        Returns:
            Score de sentimiento (0-100, donde 50 es neutral)
        """
        if sentiment_data.empty:
            return 50.0  # Neutral por defecto
        
        try:
            # Filtrar últimos 7 días - usar UTC para consistencia
            recent_cutoff = datetime.now(pytz.UTC) - timedelta(days=7)
            
            # Asegurar que las fechas sean timezone-aware
            published_dates = pd.to_datetime(sentiment_data['published_at'])
            if published_dates.dt.tz is None:
                # Si las fechas no tienen timezone, asumir UTC
                published_dates = published_dates.dt.tz_localize('UTC')
            else:
                # Convertir a UTC si tienen otro timezone
                published_dates = published_dates.dt.tz_convert('UTC')
            
            recent_sentiment = sentiment_data[published_dates >= recent_cutoff]
            
            if recent_sentiment.empty:
                return 50.0
            
            # Calcular score ponderado
            sentiment_scores = recent_sentiment['sentiment_score'].astype(float)
            
            # Convertir de rango [-1, 1] a [0, 100]
            normalized_scores = (sentiment_scores + 1) * 50
            
            # Aplicar bonus/penalización por emoción primaria (opcional)
            if 'primary_emotion' in recent_sentiment.columns:
                emotion_bonus = self._calculate_emotion_bonus(recent_sentiment['primary_emotion'])
                normalized_scores = normalized_scores + emotion_bonus
            
            # Media ponderada por recencia (más peso a noticias recientes)
            weights = self._calculate_recency_weights(recent_sentiment['published_at'])
            weighted_score = (normalized_scores * weights).sum() / weights.sum()
            
            # Aplicar factor de volumen (más noticias = más confianza en el score)
            volume_factor = min(len(recent_sentiment) / 20, 1.0)  # Máximo factor en 20+ noticias
            final_score = 50 + (weighted_score - 50) * volume_factor
            
            return max(0, min(100, final_score))
            
        except Exception as e:
            logger.warning(f"Error calculando sentiment score: {e}")
            return 50.0

    def _calculate_emotion_bonus(self, emotions: pd.Series) -> pd.Series:
        """
        Calcula bonus/penalización basado en la emoción primaria.
        
        Args:
            emotions: Series con emociones primarias
            
        Returns:
            Series con bonus/penalización para el score
        """
        try:
            # Mapeo de emociones a modificadores de score (-10 a +10)
            emotion_modifiers = {
                'Optimismo': 8,
                'Alegría': 10,
                'Confianza': 6,
                'Esperanza': 7,
                'Satisfacción': 5,
                'Neutral': 0,
                'Preocupación': -3,
                'Incertidumbre': -4,
                'Frustración': -6,
                'Miedo': -8,
                'Pánico': -10,
                'Desconfianza': -7
            }
            
            # Aplicar modificadores
            bonus = emotions.map(emotion_modifiers).fillna(0)
            return bonus
            
        except Exception as e:
            logger.warning(f"Error calculando emotion bonus: {e}")
            return pd.Series([0] * len(emotions))
    
    def _fetch_bigquery_sentiment(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Obtiene datos de sentimiento desde BigQuery.
        
        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            DataFrame con datos de sentimiento
        """
        try:
            # Importar pandas_gbq aquí para evitar problemas de importación si no está disponible
            from pandas_gbq import read_gbq
            
            query = f"""
            SELECT 
                published_at,
                sentiment_score,
                primary_emotion,
                news_category,
                source,
                headline
            FROM `{self.project_id}.{self.dataset}.noticias_historicas`
            WHERE DATE(published_at) BETWEEN '{start_date.date()}' AND '{end_date.date()}'
            ORDER BY published_at
            """
            
            df = read_gbq(query, project_id=self.project_id)
            
            if df.empty:
                logger.info("📰 No se encontraron datos de sentimiento en BigQuery")
                return pd.DataFrame()
            
            # Limpiar y preparar datos
            df['published_at'] = pd.to_datetime(df['published_at'])
            df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
            df = df.dropna(subset=['sentiment_score'])
            
            return df
            
        except ImportError:
            logger.warning("⚠️ pandas_gbq no está disponible, omitiendo datos de BigQuery")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Error obteniendo datos de BigQuery: {e}")
            return pd.DataFrame()
    
    def _fetch_recent_sentiment(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Obtiene datos recientes de sentimiento desde DATABASE_URL.
        
        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            DataFrame con datos recientes de sentimiento
        """
        try:
            import psycopg2
            import sqlalchemy
            
            if not settings.DATABASE_URL:
                logger.warning("⚠️ DATABASE_URL no configurado, omitiendo datos recientes")
                return pd.DataFrame()
            
            # Crear conexión
            engine = sqlalchemy.create_engine(settings.DATABASE_URL)
            
            # Convertir fechas a string para comparación con campo VARCHAR
            start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
            end_date_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
            
            query = """
            SELECT 
                published_at,
                sentiment_score,
                primary_emotion,
                news_category,
                source,
                headline
            FROM noticias
            WHERE published_at >= %(start_date)s AND published_at <= %(end_date)s
            ORDER BY published_at
            """
            
            df = pd.read_sql_query(
                query, 
                engine, 
                params={
                    'start_date': start_date_str,
                    'end_date': end_date_str
                }
            )
            
            if df.empty:
                logger.info("📰 No se encontraron datos recientes de sentimiento")
                return pd.DataFrame()
            
            # Limpiar y preparar datos
            df['published_at'] = pd.to_datetime(df['published_at'])
            df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
            df = df.dropna(subset=['sentiment_score'])
            
            return df
            
        except ImportError:
            logger.warning("⚠️ psycopg2 no está disponible, omitiendo datos de DATABASE_URL")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Error obteniendo datos recientes: {e}")
            return pd.DataFrame()
    
    def _calculate_recency_weights(self, dates: pd.Series) -> pd.Series:
        """
        Calcula pesos basados en la recencia de las noticias.
        
        Args:
            dates: Series con fechas de publicación
            
        Returns:
            Series con pesos (más peso = más reciente)
        """
        try:
            dates = pd.to_datetime(dates)
            now = datetime.now(pytz.UTC)
            
            # Asegurar que las fechas sean timezone-aware
            if dates.dt.tz is None:
                dates = dates.dt.tz_localize('UTC')
            else:
                dates = dates.dt.tz_convert('UTC')
            
            # Calcular días de antigüedad
            days_ago = (now - dates).dt.total_seconds() / (24 * 3600)
            
            # Peso exponencial decreciente (decay factor = 0.1 por día)
            weights = np.exp(-0.1 * days_ago)
            
            return weights
            
        except Exception as e:
            logger.warning(f"Error calculando pesos de recencia: {e}")
            return pd.Series([1.0] * len(dates))
    
    def _create_empty_sentiment_dataframe(self) -> pd.DataFrame:
        """Crea un DataFrame vacío con las columnas correctas."""
        return pd.DataFrame(columns=[
            'published_at', 'sentiment_score', 'primary_emotion', 
            'news_category', 'source', 'headline'
        ])
    
    def _is_cache_valid(self) -> bool:
        """Verifica si el cache es válido (menos de 1 hora de antigüedad)."""
        if self.sentiment_cache is None or self.cache_date is None:
            return False
        
        cache_age = datetime.now() - self.cache_date
        return cache_age.total_seconds() < 3600  # 1 hora 