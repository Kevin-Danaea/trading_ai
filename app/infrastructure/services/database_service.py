"""
Database Service - Servicio de Base de Datos
============================================

Servicio de infraestructura que maneja las operaciones de base de datos
para el sistema de recomendaciones diarias de trading.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from decimal import Decimal
import sqlalchemy
from sqlalchemy import text, Table, Column, Integer, String, DateTime, Boolean, Numeric, JSON, MetaData
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
import json

from app.infrastructure.config.settings import settings
from app.domain.entities.daily_recommendation import RecomendacionDiaria

logger = logging.getLogger(__name__)


class DatabaseService:
    """
    Servicio de base de datos para manejar operaciones CRUD de recomendaciones diarias.
    
    Maneja:
    - Conexión a PostgreSQL/Neon
    - Creación automática de tablas
    - Operaciones CRUD para recomendaciones diarias
    - Consultas especializadas para reportes
    """
    
    def __init__(self):
        """Inicializa el servicio de base de datos."""
        self.engine = None
        self.metadata = MetaData()
        self._initialize_connection()
        self._create_tables()
    
    def _initialize_connection(self):
        """Inicializa la conexión a la base de datos."""
        try:
            if not settings.DATABASE_URL:
                logger.warning("⚠️ DATABASE_URL no configurado")
                return
            
            self.engine = sqlalchemy.create_engine(
                settings.DATABASE_URL,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False  # Cambiar a True para debugging
            )
            
            # Probar conexión
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.info("✅ Conexión a base de datos establecida")
                
        except Exception as e:
            logger.error(f"❌ Error conectando a la base de datos: {e}")
            self.engine = None
    
    def _create_tables(self):
        """Crea las tablas necesarias si no existen."""
        if not self.engine:
            return
        
        try:
            # Definir tabla recomendaciones_diarias
            self.recomendaciones_table = Table(
                'recomendaciones_diarias',
                self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('fecha', DateTime, nullable=False),
                Column('simbolo', String(20), nullable=False),
                Column('estrategia_recomendada', String(10), nullable=False),
                Column('estrategia_gemini', String(10), nullable=False),
                Column('parametros_optimizados', JSON, nullable=False),
                Column('roi_porcentaje', Numeric(10, 2), nullable=False),
                Column('sharpe_ratio', Numeric(10, 4), nullable=False),
                Column('max_drawdown_porcentaje', Numeric(10, 2), nullable=False),
                Column('win_rate_porcentaje', Numeric(10, 2), nullable=False),
                Column('total_trades', Integer, nullable=False),
                Column('score_final', Numeric(5, 2), nullable=False),
                Column('score_ajustado_riesgo', Numeric(5, 2), nullable=False),
                Column('razon_gemini', String(2000), nullable=False),
                Column('fortalezas_gemini', String(2000), nullable=False),
                Column('riesgos_gemini', String(2000), nullable=False),
                Column('condiciones_mercado_gemini', String(1000), nullable=False),
                Column('score_confianza_gemini', Numeric(5, 2), nullable=False),
                Column('consenso_estrategia', Boolean, nullable=False),
                Column('diferencia_scores', Numeric(5, 2), nullable=True),
                Column('categoria_rendimiento', String(20), nullable=False),
                Column('condiciones_mercado', String(20), nullable=False),
                Column('periodo_backtesting_dias', Integer, nullable=False),
                Column('creado_en', DateTime, nullable=False),
                Column('version_pipeline', String(10), nullable=False),
                extend_existing=True
            )
            
            # Crear tabla si no existe
            self.metadata.create_all(self.engine)
            logger.info("✅ Tablas de base de datos verificadas/creadas")
            
        except Exception as e:
            logger.error(f"❌ Error creando tablas: {e}")
    
    @contextmanager
    def get_connection(self):
        """Context manager para obtener conexión a la base de datos."""
        if not self.engine:
            raise RuntimeError("Base de datos no configurada")
        
        connection = self.engine.connect()
        try:
            yield connection
        finally:
            connection.close()
    
    def save_recommendation(self, recommendation: RecomendacionDiaria) -> bool:
        """
        Guarda una recomendación diaria en la base de datos.
        
        Args:
            recommendation: Recomendación a guardar
            
        Returns:
            True si se guardó exitosamente, False en caso contrario
        """
        try:
            if not self.engine:
                logger.warning("⚠️ Base de datos no configurada, no se puede guardar la recomendación")
                return False
            
            with self.get_connection() as conn:
                # Convertir recomendación a registro de base de datos
                db_record = recommendation.get_database_record()
                
                # Convertir parametros_optimizados a JSON string
                db_record['parametros_optimizados'] = json.dumps(db_record['parametros_optimizados'])
                
                # Insertar en la tabla
                insert_stmt = self.recomendaciones_table.insert().values(**db_record)
                result = conn.execute(insert_stmt)
                conn.commit()
                
                logger.info(f"✅ Recomendación guardada: {recommendation.simbolo} - {recommendation.estrategia_recomendada}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"❌ Error SQL guardando recomendación: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error guardando recomendación: {e}")
            return False
    
    def save_recommendations(self, recommendations: List[RecomendacionDiaria]) -> Dict[str, Any]:
        """
        Guarda múltiples recomendaciones diarias en una transacción.
        
        Args:
            recommendations: Lista de recomendaciones a guardar
            
        Returns:
            Diccionario con estadísticas del proceso
        """
        if not self.engine:
            logger.warning("⚠️ Base de datos no configurada")
            return {
                'total': len(recommendations),
                'guardadas': 0,
                'errores': len(recommendations),
                'exitoso': False
            }
        
        guardadas = 0
        errores = 0
        
        try:
            with self.get_connection() as conn:
                trans = conn.begin()
                
                try:
                    for recommendation in recommendations:
                        db_record = recommendation.get_database_record()
                        db_record['parametros_optimizados'] = json.dumps(db_record['parametros_optimizados'])
                        
                        insert_stmt = self.recomendaciones_table.insert().values(**db_record)
                        conn.execute(insert_stmt)
                        guardadas += 1
                    
                    trans.commit()
                    logger.info(f"✅ {guardadas} recomendaciones guardadas en lote")
                    
                except Exception as e:
                    trans.rollback()
                    logger.error(f"❌ Error en lote, rollback ejecutado: {e}")
                    errores = len(recommendations)
                    guardadas = 0
                    
        except Exception as e:
            logger.error(f"❌ Error estableciendo transacción: {e}")
            errores = len(recommendations)
        
        return {
            'total': len(recommendations),
            'guardadas': guardadas,
            'errores': errores,
            'exitoso': guardadas > 0
        }
    
    def get_recommendations_by_date(self, fecha: date) -> List[Dict[str, Any]]:
        """
        Obtiene recomendaciones de una fecha específica.
        
        Args:
            fecha: Fecha a consultar
            
        Returns:
            Lista de recomendaciones para esa fecha
        """
        try:
            if not self.engine:
                logger.warning("⚠️ Base de datos no configurada")
                return []
            
            with self.get_connection() as conn:
                query = text("""
                    SELECT * FROM recomendaciones_diarias 
                    WHERE DATE(fecha) = :fecha
                    ORDER BY score_final DESC
                """)
                
                result = conn.execute(query, {'fecha': fecha})
                recommendations = []
                
                for row in result:
                    rec = dict(row._mapping)
                    # Convertir JSON string de vuelta a dict
                    rec['parametros_optimizados'] = json.loads(rec['parametros_optimizados'])
                    recommendations.append(rec)
                
                logger.info(f"📊 Encontradas {len(recommendations)} recomendaciones para {fecha}")
                return recommendations
                
        except Exception as e:
            logger.error(f"❌ Error consultando recomendaciones por fecha: {e}")
            return []
    
    def get_latest_recommendations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtiene las recomendaciones más recientes.
        
        Args:
            limit: Número máximo de recomendaciones a retornar
            
        Returns:
            Lista de recomendaciones ordenadas por fecha descendente
        """
        try:
            if not self.engine:
                logger.warning("⚠️ Base de datos no configurada")
                return []
            
            with self.get_connection() as conn:
                query = text("""
                    SELECT * FROM recomendaciones_diarias 
                    ORDER BY fecha DESC, score_final DESC
                    LIMIT :limit
                """)
                
                result = conn.execute(query, {'limit': limit})
                recommendations = []
                
                for row in result:
                    rec = dict(row._mapping)
                    rec['parametros_optimizados'] = json.loads(rec['parametros_optimizados'])
                    recommendations.append(rec)
                
                logger.info(f"📊 Encontradas {len(recommendations)} recomendaciones recientes")
                return recommendations
                
        except Exception as e:
            logger.error(f"❌ Error consultando recomendaciones recientes: {e}")
            return []
    
    def get_recommendations_statistics(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Obtiene estadísticas de recomendaciones de los últimos días.
        
        Args:
            days_back: Número de días hacia atrás para las estadísticas
            
        Returns:
            Diccionario con estadísticas
        """
        try:
            if not self.engine:
                logger.warning("⚠️ Base de datos no configurada")
                return {}
            
            with self.get_connection() as conn:
                query = text("""
                    SELECT 
                        COUNT(*) as total_recomendaciones,
                        AVG(score_final) as score_promedio,
                        AVG(roi_porcentaje) as roi_promedio,
                        AVG(score_confianza_gemini) as confianza_promedio,
                        SUM(CASE WHEN consenso_estrategia = true THEN 1 ELSE 0 END) as consenso_count,
                        COUNT(DISTINCT simbolo) as simbolos_unicos,
                        COUNT(DISTINCT DATE(fecha)) as dias_activos
                    FROM recomendaciones_diarias 
                    WHERE fecha >= NOW() - INTERVAL ':days_back days'
                """)
                
                result = conn.execute(query, {'days_back': days_back})
                row = result.fetchone()
                
                if row:
                    stats = dict(row._mapping)
                    stats['tasa_consenso'] = (stats['consenso_count'] / stats['total_recomendaciones']) * 100 if stats['total_recomendaciones'] > 0 else 0
                    return stats
                
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error obteniendo estadísticas: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Verifica el estado de la conexión a la base de datos.
        
        Returns:
            Diccionario con estado de salud
        """
        try:
            if not self.engine:
                return {
                    'status': 'error',
                    'message': 'Base de datos no configurada',
                    'configured': False
                }
            
            with self.get_connection() as conn:
                conn.execute(text("SELECT 1"))
                
                # Contar registros recientes
                query = text("""
                    SELECT COUNT(*) as recent_count
                    FROM recomendaciones_diarias
                    WHERE fecha >= NOW() - INTERVAL '24 hours'
                """)
                result = conn.execute(query)
                row = result.fetchone()
                recent_count = row[0] if row else 0
                
                return {
                    'status': 'ok',
                    'message': 'Base de datos operativa',
                    'configured': True,
                    'recent_recommendations': recent_count,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error de conexión: {str(e)}',
                'configured': bool(self.engine),
                'timestamp': datetime.now().isoformat()
            } 