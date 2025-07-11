#!/usr/bin/env python3
"""
Settings - Configuraciones del Proyecto
=======================================

Archivo de configuraciones centralizadas para el proyecto de Trading AI.
Maneja variables de entorno y configuraciones por defecto.
"""

import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


def _clean_env_value(env_value: str) -> str:
    """
    Limpia un valor de variable de entorno eliminando comentarios y espacios.
    
    Args:
        env_value: Valor crudo de la variable de entorno
        
    Returns:
        Valor limpio sin comentarios ni espacios extra
    """
    if not env_value:
        return env_value
    
    # Eliminar comentarios (todo lo que está después de #)
    if '#' in env_value:
        env_value = env_value.split('#')[0]
    
    # Eliminar espacios al principio y al final
    return env_value.strip()


class Settings:
    """
    Clase de configuraciones centralizadas.
    
    Maneja todas las configuraciones del proyecto, incluyendo:
    - APIs de exchanges (Binance)
    - Configuración de Google Cloud (BigQuery)
    - Configuraciones de base de datos
    - Parámetros por defecto
    """
    
    # === GOOGLE CLOUD CONFIGURATION ===
    GOOGLE_CLOUD_PROJECT_ID: str = os.getenv('GOOGLE_CLOUD_PROJECT_ID', 'tu-proyecto-id')
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')
    
    # === BINANCE API CONFIGURATION ===
    BINANCE_API_KEY: str = os.getenv('BINANCE_API_KEY', '')
    BINANCE_API_SECRET: str = os.getenv('BINANCE_API_SECRET', '')
    
    # === DATABASE CONFIGURATION ===
    DATABASE_URL: str = os.getenv('DATABASE_URL', '')
    
    # === TRADING CONFIGURATION ===
    DEFAULT_COMMISSION: float = float(_clean_env_value(os.getenv('DEFAULT_COMMISSION', '0.001')))  # 0.1%
    DEFAULT_INITIAL_CAPITAL: float = float(_clean_env_value(os.getenv('DEFAULT_INITIAL_CAPITAL', '1000.0')))
    
    # === RATE LIMITING ===
    API_RATE_LIMIT_DELAY: float = float(_clean_env_value(os.getenv('API_RATE_LIMIT_DELAY', '0.1')))
    MAX_API_RETRIES: int = int(_clean_env_value(os.getenv('MAX_API_RETRIES', '3')))
    
    # === LOGGING ===
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    def __init__(self):
        """Inicializar configuraciones y validar variables críticas."""
        self.validate_required_settings()
    
    def validate_required_settings(self):
        """
        Valida que las configuraciones críticas estén presentes.
        
        Raises:
            ValueError: Si falta alguna configuración crítica
        """
        # Para desarrollo, estas validaciones son opcionales
        # En producción, puedes descommentar las que necesites
        
        # if not self.BINANCE_API_KEY:
        #     raise ValueError("BINANCE_API_KEY no está configurada en variables de entorno")
        
        # if not self.BINANCE_API_SECRET:
        #     raise ValueError("BINANCE_API_SECRET no está configurada en variables de entorno")
        
        # if not self.GOOGLE_CLOUD_PROJECT_ID or self.GOOGLE_CLOUD_PROJECT_ID == 'tu-proyecto-id':
        #     raise ValueError("GOOGLE_CLOUD_PROJECT_ID no está configurada correctamente")
        
        pass  # Para desarrollo, permitir ejecutar sin todas las configuraciones
    
    def get_binance_config(self) -> dict:
        """
        Retorna la configuración para CCXT Binance.
        
        Returns:
            Diccionario con configuración de Binance
        """
        return {
            'apiKey': self.BINANCE_API_KEY,
            'secret': self.BINANCE_API_SECRET,
            'sandbox': False,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        }
    
    def get_bigquery_config(self) -> dict:
        """
        Retorna la configuración para BigQuery.
        
        Returns:
            Diccionario con configuración de BigQuery
        """
        return {
            'project_id': self.GOOGLE_CLOUD_PROJECT_ID,
            'credentials': self.GOOGLE_APPLICATION_CREDENTIALS
        }


# Instancia global de configuraciones
settings = Settings() 