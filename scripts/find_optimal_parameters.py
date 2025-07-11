#!/usr/bin/env python3
"""
Find Optimal Parameters - Script Principal
==========================================

Script para orquestar la simulación masiva y encontrar la mejor configuración
para la estrategia de grid trading con filtros de indicadores técnicos.

Este script:
1. Prepara el dataset maestro de 5 años (UNA SOLA VEZ)
2. Define el espacio de búsqueda de parámetros
3. Ejecuta simulaciones masivas con todas las combinaciones
4. Genera reporte final con los mejores resultados

Características:
- Simulación masiva con filtros ADX y volatilidad
- Búsqueda exhaustiva en el espacio de parámetros
- Reporte detallado de performance
- Optimización para ETH/USDT
"""

import os
import sys
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import logging

# Agregar el directorio padre al path para importar módulos del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar funciones y clases necesarias
from data_collector import fetch_and_prepare_data
from backtesting_engine import GridBotSimulator, DCABotSimulator, DCAShortSimulator
from shared.config.settings import settings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURACIÓN DEL ESPACIO DE BÚSQUEDA
# ============================================================================

# Universo de Monedas a Analizar (~20 pares representativos)
UNIVERSO_MONEDAS = [
    'BTC/USDT',     # Bitcoin - La más importante
    'ETH/USDT',     # Ethereum - DeFi leader
    'SOL/USDT',     # Solana - Alta performance
    'AVAX/USDT',    # Avalanche - Competitor
    'MATIC/USDT',   # Polygon - Layer 2
    'LINK/USDT',    # Chainlink - Oracle líder
    'DOT/USDT',     # Polkadot - Interoperabilidad
    'ADA/USDT',     # Cardano - Proof of Stake
    'DOGE/USDT',    # Dogecoin - Meme coin principal
    'XRP/USDT',     # Ripple - Pagos tradicionales
    'LTC/USDT',     # Litecoin - Bitcoin alternativo
    'BCH/USDT',     # Bitcoin Cash - Fork de Bitcoin
    'UNI/USDT',     # Uniswap - DEX líder
    'AAVE/USDT',    # AAVE - Lending protocol
    'CRV/USDT',     # Curve - Stablecoin DEX
    'MKR/USDT',     # Maker - DAI stablecoin
    'COMP/USDT',    # Compound - Lending
    'YFI/USDT',     # Yearn Finance - Yield farming
    'SUSHI/USDT',   # SushiSwap - DEX
    'NEAR/USDT'     # NEAR Protocol - Layer 1
]

# Rangos de parámetros para el "Panel de Instrumentos" - Nueva Metodología Grid Step
NIVELES_DE_GRID = [20, 30, 50, 80, 100, 150]        # Número de niveles del grid (ampliado)
PASO_DE_GRID_PERCENT = [0.4, 0.6, 0.8, 1.0, 1.5]   # Ganancia porcentual por operación
UMBRALES_ADX = [15, 20, 25, 30, 35]                 # Umbrales para filtro ADX
UMBRALES_VOLATILIDAD = [0.02, 0.025, 0.03, 0.035, 0.04]  # Umbrales de volatilidad
UMBRALES_SENTIMIENTO = [-0.2, -0.1, 0.0, 0.1]       # Umbrales para filtro de sentimiento

# Nuevos parámetros para estrategia BTD_SHORT (Bajista)
RIP_THRESHOLDS_PERCENT = [0.03, 0.05, 0.07, 0.10]   # Subida desde mínimo para vender: +3%, +5%, +7%, +10%
TAKE_PROFIT_THRESHOLDS_PERCENT = [0.03, 0.05, 0.08, 0.12]  # Caída para recomprar: -3%, -5%, -8%, -12%
UMBRALES_SENTIMIENTO_SHORT = [-0.1, 0.0, 0.1, 0.2]  # Sentimiento para BTD Short
INITIAL_CRYPTO_RATIOS = [0.5]                        # Porcentaje inicial en crypto (50% fijo)
SALE_AMOUNTS = [0.1]                                 # Cantidad fija por venta (10% del holdings inicial)

# Configuración fija
CONFIG_FIJA = {
    'commission': 0.001,        # 0.1% de comisión
    'initial_capital': 1000.0,  # $1000 inicial
    'timeframe': '4h'           # Removido 'pair' porque ahora iteramos sobre múltiples
}

# Configuración de fechas (5 años de datos)
FECHAS_CONFIG = {
    'start_date': datetime(2020, 1, 1),
    'end_date': datetime(2024, 12, 31),
    'timeframe': '1d'  # Timeframe por defecto para datos diarios
}

# Configuraciones específicas por estrategia
TIMEFRAMES_POR_ESTRATEGIA = {
    'GRID': '4h',        # Mayor granularidad para trades frecuentes
    'DCA_LONG': '1d',    # Estrategia a largo plazo
    'BTD_SHORT': '1d'    # Estrategia a largo plazo
}


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def calcular_combinaciones_totales_grid() -> int:
    """Calcula el número total de combinaciones para estrategia GRID."""
    total = len(NIVELES_DE_GRID) * len(PASO_DE_GRID_PERCENT) * len(UMBRALES_ADX) * len(UMBRALES_VOLATILIDAD) * len(UMBRALES_SENTIMIENTO)
    return total


def calcular_combinaciones_totales_dca() -> int:
    """Calcula el número total de combinaciones para estrategia DCA (Buy The Dip)."""
    # Nuevos parámetros para Buy The Dip
    dip_thresholds = [0.03, 0.05, 0.07, 0.10]  # -3%, -5%, -7%, -10%
    take_profit_thresholds = [0.05, 0.08, 0.10, 0.15]  # +5%, +8%, +10%, +15%
    sma_fast_periods = [50]  # SMA rápida
    sma_slow_periods = [200]  # SMA lenta
    
    return (len(dip_thresholds) * len(take_profit_thresholds) * 
            len(sma_fast_periods) * len(sma_slow_periods) * 
            len(UMBRALES_ADX) * len(UMBRALES_VOLATILIDAD) * len(UMBRALES_SENTIMIENTO))


def calcular_combinaciones_totales_btd_short() -> int:
    """Calcula el número total de combinaciones para estrategia BTD_SHORT."""
    sma_fast_periods = [50]   # SMA rápida (fijo)
    sma_slow_periods = [200]  # SMA lenta (fijo)
    
    return (len(RIP_THRESHOLDS_PERCENT) * len(TAKE_PROFIT_THRESHOLDS_PERCENT) * 
            len(sma_fast_periods) * len(sma_slow_periods) * 
            len(INITIAL_CRYPTO_RATIOS) * len(SALE_AMOUNTS) *
            len(UMBRALES_ADX) * len(UMBRALES_VOLATILIDAD) * len(UMBRALES_SENTIMIENTO_SHORT))


def crear_configuracion_grid(
    levels: int,
    paso_grid_percent: float,
    umbral_adx: float,
    umbral_volatilidad: float,
    umbral_sentimiento: float
) -> Dict[str, Any]:
    """
    Crea la configuración completa para una simulación.
    
    Args:
        levels: Número de niveles
        paso_grid_percent: Ganancia porcentual por operación
        umbral_adx: Umbral para filtro ADX
        umbral_volatilidad: Umbral para filtro de volatilidad
        umbral_sentimiento: Umbral para filtro de sentimiento
        
    Returns:
        Diccionario con configuración completa
    """
    return {
        'levels': levels,
        'paso_grid_percent': paso_grid_percent,
        'commission': CONFIG_FIJA['commission'],
        'initial_capital': CONFIG_FIJA['initial_capital'],
        'umbral_adx': umbral_adx,
        'umbral_volatilidad': umbral_volatilidad,
        'umbral_sentimiento': umbral_sentimiento
    }


def filtrar_dataframe_por_indicadores(
    df: pd.DataFrame,
    umbral_adx: float,
    umbral_volatilidad: float,
    umbral_sentimiento: float
) -> pd.DataFrame:
    """
    Filtra el DataFrame maestro según los umbrales de indicadores.
    
    Args:
        df: DataFrame maestro con todos los datos
        umbral_adx: Umbral para filtro ADX
        umbral_volatilidad: Umbral para filtro de volatilidad
        umbral_sentimiento: Umbral para filtro de sentimiento
        
    Returns:
        DataFrame filtrado
    """
    # Aplicar filtros de indicadores técnicos
    df_filtrado = df[
        (df['adx'] < umbral_adx) &  # ADX bajo = mercado lateral
        (df['volatility'] > umbral_volatilidad) &  # Alta volatilidad
        (df['sentiment_ma7'] > umbral_sentimiento)  # Sentimiento positivo
    ].copy()
    
    return df_filtrado  # type: ignore


def ejecutar_simulacion_unica(
    df_maestro: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Ejecuta una simulación única con la configuración dada.
    LÓGICA CORREGIDA: Pasa el DataFrame completo al simulador junto con parámetros de filtrado.
    El simulador debe decidir día a día si operar o no basándose en los filtros.
    
    Args:
        df_maestro: DataFrame maestro con todos los datos históricos (sin filtrar)
        config: Configuración de la simulación incluyendo parámetros de filtrado
        
    Returns:
        Resultados de la simulación
    """
    try:
        # ELIMINADO: Pre-filtrado de datos históricos (bug corregido)
        # Ya no filtramos df_maestro antes de pasarlo al simulador
        
        # ELIMINADO: Verificación de datos insuficientes (ya no aplica)
        # El simulador recibirá el DataFrame completo
        
        # Calcular el rango dinámicamente según la nueva metodología Grid Step
        # Fórmula: rango_total_calculado = niveles_actuales * (paso_actual / 100)
        rango_total_calculado = config['levels'] * (config['paso_grid_percent'] / 100)
        
        # Crear configuración completa para GridBotSimulator
        # NUEVO: Incluir parámetros de filtrado para que el simulador los aplique internamente
        grid_config = {
            'range_percent': rango_total_calculado,
            'levels': config['levels'],
            'commission': config['commission'],
            'initial_capital': config['initial_capital'],
            # NUEVOS PARÁMETROS DE FILTRADO (para aplicar día a día)
            'umbral_adx': config['umbral_adx'],
            'umbral_volatilidad': config['umbral_volatilidad'],
            'umbral_sentimiento': config['umbral_sentimiento']
        }
        
        # Ejecutar simulación con DataFrame completo
        simulator = GridBotSimulator(df_maestro, grid_config)
        results = simulator.run_simulation()
        
        # Agregar información de configuración
        results.update({
            'levels': config['levels'],
            'paso_grid_percent': config['paso_grid_percent'],
            'rango_calculado': rango_total_calculado,
            'umbral_adx': config['umbral_adx'],
            'umbral_volatilidad': config['umbral_volatilidad'],
            'umbral_sentimiento': config['umbral_sentimiento'],
            'datos_totales': len(df_maestro),
            'datos_utilizados_simulacion': len(df_maestro)  # Ahora siempre usa todos los datos
        })
        
        return results
        
    except Exception as e:
        logger.error(f"Error en simulación: {e}")
        return {
            'error': str(e),
            'config': config
        }


def ejecutar_simulacion_dca(
    df_maestro: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Ejecuta una simulación Buy The Dip usando DCABotSimulator.
    LÓGICA CORREGIDA: Pasa el DataFrame completo al simulador junto con parámetros de filtrado.
    El simulador debe decidir día a día si operar o no basándose en los filtros.
    
    Args:
        df_maestro: DataFrame maestro con todos los datos históricos (sin filtrar)
        config: Configuración de Buy The Dip (dip_threshold, take_profit, filtros)
        
    Returns:
        Resultados de la simulación Buy The Dip
    """
    try:
        # ELIMINADO: Pre-filtrado de datos históricos (bug corregido)
        # Ya no filtramos df_maestro antes de pasarlo al simulador
        
        # ELIMINADO: Verificación de datos insuficientes (ya no aplica)
        # El simulador recibirá el DataFrame completo
        
        # Crear configuración completa para DCABotSimulator (Buy The Dip)
        # NUEVO: Incluir parámetros de filtrado para que el simulador los aplique internamente
        dca_config = {
            'dip_threshold': config['dip_threshold'],
            'take_profit_threshold': config['take_profit_threshold'],
            'purchase_amount': config['monto_dca'],
            'sma_fast': config['sma_fast'],
            'sma_slow': config['sma_slow'],
            'lookback_days': config.get('lookback_days', 20),  # Días para calcular máximo
            'commission': config['commission'],
            'initial_capital': config['initial_capital'],
            # NUEVOS PARÁMETROS DE FILTRADO (para aplicar día a día)
            'umbral_adx': config['umbral_adx'],
            'umbral_volatilidad': config['umbral_volatilidad'],
            'umbral_sentimiento': config['umbral_sentimiento']
        }
        
        # Ejecutar simulación con DataFrame completo
        simulator = DCABotSimulator(df_maestro, dca_config)
        results = simulator.run_simulation()
        
        # Agregar información de configuración
        results.update({
            'strategy_type': 'Buy_The_Dip',
            'dip_threshold': config['dip_threshold'],
            'take_profit_threshold': config['take_profit_threshold'],
            'monto_dca': config['monto_dca'],
            'sma_fast': config['sma_fast'],
            'sma_slow': config['sma_slow'],
            'umbral_adx': config['umbral_adx'],
            'umbral_volatilidad': config['umbral_volatilidad'],
            'umbral_sentimiento': config['umbral_sentimiento'],
            'datos_totales': len(df_maestro),
            'datos_utilizados_simulacion': len(df_maestro)  # Ahora siempre usa todos los datos
        })
        
        return results
        
    except Exception as e:
        logger.error(f"Error en simulación Buy The Dip: {e}")
        return {
            'error': str(e),
            'config': config
        }


def ejecutar_simulacion_btd_short(
    df_maestro: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Ejecuta una simulación BTD Short usando DCAShortSimulator.
    LÓGICA CORREGIDA: Pasa el DataFrame completo al simulador junto con parámetros de filtrado.
    El simulador debe decidir día a día si operar o no basándose en los filtros.
    
    Args:
        df_maestro: DataFrame maestro con todos los datos históricos (sin filtrar)
        config: Configuración de BTD Short (rip_threshold, take_profit, filtros)
        
    Returns:
        Resultados de la simulación BTD Short
    """
    try:
        # ELIMINADO: Pre-filtrado de datos históricos (bug corregido)
        # Ya no filtramos df_maestro antes de pasarlo al simulador
        
        # ELIMINADO: Verificación de datos insuficientes (ya no aplica)
        # El simulador recibirá el DataFrame completo
        
        # Crear configuración completa para DCAShortSimulator (BTD Short)
        # NUEVO: Incluir parámetros de filtrado para que el simulador los aplique internamente
        short_config = {
            'rip_threshold': config['rip_threshold'],
            'take_profit_threshold': config['take_profit_threshold'],
            'sale_amount': config['sale_amount'],
            'initial_crypto_ratio': config['initial_crypto_ratio'],
            'sma_fast': config['sma_fast'],
            'sma_slow': config['sma_slow'],
            'lookback_days': config.get('lookback_days', 20),  # Días para calcular mínimo
            'commission': config['commission'],
            'initial_capital': config['initial_capital'],
            # NUEVOS PARÁMETROS DE FILTRADO (para aplicar día a día)
            'umbral_adx': config['umbral_adx'],
            'umbral_volatilidad': config['umbral_volatilidad'],
            'umbral_sentimiento': config['umbral_sentimiento']
        }
        
        # Ejecutar simulación con DataFrame completo
        simulator = DCAShortSimulator(df_maestro, short_config)
        results = simulator.run_simulation()
        
        # Agregar información de configuración
        results.update({
            'strategy_type': 'BTD_SHORT',
            'rip_threshold': config['rip_threshold'],
            'take_profit_threshold': config['take_profit_threshold'],
            'sale_amount': config['sale_amount'],
            'initial_crypto_ratio': config['initial_crypto_ratio'],
            'sma_fast': config['sma_fast'],
            'sma_slow': config['sma_slow'],
            'umbral_adx': config['umbral_adx'],
            'umbral_volatilidad': config['umbral_volatilidad'],
            'umbral_sentimiento': config['umbral_sentimiento'],
            'datos_totales': len(df_maestro),
            'datos_utilizados_simulacion': len(df_maestro)  # Ahora siempre usa todos los datos
        })
        
        return results
        
    except Exception as e:
        logger.error(f"Error en simulación BTD Short: {e}")
        return {
            'error': str(e),
            'config': config
        }


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def generar_reporte_final(resultados: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Genera el reporte final con los mejores resultados.
    
    Args:
        resultados: Lista con todos los resultados de las simulaciones
        
    Returns:
        DataFrame con los resultados ordenados por performance
    """
    logger.info("📊 Generando reporte final...")
    
    # Filtrar resultados exitosos (sin errores)
    resultados_exitosos = [r for r in resultados if 'error' not in r]
    
    if not resultados_exitosos:
        logger.error("❌ No hay resultados exitosos para analizar")
        return pd.DataFrame()
    
    # Convertir a DataFrame
    df_resultados = pd.DataFrame(resultados_exitosos)
    
    # Ordenar por retorno porcentual (descendente)
    df_resultados = df_resultados.sort_values('return_percentage', ascending=False)
    
    # Agregar ranking
    df_resultados['ranking'] = range(1, len(df_resultados) + 1)
    
    # Mostrar top 10
    logger.info("\n🏆 TOP 10 MEJORES CONFIGURACIONES:")
    logger.info("=" * 80)
    
    for i, row in df_resultados.head(10).iterrows():
        strategy_type = row.get('strategy_type', 'GRID')
        
        if strategy_type == 'GRID':
            # Formateo para estrategia GRID
            logger.info(f"#{int(row['ranking']):2d} | "
                       f"Retorno: {row['return_percentage']:6.2f}% | "
                       f"GRID: {int(row['levels'])} niveles, Paso {row['paso_grid_percent']}% (Rango {row['rango_calculado']:.1f}%) | "
                       f"ADX < {row['umbral_adx']:2.0f}, Vol > {row['umbral_volatilidad']:.3f} | "
                       f"Sentimiento: {row['umbral_sentimiento']:.2f} | "
                       f"Trades: {int(row['total_trades']):3d} | "
                       f"Drawdown: {row['max_drawdown']:5.2f}%")
        elif strategy_type == 'Buy_The_Dip':
            # Formateo para estrategia Buy The Dip
            logger.info(f"#{int(row['ranking']):2d} | "
                       f"Retorno: {row['return_percentage']:6.2f}% | "
                       f"BUY_DIP: Dip {row['dip_threshold']*100:.1f}%, TP {row['take_profit_threshold']*100:.1f}%, SMA {row['sma_fast']}/{row['sma_slow']} | "
                       f"ADX < {row['umbral_adx']:2.0f}, Vol > {row['umbral_volatilidad']:.3f} | "
                       f"Sentimiento: {row['umbral_sentimiento']:.2f} | "
                       f"Trades: {int(row['total_trades']):3d} | "
                       f"Drawdown: {row['max_drawdown']:5.2f}%")
        elif strategy_type == 'BTD_SHORT':
            # Formateo para estrategia BTD Short
            logger.info(f"#{int(row['ranking']):2d} | "
                       f"Retorno: {row['return_percentage']:6.2f}% | "
                       f"BTD_SHORT: RIP {row['rip_threshold']*100:.1f}%, TP {row['take_profit_threshold']*100:.1f}%, SMA {row['sma_fast']}/{row['sma_slow']} | "
                       f"ADX < {row['umbral_adx']:2.0f}, Vol > {row['umbral_volatilidad']:.3f} | "
                       f"Sentimiento: {row['umbral_sentimiento']:.2f} | "
                       f"Trades: {int(row['total_trades']):3d} | "
                       f"Drawdown: {row['max_drawdown']:5.2f}%")
    
    return df_resultados


def guardar_resultados_bigquery(
    df_resultados: pd.DataFrame, 
    pair: str, 
    project_id: Optional[str] = None,
    dataset: str = 'oraculo_data',
    table: str = 'estrategias_optimizadas'
) -> bool:
    """
    Guarda los resultados en Google BigQuery usando la nueva estructura JSON.
    
    Nueva estructura de tabla:
    - simulation_id: STRING (UUID único)
    - run_timestamp: TIMESTAMP (cuándo se corrió)
    - coin_pair: STRING (ej. 'ETH/USDT')
    - strategy_type: STRING ('GRID', 'DCA_LONG', 'BTD_SHORT')
    - parameters: JSON (objeto con parámetros específicos de cada estrategia)
    - result_pnl_percent: FLOAT
    - result_sharpe_ratio: FLOAT
    - result_max_drawdown: FLOAT
    - result_total_trades: INTEGER
    
    Args:
        df_resultados: DataFrame con los resultados
        pair: Par de moneda analizado (ej: 'BTC/USDT')
        project_id: ID del proyecto de Google Cloud
        dataset: Nombre del dataset en BigQuery
        table: Nombre de la tabla en BigQuery
        
    Returns:
        True si fue exitoso, False en caso de error
    """
    try:
        import uuid
        import pandas_gbq
        import json
        
        # Usar project_id de configuración si no se proporciona
        if project_id is None:
            project_id = settings.GOOGLE_CLOUD_PROJECT_ID
            
        # Validar que tenemos un project_id válido
        if not project_id or project_id == 'tu-proyecto-id':
            logger.error("❌ Error: GOOGLE_CLOUD_PROJECT_ID no está configurado correctamente")
            return False
        
        if df_resultados.empty:
            logger.warning("⚠️ DataFrame vacío, no hay datos para guardar")
            return True
        
        # Crear DataFrame normalizado para BigQuery
        df_normalizado = pd.DataFrame()
        run_timestamp = datetime.now()
        
        for index, row in df_resultados.iterrows():
            # Generar ID único para cada simulación
            simulation_id = str(uuid.uuid4())
            
            # Mapear strategy_type a nomenclatura nueva
            strategy_original = row.get('strategy_type', 'GRID')
            if strategy_original == 'GRID':
                strategy_type = 'GRID'
            elif strategy_original == 'Buy_The_Dip':
                strategy_type = 'DCA_LONG'  # Buy The Dip Long
            elif strategy_original == 'BTD_SHORT':
                strategy_type = 'BTD_SHORT'  # Buy The Dip Short
            else:
                strategy_type = strategy_original
            
            # Crear objeto JSON de parámetros según el tipo de estrategia
            parameters = {}
            
            if strategy_type == 'GRID':
                parameters = {
                    "grid_levels": int(row.get('levels') or 0),
                    "grid_step_percent": float(row.get('paso_grid_percent') or 0.0),
                    "adx_threshold": int(row.get('umbral_adx') or 0),
                    "volatility_threshold": float(row.get('umbral_volatilidad') or 0.0),
                    "sentiment_threshold": float(row.get('umbral_sentimiento') or 0.0)
                }
            elif strategy_type == 'DCA_LONG':
                parameters = {
                    "dip_threshold_percent": float(row.get('dip_threshold') or 0.0),
                    "take_profit_percent": float(row.get('take_profit_threshold') or 0.0),
                    "sma_fast": int(row.get('sma_fast') or 50),
                    "sma_slow": int(row.get('sma_slow') or 200),
                    "adx_threshold": int(row.get('umbral_adx') or 0),
                    "volatility_threshold": float(row.get('umbral_volatilidad') or 0.0),
                    "sentiment_threshold": float(row.get('umbral_sentimiento') or 0.0)
                }
            elif strategy_type == 'BTD_SHORT':
                parameters = {
                    "rip_threshold_percent": float(row.get('rip_threshold') or 0.0),
                    "take_profit_percent": float(row.get('take_profit_threshold') or 0.0),
                    "sma_fast": int(row.get('sma_fast') or 50),
                    "sma_slow": int(row.get('sma_slow') or 200),
                    "initial_crypto_ratio": float(row.get('initial_crypto_ratio') or 0.5),
                    "sale_amount": float(row.get('sale_amount') or 0.1),
                    "adx_threshold": int(row.get('umbral_adx') or 0),
                    "volatility_threshold": float(row.get('umbral_volatilidad') or 0.0),
                    "sentiment_threshold": float(row.get('umbral_sentimiento') or 0.0)
                }
            
            # Crear registro con nueva estructura
            registro = {
                'simulation_id': simulation_id,
                'run_timestamp': run_timestamp,
                'coin_pair': pair,
                'strategy_type': strategy_type,
                'parameters': json.dumps(parameters),  # Convertir dict a JSON string
                'result_pnl_percent': float(row.get('return_percentage') or 0.0),
                'result_sharpe_ratio': float(row.get('sharpe_ratio') or 0.0),
                'result_max_drawdown': float(row.get('max_drawdown') or 0.0),
                'result_total_trades': int(row.get('total_trades') or 0)
            }
            
            # Agregar registro al DataFrame normalizado
            df_normalizado = pd.concat([df_normalizado, pd.DataFrame([registro])], ignore_index=True)
        
        # Tabla completa
        tabla_completa = f"{project_id}.{dataset}.{table}"
        
        # Guardar en BigQuery con append
        pandas_gbq.to_gbq(
            df_normalizado,
            destination_table=tabla_completa,
            project_id=project_id,
            if_exists='append',
            progress_bar=False
        )
        
        logger.info(f"💾 Guardado exitoso: {len(df_normalizado)} registros para {pair} en BigQuery (estructura JSON)")
        logger.info(f"📊 Estrategias guardadas: {df_normalizado['strategy_type'].value_counts().to_dict()}")
        logger.info(f"🗃️ Tabla: {tabla_completa}")
        return True
        
    except ImportError:
        logger.error("❌ Error: pandas-gbq no está instalado. Ejecuta: pip install pandas-gbq")
        return False
    except Exception as e:
        logger.error(f"❌ Error guardando en BigQuery: {e}")
        return False


def ejecutar_backtesting_para_moneda(pair: str, sentiment_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Ejecuta el backtesting completo para una moneda específica.
    
    Args:
        pair: Par de moneda a analizar (ej: 'BTC/USDT')
        sentiment_data: DataFrame de sentimientos precargado (opcional)
        
    Returns:
        Dict con resultados del backtesting
    """
    try:
        logger.info(f"📊 Iniciando análisis para {pair}...")
        
        # Fase 1: Ejecutar estrategia GRID (timeframe 4h)
        logger.info(f"🔄 Backtesting de estrategia GRID para {pair} (timeframe: 4h)...")
        
        # Obtener datos específicos para GRID (4h)
        if sentiment_data is not None:
            from data_collector import fetch_and_prepare_data_optimized
            df_grid = fetch_and_prepare_data_optimized(
                pair=pair,
                timeframe=TIMEFRAMES_POR_ESTRATEGIA['GRID'],
                start_date=FECHAS_CONFIG['start_date'],
                end_date=FECHAS_CONFIG['end_date'],
                sentiment_data=sentiment_data
            )
        else:
            from data_collector import fetch_and_prepare_data
            df_grid = fetch_and_prepare_data(
                pair=pair,
                timeframe=TIMEFRAMES_POR_ESTRATEGIA['GRID'],
                start_date=FECHAS_CONFIG['start_date'],
                end_date=FECHAS_CONFIG['end_date']
            )
        
        if df_grid.empty:
            logger.error(f"❌ No hay datos suficientes para GRID {pair}")
            configuraciones_grid = 0
            df_grid_final = pd.DataFrame()
        else:
            logger.info(f"✅ Datos GRID descargados para {pair}: {len(df_grid)} registros (4h)")
            resultados_grid = ejecutar_optimizacion_grid(df_grid)
            df_grid_final = generar_reporte_final(resultados_grid)
            
            # Agregar tipo de estrategia
            if not df_grid_final.empty:
                df_grid_final['strategy_type'] = 'GRID'
            
            configuraciones_grid = len(df_grid_final)
            logger.info(f"✅ Backtesting de estrategia GRID para {pair} completado. {configuraciones_grid} configuraciones probadas.")
        
        # Fase 2: Ejecutar estrategias DCA_LONG y BTD_SHORT (timeframe 1d)
        logger.info(f"🔄 Backtesting de estrategias DCA_LONG y BTD_SHORT para {pair} (timeframe: 1d)...")
        
        # Obtener datos específicos para DCA_LONG y BTD_SHORT (1d)
        if sentiment_data is not None:
            from data_collector import fetch_and_prepare_data_optimized
            df_daily = fetch_and_prepare_data_optimized(
                pair=pair,
                timeframe=TIMEFRAMES_POR_ESTRATEGIA['DCA_LONG'],
                start_date=FECHAS_CONFIG['start_date'],
                end_date=FECHAS_CONFIG['end_date'],
                sentiment_data=sentiment_data
            )
        else:
            from data_collector import fetch_and_prepare_data
            df_daily = fetch_and_prepare_data(
                pair=pair,
                timeframe=TIMEFRAMES_POR_ESTRATEGIA['DCA_LONG'],
                start_date=FECHAS_CONFIG['start_date'],
                end_date=FECHAS_CONFIG['end_date']
            )
        
        if df_daily.empty:
            logger.error(f"❌ No hay datos suficientes para DCA/BTD_SHORT {pair}")
            configuraciones_dca = 0
            configuraciones_btd_short = 0
            df_dca_final = pd.DataFrame()
            df_btd_short_final = pd.DataFrame()
        else:
            logger.info(f"✅ Datos Daily descargados para {pair}: {len(df_daily)} registros (1d)")
            
            # Ejecutar DCA_LONG
            logger.info(f"📈 Procesando estrategia Buy The Dip (DCA_LONG)...")
            resultados_dca = ejecutar_optimizacion_dca(df_daily)
            df_dca_final = generar_reporte_final(resultados_dca)
            
            # Agregar tipo de estrategia
            if not df_dca_final.empty:
                df_dca_final['strategy_type'] = 'Buy_The_Dip'
                
            configuraciones_dca = len(df_dca_final)
            logger.info(f"✅ Backtesting de estrategia Buy The Dip completado. {configuraciones_dca} configuraciones probadas.")
            
            # Ejecutar BTD_SHORT
            logger.info(f"📉 Procesando estrategia BTD Short...")
            resultados_btd_short = ejecutar_optimizacion_btd_short(df_daily)
            df_btd_short_final = generar_reporte_final(resultados_btd_short)
            
            # Agregar tipo de estrategia
            if not df_btd_short_final.empty:
                df_btd_short_final['strategy_type'] = 'BTD_SHORT'
                
            configuraciones_btd_short = len(df_btd_short_final)
            logger.info(f"✅ Backtesting de estrategia BTD Short completado. {configuraciones_btd_short} configuraciones probadas.")
        
        # Fase 3: Combinar resultados
        df_combinado = pd.concat([df_grid_final, df_dca_final, df_btd_short_final], ignore_index=True)
        total_configuraciones = len(df_combinado)
        
        # Fase 4: Guardar en BigQuery
        if not df_combinado.empty:
            logger.info(f"💾 Guardando {total_configuraciones} resultados de {pair} en BigQuery...")
            exito = guardar_resultados_bigquery(df_combinado, pair)
            
            if exito:
                logger.info(f"✅ Análisis de {pair} finalizado. Datos guardados en BigQuery.")
            else:
                logger.error(f"❌ Error guardando datos de {pair} en BigQuery")
                return {
                    'pair': pair,
                    'success': False,
                    'error': 'Error guardando en BigQuery'
                }
        else:
            logger.warning(f"⚠️ No hay resultados válidos para {pair}")
            return {
                'pair': pair,
                'success': False,
                'error': 'No hay resultados válidos'
            }
        
        return {
            'pair': pair,
            'success': True,
            'configuraciones_grid': configuraciones_grid,
            'configuraciones_dca': configuraciones_dca,
            'configuraciones_btd_short': configuraciones_btd_short,
            'total_configuraciones': total_configuraciones,
            'registros_grid_4h': len(df_grid) if not df_grid.empty else 0,
            'registros_daily_1d': len(df_daily) if not df_daily.empty else 0
        }
        
    except Exception as e:
        logger.error(f"❌ Error procesando {pair}: {e}")
        return {
            'pair': pair,
            'success': False,
            'error': str(e)
        }


def ejecutar_optimizacion_grid(df_maestro: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Ejecuta la optimización masiva para estrategia GRID.
    
    Args:
        df_maestro: DataFrame maestro con todos los datos
        
    Returns:
        Lista con todos los resultados de las simulaciones GRID
    """
    # Generar todas las combinaciones para GRID
    combinaciones = list(itertools.product(
        NIVELES_DE_GRID,
        PASO_DE_GRID_PERCENT,
        UMBRALES_ADX,
        UMBRALES_VOLATILIDAD,
        UMBRALES_SENTIMIENTO
    ))
    
    resultados = []
    
    # Ejecutar simulaciones GRID
    for levels, paso_grid_percent, umbral_adx, umbral_volatilidad, umbral_sentimiento in combinaciones:
        config = crear_configuracion_grid(levels, paso_grid_percent, umbral_adx, umbral_volatilidad, umbral_sentimiento)
        resultado = ejecutar_simulacion_unica(df_maestro, config)
        
        # Agregar tipo de estrategia
        if 'error' not in resultado:
            resultado['strategy_type'] = 'GRID'
            
        resultados.append(resultado)
    
    return resultados


def ejecutar_optimizacion_dca(df_maestro: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Ejecuta la optimización masiva para estrategia Buy The Dip.
    
    Args:
        df_maestro: DataFrame maestro con todos los datos
        
    Returns:
        Lista con todos los resultados de las simulaciones Buy The Dip
    """
    # Nuevos parámetros para Buy The Dip
    dip_thresholds = [0.03, 0.05, 0.07, 0.10]  # -3%, -5%, -7%, -10%
    take_profit_thresholds = [0.05, 0.08, 0.10, 0.15]  # +5%, +8%, +10%, +15%
    sma_fast_periods = [50]  # SMA rápida
    sma_slow_periods = [200]  # SMA lenta
    montos_dca = [100]  # USD por compra (fijo para Buy The Dip)
    
    # Generar todas las combinaciones para Buy The Dip
    combinaciones = list(itertools.product(
        dip_thresholds,
        take_profit_thresholds,
        sma_fast_periods,
        sma_slow_periods,
        montos_dca,
        UMBRALES_ADX,
        UMBRALES_VOLATILIDAD,
        UMBRALES_SENTIMIENTO
    ))
    
    resultados = []
    
    # Ejecutar simulaciones Buy The Dip
    for dip_threshold, take_profit_threshold, sma_fast, sma_slow, monto, umbral_adx, umbral_volatilidad, umbral_sentimiento in combinaciones:
        config = {
            'dip_threshold': dip_threshold,
            'take_profit_threshold': take_profit_threshold,
            'sma_fast': sma_fast,
            'sma_slow': sma_slow,
            'monto_dca': monto,
            'commission': CONFIG_FIJA['commission'],
            'initial_capital': CONFIG_FIJA['initial_capital'],
            'umbral_adx': umbral_adx,
            'umbral_volatilidad': umbral_volatilidad,
            'umbral_sentimiento': umbral_sentimiento
        }
        
        resultado = ejecutar_simulacion_dca(df_maestro, config)
        resultados.append(resultado)
    
    return resultados


def ejecutar_optimizacion_btd_short(df_maestro: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Ejecuta la optimización masiva para estrategia BTD Short.
    
    Args:
        df_maestro: DataFrame maestro con todos los datos
        
    Returns:
        Lista con todos los resultados de las simulaciones BTD Short
    """
    # Nuevos parámetros para BTD Short
    rip_thresholds = [0.03, 0.05, 0.07, 0.10]  # Subida desde mínimo para vender: +3%, +5%, +7%, +10%
    take_profit_thresholds = [0.03, 0.05, 0.08, 0.12]  # Caída para recomprar: -3%, -5%, -8%, -12%
    sma_fast_periods = [50]  # SMA rápida (fijo)
    sma_slow_periods = [200]  # SMA lenta (fijo)
    initial_crypto_ratios = [0.5]  # Porcentaje inicial en crypto (50% fijo)
    sale_amounts = [0.1]  # Cantidad fija por venta (10% del holdings inicial)
    
    # Generar todas las combinaciones para BTD Short
    combinaciones = list(itertools.product(
        rip_thresholds,
        take_profit_thresholds,
        sma_fast_periods,
        sma_slow_periods,
        initial_crypto_ratios,
        sale_amounts,
        UMBRALES_ADX,
        UMBRALES_VOLATILIDAD,
        UMBRALES_SENTIMIENTO_SHORT
    ))
    
    resultados = []
    
    # Ejecutar simulaciones BTD Short
    for rip_threshold, take_profit_threshold, sma_fast, sma_slow, initial_crypto_ratio, sale_amount, umbral_adx, umbral_volatilidad, umbral_sentimiento_short in combinaciones:
        config = {
            'rip_threshold': rip_threshold,
            'take_profit_threshold': take_profit_threshold,
            'sma_fast': sma_fast,
            'sma_slow': sma_slow,
            'initial_crypto_ratio': initial_crypto_ratio,
            'sale_amount': sale_amount,
            'commission': CONFIG_FIJA['commission'],
            'initial_capital': CONFIG_FIJA['initial_capital'],
            'umbral_adx': umbral_adx,
            'umbral_volatilidad': umbral_volatilidad,
            'umbral_sentimiento': umbral_sentimiento_short
        }
        
        resultado = ejecutar_simulacion_btd_short(df_maestro, config)
        resultados.append(resultado)
    
    return resultados


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main() -> None:
    """
    Función principal que orquesta todo el proceso de optimización masiva.
    Itera sobre múltiples monedas y estrategias, guardando resultados en BigQuery.
    """
    logger.info("🎯 INICIANDO PROCESO DE OPTIMIZACIÓN MASIVA DE ESTRATEGIAS")
    logger.info("=" * 80)
    
    # Mostrar configuración inicial
    logger.info(f"📊 Universo de monedas: {len(UNIVERSO_MONEDAS)} pares")
    logger.info(f"🎯 Estrategias: GRID, Buy The Dip y BTD Short")
    logger.info(f"📈 Combinaciones GRID: {calcular_combinaciones_totales_grid()}")
    logger.info(f"📉 Combinaciones Buy The Dip: {calcular_combinaciones_totales_dca()}")
    logger.info(f"📉 Combinaciones BTD Short: {calcular_combinaciones_totales_btd_short()}")
    logger.info(f"📅 Período de análisis: {FECHAS_CONFIG['start_date']} - {FECHAS_CONFIG['end_date']}")
    logger.info("=" * 80)
    
    resultados_finales = []
    inicio_tiempo_total = time.time()
    
    try:
        # OPTIMIZACIÓN: Descargar sentimientos una sola vez al inicio
        logger.info("📰 Descargando datos de sentimiento (una sola vez para todos los pares)...")
        from data_collector import fetch_sentiment_data
        sentiment_data_global = fetch_sentiment_data(
            start_date=FECHAS_CONFIG['start_date'],
            end_date=FECHAS_CONFIG['end_date']
        )
        
        if not sentiment_data_global.empty:
            logger.info(f"✅ Sentimientos descargados: {len(sentiment_data_global)} registros")
            logger.info("🚀 Este dataset se reutilizará para todas las monedas, ahorrando tiempo y recursos")
        else:
            logger.warning("⚠️ No se encontraron datos de sentimiento para el período")
            sentiment_data_global = None
        
        # Iterar sobre cada moneda del universo
        for i, pair in enumerate(UNIVERSO_MONEDAS, 1):
            logger.info(f"\n🚀 [{i}/{len(UNIVERSO_MONEDAS)}] Procesando {pair}...")
            
            # Pasar los sentimientos compartidos a cada par
            resultado_moneda = ejecutar_backtesting_para_moneda(pair, sentiment_data_global)
            resultados_finales.append(resultado_moneda)
            
            if resultado_moneda['success']:
                logger.info(f"✅ {pair} completado exitosamente")
                logger.info(f"   📊 Configuraciones GRID: {resultado_moneda['configuraciones_grid']}")
                logger.info(f"   📉 Configuraciones Buy The Dip: {resultado_moneda['configuraciones_dca']}")
                logger.info(f"   📉 Configuraciones BTD Short: {resultado_moneda['configuraciones_btd_short']}")
                logger.info(f"   💾 Total guardado: {resultado_moneda['total_configuraciones']}")
            else:
                logger.error(f"❌ {pair} falló: {resultado_moneda.get('error', 'Error desconocido')}")
            
            # Pequeña pausa entre monedas para evitar rate limits
            if i < len(UNIVERSO_MONEDAS):
                logger.info(f"⏳ Pasando a siguiente moneda en 2 segundos...")
                time.sleep(2)
        
        # Reporte final
        tiempo_total = time.time() - inicio_tiempo_total
        exitosos = [r for r in resultados_finales if r['success']]
        fallidos = [r for r in resultados_finales if not r['success']]
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 PROCESO MASIVO COMPLETADO")
        logger.info("=" * 80)
        logger.info(f"⏱️  Tiempo total: {tiempo_total/60:.1f} minutos")
        logger.info(f"✅ Monedas procesadas exitosamente: {len(exitosos)}/{len(UNIVERSO_MONEDAS)}")
        logger.info(f"❌ Monedas con errores: {len(fallidos)}")
        
        if exitosos:
            total_configuraciones = sum(r['total_configuraciones'] for r in exitosos)
            logger.info(f"📊 Total configuraciones analizadas: {total_configuraciones}")
            logger.info(f"💾 Todos los resultados han sido guardados en BigQuery tabla 'estrategias_optimizadas'")
            logger.info(f"⚡ OPTIMIZACIÓN: Sentimientos descargados una sola vez y reutilizados {len(exitosos)} veces")
        
        if fallidos:
            logger.info(f"\n❌ Monedas que fallaron:")
            for r in fallidos:
                logger.info(f"   • {r['pair']}: {r.get('error', 'Error desconocido')}")
                
        logger.info("\n🏆 PROCESO MASIVO FINALIZADO EXITOSAMENTE")
        
    except Exception as e:
        logger.error(f"❌ Error crítico en el proceso masivo: {e}")
        raise


if __name__ == "__main__":
    main() 