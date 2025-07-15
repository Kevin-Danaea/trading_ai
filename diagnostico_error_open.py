#!/usr/bin/env python3
"""
Diagnóstico del Error 'open' en Backtesting de Futuros
=====================================================

Script para identificar y corregir el problema con el campo 'open'
en el backtesting de futuros.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Configura logging para el diagnóstico."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('diagnostico_error_open.log', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

def crear_datos_prueba():
    """Crea datos de prueba para diagnosticar el problema."""
    logger = logging.getLogger(__name__)
    
    # Crear datos de prueba con diferentes formatos
    fechas = pd.date_range(start='2024-01-01', end='2024-06-01', freq='D')
    
    # Formato 1: Columnas en minúsculas
    datos_1 = pd.DataFrame({
        'open': np.random.uniform(100, 200, len(fechas)),
        'high': np.random.uniform(200, 300, len(fechas)),
        'low': np.random.uniform(50, 100, len(fechas)),
        'close': np.random.uniform(100, 200, len(fechas)),
        'volume': np.random.uniform(1000, 10000, len(fechas))
    }, index=fechas)
    
    # Formato 2: Columnas en mayúsculas
    datos_2 = pd.DataFrame({
        'Open': np.random.uniform(100, 200, len(fechas)),
        'High': np.random.uniform(200, 300, len(fechas)),
        'Low': np.random.uniform(50, 100, len(fechas)),
        'Close': np.random.uniform(100, 200, len(fechas)),
        'Volume': np.random.uniform(1000, 10000, len(fechas))
    }, index=fechas)
    
    # Formato 3: Columnas mixtas
    datos_3 = pd.DataFrame({
        'OPEN': np.random.uniform(100, 200, len(fechas)),
        'HIGH': np.random.uniform(200, 300, len(fechas)),
        'LOW': np.random.uniform(50, 100, len(fechas)),
        'CLOSE': np.random.uniform(100, 200, len(fechas)),
        'VOLUME': np.random.uniform(1000, 10000, len(fechas))
    }, index=fechas)
    
    logger.info(f"✅ Datos de prueba creados:")
    logger.info(f"   Formato 1 (minúsculas): {datos_1.shape}")
    logger.info(f"   Formato 2 (mayúsculas): {datos_2.shape}")
    logger.info(f"   Formato 3 (MAYÚSCULAS): {datos_3.shape}")
    
    return datos_1, datos_2, datos_3

def diagnosticar_formato_datos(df: pd.DataFrame, nombre: str):
    """Diagnostica el formato de los datos."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"\n🔍 Diagnóstico de {nombre}:")
    logger.info(f"   Columnas: {list(df.columns)}")
    logger.info(f"   Tipos: {df.dtypes.to_dict()}")
    logger.info(f"   Shape: {df.shape}")
    logger.info(f"   Índice: {type(df.index)}")
    
    # Verificar columnas OHLC
    columnas_ohlc = ['open', 'high', 'low', 'close']
    columnas_ohlc_upper = ['Open', 'High', 'Low', 'Close']
    columnas_ohlc_upper_all = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
    
    # Verificar en minúsculas
    columnas_lower = [col.lower() for col in df.columns]
    ohlc_encontradas_lower = [col for col in columnas_ohlc if col in columnas_lower]
    
    # Verificar en mayúsculas
    columnas_upper = [col for col in df.columns]
    ohlc_encontradas_upper = [col for col in columnas_ohlc_upper if col in columnas_upper]
    
    # Verificar en MAYÚSCULAS
    ohlc_encontradas_upper_all = [col for col in columnas_ohlc_upper_all if col in columnas_upper]
    
    logger.info(f"   OHLC en minúsculas encontradas: {ohlc_encontradas_lower}")
    logger.info(f"   OHLC en mayúsculas encontradas: {ohlc_encontradas_upper}")
    logger.info(f"   OHLC en MAYÚSCULAS encontradas: {ohlc_encontradas_upper_all}")
    
    # Verificar valores
    if 'open' in columnas_lower or 'Open' in columnas_upper or 'OPEN' in columnas_upper:
        col_open = None
        if 'open' in columnas_lower:
            col_open = 'open'
        elif 'Open' in columnas_upper:
            col_open = 'Open'
        elif 'OPEN' in columnas_upper:
            col_open = 'OPEN'
        
        if col_open:
            valores_open = df[col_open]
            logger.info(f"   Valores de {col_open}:")
            logger.info(f"     Min: {valores_open.min()}")
            logger.info(f"     Max: {valores_open.max()}")
            logger.info(f"     NaN: {valores_open.isna().sum()}")
            logger.info(f"     Inf: {np.isinf(valores_open).sum()}")
    
    return {
        'columnas': list(df.columns),
        'ohlc_lower': ohlc_encontradas_lower,
        'ohlc_upper': ohlc_encontradas_upper,
        'ohlc_upper_all': ohlc_encontradas_upper_all,
        'tiene_open': len(ohlc_encontradas_lower) > 0 or len(ohlc_encontradas_upper) > 0 or len(ohlc_encontradas_upper_all) > 0
    }

def probar_backtesting_service(df: pd.DataFrame, nombre: str):
    """Prueba el servicio de backtesting con los datos."""
    logger = logging.getLogger(__name__)
    
    try:
        from app.infrastructure.services.futures_backtesting_service import FuturesBacktestingService
        
        logger.info(f"\n🧪 Probando FuturesBacktestingService con {nombre}...")
        
        # Crear servicio
        service = FuturesBacktestingService()
        
        # Configuración de prueba
        config = {
            'levels': 4,
            'range_percent': 8.0,
            'leverage': 10,
            'umbral_adx': 25.0,
            'umbral_volatilidad': 0.02,
            'umbral_sentimiento': 0.0,
            'maintenance_margin_rate': 0.01
        }
        
        # Ejecutar simulación
        resultados = service.run_futures_grid_simulation(df, config)
        
        if 'error' in resultados:
            logger.error(f"❌ Error en {nombre}: {resultados['error']}")
            return False
        else:
            logger.info(f"✅ {nombre} exitoso:")
            logger.info(f"   Retorno: {resultados.get('Return [%]', 0):.2f}%")
            logger.info(f"   Liquidado: {resultados.get('Was Liquidated', False)}")
            logger.info(f"   Trades: {resultados.get('# Trades', 0)}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error probando {nombre}: {str(e)}")
        return False

def corregir_formato_datos(df: pd.DataFrame, nombre: str):
    """Corrige el formato de los datos para que sea compatible."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"\n🔧 Corrigiendo formato de {nombre}...")
    
    # Crear copia
    df_corregido = df.copy()
    
    # Normalizar nombres de columnas a minúsculas
    df_corregido.columns = [col.lower() for col in df_corregido.columns]
    
    # Verificar que tenemos las columnas necesarias
    columnas_requeridas = ['open', 'high', 'low', 'close']
    columnas_faltantes = [col for col in columnas_requeridas if col not in df_corregido.columns]
    
    if columnas_faltantes:
        logger.error(f"❌ Columnas faltantes después de normalización: {columnas_faltantes}")
        return None
    
    # Convertir a formato requerido por backtesting.py
    column_mapping = {
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close'
    }
    
    df_corregido = df_corregido.rename(columns=column_mapping)
    
    # Manejar columna Volume si existe
    if 'volume' in df_corregido.columns:
        df_corregido = df_corregido.rename(columns={'volume': 'Volume'})
    
    # Asegurar columna sentiment
    if 'sentiment_score' in df_corregido.columns and 'sentiment' not in df_corregido.columns:
        df_corregido['sentiment'] = df_corregido['sentiment_score']
    elif 'sentiment' not in df_corregido.columns:
        df_corregido['sentiment'] = 0.0
    
    # Limpiar datos
    df_corregido = df_corregido.dropna(subset=['Open', 'High', 'Low', 'Close'])
    
    # Validar precios
    df_corregido = df_corregido[(df_corregido['High'] >= df_corregido['Low']) & (df_corregido['Close'] > 0)].copy()
    
    # Limpiar valores NaN
    df_corregido = df_corregido.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
    
    logger.info(f"✅ {nombre} corregido:")
    logger.info(f"   Columnas finales: {list(df_corregido.columns)}")
    logger.info(f"   Shape final: {df_corregido.shape}")
    
    return df_corregido

def main():
    """Función principal del diagnóstico."""
    logger = setup_logging()
    
    logger.info("🚀 Iniciando diagnóstico del error 'open' en backtesting de futuros...")
    logger.info("=" * 60)
    
    # Crear datos de prueba
    datos_1, datos_2, datos_3 = crear_datos_prueba()
    
    # Diagnosticar cada formato
    diagnosticos = {}
    diagnosticos['datos_1'] = diagnosticar_formato_datos(datos_1, "Datos 1 (minúsculas)")
    diagnosticos['datos_2'] = diagnosticar_formato_datos(datos_2, "Datos 2 (mayúsculas)")
    diagnosticos['datos_3'] = diagnosticar_formato_datos(datos_3, "Datos 3 (MAYÚSCULAS)")
    
    # Probar backtesting con cada formato
    resultados_prueba = {}
    resultados_prueba['datos_1'] = probar_backtesting_service(datos_1, "Datos 1 (minúsculas)")
    resultados_prueba['datos_2'] = probar_backtesting_service(datos_2, "Datos 2 (mayúsculas)")
    resultados_prueba['datos_3'] = probar_backtesting_service(datos_3, "Datos 3 (MAYÚSCULAS)")
    
    # Corregir formatos que fallaron
    datos_corregidos = {}
    for nombre, resultado in resultados_prueba.items():
        if not resultado:
            datos_original = {'datos_1': datos_1, 'datos_2': datos_2, 'datos_3': datos_3}[nombre]
            datos_corregidos[nombre] = corregir_formato_datos(datos_original, nombre)
    
    # Probar con datos corregidos
    resultados_corregidos = {}
    for nombre, df_corregido in datos_corregidos.items():
        if df_corregido is not None:
            resultados_corregidos[nombre] = probar_backtesting_service(df_corregido, f"{nombre} (corregido)")
    
    # Resumen final
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESUMEN DEL DIAGNÓSTICO")
    logger.info("=" * 60)
    
    total_formatos = len(resultados_prueba)
    exitosos_original = sum(resultados_prueba.values())
    exitosos_corregidos = sum(resultados_corregidos.values())
    
    logger.info(f"✅ Formatos exitosos originalmente: {exitosos_original}/{total_formatos}")
    logger.info(f"✅ Formatos exitosos después de corrección: {exitosos_corregidos}/{total_formatos}")
    
    # Identificar el problema
    if exitosos_original == 0:
        logger.error("🚨 PROBLEMA IDENTIFICADO: Ningún formato funciona originalmente")
        logger.info("🔧 SOLUCIÓN: El problema está en el procesamiento de columnas en futures_backtesting_service.py")
        
        # Mostrar recomendaciones
        logger.info("\n📋 RECOMENDACIONES:")
        logger.info("1. Verificar que los datos tengan columnas OHLC en cualquier formato")
        logger.info("2. Mejorar la normalización de columnas en futures_backtesting_service.py")
        logger.info("3. Agregar más logging para identificar el punto exacto del error")
        logger.info("4. Implementar manejo de errores más robusto")
        
    elif exitosos_original < total_formatos:
        logger.warning("⚠️ PROBLEMA PARCIAL: Algunos formatos funcionan, otros no")
        logger.info("🔧 SOLUCIÓN: Mejorar la detección y normalización de formatos")
        
    else:
        logger.info("✅ NO SE DETECTÓ PROBLEMA: Todos los formatos funcionan")
    
    # Mostrar detalles por formato
    logger.info("\n📋 DETALLES POR FORMATO:")
    for nombre, resultado in resultados_prueba.items():
        estado = "✅ EXITOSO" if resultado else "❌ FALLÓ"
        logger.info(f"   {nombre}: {estado}")
        
        if nombre in resultados_corregidos:
            estado_corregido = "✅ EXITOSO" if resultados_corregidos[nombre] else "❌ SIGUE FALLANDO"
            logger.info(f"   {nombre} (corregido): {estado_corregido}")
    
    logger.info("=" * 60)
    
    return exitosos_original > 0 or exitosos_corregidos > 0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error en diagnóstico: {e}")
        sys.exit(1) 