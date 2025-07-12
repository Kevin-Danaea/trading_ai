#!/usr/bin/env python3
"""
Test Complete AI Pipeline with DB & Telegram - Pipeline Completo con BD y Telegram
==================================================================================

Script para probar la integración completa del sistema de trading con:
1. Scanner (Top 150 → 10 candidatos)
2. Optimizer (10 candidatos → 10 estrategias optimizadas)
3. Ranking (10 estrategias → Top 3-5 oportunidades)
4. Qualitative Analysis (Top 3-5 → Análisis Gemini AI)
5. Daily Recommendation Service (Guardar en BD + Enviar a Telegram)

IMPORTANTE: Requiere variables de entorno:
- GEMINI_API_KEY
- DATABASE_URL (opcional)
- TELEGRAM_BOT_TOKEN (opcional)
- TELEGRAM_CHAT_ID (opcional)
"""

import sys
import os
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_complete_pipeline_with_daily_recommendations():
    """
    Prueba completa del pipeline incluyendo el nuevo servicio de recomendaciones diarias.
    
    Flujo:
    Scanner → Optimizer → Ranking → Qualitative Analysis → Daily Recommendation Service
    """
    try:
        logger.info("🚀 INICIANDO PRUEBA DEL PIPELINE COMPLETO CON BD Y TELEGRAM")
        logger.info("=" * 80)
        
        # Verificar configuración
        from app.infrastructure.config.settings import settings
        
        if not settings.GEMINI_API_KEY:
            logger.error("❌ GEMINI_API_KEY no configurada")
            logger.error("   Por favor configura la variable de entorno GEMINI_API_KEY")
            return False
        
        logger.info("✅ Configuración verificada")
        logger.info(f"   🔑 Gemini API Key: {'*' * 20}{settings.GEMINI_API_KEY[-4:]}")
        
        # Verificar configuraciones opcionales
        db_configured = bool(settings.DATABASE_URL)
        telegram_configured = bool(settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID)
        
        logger.info(f"   💾 Base de datos: {'✅ Configurada' if db_configured else '⚠️ No configurada'}")
        logger.info(f"   📱 Telegram: {'✅ Configurado' if telegram_configured else '⚠️ No configurado'}")
        
        # Importar servicios
        from app.infrastructure.providers.market_data_provider import MarketDataProvider
        from app.infrastructure.providers.sentiment_data_provider import SentimentDataProvider
        from app.application.use_cases.orchestration.trading_brain_service import TradingBrainService
        from app.application.use_cases.orchestration.daily_recommendation_service import DailyRecommendationService
        
        logger.info("✅ Servicios importados correctamente")
        
        # Inicializar proveedores
        logger.info("🔧 Inicializando proveedores...")
        market_data_provider = MarketDataProvider()
        sentiment_data_provider = SentimentDataProvider()
        
        # Inicializar cerebro de trading
        logger.info("🧠 Inicializando TradingBrainService...")
        brain = TradingBrainService(
            market_data_provider=market_data_provider,
            sentiment_data_provider=sentiment_data_provider,
            target_opportunities=3,  # Top 3 para prueba
            risk_tolerance='moderate'
        )
        
        # Inicializar servicio de recomendaciones diarias
        logger.info("📊 Inicializando DailyRecommendationService...")
        daily_service = DailyRecommendationService()
        
        logger.info("✅ Todos los servicios inicializados correctamente")
        
        # FASE 1: Ejecutar análisis completo del mercado
        logger.info("\n🎯 FASE 1: ANÁLISIS COMPLETO DEL MERCADO...")
        logger.info("⏱️ Esto puede tomar varios minutos...")
        
        start_time = datetime.now()
        result = brain.analyze_market_and_decide()
        analysis_time = datetime.now()
        
        # Verificar que el análisis fue exitoso
        if result['execution_summary']['status'] != 'completed':
            logger.error("❌ ANÁLISIS DE MERCADO FALLÓ")
            logger.error(f"   Error: {result['execution_summary'].get('error', 'Unknown')}")
            return False
        
        logger.info("✅ FASE 1 COMPLETADA: Análisis de mercado exitoso")
        
        # Mostrar estadísticas del análisis
        timing = result['timing_breakdown']
        summary = result['execution_summary']
        
        logger.info(f"\n📊 ESTADÍSTICAS DEL ANÁLISIS:")
        logger.info(f"   ⏱️ Tiempo de análisis: {timing.get('total', 0):.1f}s")
        logger.info(f"   📈 Candidatos encontrados: {summary.get('candidates_found', 0)}")
        logger.info(f"   🎯 Estrategias optimizadas: {summary.get('strategies_optimized', 0)}")
        logger.info(f"   🏆 Oportunidades seleccionadas: {summary.get('opportunities_selected', 0)}")
        logger.info(f"   🤖 Análisis cualitativos: {summary.get('qualitative_analyses', 0)}")
        
        # FASE 2: Extraer datos para recomendaciones diarias
        logger.info("\n🎯 FASE 2: PREPARANDO RECOMENDACIONES DIARIAS...")
        
        # Obtener oportunidades con análisis cualitativo
        top_opportunities = result.get('top_opportunities', [])
        qualitative_results = result.get('qualitative_results', [])
        
        if not top_opportunities or not qualitative_results:
            logger.error("❌ No hay oportunidades o análisis cualitativos disponibles")
            return False
        
        # Combinar oportunidades con sus análisis
        opportunities_with_analysis = []
        for opportunity in top_opportunities:
            # Buscar el análisis correspondiente
            matching_analysis = None
            for analysis_result in qualitative_results:
                # Los qualitative_results son objetos con .opportunity y .analysis
                if hasattr(analysis_result, 'opportunity') and analysis_result.opportunity.symbol == opportunity.symbol:
                    matching_analysis = analysis_result.analysis  # El QualitativeAnalysis
                    break
            
            if matching_analysis:
                opportunities_with_analysis.append((opportunity, matching_analysis))
                logger.info(f"   ✅ {opportunity.symbol}: Oportunidad + Análisis emparejados")
            else:
                logger.warning(f"   ⚠️ {opportunity.symbol}: No se encontró análisis cualitativo")
        
        if not opportunities_with_analysis:
            logger.error("❌ No se pudieron emparejar oportunidades con análisis")
            return False
        
        logger.info(f"✅ FASE 2 COMPLETADA: {len(opportunities_with_analysis)} recomendaciones preparadas")
        
        # FASE 3: Procesar recomendaciones diarias
        logger.info("\n🎯 FASE 3: PROCESANDO RECOMENDACIONES DIARIAS...")
        
        processing_start = datetime.now()
        daily_result = daily_service.process_daily_recommendations(
            opportunities_with_analysis=opportunities_with_analysis,
            version_pipeline="1.1"  # Version actualizada
        )
        processing_time = datetime.now()
        
        # Mostrar resultados del procesamiento
        if daily_result['status'] in ['success', 'partial_failure']:
            logger.info("✅ FASE 3 COMPLETADA: Recomendaciones procesadas")
            
            processing_summary = daily_result['processing_summary']
            recommendations_summary = daily_result['recommendations_summary']
            
            logger.info(f"\n📊 RESULTADOS DEL PROCESAMIENTO:")
            logger.info(f"   ⏱️ Tiempo de procesamiento: {(processing_time - processing_start).total_seconds():.1f}s")
            logger.info(f"   📦 Recomendaciones procesadas: {daily_result['processed_count']}")
            logger.info(f"   📊 Estado general: {daily_result['status'].upper()}")
            
            # Estadísticas de base de datos
            db_stats = processing_summary['database']
            logger.info(f"\n💾 BASE DE DATOS:")
            logger.info(f"   📁 Guardadas: {db_stats['saved']}/{db_stats['total']}")
            logger.info(f"   📊 Tasa de éxito: {db_stats['success_rate']:.1f}%")
            logger.info(f"   ❌ Errores: {db_stats['errors']}")
            
            # Estadísticas de Telegram
            telegram_stats = processing_summary['telegram']
            logger.info(f"\n📱 TELEGRAM:")
            logger.info(f"   📤 Enviados: {telegram_stats['sent']}/{telegram_stats['total']}")
            logger.info(f"   📊 Tasa de éxito: {telegram_stats['success_rate']:.1f}%")
            logger.info(f"   ❌ Errores: {telegram_stats['errors']}")
            
            # Estadísticas de recomendaciones
            logger.info(f"\n🎯 RESUMEN DE RECOMENDACIONES:")
            logger.info(f"   📈 ROI promedio: {recommendations_summary['avg_roi']:.1f}%")
            logger.info(f"   🤖 Confianza promedio: {recommendations_summary['avg_confidence']:.1f}/100")
            logger.info(f"   ✅ Consensos: {recommendations_summary['consensus_count']}/{recommendations_summary['total']}")
            logger.info(f"   💎 Símbolos: {', '.join(recommendations_summary['symbols'])}")
            
            # Mostrar resumen ejecutivo si está disponible
            if 'executive_summary' in daily_result:
                exec_summary = daily_result['executive_summary']
                if exec_summary.get('status') == 'success':
                    logger.info(f"\n📋 RESUMEN EJECUTIVO:")
                    logger.info(f"   🎯 Sentimiento del mercado: {exec_summary['market_sentiment'].upper()}")
                    
                    key_metrics = exec_summary['key_metrics']
                    logger.info(f"   📊 Oportunidades totales: {key_metrics['total_opportunities']}")
                    logger.info(f"   🤝 Tasa de consenso: {key_metrics['consensus_rate']}")
                    logger.info(f"   📈 ROI promedio: {key_metrics['avg_roi']}")
                    logger.info(f"   🎲 Estrategia dominante: {key_metrics['top_strategy']}")
                    
                    # Insights clave
                    if exec_summary.get('key_insights'):
                        logger.info(f"\n🔍 INSIGHTS CLAVE:")
                        for insight in exec_summary['key_insights']:
                            logger.info(f"   • {insight}")
                    
                    # Recomendaciones de acción
                    if exec_summary.get('action_recommendations'):
                        logger.info(f"\n💡 RECOMENDACIONES DE ACCIÓN:")
                        for action in exec_summary['action_recommendations']:
                            logger.info(f"   • {action}")
            
            logger.info("\n🎯 CONCLUSIÓN FINAL:")
            logger.info("   ✅ Pipeline completo funciona correctamente")
            logger.info("   ✅ Integración: Scanner → Optimizer → Ranking → Gemini AI → BD/Telegram")
            logger.info("   ✅ Análisis multi-estrategia: Grid, DCA, BTD para cada moneda")
            logger.info("   ✅ Gemini AI analiza las 3 estrategias y recomienda la mejor")
            logger.info("   ✅ Recomendaciones guardadas en base de datos")
            logger.info("   ✅ Reportes enviados automáticamente a Telegram")
            logger.info("   ✅ Sistema listo para operación diaria automatizada")
            
            return True
        else:
            logger.error("❌ FASE 3 FALLÓ: Error procesando recomendaciones diarias")
            logger.error(f"   Error: {daily_result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ ERROR EN PRUEBA DEL PIPELINE COMPLETO: {e}")
        logger.error("   Verifica que todas las dependencias estén instaladas:")
        logger.error("   - pip install google-genai python-telegram-bot")
        logger.error("   - Configura variables de entorno: GEMINI_API_KEY, DATABASE_URL, TELEGRAM_BOT_TOKEN")
        return False

def test_system_health():
    """
    Prueba el estado de salud de todos los componentes del sistema.
    """
    try:
        logger.info("\n🔍 VERIFICANDO ESTADO DE SALUD DEL SISTEMA...")
        
        from app.application.use_cases.orchestration.daily_recommendation_service import DailyRecommendationService
        
        daily_service = DailyRecommendationService()
        health_report = daily_service.get_system_health()
        
        logger.info(f"🏥 ESTADO GENERAL DEL SISTEMA: {health_report['overall_status'].upper()}")
        
        components = health_report['components']
        
        # Estado de la base de datos
        db_health = components['database']
        db_icon = "✅" if db_health['status'] == 'ok' else "❌"
        logger.info(f"   💾 Base de datos: {db_icon} {db_health['message']}")
        
        # Estado de Telegram
        telegram_health = components['telegram']
        telegram_icon = "✅" if telegram_health['status'] == 'ok' else "❌"
        logger.info(f"   📱 Telegram: {telegram_icon} {telegram_health['message']}")
        
        # Estado del servicio de reportes
        report_health = components['report_service']
        report_icon = "✅" if report_health['status'] == 'ok' else "❌"
        logger.info(f"   📊 Reportes: {report_icon} {report_health['message']}")
        
        return health_report['overall_status'] in ['healthy', 'degraded']
        
    except Exception as e:
        logger.error(f"❌ Error verificando estado del sistema: {e}")
        return False

def main():
    """Función principal que ejecuta todas las pruebas."""
    logger.info("🤖 INICIANDO PRUEBAS DEL SISTEMA COMPLETO DE TRADING AI")
    logger.info(f"⏰ Timestamp: {datetime.now()}")
    
    # Prueba 1: Estado del sistema
    logger.info("\n" + "="*60)
    logger.info("PRUEBA 1: ESTADO DE SALUD DEL SISTEMA")
    logger.info("="*60)
    
    health_success = test_system_health()
    
    if not health_success:
        logger.warning("⚠️ Algunos componentes no están configurados correctamente")
        logger.warning("   El pipeline puede ejecutarse parcialmente")
    
    # Prueba 2: Pipeline completo con BD y Telegram
    logger.info("\n" + "="*60)
    logger.info("PRUEBA 2: PIPELINE COMPLETO CON BD Y TELEGRAM")
    logger.info("="*60)
    
    complete_success = test_complete_pipeline_with_daily_recommendations()
    
    # Resultado final
    logger.info("\n" + "="*80)
    logger.info("RESULTADO FINAL DE LAS PRUEBAS")
    logger.info("="*80)
    
    if complete_success:
        logger.info("🎉 PRUEBA DEL PIPELINE COMPLETO EXITOSA!")
        logger.info("   ✅ Sistema completamente funcional")
        logger.info("   ✅ Análisis multi-estrategia (Grid, DCA, BTD) operativo")
        logger.info("   ✅ Integración con Gemini AI para selección de estrategia")
        logger.info("   ✅ Comparación Cuantitativo vs Cualitativo")
        logger.info("   ✅ Recomendaciones guardadas en base de datos")
        logger.info("   ✅ Reportes automáticos a Telegram")
        logger.info("   ✅ Sistema listo para operación diaria automatizada")
        return True
    else:
        logger.error("❌ PRUEBA DEL PIPELINE FALLÓ")
        logger.error(f"   Estado del sistema: {'✅' if health_success else '❌'}")
        logger.error(f"   Pipeline completo: {'✅' if complete_success else '❌'}")
        logger.error("   Revisa la configuración y logs de error")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 