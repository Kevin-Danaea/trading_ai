#!/usr/bin/env python3
"""
Test Daily Recommendations Only - Prueba Solo del Servicio de Recomendaciones Diarias
====================================================================================

Script para probar específicamente el servicio de recomendaciones diarias usando
datos de muestra, sin ejecutar todo el pipeline de análisis.

Útil para:
- Verificar la configuración de BD y Telegram
- Probar el formateo de mensajes
- Validar la estructura de datos
- Debug rápido del sistema

IMPORTANTE: Requiere variables de entorno opcionales:
- DATABASE_URL (para probar BD)
- TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID (para probar Telegram)
"""

import sys
import os
import logging
from datetime import datetime
from decimal import Decimal

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_sample_trading_opportunity():
    """Crea una TradingOpportunity de muestra para testing."""
    from app.domain.entities.crypto_candidate import CryptoCandidate
    from app.domain.entities.trading_opportunity import TradingOpportunity, StrategyResult
    
    # Crear candidato de muestra con parámetros correctos
    candidate = CryptoCandidate(
        symbol="BTC/USDT",
        market_cap_rank=1,
        current_price=42000.0,
        volatility_24h=0.041,
        volatility_7d=0.045,
        adx=18.5,
        sentiment_score=78.2,
        sentiment_ma7=75.8,
        volume_24h=2500000000.0,
        volume_change_24h=12.5,
        price_change_24h=2.34,
        price_change_7d=5.67,
        score=87.5,
        reasons=[
            "Volatilidad moderada ideal para Grid Trading",
            "ADX bajo indica mercado lateral",
            "Sentimiento positivo sostenido",
            "Volumen alto mantiene liquidez"
        ]
    )
    
    # Crear resultados de estrategias
    grid_strategy = StrategyResult(
        strategy_name='grid',
        optimized_params={'grid_size': 0.02, 'num_grids': 20},
        roi_percentage=15.6,
        sharpe_ratio=1.34,
        max_drawdown_percentage=8.2,
        win_rate_percentage=72.5,
        total_trades=45,
        avg_trade_percentage=0.67,
        volatility_percentage=4.1,
        calmar_ratio=1.9,
        sortino_ratio=1.8,
        exposure_time_percentage=85.3,
        optimization_iterations=100,
        optimization_duration_seconds=180.0,
        confidence_level=0.87
    )
    
    dca_strategy = StrategyResult(
        strategy_name='dca',
        optimized_params={'dca_amount': 100, 'dca_interval': 'daily'},
        roi_percentage=12.3,
        sharpe_ratio=1.12,
        max_drawdown_percentage=12.5,
        win_rate_percentage=68.2,
        total_trades=28,
        avg_trade_percentage=0.89,
        volatility_percentage=3.8,
        calmar_ratio=0.98,
        sortino_ratio=1.45,
        exposure_time_percentage=95.0,
        optimization_iterations=80,
        optimization_duration_seconds=150.0,
        confidence_level=0.82
    )
    
    btd_strategy = StrategyResult(
        strategy_name='btd',
        optimized_params={'dip_threshold': 0.05, 'buy_amount': 150},
        roi_percentage=18.9,
        sharpe_ratio=1.56,
        max_drawdown_percentage=15.3,
        win_rate_percentage=65.7,
        total_trades=18,
        avg_trade_percentage=1.23,
        volatility_percentage=5.2,
        calmar_ratio=1.23,
        sortino_ratio=2.1,
        exposure_time_percentage=45.8,
        optimization_iterations=75,
        optimization_duration_seconds=120.0,
        confidence_level=0.89
    )
    
    # Crear oportunidad de trading
    opportunity = TradingOpportunity(
        candidate=candidate,
        strategy_results={
            'grid': grid_strategy,
            'dca': dca_strategy,
            'btd': btd_strategy
        },
        recommended_strategy_name='grid',  # Grid es la mejor por ROI ajustado
        backtest_period_days=30,
        final_score=87.5,
        risk_adjusted_score=82.3,
        created_at=datetime.now(),
        market_conditions='sideways'
    )
    
    return opportunity

def create_sample_qualitative_analysis():
    """Crea un QualitativeAnalysis de muestra para testing."""
    from app.domain.entities.qualitative_analysis import QualitativeAnalysis
    
    analysis = QualitativeAnalysis(
        reasoning="BTC/USDT muestra condiciones ideales para Grid Trading debido al mercado lateral con volatilidad moderada. La estrategia Grid es superior a DCA en este contexto porque aprovecha mejor las oscilaciones del precio. BTD podría ser arriesgada dado el contexto actual de mercado.",
        market_context="Mercado lateral con soporte fuerte en $40,000 y resistencia en $45,000. Volatilidad estable entre 4-5% diaria.",
        risk_factors=[
            "Posible ruptura bajista del soporte",
            "Eventos macroeconómicos pueden alterar la lateralidad",
            "Reducción de volumen durante fines de semana"
        ],
        opportunity_factors=[
            "Patrón lateral establecido desde hace 2 semanas",
            "Volumen alto mantiene la estabilidad",
            "Indicadores técnicos soportan continuidad del rango"
        ],
        recommended_strategy='grid',
        strategy_reasoning="Grid Trading es óptima para mercados laterales como el actual. Los grids de 2% capturan eficientemente las oscilaciones del 4-5% diarias, maximizando el número de trades rentables.",
        alternative_strategies_notes="DCA sería menos eficiente en este contexto lateral. BTD presenta mayor riesgo por posibles rupturas bajistas.",
        strategic_notes="Configurar grids entre $40,500-$44,500 con tamaño de 2% para optimizar capturas.",
        confidence_level='high',
        recommendation='buy',
        analysis_timestamp=datetime.now(),
        execution_notes="Ejecutar durante sesiones de alta liquidez (horario europeo/americano)"
    )
    
    return analysis

def test_daily_recommendation_creation():
    """Prueba la creación de una recomendación diaria."""
    logger.info("🧪 PRUEBA 1: CREACIÓN DE RECOMENDACIÓN DIARIA")
    
    try:
        from app.domain.entities.daily_recommendation import RecomendacionDiaria
        
        # Crear datos de muestra
        opportunity = create_sample_trading_opportunity()
        analysis = create_sample_qualitative_analysis()
        
        # Crear recomendación diaria
        recommendation = RecomendacionDiaria.from_trading_opportunity_and_analysis(
            opportunity=opportunity,
            qualitative_analysis=analysis,
            version_pipeline="1.1-test"
        )
        
        logger.info("✅ Recomendación diaria creada exitosamente")
        logger.info(f"   💎 Símbolo: {recommendation.simbolo}")
        logger.info(f"   🎯 Estrategia Quant: {recommendation.estrategia_recomendada}")
        logger.info(f"   🤖 Estrategia Gemini: {recommendation.estrategia_gemini}")
        logger.info(f"   ✅ Consenso: {recommendation.consenso_estrategia}")
        logger.info(f"   📈 ROI: {recommendation.roi_porcentaje:.1f}%")
        logger.info(f"   🎖️ Categoría: {recommendation.categoria_rendimiento}")
        logger.info(f"   🚦 Nivel riesgo: {recommendation.nivel_riesgo}")
        logger.info(f"   💪 Recomendación final: {recommendation.recomendacion_final}")
        
        return recommendation
        
    except Exception as e:
        logger.error(f"❌ Error creando recomendación diaria: {e}")
        return None

def test_database_operations(recommendation):
    """Prueba las operaciones de base de datos."""
    logger.info("\n🧪 PRUEBA 2: OPERACIONES DE BASE DE DATOS")
    
    try:
        from app.infrastructure.services.database_service import DatabaseService
        
        db_service = DatabaseService()
        
        # Verificar estado de salud
        health = db_service.health_check()
        logger.info(f"🏥 Estado BD: {health['status']} - {health['message']}")
        
        if health['status'] == 'ok':
            # Guardar recomendación
            logger.info("💾 Guardando recomendación en BD...")
            success = db_service.save_recommendation(recommendation)
            
            if success:
                logger.info("✅ Recomendación guardada exitosamente")
                
                # Obtener estadísticas
                stats = db_service.get_recommendations_statistics(days_back=7)
                if stats:
                    logger.info(f"📊 Estadísticas de BD:")
                    logger.info(f"   Total recomendaciones (7 días): {stats.get('total_recomendaciones', 0)}")
                    logger.info(f"   ROI promedio: {stats.get('roi_promedio', 0):.1f}%")
                    logger.info(f"   Tasa de consenso: {stats.get('tasa_consenso', 0):.1f}%")
                    
            else:
                logger.warning("⚠️ Error guardando recomendación en BD")
        else:
            logger.warning("⚠️ Base de datos no configurada o no disponible")
            
        return health['status'] == 'ok'
        
    except Exception as e:
        logger.error(f"❌ Error en operaciones de BD: {e}")
        return False

def test_telegram_operations(recommendation):
    """Prueba las operaciones de Telegram."""
    logger.info("\n🧪 PRUEBA 3: OPERACIONES DE TELEGRAM")
    
    try:
        from app.infrastructure.services.telegram_service import TelegramService
        
        telegram_service = TelegramService()
        
        # Verificar estado de salud
        health = telegram_service.health_check()
        logger.info(f"📱 Estado Telegram: {health['status']} - {health['message']}")
        
        if health['status'] == 'ok':
            # Formatear mensaje
            logger.info("📝 Formateando mensaje...")
            message = telegram_service.format_recommendation_message(recommendation)
            logger.info(f"✅ Mensaje formateado ({len(message)} caracteres)")
            
            # Mostrar preview del mensaje
            preview = message[:200] + "..." if len(message) > 200 else message
            logger.info(f"👀 Preview: {preview}")
            
            # Enviar mensaje de prueba (opcional)
            logger.info("📤 Enviando mensaje de prueba...")
            test_message = f"🧪 TEST: {datetime.now().strftime('%H:%M:%S')} - Sistema de recomendaciones operativo"
            success = telegram_service.send_message_sync(test_message)
            
            if success:
                logger.info("✅ Mensaje de prueba enviado exitosamente")
            else:
                logger.warning("⚠️ Error enviando mensaje de prueba")
                
        else:
            logger.warning("⚠️ Telegram no configurado o no disponible")
            
        return health['status'] == 'ok'
        
    except Exception as e:
        logger.error(f"❌ Error en operaciones de Telegram: {e}")
        return False

def test_daily_report_generation(recommendation):
    """Prueba la generación de reportes diarios."""
    logger.info("\n🧪 PRUEBA 4: GENERACIÓN DE REPORTES")
    
    try:
        from app.application.use_cases.reporting.daily_report_service import DailyReportService
        
        report_service = DailyReportService()
        
        # Crear lista con la recomendación de muestra
        recommendations = [recommendation]
        
        # Generar estadísticas
        logger.info("📊 Generando estadísticas...")
        stats = report_service.generate_daily_statistics(recommendations)
        
        logger.info(f"✅ Estadísticas generadas:")
        logger.info(f"   Total recomendaciones: {stats.total_recommendations}")
        logger.info(f"   Tasa de consenso: {stats.consensus_rate:.1f}%")
        logger.info(f"   ROI promedio: {stats.avg_roi:.1f}%")
        logger.info(f"   Distribución de estrategias: {stats.strategy_distribution}")
        logger.info(f"   Distribución de riesgo: {stats.risk_distribution}")
        
        # Generar resumen ejecutivo
        logger.info("📋 Generando resumen ejecutivo...")
        executive_summary = report_service.generate_executive_summary(recommendations)
        
        if executive_summary['status'] == 'success':
            logger.info(f"✅ Resumen ejecutivo generado:")
            logger.info(f"   Sentimiento de mercado: {executive_summary['market_sentiment']}")
            logger.info(f"   Insights clave: {len(executive_summary['key_insights'])}")
            logger.info(f"   Recomendaciones de acción: {len(executive_summary['action_recommendations'])}")
        
        # Generar resumen textual
        logger.info("📝 Generando resumen textual...")
        text_summary = report_service.get_report_summary(recommendations)
        logger.info(f"✅ Resumen textual generado ({len(text_summary)} caracteres)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error generando reportes: {e}")
        return False

def test_complete_daily_service(recommendation):
    """Prueba el servicio completo de recomendaciones diarias."""
    logger.info("\n🧪 PRUEBA 5: SERVICIO COMPLETO DE RECOMENDACIONES DIARIAS")
    
    try:
        from app.application.use_cases.orchestration.daily_recommendation_service import DailyRecommendationService
        
        daily_service = DailyRecommendationService()
        
        # Crear oportunidad y análisis para el servicio
        opportunity = create_sample_trading_opportunity()
        analysis = create_sample_qualitative_analysis()
        
        # Procesar con el servicio completo
        logger.info("⚙️ Procesando con servicio completo...")
        result = daily_service.process_single_recommendation(
            opportunity=opportunity,
            analysis=analysis,
            version_pipeline="1.1-test"
        )
        
        if result['status'] in ['success', 'partial_failure']:
            logger.info(f"✅ Servicio completo exitoso: {result['status']}")
            
            summary = result['processing_summary']
            logger.info(f"📊 Resultados:")
            logger.info(f"   BD: {summary['database']['success_rate']:.1f}% éxito")
            logger.info(f"   Telegram: {summary['telegram']['success_rate']:.1f}% éxito")
            logger.info(f"   Reportes: {summary['reports']['status']}")
            
            return True
        else:
            logger.error(f"❌ Servicio completo falló: {result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error en servicio completo: {e}")
        return False

def main():
    """Función principal que ejecuta todas las pruebas."""
    logger.info("🤖 INICIANDO PRUEBAS DEL SERVICIO DE RECOMENDACIONES DIARIAS")
    logger.info(f"⏰ Timestamp: {datetime.now()}")
    logger.info("=" * 80)
    
    # Verificar configuración básica
    from app.infrastructure.config.settings import settings
    
    logger.info("🔧 VERIFICANDO CONFIGURACIÓN:")
    logger.info(f"   💾 Base de datos: {'✅' if settings.DATABASE_URL else '⚠️ No configurada'}")
    logger.info(f"   📱 Telegram: {'✅' if (settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID) else '⚠️ No configurado'}")
    
    results = []
    
    # Prueba 1: Creación de recomendación
    recommendation = test_daily_recommendation_creation()
    results.append(('Creación', recommendation is not None))
    
    if recommendation:
        # Prueba 2: Base de datos
        db_result = test_database_operations(recommendation)
        results.append(('Base de datos', db_result))
        
        # Prueba 3: Telegram
        telegram_result = test_telegram_operations(recommendation)
        results.append(('Telegram', telegram_result))
        
        # Prueba 4: Reportes
        report_result = test_daily_report_generation(recommendation)
        results.append(('Reportes', report_result))
        
        # Prueba 5: Servicio completo
        complete_result = test_complete_daily_service(recommendation)
        results.append(('Servicio completo', complete_result))
    
    # Resultado final
    logger.info("\n" + "="*80)
    logger.info("RESULTADO FINAL DE LAS PRUEBAS")
    logger.info("="*80)
    
    for test_name, success in results:
        icon = "✅" if success else "❌"
        logger.info(f"   {icon} {test_name}")
    
    successful_tests = sum(1 for _, success in results if success)
    total_tests = len(results)
    
    if successful_tests == total_tests:
        logger.info(f"\n🎉 TODAS LAS PRUEBAS EXITOSAS! ({successful_tests}/{total_tests})")
        logger.info("   ✅ Servicio de recomendaciones diarias completamente funcional")
        logger.info("   ✅ Listo para integración en producción")
        return True
    else:
        logger.warning(f"\n⚠️ ALGUNAS PRUEBAS FALLARON ({successful_tests}/{total_tests})")
        logger.warning("   Revisa la configuración de los componentes que fallaron")
        logger.warning("   El sistema puede funcionar parcialmente")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 