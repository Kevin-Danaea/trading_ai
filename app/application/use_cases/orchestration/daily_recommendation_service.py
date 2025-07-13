"""
Weekly Recommendation Service - Servicio de Recomendaciones Semanales
====================================================================

Servicio orquestador que maneja el flujo completo de recomendaciones semanales:
- Procesar recomendaciones enriquecidas con filtro semanal
- Seleccionar cartera balanceada (1 GRID, 2 DCA/BTD, 1 GRID Futuros, 1 DCA Futuros)
- Guardar en base de datos
- Generar reportes
- Enviar notificaciones a Telegram
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.domain.entities.trading_opportunity import TradingOpportunity
from app.domain.entities.qualitative_analysis import QualitativeAnalysis
from app.domain.entities.daily_recommendation import RecomendacionDiaria
from app.infrastructure.services.database_service import DatabaseService
from app.infrastructure.services.telegram_service import TelegramService
from app.application.use_cases.reporting.daily_report_service import DailyReportService
from app.application.use_cases.orchestration.weekly_portfolio_service import WeeklyPortfolioService, WeeklyPortfolioSelection
from app.application.use_cases.qualitative_analysis.qualitative_filter_service import QualitativeResult
from app.application.use_cases.qualitative_analysis.futures_analysis_service import FuturesAnalysisService
from app.infrastructure.providers.sentiment_data_provider import SentimentDataProvider

logger = logging.getLogger(__name__)


class WeeklyRecommendationService:
    """
    Servicio orquestador para el flujo completo de recomendaciones semanales.
    
    Coordina:
    - Selección de cartera semanal balanceada
    - Almacenamiento en base de datos
    - Generación de reportes
    - Envío de notificaciones
    """
    
    def __init__(self):
        """Inicializa el servicio con sus dependencias."""
        self.database_service = DatabaseService()
        self.telegram_service = TelegramService()
        self.report_service = DailyReportService()
        
        # Inicializar servicios especializados
        self.sentiment_provider = SentimentDataProvider()
        self.futures_service = FuturesAnalysisService(self.sentiment_provider)
        self.weekly_portfolio_service = WeeklyPortfolioService(futures_service=self.futures_service)
        
        self.logger = logging.getLogger(__name__)
    
    def process_weekly_recommendations(
        self,
        qualitative_results: List[QualitativeResult],
        version_pipeline: str = "weekly_v1.0"
    ) -> Dict[str, Any]:
        """
        Procesa las recomendaciones semanales con filtro de cartera balanceada.
        
        Args:
            qualitative_results: Lista de resultados cualitativos
            version_pipeline: Versión del pipeline para tracking
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        try:
            logger.info(f"🚀 Iniciando procesamiento semanal de {len(qualitative_results)} candidatos")
            
            # 1. Seleccionar cartera semanal balanceada
            weekly_selection = self.weekly_portfolio_service.select_weekly_portfolio(
                qualitative_results=qualitative_results,
                force_selection=False
            )
            
            if not weekly_selection.get_all_recommendations():
                logger.warning("⚠️ No se pudo seleccionar cartera semanal")
                return {
                    'status': 'error',
                    'message': 'No se pudo seleccionar cartera semanal',
                    'processed_count': 0,
                    'weekly_selection': weekly_selection
                }
            
            # 2. Validar selección
            validation_result = self.weekly_portfolio_service.validate_weekly_selection(weekly_selection)
            
            if not validation_result['is_valid']:
                logger.error(f"❌ Selección semanal no válida: {validation_result['errors']}")
                return {
                    'status': 'error',
                    'message': 'Selección semanal no válida',
                    'processed_count': 0,
                    'validation_result': validation_result
                }
            
            # 3. Obtener recomendaciones finales
            recommendations = weekly_selection.get_all_recommendations()
            
            # 4. Guardar en base de datos
            db_result = self._save_to_database(recommendations)
            
            # 5. Generar reportes
            report_result = self._generate_weekly_reports(recommendations, weekly_selection)
            
            # 6. Enviar a Telegram
            telegram_result = self._send_weekly_to_telegram(recommendations, weekly_selection)
            
            # 7. Consolidar resultados
            result = self._consolidate_weekly_results(
                recommendations, 
                weekly_selection,
                db_result, 
                report_result, 
                telegram_result
            )
            
            logger.info(f"✅ Procesamiento semanal completado: {result['processed_count']} recomendaciones")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error procesando recomendaciones semanales: {e}")
            return {
                'status': 'error',
                'message': f'Error procesando recomendaciones semanales: {str(e)}',
                'processed_count': 0,
                'timestamp': datetime.now().isoformat()
            }
    
    def _convert_to_recommendations(
        self,
        opportunities_with_analysis: List[tuple[TradingOpportunity, QualitativeAnalysis]],
        version_pipeline: str
    ) -> List[RecomendacionDiaria]:
        """
        Convierte oportunidades y análisis a recomendaciones diarias.
        
        Args:
            opportunities_with_analysis: Lista de tuplas (oportunidad, análisis)
            version_pipeline: Versión del pipeline
            
        Returns:
            Lista de recomendaciones diarias
        """
        recommendations = []
        
        for opportunity, analysis in opportunities_with_analysis:
            try:
                recommendation = RecomendacionDiaria.from_trading_opportunity_and_analysis(
                    opportunity=opportunity,
                    qualitative_analysis=analysis,
                    version_pipeline=version_pipeline
                )
                recommendations.append(recommendation)
                
            except Exception as e:
                logger.error(f"❌ Error convirtiendo {opportunity.symbol}: {e}")
                continue
        
        logger.info(f"📊 Convertidas {len(recommendations)} recomendaciones exitosamente")
        return recommendations
    
    def _save_to_database(self, recommendations: List[RecomendacionDiaria]) -> Dict[str, Any]:
        """
        Guarda las recomendaciones en la base de datos.
        
        Args:
            recommendations: Lista de recomendaciones
            
        Returns:
            Resultado del guardado
        """
        try:
            logger.info(f"💾 Guardando {len(recommendations)} recomendaciones en base de datos")
            
            db_result = self.database_service.save_recommendations(recommendations)
            
            if db_result['exitoso']:
                logger.info(f"✅ Base de datos: {db_result['guardadas']}/{db_result['total']} guardadas")
            else:
                logger.error(f"❌ Error guardando en base de datos: {db_result['errores']} errores")
            
            return db_result
            
        except Exception as e:
            logger.error(f"❌ Error en guardado de base de datos: {e}")
            return {
                'total': len(recommendations),
                'guardadas': 0,
                'errores': len(recommendations),
                'exitoso': False,
                'error_message': str(e)
            }
    
    def _generate_reports(self, recommendations: List[RecomendacionDiaria]) -> Dict[str, Any]:
        """
        Genera los reportes de las recomendaciones.
        
        Args:
            recommendations: Lista de recomendaciones
            
        Returns:
            Resultado de la generación de reportes
        """
        try:
            logger.info(f"📊 Generando reportes para {len(recommendations)} recomendaciones")
            
            # Generar estadísticas
            stats = self.report_service.generate_daily_statistics(recommendations)
            
            # Generar resumen ejecutivo
            executive_summary = self.report_service.generate_executive_summary(recommendations)
            
            # Generar reporte detallado
            detailed_report = self.report_service.generate_detailed_report(recommendations)
            
            logger.info("✅ Reportes generados exitosamente")
            
            return {
                'status': 'success',
                'statistics': stats,
                'executive_summary': executive_summary,
                'detailed_report': detailed_report,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error generando reportes: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    def _send_to_telegram(self, recommendations: List[RecomendacionDiaria]) -> Dict[str, Any]:
        """
        Envía las recomendaciones a Telegram.
        
        Args:
            recommendations: Lista de recomendaciones
            
        Returns:
            Resultado del envío
        """
        try:
            logger.info(f"📱 Enviando {len(recommendations)} recomendaciones a Telegram")
            
            telegram_result = self.telegram_service.send_daily_report(recommendations)
            
            if telegram_result['success_rate'] > 80:
                logger.info(f"✅ Telegram: {telegram_result['sent_successfully']}/{telegram_result['total_messages']} mensajes enviados")
            else:
                logger.warning(f"⚠️ Telegram: Solo {telegram_result['success_rate']:.1f}% de mensajes enviados")
            
            return telegram_result
            
        except Exception as e:
            logger.error(f"❌ Error enviando a Telegram: {e}")
            return {
                'total_messages': len(recommendations) + 1,
                'sent_successfully': 0,
                'errors': len(recommendations) + 1,
                'success_rate': 0,
                'error_message': str(e)
            }
    
    def _consolidate_results(
        self,
        recommendations: List[RecomendacionDiaria],
        db_result: Dict[str, Any],
        report_result: Dict[str, Any],
        telegram_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Consolida los resultados de todos los procesos.
        
        Args:
            recommendations: Lista de recomendaciones procesadas
            db_result: Resultado del guardado en BD
            report_result: Resultado de la generación de reportes
            telegram_result: Resultado del envío a Telegram
            
        Returns:
            Resultado consolidado
        """
        try:
            # Determinar estado general
            overall_status = 'success'
            if not db_result.get('exitoso', False):
                overall_status = 'partial_failure'
            if telegram_result.get('success_rate', 0) < 50:
                overall_status = 'partial_failure'
            if report_result.get('status') == 'error':
                overall_status = 'partial_failure'
            
            # Calcular métricas de éxito
            total_recommendations = len(recommendations)
            db_success_rate = (db_result.get('guardadas', 0) / total_recommendations) * 100 if total_recommendations > 0 else 0
            telegram_success_rate = telegram_result.get('success_rate', 0)
            
            # Crear resumen consolidado
            consolidated_result = {
                'status': overall_status,
                'processed_count': total_recommendations,
                'processing_summary': {
                    'database': {
                        'saved': db_result.get('guardadas', 0),
                        'total': db_result.get('total', 0),
                        'success_rate': db_success_rate,
                        'errors': db_result.get('errores', 0)
                    },
                    'telegram': {
                        'sent': telegram_result.get('sent_successfully', 0),
                        'total': telegram_result.get('total_messages', 0),
                        'success_rate': telegram_success_rate,
                        'errors': telegram_result.get('errors', 0)
                    },
                    'reports': {
                        'generated': report_result.get('status') == 'success',
                        'status': report_result.get('status', 'unknown')
                    }
                },
                'recommendations_summary': {
                    'total': total_recommendations,
                    'consensus_count': sum(1 for r in recommendations if r.consenso_estrategia),
                    'avg_roi': sum(r.roi_porcentaje for r in recommendations) / total_recommendations if total_recommendations > 0 else 0,
                    'avg_confidence': sum(r.score_confianza_gemini for r in recommendations) / total_recommendations if total_recommendations > 0 else 0,
                    'symbols': [r.simbolo for r in recommendations]
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Agregar detalles adicionales si están disponibles
            if report_result.get('executive_summary'):
                consolidated_result['executive_summary'] = report_result['executive_summary']
            
            return consolidated_result
            
        except Exception as e:
            logger.error(f"❌ Error consolidando resultados: {e}")
            return {
                'status': 'error',
                'message': f'Error consolidando resultados: {str(e)}',
                'processed_count': len(recommendations),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_weekly_reports(self, recommendations: List[RecomendacionDiaria], weekly_selection: WeeklyPortfolioSelection) -> Dict[str, Any]:
        """
        Genera los reportes semanales de las recomendaciones.
        
        Args:
            recommendations: Lista de recomendaciones
            weekly_selection: Selección semanal
            
        Returns:
            Resultado de la generación de reportes
        """
        try:
            logger.info(f"📊 Generando reportes semanales para {len(recommendations)} recomendaciones")
            
            # Generar estadísticas
            stats = self.report_service.generate_daily_statistics(recommendations)
            
            # Generar resumen ejecutivo semanal
            executive_summary = self._generate_weekly_executive_summary(recommendations, weekly_selection)
            
            # Generar reporte detallado
            detailed_report = self.report_service.generate_detailed_report(recommendations)
            
            logger.info("✅ Reportes semanales generados exitosamente")
            
            return {
                'status': 'success',
                'statistics': stats,
                'executive_summary': executive_summary,
                'detailed_report': detailed_report,
                'weekly_selection_summary': weekly_selection.get_completion_status(),
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error generando reportes semanales: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    def _send_weekly_to_telegram(self, recommendations: List[RecomendacionDiaria], weekly_selection: WeeklyPortfolioSelection) -> Dict[str, Any]:
        """
        Envía las recomendaciones semanales a Telegram.
        
        Args:
            recommendations: Lista de recomendaciones
            weekly_selection: Selección semanal
            
        Returns:
            Resultado del envío
        """
        try:
            logger.info(f"📱 Enviando {len(recommendations)} recomendaciones semanales a Telegram")
            
            # Enviar con formato semanal
            telegram_result = self.telegram_service.send_weekly_report(recommendations, weekly_selection)
            
            if telegram_result.get('success_rate', 0) > 80:
                logger.info(f"✅ Telegram semanal: {telegram_result['sent_successfully']}/{telegram_result['total_messages']} mensajes enviados")
            else:
                logger.warning(f"⚠️ Telegram semanal: Solo {telegram_result.get('success_rate', 0):.1f}% de mensajes enviados")
            
            return telegram_result
            
        except Exception as e:
            logger.error(f"❌ Error enviando recomendaciones semanales a Telegram: {e}")
            return {
                'total_messages': len(recommendations) + 1,
                'sent_successfully': 0,
                'errors': len(recommendations) + 1,
                'success_rate': 0,
                'error_message': str(e)
            }
    
    def _consolidate_weekly_results(
        self,
        recommendations: List[RecomendacionDiaria],
        weekly_selection: WeeklyPortfolioSelection,
        db_result: Dict[str, Any],
        report_result: Dict[str, Any],
        telegram_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Consolida los resultados del procesamiento semanal.
        
        Args:
            recommendations: Lista de recomendaciones procesadas
            weekly_selection: Selección semanal
            db_result: Resultado del guardado en BD
            report_result: Resultado de la generación de reportes
            telegram_result: Resultado del envío a Telegram
            
        Returns:
            Resultado consolidado
        """
        try:
            # Determinar estado general
            overall_status = 'success'
            if not db_result.get('exitoso', False):
                overall_status = 'partial_failure'
            if telegram_result.get('success_rate', 0) < 50:
                overall_status = 'partial_failure'
            if report_result.get('status') == 'error':
                overall_status = 'partial_failure'
            
            # Calcular métricas de éxito
            total_recommendations = len(recommendations)
            db_success_rate = (db_result.get('guardadas', 0) / total_recommendations) * 100 if total_recommendations > 0 else 0
            telegram_success_rate = telegram_result.get('success_rate', 0)
            
            # Crear resumen consolidado
            consolidated_result = {
                'status': overall_status,
                'processed_count': total_recommendations,
                'weekly_selection_quality': weekly_selection.selection_quality,
                'weekly_selection_complete': weekly_selection.is_complete(),
                'processing_summary': {
                    'database': {
                        'saved': db_result.get('guardadas', 0),
                        'total': db_result.get('total', 0),
                        'success_rate': db_success_rate,
                        'errors': db_result.get('errores', 0)
                    },
                    'telegram': {
                        'sent': telegram_result.get('sent_successfully', 0),
                        'total': telegram_result.get('total_messages', 0),
                        'success_rate': telegram_success_rate,
                        'errors': telegram_result.get('errors', 0)
                    },
                    'reports': {
                        'generated': report_result.get('status') == 'success',
                        'status': report_result.get('status', 'unknown')
                    }
                },
                'weekly_portfolio_summary': {
                    'grid_spot': weekly_selection.grid_spot.simbolo if weekly_selection.grid_spot else None,
                    'dca_btd_spot': [r.simbolo for r in weekly_selection.dca_btd_spot],
                    'grid_futures': weekly_selection.grid_futures.simbolo if weekly_selection.grid_futures else None,
                    'dca_futures': weekly_selection.dca_futures.simbolo if weekly_selection.dca_futures else None,
                    'total_selected': weekly_selection.total_selected,
                    'valid_until': weekly_selection.valid_until.isoformat() if weekly_selection.valid_until else None
                },
                'recommendations_summary': {
                    'total': total_recommendations,
                    'avg_roi': sum(r.roi_porcentaje for r in recommendations) / total_recommendations if total_recommendations > 0 else 0,
                    'avg_confidence': sum(r.score_confianza_gemini for r in recommendations) / total_recommendations if total_recommendations > 0 else 0,
                    'symbols': [r.simbolo for r in recommendations]
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Agregar detalles adicionales si están disponibles
            if report_result.get('executive_summary'):
                consolidated_result['executive_summary'] = report_result['executive_summary']
            
            return consolidated_result
            
        except Exception as e:
            logger.error(f"❌ Error consolidando resultados semanales: {e}")
            return {
                'status': 'error',
                'message': f'Error consolidando resultados semanales: {str(e)}',
                'processed_count': len(recommendations),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_weekly_executive_summary(self, recommendations: List[RecomendacionDiaria], weekly_selection: WeeklyPortfolioSelection) -> Dict[str, Any]:
        """
        Genera resumen ejecutivo específico para cartera semanal.
        
        Args:
            recommendations: Recomendaciones seleccionadas
            weekly_selection: Selección semanal
            
        Returns:
            Resumen ejecutivo semanal
        """
        try:
            return {
                'selection_quality': weekly_selection.selection_quality,
                'portfolio_composition': {
                    'grid_spot': weekly_selection.grid_spot.simbolo if weekly_selection.grid_spot else None,
                    'dca_btd_spot': [r.simbolo for r in weekly_selection.dca_btd_spot],
                    'grid_futures': weekly_selection.grid_futures.simbolo if weekly_selection.grid_futures else None,
                    'dca_futures': weekly_selection.dca_futures.simbolo if weekly_selection.dca_futures else None
                },
                'expected_performance': {
                    'avg_roi': sum(r.roi_porcentaje for r in recommendations) / len(recommendations) if recommendations else 0,
                    'avg_sharpe': sum(r.sharpe_ratio for r in recommendations) / len(recommendations) if recommendations else 0,
                    'max_drawdown': max(r.max_drawdown_porcentaje for r in recommendations) if recommendations else 0
                },
                'risk_assessment': {
                    'low_risk_count': sum(1 for r in recommendations if r.nivel_riesgo == 'BAJO'),
                    'medium_risk_count': sum(1 for r in recommendations if r.nivel_riesgo == 'MEDIO'),
                    'high_risk_count': sum(1 for r in recommendations if r.nivel_riesgo == 'ALTO')
                },
                'valid_period': {
                    'start_date': weekly_selection.selection_date.isoformat() if weekly_selection.selection_date else None,
                    'end_date': weekly_selection.valid_until.isoformat() if weekly_selection.valid_until else None
                }
            }
        except Exception as e:
            logger.error(f"❌ Error generando resumen ejecutivo semanal: {e}")
            return {'error': str(e)}
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Verifica el estado de salud de todos los servicios.
        
        Returns:
            Estado de salud del sistema
        """
        try:
            logger.info("🔍 Verificando estado de salud del sistema")
            
            # Verificar base de datos
            db_health = self.database_service.health_check()
            
            # Verificar Telegram
            telegram_health = self.telegram_service.health_check()
            
            # Estado general
            overall_status = 'healthy'
            if db_health['status'] != 'ok' or telegram_health['status'] != 'ok':
                overall_status = 'degraded'
            
            health_report = {
                'overall_status': overall_status,
                'components': {
                    'database': db_health,
                    'telegram': telegram_health,
                    'report_service': {
                        'status': 'ok',
                        'message': 'Servicio de reportes operativo'
                    }
                },
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Estado del sistema: {overall_status}")
            return health_report
            
        except Exception as e:
            logger.error(f"❌ Error verificando estado del sistema: {e}")
            return {
                'overall_status': 'error',
                'message': f'Error verificando estado: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def process_single_recommendation(
        self,
        qualitative_result: QualitativeResult,
        version_pipeline: str = "weekly_v1.0"
    ) -> Dict[str, Any]:
        """
        Procesa una sola recomendación (útil para testing).
        
        Args:
            qualitative_result: Resultado cualitativo
            version_pipeline: Versión del pipeline
            
        Returns:
            Resultado del procesamiento
        """
        return self.process_weekly_recommendations(
            [qualitative_result], 
            version_pipeline
        )
    
    def get_daily_summary(self, recommendations: List[RecomendacionDiaria]) -> str:
        """
        Genera un resumen textual del día.
        
        Args:
            recommendations: Lista de recomendaciones
            
        Returns:
            Resumen textual
        """
        try:
            return self.report_service.get_report_summary(recommendations)
        except Exception as e:
            logger.error(f"❌ Error generando resumen: {e}")
            return f"Error generando resumen: {str(e)}" 