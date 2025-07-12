"""
Daily Report Service - Servicio de Reportes Diarios
==================================================

Servicio de aplicación que maneja la generación de reportes diarios
formateados para diferentes canales de comunicación.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from dataclasses import dataclass

from app.domain.entities.daily_recommendation import RecomendacionDiaria

logger = logging.getLogger(__name__)


@dataclass
class ReportStatistics:
    """Estadísticas de un reporte diario."""
    total_recommendations: int
    consensus_count: int
    consensus_rate: float
    avg_roi: float
    avg_confidence: float
    strategy_distribution: Dict[str, int]
    risk_distribution: Dict[str, int]
    category_distribution: Dict[str, int]
    top_opportunities: List[RecomendacionDiaria]


class DailyReportService:
    """
    Servicio de generación de reportes diarios.
    
    Responsabilidades:
    - Analizar recomendaciones diarias
    - Generar estadísticas y métricas
    - Formatear reportes para diferentes canales
    - Crear resúmenes ejecutivos
    """
    
    def __init__(self):
        """Inicializa el servicio de reportes."""
        self.logger = logging.getLogger(__name__)
    
    def generate_daily_statistics(self, recommendations: List[RecomendacionDiaria]) -> ReportStatistics:
        """
        Genera estadísticas completas de las recomendaciones diarias.
        
        Args:
            recommendations: Lista de recomendaciones del día
            
        Returns:
            ReportStatistics con métricas calculadas
        """
        try:
            if not recommendations:
                return ReportStatistics(
                    total_recommendations=0,
                    consensus_count=0,
                    consensus_rate=0.0,
                    avg_roi=0.0,
                    avg_confidence=0.0,
                    strategy_distribution={},
                    risk_distribution={},
                    category_distribution={},
                    top_opportunities=[]
                )
            
            total = len(recommendations)
            
            # Consenso cuantitativo vs cualitativo
            consensus_count = sum(1 for r in recommendations if r.consenso_estrategia)
            consensus_rate = (consensus_count / total) * 100 if total > 0 else 0
            
            # Métricas promedio
            avg_roi = sum(r.roi_porcentaje for r in recommendations) / total
            avg_confidence = sum(r.score_confianza_gemini for r in recommendations) / total
            
            # Distribuciones
            strategy_distribution = self._calculate_distribution(
                recommendations, 
                lambda r: r.estrategia_gemini
            )
            
            risk_distribution = self._calculate_distribution(
                recommendations, 
                lambda r: r.nivel_riesgo
            )
            
            category_distribution = self._calculate_distribution(
                recommendations, 
                lambda r: r.categoria_rendimiento
            )
            
            # Top oportunidades (top 5 por score final)
            top_opportunities = sorted(
                recommendations, 
                key=lambda x: x.score_final, 
                reverse=True
            )[:5]
            
            return ReportStatistics(
                total_recommendations=total,
                consensus_count=consensus_count,
                consensus_rate=consensus_rate,
                avg_roi=avg_roi,
                avg_confidence=avg_confidence,
                strategy_distribution=strategy_distribution,
                risk_distribution=risk_distribution,
                category_distribution=category_distribution,
                top_opportunities=top_opportunities
            )
            
        except Exception as e:
            logger.error(f"❌ Error generando estadísticas: {e}")
            raise
    
    def _calculate_distribution(self, recommendations: List[RecomendacionDiaria], key_func) -> Dict[str, int]:
        """
        Calcula la distribución de una característica en las recomendaciones.
        
        Args:
            recommendations: Lista de recomendaciones
            key_func: Función para extraer la característica
            
        Returns:
            Diccionario con la distribución
        """
        distribution = {}
        for recommendation in recommendations:
            key = key_func(recommendation)
            distribution[key] = distribution.get(key, 0) + 1
        return distribution
    
    def generate_executive_summary(self, recommendations: List[RecomendacionDiaria]) -> Dict[str, Any]:
        """
        Genera un resumen ejecutivo de las recomendaciones.
        
        Args:
            recommendations: Lista de recomendaciones del día
            
        Returns:
            Diccionario con resumen ejecutivo
        """
        try:
            if not recommendations:
                return {
                    'status': 'no_recommendations',
                    'message': 'No hay recomendaciones para hoy',
                    'date': datetime.now().strftime('%Y-%m-%d')
                }
            
            stats = self.generate_daily_statistics(recommendations)
            
            # Determinar el sentimiento general del mercado
            market_sentiment = self._determine_market_sentiment(stats)
            
            # Identificar insights clave
            key_insights = self._generate_key_insights(stats)
            
            # Recomendaciones de acción
            action_recommendations = self._generate_action_recommendations(stats)
            
            return {
                'status': 'success',
                'date': recommendations[0].fecha.strftime('%Y-%m-%d'),
                'market_sentiment': market_sentiment,
                'key_metrics': {
                    'total_opportunities': stats.total_recommendations,
                    'consensus_rate': f"{stats.consensus_rate:.1f}%",
                    'avg_roi': f"{stats.avg_roi:.1f}%",
                    'avg_confidence': f"{stats.avg_confidence:.1f}/100",
                    'top_strategy': max(stats.strategy_distribution.keys(), key=lambda k: stats.strategy_distribution[k]) if stats.strategy_distribution else 'N/A'
                },
                'key_insights': key_insights,
                'action_recommendations': action_recommendations,
                'top_opportunities': [
                    {
                        'symbol': opp.simbolo,
                        'roi': f"{opp.roi_porcentaje:.1f}%",
                        'strategy': opp.estrategia_gemini,
                        'consensus': opp.consenso_estrategia,
                        'risk': opp.nivel_riesgo
                    }
                    for opp in stats.top_opportunities
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Error generando resumen ejecutivo: {e}")
            return {
                'status': 'error',
                'message': f'Error generando resumen: {str(e)}',
                'date': datetime.now().strftime('%Y-%m-%d')
            }
    
    def _determine_market_sentiment(self, stats: ReportStatistics) -> str:
        """
        Determina el sentimiento general del mercado basado en las métricas.
        
        Args:
            stats: Estadísticas del reporte
            
        Returns:
            Sentimiento del mercado ('bullish', 'bearish', 'neutral')
        """
        if stats.total_recommendations == 0:
            return 'neutral'
        
        # Factores para determinar sentimiento
        high_roi_count = sum(1 for opp in stats.top_opportunities if opp.roi_porcentaje > 15)
        low_risk_count = sum(1 for opp in stats.top_opportunities if opp.nivel_riesgo == 'BAJO')
        high_consensus_rate = stats.consensus_rate > 70
        
        if high_roi_count >= 3 and low_risk_count >= 2 and high_consensus_rate:
            return 'bullish'
        elif stats.avg_roi < 5 or stats.consensus_rate < 30:
            return 'bearish'
        else:
            return 'neutral'
    
    def _generate_key_insights(self, stats: ReportStatistics) -> List[str]:
        """
        Genera insights clave basados en las estadísticas.
        
        Args:
            stats: Estadísticas del reporte
            
        Returns:
            Lista de insights clave
        """
        insights = []
        
        # Insight sobre consenso
        if stats.consensus_rate > 80:
            insights.append(f"Excelente consenso entre análisis cuantitativo y cualitativo ({stats.consensus_rate:.1f}%)")
        elif stats.consensus_rate < 50:
            insights.append(f"Baja concordancia entre análisis cuantitativo y cualitativo ({stats.consensus_rate:.1f}%)")
        
        # Insight sobre ROI
        if stats.avg_roi > 20:
            insights.append(f"ROI promedio muy alto ({stats.avg_roi:.1f}%) - Oportunidades excepcionales")
        elif stats.avg_roi < 5:
            insights.append(f"ROI promedio bajo ({stats.avg_roi:.1f}%) - Mercado poco atractivo")
        
        # Insight sobre estrategias
        if stats.strategy_distribution:
            dominant_strategy = max(stats.strategy_distribution.keys(), key=lambda k: stats.strategy_distribution[k])
            strategy_count = stats.strategy_distribution[dominant_strategy]
            if strategy_count > stats.total_recommendations * 0.6:
                strategy_names = {'grid': 'Grid Trading', 'dca': 'DCA', 'btd': 'Buy The Dip'}
                insights.append(f"Dominio de estrategia {strategy_names.get(dominant_strategy, dominant_strategy)} ({strategy_count}/{stats.total_recommendations})")
        
        # Insight sobre riesgo
        if stats.risk_distribution:
            low_risk_count = stats.risk_distribution.get('BAJO', 0)
            high_risk_count = stats.risk_distribution.get('ALTO', 0)
            if low_risk_count > stats.total_recommendations * 0.7:
                insights.append(f"Mayoría de oportunidades de bajo riesgo ({low_risk_count}/{stats.total_recommendations})")
            elif high_risk_count > stats.total_recommendations * 0.5:
                insights.append(f"Muchas oportunidades de alto riesgo ({high_risk_count}/{stats.total_recommendations})")
        
        return insights
    
    def _generate_action_recommendations(self, stats: ReportStatistics) -> List[str]:
        """
        Genera recomendaciones de acción basadas en las estadísticas.
        
        Args:
            stats: Estadísticas del reporte
            
        Returns:
            Lista de recomendaciones de acción
        """
        actions = []
        
        # Recomendaciones basadas en consenso
        if stats.consensus_rate < 50:
            actions.append("Revisar discrepancias entre análisis cuantitativo y cualitativo")
        
        # Recomendaciones basadas en ROI
        if stats.avg_roi > 15:
            actions.append("Considerar aumentar el capital destinado a trading")
        elif stats.avg_roi < 5:
            actions.append("Evaluar condiciones de mercado antes de invertir")
        
        # Recomendaciones basadas en riesgo
        if stats.risk_distribution:
            high_risk_count = stats.risk_distribution.get('ALTO', 0)
            if high_risk_count > stats.total_recommendations * 0.6:
                actions.append("Implementar gestión de riesgo estricta")
        
        # Recomendaciones basadas en diversificación
        if len(stats.strategy_distribution) == 1:
            actions.append("Considerar diversificar estrategias de trading")
        
        return actions
    
    def generate_detailed_report(self, recommendations: List[RecomendacionDiaria]) -> Dict[str, Any]:
        """
        Genera un reporte detallado completo.
        
        Args:
            recommendations: Lista de recomendaciones del día
            
        Returns:
            Diccionario con reporte detallado
        """
        try:
            stats = self.generate_daily_statistics(recommendations)
            executive_summary = self.generate_executive_summary(recommendations)
            
            return {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'report_type': 'daily_detailed',
                    'version': '1.0'
                },
                'executive_summary': executive_summary,
                'detailed_statistics': {
                    'total_recommendations': stats.total_recommendations,
                    'consensus_metrics': {
                        'consensus_count': stats.consensus_count,
                        'consensus_rate': stats.consensus_rate,
                        'non_consensus_count': stats.total_recommendations - stats.consensus_count
                    },
                    'performance_metrics': {
                        'avg_roi': stats.avg_roi,
                        'avg_confidence': stats.avg_confidence,
                        'roi_range': {
                            'min': min(r.roi_porcentaje for r in recommendations) if recommendations else 0,
                            'max': max(r.roi_porcentaje for r in recommendations) if recommendations else 0
                        }
                    },
                    'distributions': {
                        'strategies': stats.strategy_distribution,
                        'risk_levels': stats.risk_distribution,
                        'categories': stats.category_distribution
                    }
                },
                'recommendations_detail': [
                    {
                        'symbol': r.simbolo,
                        'strategy_recommended': r.estrategia_recomendada,
                        'strategy_gemini': r.estrategia_gemini,
                        'consensus': r.consenso_estrategia,
                        'roi': r.roi_porcentaje,
                        'risk_level': r.nivel_riesgo,
                        'category': r.categoria_rendimiento,
                        'final_recommendation': r.recomendacion_final,
                        'confidence_score': r.score_confianza_gemini,
                        'final_score': r.score_final
                    }
                    for r in recommendations
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Error generando reporte detallado: {e}")
            return {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'report_type': 'daily_detailed',
                    'version': '1.0'
                },
                'error': str(e),
                'status': 'error'
            }
    
    def get_report_summary(self, recommendations: List[RecomendacionDiaria]) -> str:
        """
        Genera un resumen textual del reporte.
        
        Args:
            recommendations: Lista de recomendaciones del día
            
        Returns:
            Resumen textual del reporte
        """
        try:
            if not recommendations:
                return "No hay recomendaciones para el día de hoy."
            
            stats = self.generate_daily_statistics(recommendations)
            
            summary = f"""
RESUMEN DIARIO DE TRADING - {recommendations[0].fecha.strftime('%Y-%m-%d')}

📊 ESTADÍSTICAS GENERALES:
• Total de recomendaciones: {stats.total_recommendations}
• Consenso Quant-AI: {stats.consensus_count}/{stats.total_recommendations} ({stats.consensus_rate:.1f}%)
• ROI promedio: {stats.avg_roi:.1f}%
• Confianza promedio: {stats.avg_confidence:.1f}/100

🎯 DISTRIBUCIÓN POR ESTRATEGIA:
{self._format_distribution_text(stats.strategy_distribution)}

⚡ DISTRIBUCIÓN POR RIESGO:
{self._format_distribution_text(stats.risk_distribution)}

🏆 TOP 3 OPORTUNIDADES:
{self._format_top_opportunities_text(stats.top_opportunities[:3])}

🔍 INSIGHTS CLAVE:
{chr(10).join(f'• {insight}' for insight in self._generate_key_insights(stats))}
"""
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"❌ Error generando resumen: {e}")
            return f"Error generando resumen: {str(e)}"
    
    def _format_distribution_text(self, distribution: Dict[str, int]) -> str:
        """Formatea una distribución como texto."""
        if not distribution:
            return "• No hay datos disponibles"
        
        lines = []
        for key, count in distribution.items():
            lines.append(f"• {key}: {count}")
        return "\n".join(lines)
    
    def _format_top_opportunities_text(self, opportunities: List[RecomendacionDiaria]) -> str:
        """Formatea las top oportunidades como texto."""
        if not opportunities:
            return "• No hay oportunidades disponibles"
        
        lines = []
        for i, opp in enumerate(opportunities, 1):
            consensus = "✅" if opp.consenso_estrategia else "⚠️"
            lines.append(f"{i}. {opp.simbolo} - {opp.roi_porcentaje:.1f}% {consensus}")
        return "\n".join(lines) 