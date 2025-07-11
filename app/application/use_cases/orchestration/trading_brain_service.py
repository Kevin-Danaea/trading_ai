"""
Trading Brain Service - Servicio Cerebro de Trading
==================================================

El cerebro principal del sistema de trading que orquesta todo el flujo:
Scanner → Optimizer → Ranking → Decisión

Este servicio coordina:
1. Scanner: Encuentra las mejores 10 monedas
2. Optimizer: Optimiza estrategias para cada moneda
3. Ranking: Selecciona las Top 3-5 oportunidades
4. Decision: Prepara las decisiones finales de trading
"""

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from app.application.use_cases.scanning.crypto_scanner_service import CryptoScannerService
from app.application.use_cases.optimization.bayesian_optimizer_service import BayesianOptimizerService
from app.application.use_cases.ranking.opportunity_ranking_service import OpportunityRankingService
from app.domain.entities import CryptoCandidate, TradingOpportunity, RankingResult
from app.infrastructure.providers.market_data_provider import MarketDataProvider
from app.infrastructure.providers.sentiment_data_provider import SentimentDataProvider

logger = logging.getLogger(__name__)


class TradingBrainService:
    """
    El cerebro principal que orquesta todo el flujo de trading inteligente.
    
    Flujo completo:
    1. 🔍 Scanner: Top 150 → Mejores 10 candidatos
    2. 🧪 Optimizer: 10 candidatos → 10 estrategias optimizadas
    3. 🧠 Ranking: 10 estrategias → Top 3-5 oportunidades
    4. 💡 Decision: Top 3-5 → Decisiones de trading listas
    """
    
    def __init__(self,
                 market_data_provider: MarketDataProvider,
                 sentiment_data_provider: SentimentDataProvider,
                 target_opportunities: int = 5,
                 risk_tolerance: str = 'moderate'):
        """
        Inicializa el cerebro de trading.
        
        Args:
            market_data_provider: Proveedor de datos de mercado
            sentiment_data_provider: Proveedor de datos de sentimiento
            target_opportunities: Número objetivo de oportunidades finales (3-5)
            risk_tolerance: Tolerancia al riesgo ('conservative', 'moderate', 'aggressive')
        """
        self.market_data_provider = market_data_provider
        self.sentiment_data_provider = sentiment_data_provider
        self.target_opportunities = max(3, min(5, target_opportunities))
        self.risk_tolerance = risk_tolerance
        
        # Inicializar servicios
        self.scanner = CryptoScannerService(
            market_data_provider=market_data_provider,
            sentiment_data_provider=sentiment_data_provider,
            top_n=150,  # Analizar Top 150
            target_candidates=10  # Seleccionar 10 candidatos
        )
        
        # Importar servicios necesarios para el optimizer
        from app.infrastructure.services.backtesting_service import BacktestingService
        
        # Crear servicios para el optimizer
        backtesting_service = BacktestingService()
        
        self.optimizer = BayesianOptimizerService(
            backtesting_service=backtesting_service,
            data_service=market_data_provider,  # Usar como data service
            optimization_window_months=9
        )
        
        self.ranker = OpportunityRankingService(
            target_count=target_opportunities,
            min_score_threshold=70.0,
            risk_tolerance=risk_tolerance
        )
        
        logger.info("🧠 TradingBrainService inicializado:")
        logger.info(f"   🎯 Target: {self.target_opportunities} oportunidades")
        logger.info(f"   ⚖️ Riesgo: {self.risk_tolerance}")
    
    def analyze_market_and_decide(self) -> Dict[str, Any]:
        """
        Ejecuta el análisis completo del mercado y toma decisiones de trading.
        
        Returns:
            Diccionario con:
            - top_opportunities: Lista de mejores oportunidades
            - ranking_result: Resultado completo del ranking
            - execution_summary: Resumen de la ejecución
            - timing_breakdown: Desglose de tiempos
        """
        logger.info("🚀 INICIANDO ANÁLISIS COMPLETO DEL MERCADO...")
        start_time = time.time()
        
        timing = {}
        execution_summary = {
            'started_at': datetime.now(),
            'status': 'running',
            'phase': 'initialization',
            'errors': []
        }
        
        try:
            # FASE 1: SCANNING - Encontrar mejores candidatos
            execution_summary['phase'] = 'scanning'
            logger.info("🔍 FASE 1: SCANNING - Buscando mejores candidatos...")
            
            phase_start = time.time()
            candidates = self.scanner.scan_market()
            timing['scanning'] = time.time() - phase_start
            
            if not candidates:
                raise ValueError("No se encontraron candidatos válidos en el scanning")
            
            logger.info(f"✅ SCANNING completado: {len(candidates)} candidatos encontrados")
            
            # FASE 2: OPTIMIZATION - Optimizar estrategias
            execution_summary['phase'] = 'optimization'
            logger.info("🧪 FASE 2: OPTIMIZATION - Optimizando estrategias...")
            
            phase_start = time.time()
            optimization_results = self._optimize_candidates(candidates)
            timing['optimization'] = time.time() - phase_start
            
            if not optimization_results:
                raise ValueError("No se pudieron optimizar estrategias para los candidatos")
            
            logger.info(f"✅ OPTIMIZATION completado: {len(optimization_results)} estrategias optimizadas")
            
            # FASE 3: RANKING - Seleccionar mejores oportunidades
            execution_summary['phase'] = 'ranking'
            logger.info("🧠 FASE 3: RANKING - Seleccionando mejores oportunidades...")
            
            phase_start = time.time()
            ranking_result = self.ranker.rank_opportunities(candidates, optimization_results)
            timing['ranking'] = time.time() - phase_start
            
            if not ranking_result.top_opportunities:
                raise ValueError("No se encontraron oportunidades válidas después del ranking")
            
            logger.info(f"✅ RANKING completado: {len(ranking_result.top_opportunities)} oportunidades seleccionadas")
            
            # FASE 4: FINALIZATION - Preparar resultado final
            execution_summary['phase'] = 'finalization'
            total_time = time.time() - start_time
            timing['total'] = total_time
            
            execution_summary.update({
                'status': 'completed',
                'phase': 'completed',
                'completed_at': datetime.now(),
                'candidates_found': len(candidates),
                'strategies_optimized': len(optimization_results),
                'opportunities_selected': len(ranking_result.top_opportunities),
                'total_duration_seconds': total_time
            })
            
            # Crear resultado final
            result = {
                'top_opportunities': ranking_result.top_opportunities,
                'ranking_result': ranking_result,
                'execution_summary': execution_summary,
                'timing_breakdown': timing,
                'recommendations': self._generate_recommendations(ranking_result)
            }
            
            self._log_final_results(result)
            return result
            
        except Exception as e:
            execution_summary.update({
                'status': 'failed',
                'error': str(e),
                'failed_at': datetime.now(),
                'total_duration_seconds': time.time() - start_time
            })
            execution_summary['errors'].append(str(e))
            
            logger.error(f"❌ Error en análisis de mercado: {e}")
            
            return {
                'top_opportunities': [],
                'ranking_result': None,
                'execution_summary': execution_summary,
                'timing_breakdown': timing,
                'recommendations': []
            }
    
    def _optimize_candidates(self, candidates: List[CryptoCandidate]) -> Dict[str, Dict[str, Any]]:
        """Optimiza estrategias para todos los candidatos."""
        optimization_results = {}
        
        logger.info(f"🧪 Optimizando estrategias para {len(candidates)} candidatos...")
        
        for i, candidate in enumerate(candidates, 1):
            symbol = candidate.symbol
            logger.info(f"🔬 Optimizando {symbol} ({i}/{len(candidates)})...")
            
            try:
                # Ejecutar optimización bayesiana para este candidato
                # Optimizar todas las estrategias para este símbolo
                strategies = ['grid', 'dca', 'btd']
                best_results = []
                
                for strategy in strategies:
                    strategy_result = self.optimizer.optimize_symbol(
                        symbol=symbol,
                        strategy=strategy,
                        n_trials=50,  # Reducido para velocidad
                        timeout_minutes=3  # Límite de tiempo por estrategia
                    )
                    
                    if strategy_result:
                        best_results.append({
                            'strategy': strategy,
                            'params': strategy_result.best_params,
                            'value': strategy_result.best_value,
                            **strategy_result.study_stats
                        })
                
                # Crear resultado compatible con el formato esperado
                if best_results:
                    # Ordenar por valor (score) y tomar la mejor
                    best_results.sort(key=lambda x: x.get('value', 0), reverse=True)
                    best_config = best_results[0]
                    
                    result = {
                        'best_configurations': [{
                            'strategy': best_config['strategy'],
                            'params': best_config['params'],
                            'metrics': {
                                'Return [%]': best_config.get('roi', 0),
                                'Sharpe Ratio': best_config.get('sharpe_ratio', 0),
                                'Max. Drawdown [%]': best_config.get('max_drawdown', 0),
                                'Win Rate [%]': best_config.get('win_rate', 0),
                                '# Trades': best_config.get('total_trades', 0),
                                'Calmar Ratio': best_config.get('calmar_ratio', 0),
                                'Volatility [%]': 0,  # Placeholder
                                'Avg. Trade [%]': 0,  # Placeholder
                                'Sortino Ratio': 0,  # Placeholder
                                'Exposure Time [%]': 0  # Placeholder
                            }
                        }],
                        'total_iterations': sum(r.get('trials_total', 0) for r in best_results),
                        'duration_seconds': 180  # Aproximado
                    }
                else:
                    result = None
                
                if result and 'best_configurations' in result:
                    optimization_results[symbol] = result
                    logger.info(f"✅ {symbol} optimizado exitosamente")
                else:
                    logger.warning(f"⚠️ {symbol} sin resultados de optimización válidos")
                
            except Exception as e:
                logger.warning(f"⚠️ Error optimizando {symbol}: {e}")
                continue
        
        logger.info(f"✅ Optimización completada: {len(optimization_results)}/{len(candidates)} éxitos")
        return optimization_results
    
    def _generate_recommendations(self, ranking_result: RankingResult) -> List[Dict[str, Any]]:
        """Genera recomendaciones basadas en los resultados del ranking."""
        recommendations = []
        
        if not ranking_result.top_opportunities:
            return [{'type': 'warning', 'message': 'No se encontraron oportunidades válidas'}]
        
        # Recomendación de la mejor oportunidad
        best_opp = ranking_result.get_best_opportunity()
        if best_opp:
            recommendations.append({
                'type': 'primary',
                'symbol': best_opp.symbol,
                'strategy': best_opp.strategy_name,
                'message': f"MEJOR OPORTUNIDAD: {best_opp.symbol} con estrategia {best_opp.strategy_name.upper()}",
                'details': {
                    'roi_expected': f"{best_opp.roi_percentage:.1f}%",
                    'risk_level': best_opp.performance_category,
                    'confidence': f"{best_opp.confidence_level:.0%}",
                    'max_drawdown': f"{best_opp.max_drawdown_percentage:.1f}%"
                }
            })
        
        # Análisis de distribución de riesgo
        categories = ranking_result.category_distribution
        if 'PREMIUM' in categories and categories['PREMIUM'] > 0:
            recommendations.append({
                'type': 'success',
                'message': f"Excelente diversificación: {categories['PREMIUM']} oportunidades PREMIUM encontradas"
            })
        
        # Recomendaciones por tolerancia al riesgo
        if self.risk_tolerance == 'conservative':
            conservative_opps = [opp for opp in ranking_result.top_opportunities if opp.is_low_risk]
            if conservative_opps:
                recommendations.append({
                    'type': 'info',
                    'message': f"Para perfil conservador: {len(conservative_opps)} oportunidades de bajo riesgo disponibles"
                })
        
        # Alertas de riesgo
        high_risk_count = sum(1 for opp in ranking_result.top_opportunities 
                             if opp.max_drawdown_percentage > 20)
        if high_risk_count > 0:
            recommendations.append({
                'type': 'warning',
                'message': f"ATENCIÓN: {high_risk_count} oportunidades con alto riesgo (>20% drawdown)"
            })
        
        return recommendations
    
    def get_quick_market_overview(self) -> Dict[str, Any]:
        """
        Versión rápida que solo ejecuta el scanner para una vista general del mercado.
        
        Returns:
            Vista general rápida del mercado sin optimización completa
        """
        logger.info("⚡ Ejecutando vista rápida del mercado...")
        
        try:
            start_time = time.time()
            
            # Solo ejecutar scanner
            candidates = self.scanner.scan_market()
            
            total_time = time.time() - start_time
            
            return {
                'candidates': candidates,
                'candidate_count': len(candidates),
                'top_symbols': [c.symbol for c in candidates[:5]],
                'avg_score': sum(c.score for c in candidates) / len(candidates) if candidates else 0,
                'scan_duration_seconds': total_time,
                'market_sentiment': 'positive' if candidates and candidates[0].score > 85 else 'neutral',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Error en vista rápida: {e}")
            return {
                'candidates': [],
                'candidate_count': 0,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _log_final_results(self, result: Dict[str, Any]):
        """Log detallado de los resultados finales."""
        logger.info("🎉 ANÁLISIS COMPLETO FINALIZADO:")
        logger.info("=" * 80)
        
        summary = result['execution_summary']
        timing = result['timing_breakdown']
        opportunities = result['top_opportunities']
        
        # Resumen de ejecución
        logger.info(f"⏱️  Tiempo total: {timing.get('total', 0):.1f}s")
        logger.info(f"   📊 Scanning: {timing.get('scanning', 0):.1f}s")
        logger.info(f"   🧪 Optimization: {timing.get('optimization', 0):.1f}s") 
        logger.info(f"   🧠 Ranking: {timing.get('ranking', 0):.1f}s")
        
        # Estadísticas
        logger.info(f"📈 Candidatos encontrados: {summary.get('candidates_found', 0)}")
        logger.info(f"🎯 Estrategias optimizadas: {summary.get('strategies_optimized', 0)}")
        logger.info(f"🏆 Oportunidades seleccionadas: {summary.get('opportunities_selected', 0)}")
        
        # Top oportunidades
        logger.info(f"\n🥇 TOP {len(opportunities)} OPORTUNIDADES FINALES:")
        for i, opp in enumerate(opportunities, 1):
            logger.info(f"{i}. {opp.symbol} ({opp.strategy_name.upper()}) - "
                       f"Score: {opp.final_score:.1f}/100 | "
                       f"ROI: {opp.roi_percentage:.1f}% | "
                       f"Riesgo: {opp.performance_category}")
        
        # Recomendaciones
        recommendations = result.get('recommendations', [])
        if recommendations:
            logger.info(f"\n💡 RECOMENDACIONES:")
            for rec in recommendations:
                logger.info(f"   {rec['type'].upper()}: {rec['message']}")
        
        logger.info("=" * 80) 