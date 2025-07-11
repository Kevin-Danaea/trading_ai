"""
Opportunity Ranking Service - Servicio de Ranking de Oportunidades
==================================================================

El "Cerebro" del sistema que evalúa y rankea las oportunidades de trading
para seleccionar las Top 3-5 basándose en múltiples criterios.

Funcionalidad:
- Recibe 10 oportunidades con estrategias optimizadas
- Aplica algoritmo de ranking multi-criterio
- Selecciona las Top 3-5 mejores oportunidades
- Proporciona justificación y confianza para cada selección
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import asdict

from app.domain.entities.trading_opportunity import TradingOpportunity, RankingResult
from app.domain.entities.crypto_candidate import CryptoCandidate

logger = logging.getLogger(__name__)


class OpportunityRankingService:
    """
    Servicio que implementa el algoritmo de ranking inteligente para
    seleccionar las mejores oportunidades de trading.
    """
    
    def __init__(self, 
                 target_count: int = 5,
                 min_score_threshold: float = 70.0,
                 risk_tolerance: str = 'moderate'):
        """
        Inicializa el servicio de ranking.
        
        Args:
            target_count: Número objetivo de oportunidades a seleccionar (3-5)
            min_score_threshold: Score mínimo para considerar una oportunidad
            risk_tolerance: Tolerancia al riesgo ('conservative', 'moderate', 'aggressive')
        """
        self.target_count = max(3, min(5, target_count))  # Entre 3-5
        self.min_score_threshold = min_score_threshold
        self.risk_tolerance = risk_tolerance
        
        # Configurar pesos según tolerancia al riesgo
        self.ranking_criteria = self._get_ranking_criteria()
        
        logger.info(f"🧠 OpportunityRankingService inicializado:")
        logger.info(f"   📊 Target: {self.target_count} oportunidades")
        logger.info(f"   🎯 Score mínimo: {self.min_score_threshold}")
        logger.info(f"   ⚖️ Tolerancia: {self.risk_tolerance}")
    
    def rank_opportunities(self, 
                         candidates: List[CryptoCandidate],
                         optimization_results: Dict[str, Dict[str, Any]]) -> RankingResult:
        """
        Rankea las oportunidades y selecciona las mejores.
        
        Args:
            candidates: Lista de candidatos del scanner
            optimization_results: Resultados de optimización por símbolo
            
        Returns:
            RankingResult con las mejores oportunidades rankeadas
        """
        logger.info(f"🧠 Iniciando ranking de {len(candidates)} oportunidades...")
        
        try:
            # 1. Convertir candidatos + resultados en TradingOpportunity
            opportunities = self._create_trading_opportunities(candidates, optimization_results)
            
            if not opportunities:
                logger.warning("⚠️ No se pudieron crear oportunidades válidas")
                return self._create_empty_result()
            
            logger.info(f"✅ Creadas {len(opportunities)} oportunidades válidas")
            
            # 2. Calcular scores finales para cada oportunidad
            scored_opportunities = self._calculate_final_scores(opportunities)
            
            # 3. Aplicar filtros de calidad
            filtered_opportunities = self._apply_quality_filters(scored_opportunities)
            
            # 4. Seleccionar las mejores N oportunidades
            top_opportunities = self._select_top_opportunities(filtered_opportunities)
            
            # 5. Crear resultado final
            result = self._create_ranking_result(
                top_opportunities, 
                scored_opportunities, 
                filtered_opportunities
            )
            
            self._log_ranking_results(result)
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en ranking de oportunidades: {e}")
            return self._create_empty_result()
    
    def _create_trading_opportunities(self, 
                                    candidates: List[CryptoCandidate],
                                    optimization_results: Dict[str, Dict[str, Any]]) -> List[TradingOpportunity]:
        """Convierte candidatos + resultados de optimización en TradingOpportunity."""
        opportunities = []
        
        for candidate in candidates:
            symbol = candidate.symbol
            
            if symbol not in optimization_results:
                logger.warning(f"⚠️ Sin resultados de optimización para {symbol}")
                continue
            
            opt_result = optimization_results[symbol]
            
            # Extraer la mejor estrategia del resultado de optimización
            best_strategy = self._extract_best_strategy(opt_result)
            
            if not best_strategy:
                logger.warning(f"⚠️ Sin estrategia válida para {symbol}")
                continue
            
            try:
                opportunity = TradingOpportunity(
                    # Candidato base
                    candidate=candidate,
                    
                    # Estrategia optimizada
                    strategy_name=best_strategy['strategy'],
                    optimized_params=best_strategy['params'],
                    
                    # Métricas de rendimiento
                    roi_percentage=float(best_strategy.get('Return [%]', 0.0)),
                    sharpe_ratio=float(best_strategy.get('Sharpe Ratio', 0.0)),
                    max_drawdown_percentage=abs(float(best_strategy.get('Max. Drawdown [%]', 100.0))),
                    win_rate_percentage=float(best_strategy.get('Win Rate [%]', 0.0)),
                    total_trades=int(best_strategy.get('# Trades', 0)),
                    avg_trade_percentage=float(best_strategy.get('Avg. Trade [%]', 0.0)),
                    volatility_percentage=float(best_strategy.get('Volatility [%]', 0.0)),
                    
                    # Métricas adicionales
                    calmar_ratio=float(best_strategy.get('Calmar Ratio', 0.0)),
                    sortino_ratio=float(best_strategy.get('Sortino Ratio', 0.0)),
                    exposure_time_percentage=float(best_strategy.get('Exposure Time [%]', 0.0)),
                    
                    # Información del proceso
                    optimization_iterations=int(opt_result.get('total_iterations', 0)),
                    optimization_duration_seconds=float(opt_result.get('duration_seconds', 0)),
                    backtest_period_days=30,  # Placeholder
                    
                    # Scores (se calcularán después)
                    final_score=0.0,
                    risk_adjusted_score=0.0,
                    confidence_level=0.8,  # Default
                    
                    # Metadatos
                    created_at=datetime.now(),
                    market_conditions=self._determine_market_conditions(candidate)
                )
                
                opportunities.append(opportunity)
                logger.debug(f"✅ Oportunidad creada para {symbol}")
                
            except Exception as e:
                logger.warning(f"⚠️ Error creando oportunidad para {symbol}: {e}")
                continue
        
        return opportunities
    
    def _extract_best_strategy(self, opt_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extrae la mejor estrategia del resultado de optimización."""
        try:
            # El optimizador devuelve las mejores configuraciones
            best_configs = opt_result.get('best_configurations', [])
            
            if not best_configs:
                return None
            
            # Tomar la mejor configuración (primera en la lista)
            best_config = best_configs[0]
            
            return {
                'strategy': best_config.get('strategy', 'unknown'),
                'params': best_config.get('params', {}),
                **best_config.get('metrics', {})
            }
            
        except Exception as e:
            logger.warning(f"Error extrayendo mejor estrategia: {e}")
            return None
    
    def _calculate_final_scores(self, opportunities: List[TradingOpportunity]) -> List[TradingOpportunity]:
        """Calcula los scores finales usando algoritmo multi-criterio."""
        logger.info("🔢 Calculando scores finales...")
        
        scored_opportunities = []
        
        for opp in opportunities:
            try:
                # Calcular score final basado en criterios ponderados
                final_score = self._calculate_weighted_score(opp)
                
                # Calcular score ajustado por riesgo
                risk_adjusted_score = self._calculate_risk_adjusted_score(opp, final_score)
                
                # Calcular nivel de confianza
                confidence_level = self._calculate_confidence_level(opp)
                
                # Crear nueva oportunidad con scores actualizados
                updated_opp = TradingOpportunity(
                    **{k: v for k, v in asdict(opp).items() if k not in ['final_score', 'risk_adjusted_score', 'confidence_level']},
                    final_score=final_score,
                    risk_adjusted_score=risk_adjusted_score,
                    confidence_level=confidence_level
                )
                
                scored_opportunities.append(updated_opp)
                
            except Exception as e:
                logger.warning(f"Error calculando score para {opp.symbol}: {e}")
                continue
        
        return scored_opportunities
    
    def _calculate_weighted_score(self, opp: TradingOpportunity) -> float:
        """Calcula score ponderado basado en múltiples criterios."""
        criteria = self.ranking_criteria
        
        # Normalizar métricas a escala 0-100
        roi_score = min(100, max(0, (opp.roi_percentage + 50) * 1.0))  # ROI: -50% a +50% → 0-100
        sharpe_score = min(100, max(0, opp.sharpe_ratio * 25))  # Sharpe: 0-4 → 0-100
        drawdown_score = min(100, max(0, 100 - opp.max_drawdown_percentage * 2))  # Drawdown: 0-50% → 100-0
        winrate_score = min(100, max(0, opp.win_rate_percentage))  # Win Rate: 0-100%
        volatility_score = min(100, max(0, 100 - opp.volatility_percentage))  # Volatilidad: menos es mejor
        
        # Score del candidato original (scanner)
        candidate_score = opp.candidate.score
        
        # Calcular score ponderado
        weighted_score = (
            criteria['roi'] * roi_score +
            criteria['sharpe'] * sharpe_score +
            criteria['drawdown'] * drawdown_score +
            criteria['winrate'] * winrate_score +
            criteria['volatility'] * volatility_score +
            criteria['candidate'] * candidate_score
        )
        
        return min(100.0, max(0.0, weighted_score))
    
    def _calculate_risk_adjusted_score(self, opp: TradingOpportunity, base_score: float) -> float:
        """Calcula score ajustado por riesgo."""
        # Factor de ajuste por riesgo
        risk_factor = 1.0
        
        # Penalizar alto drawdown
        if opp.max_drawdown_percentage > 30:
            risk_factor *= 0.7
        elif opp.max_drawdown_percentage > 20:
            risk_factor *= 0.85
        
        # Penalizar baja cantidad de trades (menos confiable)
        if opp.total_trades < 10:
            risk_factor *= 0.8
        elif opp.total_trades < 5:
            risk_factor *= 0.6
        
        # Bonificar buen Sharpe ratio
        if opp.sharpe_ratio > 2.0:
            risk_factor *= 1.1
        elif opp.sharpe_ratio > 1.5:
            risk_factor *= 1.05
        
        return min(100.0, base_score * risk_factor)
    
    def _calculate_confidence_level(self, opp: TradingOpportunity) -> float:
        """Calcula nivel de confianza en la predicción."""
        confidence = 0.5  # Base
        
        # Más trades = más confianza
        if opp.total_trades >= 20:
            confidence += 0.3
        elif opp.total_trades >= 10:
            confidence += 0.2
        elif opp.total_trades >= 5:
            confidence += 0.1
        
        # Sharpe ratio consistente = más confianza
        if opp.sharpe_ratio > 1.5:
            confidence += 0.2
        elif opp.sharpe_ratio > 1.0:
            confidence += 0.1
        
        # Score del candidato alto = más confianza
        if opp.candidate.score > 90:
            confidence += 0.15
        elif opp.candidate.score > 80:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _apply_quality_filters(self, opportunities: List[TradingOpportunity]) -> List[TradingOpportunity]:
        """Aplica filtros de calidad para eliminar oportunidades de baja calidad."""
        filtered = []
        
        for opp in opportunities:
            # Filtro de score mínimo
            if opp.final_score < self.min_score_threshold:
                logger.debug(f"🚫 {opp.symbol} filtrado por score bajo: {opp.final_score:.1f}")
                continue
            
            # Filtro de trades mínimos
            if opp.total_trades < 3:
                logger.debug(f"🚫 {opp.symbol} filtrado por pocos trades: {opp.total_trades}")
                continue
            
            # Filtro de drawdown máximo
            max_allowable_drawdown = {'conservative': 15, 'moderate': 25, 'aggressive': 35}
            if opp.max_drawdown_percentage > max_allowable_drawdown[self.risk_tolerance]:
                logger.debug(f"🚫 {opp.symbol} filtrado por alto drawdown: {opp.max_drawdown_percentage:.1f}%")
                continue
            
            filtered.append(opp)
        
        logger.info(f"🔍 Filtros aplicados: {len(opportunities)} → {len(filtered)} oportunidades")
        return filtered
    
    def _select_top_opportunities(self, opportunities: List[TradingOpportunity]) -> List[TradingOpportunity]:
        """Selecciona las mejores N oportunidades."""
        if not opportunities:
            return []
        
        # Ordenar por score final (descendente)
        sorted_opportunities = sorted(
            opportunities,
            key=lambda x: (x.final_score, x.risk_adjusted_score, x.sharpe_ratio),
            reverse=True
        )
        
        # Tomar las mejores N
        top_count = min(self.target_count, len(sorted_opportunities))
        selected = sorted_opportunities[:top_count]
        
        logger.info(f"🏆 Seleccionadas {len(selected)} mejores oportunidades")
        return selected
    
    def _create_ranking_result(self, 
                             top_opportunities: List[TradingOpportunity],
                             all_opportunities: List[TradingOpportunity],
                             filtered_opportunities: List[TradingOpportunity]) -> RankingResult:
        """Crea el resultado final del ranking."""
        
        # Calcular estadísticas
        if all_opportunities:
            avg_score = np.mean([opp.final_score for opp in all_opportunities])
            best_score = max([opp.final_score for opp in all_opportunities])
        else:
            avg_score = 0.0
            best_score = 0.0
        
        score_threshold = self.min_score_threshold
        
        # Distribución por categorías
        category_distribution = {}
        for opp in top_opportunities:
            category = opp.performance_category
            category_distribution[category] = category_distribution.get(category, 0) + 1
        
        return RankingResult(
            top_opportunities=top_opportunities,
            all_opportunities=all_opportunities,
            ranking_criteria=self.ranking_criteria,
            total_evaluated=len(all_opportunities),
            selected_count=len(top_opportunities),
            avg_score=float(avg_score),
            best_score=best_score,
            score_threshold=score_threshold,
            category_distribution=category_distribution,
            ranked_at=datetime.now()
        )
    
    def _create_empty_result(self) -> RankingResult:
        """Crea un resultado vacío para casos de error."""
        return RankingResult(
            top_opportunities=[],
            all_opportunities=[],
            ranking_criteria=self.ranking_criteria,
            total_evaluated=0,
            selected_count=0,
            avg_score=0.0,
            best_score=0.0,
            score_threshold=self.min_score_threshold,
            category_distribution={},
            ranked_at=datetime.now()
        )
    
    def _get_ranking_criteria(self) -> Dict[str, float]:
        """Retorna los pesos de criterios según tolerancia al riesgo."""
        criteria_by_tolerance = {
            'conservative': {
                'roi': 0.15,
                'sharpe': 0.25,
                'drawdown': 0.30,
                'winrate': 0.15,
                'volatility': 0.10,
                'candidate': 0.05
            },
            'moderate': {
                'roi': 0.25,
                'sharpe': 0.20,
                'drawdown': 0.20,
                'winrate': 0.15,
                'volatility': 0.10,
                'candidate': 0.10
            },
            'aggressive': {
                'roi': 0.35,
                'sharpe': 0.15,
                'drawdown': 0.10,
                'winrate': 0.20,
                'volatility': 0.05,
                'candidate': 0.15
            }
        }
        
        return criteria_by_tolerance.get(self.risk_tolerance, criteria_by_tolerance['moderate'])
    
    def _determine_market_conditions(self, candidate: CryptoCandidate) -> str:
        """Determina las condiciones de mercado basándose en el candidato."""
        # Lógica simplificada basada en cambios de precio
        if hasattr(candidate, 'price_change_7d'):
            change_7d = getattr(candidate, 'price_change_7d', 0)
            if change_7d > 10:
                return 'bullish'
            elif change_7d < -10:
                return 'bearish'
        
        return 'sideways'
    
    def _log_ranking_results(self, result: RankingResult):
        """Log detallado de los resultados del ranking."""
        logger.info("🏆 RANKING COMPLETADO:")
        logger.info("=" * 60)
        
        summary = result.get_summary()
        logger.info(f"📊 Evaluadas: {summary['total_evaluated']} | Seleccionadas: {summary['selected_count']}")
        logger.info(f"🎯 Mejor Score: {summary['best_score']} | Promedio: {summary['avg_score']}")
        logger.info(f"📈 Distribución: {summary['category_distribution']}")
        
        logger.info("\n🥇 TOP OPORTUNIDADES:")
        for i, opp in enumerate(result.top_opportunities, 1):
            summary_opp = opp.get_summary()
            logger.info(f"{i}. {summary_opp['symbol']} ({summary_opp['strategy']}) - "
                       f"Score: {summary_opp['final_score']} | "
                       f"ROI: {summary_opp['roi']} | "
                       f"Sharpe: {summary_opp['sharpe_ratio']} | "
                       f"Cat: {summary_opp['category']}")
        
        logger.info("=" * 60) 