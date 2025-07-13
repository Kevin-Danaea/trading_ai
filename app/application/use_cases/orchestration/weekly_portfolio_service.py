"""
Weekly Portfolio Service - Servicio de Cartera Semanal
======================================================

Servicio especializado para la selección semanal de cartera de trading.
Garantiza una distribución estratégica de 5 recomendaciones:
- 1 GRID sólida (spot)
- 2 DCA/BTD sólidas (spot)  
- 1 GRID Futuros sólida
- 1 DCA Futuros sólida

Este servicio reemplaza la lógica diaria con un enfoque semanal más estable.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from app.domain.entities.trading_opportunity import TradingOpportunity
from app.domain.entities.qualitative_analysis import QualitativeAnalysis
from app.domain.entities.daily_recommendation import RecomendacionDiaria
from app.application.use_cases.qualitative_analysis.qualitative_filter_service import QualitativeResult
from app.application.use_cases.qualitative_analysis.futures_analysis_service import FuturesAnalysisService

logger = logging.getLogger(__name__)


@dataclass
class WeeklyPortfolioSelection:
    """Resultado de la selección semanal de cartera."""
    
    # Recomendaciones spot
    grid_spot: Optional[RecomendacionDiaria] = None
    dca_btd_spot: List[RecomendacionDiaria] = field(default_factory=list)
    
    # Recomendaciones futuros
    grid_futures: Optional[RecomendacionDiaria] = None
    dca_futures: Optional[RecomendacionDiaria] = None
    
    # Metadatos
    selection_date: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    total_selected: int = 0
    selection_quality: str = "unknown"
    
    def __post_init__(self):
        """Inicialización post-creación."""
        if self.selection_date is None:
            self.selection_date = datetime.now()
        if self.valid_until is None:
            self.valid_until = self.selection_date + timedelta(days=7)
        
        # Calcular total seleccionado
        self.total_selected = (
            (1 if self.grid_spot else 0) +
            len(self.dca_btd_spot) +
            (1 if self.grid_futures else 0) +
            (1 if self.dca_futures else 0)
        )
    
    def get_all_recommendations(self) -> List[RecomendacionDiaria]:
        """Obtiene todas las recomendaciones como lista."""
        recommendations = []
        
        if self.grid_spot:
            recommendations.append(self.grid_spot)
        
        recommendations.extend(self.dca_btd_spot)
        
        if self.grid_futures:
            recommendations.append(self.grid_futures)
        
        if self.dca_futures:
            recommendations.append(self.dca_futures)
        
        return recommendations
    
    def is_complete(self) -> bool:
        """Verifica si la selección está completa (5 recomendaciones)."""
        return self.total_selected == 5
    
    def get_completion_status(self) -> Dict[str, Any]:
        """Obtiene el estado de completitud de la selección."""
        return {
            'is_complete': self.is_complete(),
            'total_selected': self.total_selected,
            'target_total': 5,
            'missing': {
                'grid_spot': self.grid_spot is None,
                'dca_btd_spot': len(self.dca_btd_spot) < 2,
                'grid_futures': self.grid_futures is None,
                'dca_futures': self.dca_futures is None
            }
        }


class WeeklyPortfolioService:
    """
    Servicio para gestionar la selección semanal de cartera.
    
    Implementa la lógica de filtrado estricto para garantizar:
    - Máxima solidez en cada selección
    - Distribución estratégica balanceada
    - Validación de calidad antes de envío
    """
    
    def __init__(self, min_confidence_threshold: float = 0.75, futures_service: Optional[FuturesAnalysisService] = None):
        """
        Inicializa el servicio de cartera semanal.
        
        Args:
            min_confidence_threshold: Umbral mínimo de confianza (0.75 = 75%)
            futures_service: Servicio especializado para análisis de futuros
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.futures_service = futures_service
        self.logger = logging.getLogger(__name__)
    
    def select_weekly_portfolio(
        self,
        qualitative_results: List[QualitativeResult],
        force_selection: bool = False
    ) -> WeeklyPortfolioSelection:
        """
        Selecciona la cartera semanal desde los resultados cualitativos.
        
        Args:
            qualitative_results: Resultados del análisis cualitativo
            force_selection: Si True, selecciona lo mejor disponible aunque no cumpla todos los criterios
            
        Returns:
            Selección semanal de cartera
        """
        logger.info(f"🎯 Iniciando selección de cartera semanal desde {len(qualitative_results)} candidatos")
        
        # Filtrar solo recomendaciones BUY con alta confianza
        solid_candidates = self._filter_solid_candidates(qualitative_results)
        
        if not solid_candidates:
            logger.warning("⚠️ No se encontraron candidatos sólidos para cartera semanal")
            if not force_selection:
                return WeeklyPortfolioSelection(selection_quality="insufficient_quality")
        
        logger.info(f"✅ {len(solid_candidates)} candidatos sólidos encontrados")
        
        # Separar candidatos spot y futuros
        spot_candidates, futures_candidates = self._separate_spot_and_futures(solid_candidates)
        
        logger.info(f"📊 Candidatos spot: {len(spot_candidates)}, futuros: {len(futures_candidates)}")
        
        # Seleccionar cartera
        selection = WeeklyPortfolioSelection()
        
        # 1. Seleccionar GRID spot (1 recomendación)
        selection.grid_spot = self._select_best_grid_spot(spot_candidates)
        
        # 2. Seleccionar DCA/BTD spot (2 recomendaciones)
        selection.dca_btd_spot = self._select_best_dca_btd_spot(spot_candidates, exclude_symbol=selection.grid_spot.simbolo if selection.grid_spot else None)
        
        # 3. Seleccionar GRID futuros (1 recomendación)
        selection.grid_futures = self._select_best_grid_futures(futures_candidates)
        
        # 4. Seleccionar DCA futuros (1 recomendación)
        selection.dca_futures = self._select_best_dca_futures(futures_candidates, exclude_symbol=selection.grid_futures.simbolo if selection.grid_futures else None)
        
        # Evaluar calidad de la selección
        selection.selection_quality = self._evaluate_selection_quality(selection)
        
        logger.info(f"🎯 Selección completada: {selection.total_selected}/5 recomendaciones")
        logger.info(f"📈 Calidad de selección: {selection.selection_quality}")
        
        return selection
    
    def _filter_solid_candidates(self, qualitative_results: List[QualitativeResult]) -> List[QualitativeResult]:
        """
        Filtra candidatos sólidos según criterios diferenciados para spot y futuros.
        
        Args:
            qualitative_results: Resultados cualitativos
            
        Returns:
            Lista de candidatos sólidos
        """
        solid_candidates = []
        
        for result in qualitative_results:
            # Determinar si es candidato para futuros
            is_futures_candidate = (
                hasattr(result.opportunity.candidate, 'is_suitable_for_futures') and
                result.opportunity.candidate.is_suitable_for_futures()
            )
            
            if is_futures_candidate:
                # Criterios más flexibles para futuros (alto riesgo requiere flexibilidad)
                is_solid = self._is_solid_futures_candidate(result)
            else:
                # Criterios estándar para spot
                is_solid = self._is_solid_spot_candidate(result)
            
            if is_solid:
                solid_candidates.append(result)
        
        # Ordenar por score de confianza descendente
        solid_candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return solid_candidates
    
    def _is_solid_spot_candidate(self, result: QualitativeResult) -> bool:
        """
        Evalúa si un candidato spot es sólido.
        
        Args:
            result: Resultado cualitativo
            
        Returns:
            True si es sólido para spot
        """
        return all([
            # Recomendación debe ser BUY
            result.analysis.recommendation.lower() == 'buy',
            
            # Confianza media o alta (más flexible)
            result.analysis.confidence_level.lower() in ['medium', 'high'],
            
            # Score de confianza >= 75% (más flexible)
            result.confidence_score >= self.min_confidence_threshold * 100,
            
            # Métricas cuantitativas sólidas (mantener para gestión de riesgo)
            result.opportunity.sharpe_ratio >= 1.5,
            result.opportunity.max_drawdown_percentage <= 15.0,
            result.opportunity.win_rate_percentage >= 60.0,
            
            # ROI mínimo
            result.opportunity.roi_percentage >= 10.0
        ])
    
    def _is_solid_futures_candidate(self, result: QualitativeResult) -> bool:
        """
        Evalúa si un candidato futuros es sólido con criterios ULTRA flexibles.
        
        Los futuros requieren mayor flexibilidad debido a:
        - Apalancamiento que amplifica ganancias
        - Volatilidad que puede ser aprovechada
        - Timing más importante que métricas perfectas
        
        Args:
            result: Resultado cualitativo
            
        Returns:
            True si es sólido para futuros
        """
        # Criterios básicos más flexibles
        basic_criteria = [
            # Recomendación debe ser BUY o HOLD (más flexible)
            result.analysis.recommendation.lower() in ['buy', 'hold'],
            
            # Confianza: acepta todos los niveles para futuros
            result.analysis.confidence_level.lower() in ['low', 'medium', 'high'],
            
            # Score de confianza >= 50% (muy flexible para futuros)
            result.confidence_score >= 50.0,
        ]
        
        # Criterios cuantitativos ULTRA flexibles para futuros
        quantitative_criteria = [
            result.opportunity.sharpe_ratio >= 1.0,  # Reducido de 1.2 a 1.0
            result.opportunity.max_drawdown_percentage <= 25.0,  # Aumentado de 20% a 25%
            result.opportunity.win_rate_percentage >= 50.0,  # Reducido de 55% a 50%
            result.opportunity.roi_percentage >= 5.0  # Reducido de 8% a 5%
        ]
        
        # Para futuros, solo necesitamos criterios básicos + al menos 3 de 4 cuantitativos
        return all(basic_criteria) and sum(quantitative_criteria) >= 3
    
    def _separate_spot_and_futures(self, candidates: List[QualitativeResult]) -> Tuple[List[QualitativeResult], List[QualitativeResult]]:
        """
        Separa candidatos en spot y futuros.
        
        Args:
            candidates: Lista de candidatos
            
        Returns:
            Tupla (candidatos_spot, candidatos_futuros)
        """
        spot_candidates = []
        futures_candidates = []
        
        for candidate in candidates:
            # Usar análisis cualitativo para determinar si es apto para futuros
            if candidate.analysis.suitable_for_futures:
                futures_candidates.append(candidate)
            else:
                spot_candidates.append(candidate)
        
        # Si no hay candidatos para futuros, usar análisis especializado
        if not futures_candidates and self.futures_service:
            logger.info("🔍 No hay candidatos marcados para futuros, usando análisis especializado")
            futures_candidates = self._analyze_candidates_for_futures(candidates)
        
        return spot_candidates, futures_candidates
    
    def _analyze_candidates_for_futures(self, candidates: List[QualitativeResult]) -> List[QualitativeResult]:
        """
        Analiza candidatos usando el servicio especializado de futuros.
        
        Args:
            candidates: Lista de candidatos a analizar
            
        Returns:
            Lista de candidatos aptos para futuros
        """
        futures_candidates = []
        
        for candidate in candidates:
            try:
                # Verificar que el servicio de futuros esté disponible
                if not self.futures_service:
                    continue
                    
                # Usar el servicio especializado para re-analizar para futuros
                futures_analysis = self.futures_service.analyze_futures_opportunity(candidate.opportunity.candidate)
                
                if futures_analysis and futures_analysis.suitable_for_futures:
                    # Crear nuevo resultado con análisis de futuros
                    futures_result = QualitativeResult(
                        opportunity=candidate.opportunity,
                        analysis=futures_analysis,
                        confidence_score=futures_analysis.confidence_score,
                        strategic_recommendation=futures_analysis.recommendation,
                        risk_assessment=futures_analysis.futures_risk_level,
                        execution_priority=candidate.execution_priority
                    )
                    futures_candidates.append(futures_result)
                    
            except Exception as e:
                logger.warning(f"⚠️ Error analizando {candidate.opportunity.symbol} para futuros: {e}")
                continue
        
        logger.info(f"🎯 Análisis especializado encontró {len(futures_candidates)} candidatos para futuros")
        return futures_candidates
    
    def _select_best_grid_spot(self, spot_candidates: List[QualitativeResult]) -> Optional[RecomendacionDiaria]:
        """
        Selecciona la mejor oportunidad GRID para spot.
        
        Args:
            spot_candidates: Candidatos spot
            
        Returns:
            Mejor recomendación GRID o None
        """
        grid_candidates = [
            c for c in spot_candidates 
            if c.analysis.recommended_strategy.lower() == 'grid'
        ]
        
        if not grid_candidates:
            logger.warning("⚠️ No se encontraron candidatos GRID spot")
            return None
        
        # Seleccionar el mejor basado en score y métricas
        best_grid = max(grid_candidates, key=lambda x: (
            x.confidence_score,
            x.opportunity.sharpe_ratio,
            -x.opportunity.max_drawdown_percentage
        ))
        
        return self._convert_to_recommendation(best_grid, "GRID_SPOT")
    
    def _select_best_dca_btd_spot(self, spot_candidates: List[QualitativeResult], exclude_symbol: Optional[str] = None) -> List[RecomendacionDiaria]:
        """
        Selecciona las 2 mejores oportunidades DCA/BTD para spot.
        
        Args:
            spot_candidates: Candidatos spot
            exclude_symbol: Símbolo a excluir (ya seleccionado)
            
        Returns:
            Lista de hasta 2 recomendaciones DCA/BTD
        """
        dca_btd_candidates = [
            c for c in spot_candidates 
            if c.analysis.recommended_strategy.lower() in ['dca', 'btd'] and
            (exclude_symbol is None or c.opportunity.candidate.symbol != exclude_symbol)
        ]
        
        if not dca_btd_candidates:
            logger.warning("⚠️ No se encontraron candidatos DCA/BTD spot")
            return []
        
        # Ordenar por calidad y seleccionar top 2
        dca_btd_candidates.sort(key=lambda x: (
            x.confidence_score,
            x.opportunity.roi_percentage,
            x.opportunity.sharpe_ratio
        ), reverse=True)
        
        selected = dca_btd_candidates[:2]
        
        recommendations = []
        for candidate in selected:
            strategy_type = f"{candidate.analysis.recommended_strategy.upper()}_SPOT"
            recommendations.append(self._convert_to_recommendation(candidate, strategy_type))
        
        return recommendations
    
    def _select_best_grid_futures(self, futures_candidates: List[QualitativeResult]) -> Optional[RecomendacionDiaria]:
        """
        Selecciona la mejor oportunidad GRID para futuros.
        
        Args:
            futures_candidates: Candidatos futuros
            
        Returns:
            Mejor recomendación GRID futuros o None
        """
        grid_futures_candidates = [
            c for c in futures_candidates 
            if c.analysis.recommended_strategy.lower() == 'grid'
        ]
        
        if not grid_futures_candidates:
            logger.warning("⚠️ No se encontraron candidatos GRID futuros")
            return None
        
        # Seleccionar el mejor con criterios específicos para futuros
        best_grid_futures = max(grid_futures_candidates, key=lambda x: (
            x.confidence_score,
            x.opportunity.sharpe_ratio,
            x.opportunity.candidate.get_futures_risk_level() == 'BAJO'  # Priorizar bajo riesgo
        ))
        
        return self._convert_to_recommendation(best_grid_futures, "GRID_FUTURES")
    
    def _select_best_dca_futures(self, futures_candidates: List[QualitativeResult], exclude_symbol: Optional[str] = None) -> Optional[RecomendacionDiaria]:
        """
        Selecciona la mejor oportunidad DCA para futuros.
        
        Args:
            futures_candidates: Candidatos futuros
            exclude_symbol: Símbolo a excluir
            
        Returns:
            Mejor recomendación DCA futuros o None
        """
        dca_futures_candidates = [
            c for c in futures_candidates 
            if c.analysis.recommended_strategy.lower() == 'dca' and
            (exclude_symbol is None or c.opportunity.candidate.symbol != exclude_symbol)
        ]
        
        if not dca_futures_candidates:
            logger.warning("⚠️ No se encontraron candidatos DCA futuros")
            return None
        
        # Seleccionar el mejor
        best_dca_futures = max(dca_futures_candidates, key=lambda x: (
            x.confidence_score,
            x.opportunity.roi_percentage,
            x.opportunity.candidate.get_optimal_leverage_for_futures()
        ))
        
        return self._convert_to_recommendation(best_dca_futures, "DCA_FUTURES")
    
    def _convert_to_recommendation(self, qualitative_result: QualitativeResult, category: str) -> RecomendacionDiaria:
        """
        Convierte un resultado cualitativo a recomendación diaria.
        
        Args:
            qualitative_result: Resultado cualitativo
            category: Categoría de la recomendación
            
        Returns:
            Recomendación diaria
        """
        recommendation = RecomendacionDiaria.from_trading_opportunity_and_analysis(
            opportunity=qualitative_result.opportunity,
            qualitative_analysis=qualitative_result.analysis,
            version_pipeline="weekly_v1.0"
        )
        
        # Personalizar para categoría semanal
        recommendation.categoria = category
        recommendation.es_semanal = True
        
        return recommendation
    
    def _evaluate_selection_quality(self, selection: WeeklyPortfolioSelection) -> str:
        """
        Evalúa la calidad de la selección semanal.
        
        Args:
            selection: Selección semanal
            
        Returns:
            Calidad: 'excellent', 'good', 'acceptable', 'poor'
        """
        if selection.total_selected == 5:
            return 'excellent'
        elif selection.total_selected >= 4:
            return 'good'
        elif selection.total_selected >= 3:
            return 'acceptable'
        else:
            return 'poor'
    
    def validate_weekly_selection(self, selection: WeeklyPortfolioSelection) -> Dict[str, Any]:
        """
        Valida la selección semanal antes del envío.
        
        Args:
            selection: Selección semanal
            
        Returns:
            Resultado de validación
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'quality_score': 0
        }
        
        # Validar completitud
        if not selection.is_complete():
            validation_result['warnings'].append(f"Selección incompleta: {selection.total_selected}/5")
        
        # Validar calidad individual
        all_recommendations = selection.get_all_recommendations()
        quality_scores = []
        
        for rec in all_recommendations:
            if rec.score_confianza_gemini < 80:
                validation_result['warnings'].append(f"{rec.simbolo}: Confianza baja ({rec.score_confianza_gemini})")
            
            if rec.roi_porcentaje < 10:
                validation_result['warnings'].append(f"{rec.simbolo}: ROI bajo ({rec.roi_porcentaje}%)")
            
            quality_scores.append(rec.score_final)
        
        # Calcular score de calidad general
        if quality_scores:
            validation_result['quality_score'] = sum(quality_scores) / len(quality_scores)
        
        # Determinar si es válida para envío
        if validation_result['quality_score'] < 70:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Calidad general insuficiente")
        
        return validation_result 