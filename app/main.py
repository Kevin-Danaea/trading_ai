#!/usr/bin/env python3
"""
Trading AI - Punto de Entrada Principal
======================================

Sistema inteligente de trading con arquitectura limpia.

Este es el punto de entrada principal que orquesta todo el sistema:
- Scanner inteligente de oportunidades
- Optimización bayesiana de estrategias  
- Backtesting con estrategias modernas
- Análisis completo de portafolios

Arquitectura implementada:
- Domain: Entidades y estrategias de negocio puras
- Application: Casos de uso y servicios de aplicación
- Infrastructure: Proveedores de datos externos y configuración
"""

import sys
import os
import argparse
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingAIApplication:
    """
    Aplicación principal del sistema de Trading AI.
    
    Maneja la inicialización y coordinación de todos los componentes
    siguiendo los principios de Clean Architecture.
    """
    
    def __init__(self):
        """Inicializa la aplicación cargando todas las dependencias."""
        logger.info("🤖 Inicializando Trading AI Application...")
        
        # Cargar configuración
        self._load_configuration()
        
        # Inicializar dependencias de infraestructura
        self._initialize_infrastructure()
        
        # Inicializar servicios de aplicación
        self._initialize_application_services()
        
        logger.info("✅ Trading AI Application inicializada correctamente")
    
    def _load_configuration(self):
        """Carga la configuración de infraestructura."""
        try:
            from app.infrastructure.config.settings import Settings
            self.settings = Settings()
            logger.info("📋 Configuración cargada")
        except Exception as e:
            logger.error(f"❌ Error cargando configuración: {e}")
            raise
    
    def _initialize_infrastructure(self):
        """Inicializa los proveedores de infraestructura."""
        try:
            logger.info("🔧 Inicializando infraestructura...")
            
            # Importar proveedores reales
            from app.infrastructure.providers import MarketDataProvider, SentimentDataProvider
            from app.infrastructure.services import BacktestingService, DatabaseService, TelegramService
            
            # Inicializar proveedores reales
            self.market_data_provider = MarketDataProvider()
            self.sentiment_data_provider = SentimentDataProvider()
            self.backtesting_service = BacktestingService()
            self.data_service = self.market_data_provider  # Por simplicidad, usar MarketDataProvider como data_service
            
            # Inicializar servicios de comunicación y almacenamiento
            try:
                self.telegram_service = TelegramService()
                logger.info("📱 Servicio de Telegram inicializado")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo inicializar Telegram: {e}")
                self.telegram_service = None
            
            try:
                self.database_service = DatabaseService()
                logger.info("💾 Servicio de base de datos inicializado")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo inicializar base de datos: {e}")
                self.database_service = None
            
            logger.info("🔧 Infraestructura inicializada ✅")
        except Exception as e:
            logger.error(f"❌ Error inicializando infraestructura: {e}")
            raise
    
    def _initialize_application_services(self):
        """Inicializa los servicios de aplicación."""
        try:
            logger.info("📊 Inicializando servicios de aplicación...")
            
            # Importar servicios reales
            from app.application.use_cases.scanning.crypto_scanner_service import CryptoScannerService
            from app.application.use_cases.optimization.bayesian_optimizer_service import BayesianOptimizerService
            from app.application.use_cases.qualitative_analysis.qualitative_filter_service import QualitativeFilterService
            from app.application.use_cases.orchestration.daily_recommendation_service import WeeklyRecommendationService
            from app.application.use_cases.optimization.futures_optimization_service import FuturesOptimizationService
            
            # Inicializar servicios reales con inyección de dependencias
            self.scanner_service = CryptoScannerService(
                market_data_provider=self.market_data_provider,
                sentiment_data_provider=self.sentiment_data_provider
            )
            
            self.optimizer_service = BayesianOptimizerService(
                backtesting_service=self.backtesting_service,
                data_service=self.data_service
            )
            
            self.qualitative_service = QualitativeFilterService()
            self.weekly_service = WeeklyRecommendationService()
            self.futures_optimizer_service = FuturesOptimizationService()
            
            logger.info("📊 Servicios de aplicación inicializados ✅")
        except Exception as e:
            logger.error(f"❌ Error inicializando servicios: {e}")
            raise
    
    def run_scanner_only(self) -> List[Any]:
        """
        Ejecuta solo el scanner de mercado.
        
        Returns:
            Lista de candidatos encontrados
        """
        logger.info("🔍 Ejecutando scanner de mercado...")
        
        try:
            if self.scanner_service:
                candidates = self.scanner_service.scan_market()
                logger.info(f"✅ Scanner completado: {len(candidates)} candidatos encontrados")
                return candidates
            else:
                logger.warning("⚠️ Scanner service no disponible (modo mock)")
                return []
        except Exception as e:
            logger.error(f"❌ Error en scanner: {e}")
            return []
    
    def run_optimization_only(self, symbols: List[str], n_trials: int = 150) -> Dict[str, Any]:
        """
        Ejecuta solo optimización de estrategias con datos históricos de 12 meses.
        
        Args:
            symbols: Lista de símbolos a optimizar
            n_trials: Número de trials por símbolo
            
        Returns:
            Resultados de optimización formateados
        """
        try:
            logger.info(f"🔧 Ejecutando optimización de estrategias...")
            logger.info(f"📊 Símbolos: {symbols}")
            logger.info(f"🎯 Trials por símbolo: {n_trials}")
            logger.info(f"📅 Datos históricos: 12 meses")
            logger.info(f"🎯 Objetivo: Optimizar para trading de hoy y próximos 7 días")
            
            # Obtener datos históricos de 12 meses para cada símbolo
            data_dict = {}
            for symbol in symbols:
                try:
                    # Usar el método específico para optimización con 12 meses de datos
                    df = self.market_data_provider.fetch_optimization_historical_data(symbol, months=12)
                    if df is not None and len(df) > 180:  # Mínimo 6 meses de datos
                        data_dict[symbol] = df
                        logger.info(f"   📊 Datos cargados para {symbol}: {len(df)} registros ({len(df)/30:.1f} meses)")
                    else:
                        logger.warning(f"   ⚠️ Datos insuficientes para {symbol} (requiere mínimo 6 meses)")
                except Exception as e:
                    logger.error(f"   ❌ Error cargando datos históricos para {symbol}: {e}")
            
            if not data_dict:
                logger.error("❌ No se pudieron cargar datos históricos para ningún símbolo")
                return {'success': False, 'error': 'No historical data available'}
            
            # Ejecutar optimización para cada estrategia
            results = {}
            
            # Optimizar Grid Trading
            logger.info("🔄 Optimizando Grid Trading...")
            grid_results = []
            for symbol, df in data_dict.items():
                try:
                    # Usar el optimizador bayesiano para Grid Trading
                    optimization_result = self.optimizer_service.optimize_symbol(
                        symbol=symbol,
                        strategy='grid',
                        n_trials=n_trials
                    )
                    if optimization_result:
                        # Convertir OptimizationResult a diccionario
                        result_dict = {
                            'symbol': symbol,
                            'strategy': 'Grid Trading',
                            'roi': optimization_result.get_roi(),
                            'sharpe_ratio': optimization_result.get_sharpe_ratio(),
                            'max_drawdown': optimization_result.get_max_drawdown(),
                            'win_rate': optimization_result.get_win_rate(),
                            'total_trades': optimization_result.study_stats.get('total_trades', 0),
                            'best_params': optimization_result.best_params,
                            'score': optimization_result.best_value,
                            'iterations': optimization_result.trials_completed,
                            'duration': optimization_result.optimization_time
                        }
                        grid_results.append(result_dict)
                except Exception as e:
                    logger.error(f"❌ Error optimizando Grid Trading para {symbol}: {e}")
            
            # Optimizar DCA
            logger.info("🔄 Optimizando DCA...")
            dca_results = []
            for symbol, df in data_dict.items():
                try:
                    # Usar el optimizador bayesiano para DCA
                    optimization_result = self.optimizer_service.optimize_symbol(
                        symbol=symbol,
                        strategy='dca',
                        n_trials=n_trials
                    )
                    if optimization_result:
                        # Convertir OptimizationResult a diccionario
                        result_dict = {
                            'symbol': symbol,
                            'strategy': 'DCA',
                            'roi': optimization_result.get_roi(),
                            'sharpe_ratio': optimization_result.get_sharpe_ratio(),
                            'max_drawdown': optimization_result.get_max_drawdown(),
                            'win_rate': optimization_result.get_win_rate(),
                            'total_trades': optimization_result.study_stats.get('total_trades', 0),
                            'best_params': optimization_result.best_params,
                            'score': optimization_result.best_value,
                            'iterations': optimization_result.trials_completed,
                            'duration': optimization_result.optimization_time
                        }
                        dca_results.append(result_dict)
                except Exception as e:
                    logger.error(f"❌ Error optimizando DCA para {symbol}: {e}")
            
            # Optimizar BTD
            logger.info("🔄 Optimizando BTD...")
            btd_results = []
            for symbol, df in data_dict.items():
                try:
                    # Usar el optimizador bayesiano para BTD
                    optimization_result = self.optimizer_service.optimize_symbol(
                        symbol=symbol,
                        strategy='btd',
                        n_trials=n_trials
                    )
                    if optimization_result:
                        # Convertir OptimizationResult a diccionario
                        result_dict = {
                            'symbol': symbol,
                            'strategy': 'BTD',
                            'roi': optimization_result.get_roi(),
                            'sharpe_ratio': optimization_result.get_sharpe_ratio(),
                            'max_drawdown': optimization_result.get_max_drawdown(),
                            'win_rate': optimization_result.get_win_rate(),
                            'total_trades': optimization_result.study_stats.get('total_trades', 0),
                            'best_params': optimization_result.best_params,
                            'score': optimization_result.best_value,
                            'iterations': optimization_result.trials_completed,
                            'duration': optimization_result.optimization_time
                        }
                        btd_results.append(result_dict)
                except Exception as e:
                    logger.error(f"❌ Error optimizando BTD para {symbol}: {e}")
            
            results = {
                'grid_trading': grid_results,
                'dca': dca_results,
                'btd': btd_results
            }
            
            # Formatear resultados
            formatted_results = self._format_optimization_results(results)
            
            logger.info("✅ Optimización completada:")
            logger.info(f"   📊 Símbolos procesados: {len(data_dict)}")
            logger.info(f"   📅 Datos históricos: 12 meses por símbolo")
            logger.info(f"   🎯 Optimizado para: Trading de hoy y próximos 7 días")
            logger.info(f"   🔧 Estrategias: Grid Trading, DCA, BTD")
            
            return {
                'success': True,
                'results': formatted_results,
                'raw_results': results,
                'data_period_months': 12,
                'optimization_target': 'today_and_next_7_days',
                'symbols_processed': len(data_dict)
            }
            
        except Exception as e:
            logger.error(f"❌ Error en optimización: {e}")
            return {'success': False, 'error': str(e)}
    
    def _format_optimization_results(self, raw_results: Dict[str, List[Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Convierte los resultados de optimización a formato estandarizado.
        
        Args:
            raw_results: Resultados raw del optimizer service
            
        Returns:
            Resultados formateados
        """
        formatted = {}
        
        for strategy_name, optimization_results in raw_results.items():
            formatted[strategy_name] = {}
            
            for opt_result in optimization_results:
                symbol = opt_result.get('symbol', 'UNKNOWN')
                
                # Verificar si el resultado es exitoso
                roi = opt_result.get('roi', 0)
                is_successful = roi > 0
                
                if is_successful:
                    formatted[strategy_name][symbol] = {
                        'success': True,
                        'roi': roi,
                        'sharpe_ratio': opt_result.get('sharpe_ratio', 0),
                        'max_drawdown': opt_result.get('max_drawdown', 0),
                        'win_rate': opt_result.get('win_rate', 0),
                        'total_trades': opt_result.get('total_trades', 0),
                        'best_params': opt_result.get('best_params', {}),
                        'score': opt_result.get('score', 0),
                        'iterations': opt_result.get('iterations', 0),
                        'duration': opt_result.get('duration', 0),
                        'confidence': 0.8  # Valor por defecto
                    }
                else:
                    formatted[strategy_name][symbol] = {
                        'success': False,
                        'error': f'Optimization failed: roi={roi}'
                    }
        
        return formatted
    
    def run_full_analysis(self, 
                         force_symbols: Optional[List[str]] = None,
                         n_trials: int = 150) -> Dict[str, Any]:
        """
        Ejecuta análisis completo: Scanner + Optimización (Spot + Futuros) + Análisis Cualitativo + Selección Semanal + Notificaciones.
        
        Args:
            force_symbols: Símbolos específicos (omite scanner si se proporciona)
            n_trials: Número de trials por símbolo
            
        Returns:
            Resultados completos del análisis (spot + futuros)
        """
        logger.info("🚀 Ejecutando análisis completo del pipeline (Spot + Futuros)...")
        start_time = datetime.now()
        
        try:
            # Fase 1: Selección de símbolos
            if force_symbols:
                symbols = force_symbols
                logger.info(f"💡 Usando símbolos específicos: {symbols}")
            else:
                candidates = self.run_scanner_only()
                if not candidates:
                    logger.error("❌ No se encontraron candidatos en el scanner")
                    return {'success': False, 'error': 'No candidates found'}
                
                symbols = [c.symbol for c in candidates]
                logger.info(f"🎯 Símbolos seleccionados por scanner: {symbols}")
            
            # Fase 2A: Optimización bayesiana SPOT
            logger.info("📊 Iniciando optimización bayesiana SPOT...")
            spot_optimization_results = self.run_optimization_only(symbols, n_trials)
            
            if not spot_optimization_results.get('success'):
                logger.error("❌ Error en optimización bayesiana SPOT")
                return {'success': False, 'error': 'Spot optimization failed'}
            
            # Fase 2B: Optimización bayesiana FUTUROS
            logger.info("⚡ Iniciando optimización bayesiana FUTUROS...")
            futures_optimization_results = self._run_futures_optimization(symbols, n_trials)
            
            if not futures_optimization_results.get('success'):
                logger.error("❌ Error en optimización bayesiana FUTUROS")
                return {'success': False, 'error': 'Futures optimization failed'}
            
            # Fase 3A: Análisis cualitativo SPOT
            logger.info("🧠 Iniciando análisis cualitativo SPOT...")
            spot_qualitative_results = self._run_qualitative_analysis(spot_optimization_results)
            
            # Fase 3B: Análisis cualitativo FUTUROS
            logger.info("🧠 Iniciando análisis cualitativo FUTUROS...")
            futures_qualitative_results = self._run_futures_qualitative_analysis(futures_optimization_results)
            
            # Marcar resultados como spot o futuros para evitar duplicación
            for result in spot_qualitative_results:
                result.market_type = 'spot'
            
            for result in futures_qualitative_results:
                result.market_type = 'futures'
            
            # Combinar resultados cualitativos
            all_qualitative_results = spot_qualitative_results + futures_qualitative_results
            
            if not all_qualitative_results:
                logger.error("❌ No se obtuvieron resultados del análisis cualitativo")
                return {'success': False, 'error': 'Qualitative analysis failed'}
            
            # Fase 4: Selección de cartera semanal (combinada)
            logger.info("🎯 Iniciando selección de cartera semanal (Spot + Futuros)...")
            weekly_results = self._run_weekly_selection(all_qualitative_results)
            
            # Fase 5: Selección de mejor oportunidad de futuros
            logger.info("🎯 Seleccionando mejor oportunidad de futuros...")
            best_futures_opportunity = self._select_best_futures_opportunity(futures_qualitative_results)
            
            # Fase 6: Resultados finales
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            results = {
                'success': True,
                'market_type': 'spot_and_futures',
                'symbols_analyzed': symbols,
                'total_symbols': len(symbols),
                'spot_optimization_results': spot_optimization_results,
                'futures_optimization_results': futures_optimization_results,
                'spot_qualitative_results': len(spot_qualitative_results),
                'futures_qualitative_results': len(futures_qualitative_results),
                'total_qualitative_results': len(all_qualitative_results),
                'weekly_results': weekly_results,
                'best_futures_opportunity': best_futures_opportunity,
                'total_time_seconds': total_time,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
            
            logger.info(f"✅ Pipeline completo (Spot + Futuros) terminado en {total_time:.1f} segundos")
            logger.info(f"📈 Recomendaciones SPOT: {len(spot_qualitative_results)}")
            logger.info(f"⚡ Recomendaciones FUTUROS: {len(futures_qualitative_results)}")
            logger.info(f"🎯 Recomendaciones semanales generadas: {weekly_results.get('total_recommendations', 0)}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en pipeline completo: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_futures_pipeline(self, 
                           force_symbols: Optional[List[str]] = None,
                           n_trials: int = 100) -> Dict[str, Any]:
        """
        Ejecuta pipeline específico para futuros: Scanner + Optimización de Futuros + Análisis Cualitativo.
        
        Args:
            force_symbols: Símbolos específicos (omite scanner si se proporciona)
            n_trials: Número de trials por símbolo para optimización de futuros
            
        Returns:
            Resultados del pipeline de futuros
        """
        logger.info("🚀 Ejecutando pipeline de futuros...")
        start_time = datetime.now()
        
        try:
            # Fase 1: Selección de símbolos (usar mismo scanner)
            if force_symbols:
                symbols = force_symbols
                logger.info(f"💡 Usando símbolos específicos para futuros: {symbols}")
            else:
                candidates = self.run_scanner_only()
                if not candidates:
                    logger.error("❌ No se encontraron candidatos en el scanner")
                    return {'success': False, 'error': 'No candidates found'}
                
                # Usar top 10 como mencionó el usuario
                symbols = [c.symbol for c in candidates[:10]]
                logger.info(f"🎯 Top 10 símbolos seleccionados para futuros: {symbols}")
            
            # Fase 2: Optimización específica para futuros
            logger.info("⚡ Iniciando optimización de futuros...")
            futures_optimization_results = self._run_futures_optimization(symbols, n_trials)
            
            if not futures_optimization_results.get('success'):
                logger.error("❌ Error en optimización de futuros")
                return {'success': False, 'error': 'Futures optimization failed'}
            
            # Fase 3: Análisis cualitativo para futuros
            logger.info("🧠 Iniciando análisis cualitativo para futuros...")
            futures_qualitative_results = self._run_futures_qualitative_analysis(futures_optimization_results)
            
            if not futures_qualitative_results:
                logger.error("❌ No se obtuvieron resultados del análisis cualitativo de futuros")
                return {'success': False, 'error': 'Futures qualitative analysis failed'}
            
            # Fase 4: Selección de mejor oportunidad de futuros
            logger.info("🎯 Seleccionando mejor oportunidad de futuros...")
            best_futures_opportunity = self._select_best_futures_opportunity(futures_qualitative_results)
            
            # Fase 5: Resultados finales
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            results = {
                'success': True,
                'market_type': 'futures',
                'symbols_analyzed': symbols,
                'total_symbols': len(symbols),
                'futures_optimization_results': futures_optimization_results,
                'futures_qualitative_results': len(futures_qualitative_results),
                'best_futures_opportunity': best_futures_opportunity,
                'total_time_seconds': total_time,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
            
            logger.info(f"✅ Pipeline de futuros completado en {total_time:.1f} segundos")
            if best_futures_opportunity:
                logger.info(f"🏆 Mejor oportunidad de futuros: {best_futures_opportunity.get('symbol', 'N/A')}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en pipeline de futuros: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_qualitative_analysis(self, optimization_results: Dict[str, Any]) -> List[Any]:
        """
        Ejecuta análisis cualitativo sobre los resultados de optimización.
        
        Args:
            optimization_results: Resultados de la optimización bayesiana
            
        Returns:
            Lista de resultados cualitativos
        """
        try:
            from app.application.use_cases.qualitative_analysis.qualitative_filter_service import QualitativeFilterService
            
            # Obtener las mejores oportunidades de los resultados de optimización
            opportunities = self._extract_opportunities_from_optimization(optimization_results)
            
            if not opportunities:
                logger.warning("⚠️ No se encontraron oportunidades para análisis cualitativo")
                return []
            
            # Ejecutar análisis cualitativo
            qualitative_service = QualitativeFilterService()
            qualitative_results = qualitative_service.analyze_opportunities(opportunities)
            
            logger.info(f"🧠 Análisis cualitativo completado: {len(qualitative_results)} resultados")
            return qualitative_results
            
        except Exception as e:
            logger.error(f"❌ Error en análisis cualitativo: {e}")
            return []
    
    def _run_weekly_selection(self, qualitative_results: List[Any]) -> Dict[str, Any]:
        """
        Ejecuta selección de cartera semanal y notificaciones (Spot + Futuros).
        
        Args:
            qualitative_results: Resultados del análisis cualitativo (spot + futuros)
            
        Returns:
            Resultados de la selección semanal
        """
        try:
            from app.application.use_cases.orchestration.daily_recommendation_service import WeeklyRecommendationService
            
            # Separar oportunidades de spot y futuros
            spot_opportunities = []
            futures_opportunities = []
            
            for result in qualitative_results:
                # Verificar si es una oportunidad de futuros basándose en la estrategia
                is_futures = False
                if hasattr(result, 'market_type'):
                    is_futures = result.market_type == 'futures'
                elif hasattr(result, 'opportunity') and hasattr(result.opportunity, 'recommended_strategy_name'):
                    is_futures = result.opportunity.recommended_strategy_name == 'FuturesGrid'
                elif hasattr(result, 'recommended_strategy_name'):
                    is_futures = result.recommended_strategy_name == 'FuturesGrid'
                
                if is_futures:
                    futures_opportunities.append(result)
                else:
                    spot_opportunities.append(result)
            
            logger.info(f"📊 Procesando {len(spot_opportunities)} oportunidades SPOT y {len(futures_opportunities)} oportunidades FUTUROS")
            
            # Ejecutar selección semanal con todas las oportunidades (spot + futuros)
            weekly_service = WeeklyRecommendationService()
            weekly_results = weekly_service.process_weekly_recommendations(
                qualitative_results=qualitative_results,  # Todas las oportunidades
                version_pipeline="weekly_v2.0_spot_and_futures"
            )
            
            # Enviar notificaciones unificadas
            self._send_unified_notifications(spot_opportunities, futures_opportunities, weekly_results)
            
            # Guardar en base de datos
            self._save_to_database(spot_opportunities, futures_opportunities, weekly_results)
            
            logger.info(f"🎯 Selección semanal completada: {weekly_results.get('success', False)}")
            logger.info(f"📱 Notificaciones enviadas: SPOT ({len(spot_opportunities)}) + FUTUROS ({len(futures_opportunities)})")
            logger.info(f"💾 Datos guardados en base de datos")
            
            return weekly_results
            
        except Exception as e:
            logger.error(f"❌ Error en selección semanal: {e}")
            return {'success': False, 'error': str(e)}
    
    def _send_unified_notifications(self, spot_opportunities: List[Any], futures_opportunities: List[Any], weekly_results: Dict[str, Any]):
        """
        Envía notificaciones unificadas para spot y futuros.
        
        Args:
            spot_opportunities: Oportunidades de spot
            futures_opportunities: Oportunidades de futuros
            weekly_results: Resultados semanales
        """
        try:
            if not self.telegram_service:
                logger.warning("⚠️ Servicio de Telegram no disponible")
                return
            
            # Crear mensaje unificado
            message = "🤖 **ANÁLISIS COMPLETO TRADING AI**\n\n"
            message += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Sección SPOT
            if spot_opportunities:
                message += "📈 **OPORTUNIDADES SPOT:**\n"
                for i, opp in enumerate(spot_opportunities[:5], 1):  # Top 5
                    symbol = getattr(opp, 'opportunity', opp).candidate.symbol if hasattr(opp, 'opportunity') else getattr(opp, 'symbol', 'N/A')
                    roi = getattr(opp, 'opportunity', opp).roi_percentage if hasattr(opp, 'opportunity') else getattr(opp, 'roi_percentage', 0)
                    strategy = getattr(opp, 'opportunity', opp).recommended_strategy_name if hasattr(opp, 'opportunity') else getattr(opp, 'recommended_strategy_name', 'N/A')
                    message += f"{i}. {symbol}: {roi:.2f}% ({strategy})\n"
                message += "\n"
            else:
                message += "📈 **SPOT:** Sin oportunidades rentables\n\n"
            
            # Sección FUTUROS
            if futures_opportunities:
                message += "⚡ **OPORTUNIDADES FUTUROS:**\n"
                for i, opp in enumerate(futures_opportunities[:5], 1):  # Top 5
                    symbol = getattr(opp, 'opportunity', opp).candidate.symbol if hasattr(opp, 'opportunity') else getattr(opp, 'symbol', 'N/A')
                    roi = getattr(opp, 'opportunity', opp).roi_percentage if hasattr(opp, 'opportunity') else getattr(opp, 'roi_percentage', 0)
                    leverage = getattr(opp, 'opportunity', opp).optimized_params.get('leverage', 'x3') if hasattr(opp, 'opportunity') else getattr(opp, 'leverage', 'x3')
                    message += f"{i}. {symbol}: {roi:.2f}% ({leverage})\n"
                message += "\n"
            else:
                message += "⚡ **FUTUROS:** Sin oportunidades rentables\n\n"
            
            # Resumen semanal
            total_recommendations = weekly_results.get('total_recommendations', 0)
            message += f"🎯 **RESUMEN SEMANAL:** {total_recommendations} recomendaciones\n"
            message += f"⏱️ Tiempo total: {weekly_results.get('processing_time', 0):.1f}s\n\n"
            
            message += "🔗 Ver detalles completos en la base de datos"
            
            # Escapar caracteres especiales para Telegram
            escaped_message = self.telegram_service._escape_markdown_v2(message)
            
            # Enviar notificación
            self.telegram_service.send_message_sync(escaped_message)
            logger.info("📱 Notificación unificada enviada a Telegram")
            
        except Exception as e:
            logger.error(f"❌ Error enviando notificaciones: {e}")
    
    def _save_to_database(self, spot_opportunities: List[Any], futures_opportunities: List[Any], weekly_results: Dict[str, Any]):
        """
        Guarda resultados en la base de datos.
        
        Args:
            spot_opportunities: Oportunidades de spot
            futures_opportunities: Oportunidades de futuros
            weekly_results: Resultados semanales
        """
        try:
            if not self.database_service:
                logger.warning("⚠️ Servicio de base de datos no disponible")
                return
            
            # Guardar oportunidades de spot
            for opp in spot_opportunities:
                # Convertir a RecomendacionDiaria si es necesario
                if hasattr(opp, 'to_recommendation'):
                    recommendation = opp.to_recommendation()
                    self.database_service.save_recommendation(recommendation)
            
            # Guardar oportunidades de futuros
            for opp in futures_opportunities:
                # Convertir a RecomendacionDiaria si es necesario
                if hasattr(opp, 'to_recommendation'):
                    recommendation = opp.to_recommendation()
                    self.database_service.save_recommendation(recommendation)
            
            # Guardar resultados semanales como recomendaciones
            if weekly_results.get('recommendations'):
                for rec in weekly_results['recommendations']:
                    if hasattr(rec, 'to_recommendation'):
                        recommendation = rec.to_recommendation()
                        self.database_service.save_recommendation(recommendation)
            
            logger.info(f"💾 Guardados en DB: {len(spot_opportunities)} SPOT + {len(futures_opportunities)} FUTUROS")
            
        except Exception as e:
            logger.error(f"❌ Error guardando en base de datos: {e}")
    
    def _run_futures_optimization(self, symbols: List[str], n_trials: int) -> Dict[str, Any]:
        """
        Ejecuta optimización específica para futuros con datos históricos de 12 meses.
        
        Args:
            symbols: Lista de símbolos a optimizar
            n_trials: Número de trials por símbolo
            
        Returns:
            Resultados de optimización de futuros
        """
        try:
            logger.info(f"⚡ Optimizando {len(symbols)} símbolos para futuros con datos de 12 meses...")
            logger.info(f"🎯 Objetivo: Encontrar mejores parámetros para trading de hoy y próximos 7 días")
            
            # Obtener datos históricos de 12 meses para cada símbolo
            data_dict = {}
            for symbol in symbols:
                try:
                    # Usar el método específico para optimización con 12 meses de datos
                    df = self.market_data_provider.fetch_optimization_historical_data(symbol, months=12)
                    if df is not None and len(df) > 180:  # Mínimo 6 meses de datos
                        data_dict[symbol] = df
                        logger.info(f"   📊 Datos cargados para {symbol}: {len(df)} registros ({len(df)/30:.1f} meses)")
                        logger.info(f"   💰 Funding rate promedio: {df['funding_rate'].mean()*100:.3f}%")
                    else:
                        logger.warning(f"   ⚠️ Datos insuficientes para {symbol} (requiere mínimo 6 meses)")
                except Exception as e:
                    logger.error(f"   ❌ Error cargando datos históricos para {symbol}: {e}")
            
            if not data_dict:
                logger.error("❌ No se pudieron cargar datos históricos para ningún símbolo")
                return {'success': False, 'error': 'No historical data available'}
            
            # Ejecutar optimización de futuros con datos históricos
            logger.info(f"🔄 Ejecutando optimización bayesiana con {n_trials} trials por símbolo...")
            optimization_results = self.futures_optimizer_service.optimize_multiple_symbols(
                data_dict=data_dict,
                n_trials_per_symbol=n_trials
            )
            
            # Filtrar solo resultados rentables sin liquidación
            profitable_results = self.futures_optimizer_service.filter_profitable_no_liquidation(
                optimization_results
            )
            
            # Calcular estadísticas de funding rate
            total_funding_cost = 0
            total_symbols_with_funding = 0
            
            for result in profitable_results:
                final_results = result.get('final_results', {})
                funding_cost = final_results.get('Total Funding Cost', 0)
                if funding_cost > 0:
                    total_funding_cost += funding_cost
                    total_symbols_with_funding += 1
            
            logger.info(f"✅ Optimización de futuros completada:")
            logger.info(f"   📊 Símbolos procesados: {len(optimization_results)}")
            logger.info(f"   💰 Símbolos rentables sin liquidación: {len(profitable_results)}")
            logger.info(f"   ⚡ Datos históricos utilizados: 12 meses por símbolo")
            logger.info(f"   🎯 Optimizado para: Trading de hoy y próximos 7 días")
            logger.info(f"   💸 Costo promedio de funding: {total_funding_cost/max(total_symbols_with_funding, 1):.2f} USDT")
            
            return {
                'success': True,
                'total_symbols': len(symbols),
                'processed_symbols': len(optimization_results),
                'profitable_symbols': len(profitable_results),
                'optimization_results': optimization_results,
                'profitable_results': profitable_results,
                'data_period_months': 12,
                'optimization_target': 'today_and_next_7_days',
                'funding_statistics': {
                    'total_funding_cost': total_funding_cost,
                    'symbols_with_funding': total_symbols_with_funding,
                    'avg_funding_cost': total_funding_cost / max(total_symbols_with_funding, 1)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error en optimización de futuros: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_futures_qualitative_analysis(self, futures_optimization_results: Dict[str, Any]) -> List[Any]:
        """
        Ejecuta análisis cualitativo específico para futuros.
        
        Args:
            futures_optimization_results: Resultados de optimización de futuros
            
        Returns:
            Lista de resultados cualitativos para futuros
        """
        try:
            # Obtener resultados rentables sin liquidación
            profitable_results = futures_optimization_results.get('profitable_results', [])
            
            if not profitable_results:
                logger.warning("⚠️ No hay resultados rentables para análisis cualitativo de futuros")
                return []
            
            # Convertir a oportunidades de trading
            opportunities = self._convert_futures_results_to_opportunities(profitable_results)
            
            if not opportunities:
                logger.warning("⚠️ No se pudieron convertir resultados a oportunidades")
                return []
            
            # Ejecutar análisis cualitativo con prompt específico para futuros
            from app.application.use_cases.qualitative_analysis.qualitative_filter_service import QualitativeFilterService
            qualitative_service = QualitativeFilterService()
            qualitative_results = qualitative_service.analyze_opportunities(opportunities)
            
            logger.info(f"🧠 Análisis cualitativo de futuros completado: {len(qualitative_results)} resultados")
            return qualitative_results
            
        except Exception as e:
            logger.error(f"❌ Error en análisis cualitativo de futuros: {e}")
            return []
    
    def _select_best_futures_opportunity(self, futures_qualitative_results: List[Any]) -> Optional[Dict[str, Any]]:
        """
        Selecciona la mejor oportunidad de futuros basada en análisis cualitativo.
        
        Args:
            futures_qualitative_results: Resultados del análisis cualitativo
            
        Returns:
            Mejor oportunidad de futuros o None
        """
        try:
            if not futures_qualitative_results:
                logger.warning("⚠️ No hay resultados cualitativos para seleccionar")
                return None
            
            # Filtrar solo oportunidades aprobadas
            approved_opportunities = [
                result for result in futures_qualitative_results
                if hasattr(result, 'analysis') and result.analysis.recommendation.upper() in ['EXCELENTE', 'BUENO', 'ACEPTABLE']
            ]
            
            if not approved_opportunities:
                logger.warning("⚠️ No hay oportunidades aprobadas para futuros")
                return None
            
            # Seleccionar la mejor basada en score y sin liquidación
            best_opportunity = max(
                approved_opportunities,
                key=lambda x: (
                    getattr(x, 'confidence_score', 0),
                    getattr(x.opportunity, 'final_score', 0)
                )
            )
            
            logger.info(f"🏆 Mejor oportunidad de futuros seleccionada: {best_opportunity.opportunity.candidate.symbol}")
            return {
                'symbol': best_opportunity.opportunity.candidate.symbol,
                'roi_percentage': best_opportunity.opportunity.roi_percentage,
                'strategy': best_opportunity.analysis.recommended_strategy,
                'leverage': best_opportunity.analysis.optimal_leverage,
                'confidence_score': best_opportunity.confidence_score
            }
            
        except Exception as e:
            logger.error(f"❌ Error seleccionando mejor oportunidad de futuros: {e}")
            return None
    
    def _convert_futures_results_to_opportunities(self, futures_results: List[Dict[str, Any]]) -> List[Any]:
        """
        Convierte resultados de optimización de futuros a oportunidades de trading.
        
        Args:
            futures_results: Resultados de optimización de futuros
            
        Returns:
            Lista de oportunidades de trading
        """
        try:
            from app.domain.entities.trading_opportunity import TradingOpportunity, StrategyResult
            from app.domain.entities.crypto_candidate import CryptoCandidate
            from datetime import datetime
            
            opportunities = []
            
            for result in futures_results:
                if 'error' in result:
                    continue
                
                symbol = result.get('symbol', '')
                final_results = result.get('final_results', {})
                best_params = result.get('best_params', {})
                
                # Crear candidato
                candidate = CryptoCandidate(
                    symbol=symbol,
                    market_cap_rank=1,
                    current_price=1.0,
                    volatility_24h=0.15,
                    volatility_7d=0.20,
                    adx=20.0,
                    sentiment_score=0.0,
                    sentiment_ma7=0.0,
                    volume_24h=0.0,
                    volume_change_24h=0.0,
                    price_change_24h=0.0,
                    price_change_7d=0.0,
                    score=final_results.get('performance_score', 0.0),
                    reasons=[f"Futures optimization: {final_results.get('Return [%]', 0.0):.1f}% ROI"]
                )
                
                # Crear resultado de estrategia
                strategy_result = StrategyResult(
                    strategy_name='FuturesGrid',
                    optimized_params=best_params,
                    roi_percentage=final_results.get('Return [%]', 0.0),
                    sharpe_ratio=final_results.get('Sharpe Ratio', 0.0),
                    max_drawdown_percentage=abs(final_results.get('Max. Drawdown [%]', 0.0)),
                    win_rate_percentage=final_results.get('Win Rate [%]', 0.0),
                    total_trades=final_results.get('# Trades', 0),
                    avg_trade_percentage=final_results.get('Return [%]', 0.0) / max(final_results.get('# Trades', 1), 1),
                    volatility_percentage=final_results.get('Volatility [%]', 20.0),
                    calmar_ratio=final_results.get('Return [%]', 0.0) / max(abs(final_results.get('Max. Drawdown [%]', 1.0)), 1.0),
                    sortino_ratio=final_results.get('Sharpe Ratio', 0.0) * 1.1,  # Estimación
                    exposure_time_percentage=100.0,
                    optimization_iterations=100,
                    optimization_duration_seconds=60.0,
                    confidence_level=0.8
                )
                
                # Crear oportunidad de trading
                opportunity = TradingOpportunity(
                    candidate=candidate,
                    strategy_results={'FuturesGrid': strategy_result},
                    recommended_strategy_name='FuturesGrid',
                    backtest_period_days=1825,  # 5 años
                    final_score=min(100, max(0, final_results.get('Return [%]', 0.0) * 2)),
                    risk_adjusted_score=min(100, max(0, final_results.get('Return [%]', 0.0) * 1.5)),
                    created_at=datetime.now(),
                    market_conditions="sideways"
                )
                
                opportunities.append(opportunity)
            
            logger.info(f"🔄 Convertidos {len(opportunities)} resultados a oportunidades de futuros")
            return opportunities
            
        except Exception as e:
            logger.error(f"❌ Error convirtiendo resultados de futuros: {e}")
            return []
    
    def _extract_opportunities_from_optimization(self, optimization_results: Dict[str, Any]) -> List[Any]:
        """
        Extrae oportunidades de trading de los resultados de optimización.
        
        Args:
            optimization_results: Resultados de optimización bayesiana
            
        Returns:
            Lista de oportunidades de trading
        """
        try:
            from app.domain.entities.trading_opportunity import TradingOpportunity, StrategyResult
            from app.domain.entities.crypto_candidate import CryptoCandidate
            from datetime import datetime
            
            opportunities = []
            
            # Extraer datos de optimización
            optimization_data = optimization_results.get('results', {})
            
            logger.info(f"🔍 Datos de optimización disponibles: {list(optimization_data.keys())}")
            
            # La estructura es {strategy: {symbol: data}}, necesitamos reorganizarla
            # Crear diccionario {symbol: {strategy: data}}
            symbol_data = {}
            
            for strategy_name, strategy_results in optimization_data.items():
                logger.info(f"📊 Estrategia {strategy_name}: {list(strategy_results.keys())}")
                
                for symbol, strategy_data in strategy_results.items():
                    if symbol not in symbol_data:
                        symbol_data[symbol] = {}
                    
                    symbol_data[symbol][strategy_name] = strategy_data
            
            # Procesar cada símbolo
            for symbol, strategies in symbol_data.items():
                logger.info(f"🔍 Procesando símbolo: {symbol}")
                
                # Agrupar estrategias por símbolo
                strategy_results = {}
                best_strategy = None
                best_roi = -999
                
                for strategy_name, strategy_data in strategies.items():
                    logger.info(f"📊 Estrategia {strategy_name}: success={strategy_data.get('success')}, roi={strategy_data.get('roi', 0)}")
                    # Criterio más flexible: éxito O trades > 0 O ROI > -5%
                    has_trades = strategy_data.get('total_trades', 0) > 0
                    has_roi = strategy_data.get('roi', 0) > -5  # Permitir pequeñas pérdidas
                    is_successful = strategy_data.get('success', False)
                    
                    if (is_successful or has_trades or has_roi):
                        # Crear resultado de estrategia
                        strategy_result = StrategyResult(
                            strategy_name=strategy_name,
                            optimized_params=strategy_data.get('best_params', {}),
                            roi_percentage=strategy_data.get('roi', 0),
                            sharpe_ratio=strategy_data.get('sharpe_ratio', 0),
                            max_drawdown_percentage=abs(strategy_data.get('max_drawdown', 0)),
                            win_rate_percentage=strategy_data.get('win_rate', 0),
                            total_trades=strategy_data.get('total_trades', 0),
                            avg_trade_percentage=strategy_data.get('avg_trade', 0),
                            volatility_percentage=strategy_data.get('volatility', 0),
                            calmar_ratio=strategy_data.get('calmar_ratio', 0),
                            sortino_ratio=strategy_data.get('sortino_ratio', 0),
                            exposure_time_percentage=strategy_data.get('exposure_time', 100),
                            optimization_iterations=strategy_data.get('iterations', 50),
                            optimization_duration_seconds=strategy_data.get('duration', 60),
                            confidence_level=strategy_data.get('confidence', 0.7)
                        )
                        
                        strategy_results[strategy_name] = strategy_result
                        
                        # Encontrar mejor estrategia
                        if strategy_data.get('roi', 0) > best_roi:
                            best_roi = strategy_data.get('roi', 0)
                            # Asegurar que strategy_name sea una estrategia válida, no un símbolo
                            if strategy_name in ['grid', 'dca', 'btd', 'grid_trading', 'dca_strategy', 'btd_strategy', 'FuturesGrid']:
                                best_strategy = strategy_name
                            else:
                                # Si no es una estrategia válida, usar la primera estrategia disponible
                                best_strategy = list(strategy_results.keys())[0] if strategy_results else 'grid'
                
                # Solo crear oportunidad si hay al menos una estrategia exitosa
                if strategy_results and best_strategy:
                    # Crear candidato con valores por defecto
                    candidate = CryptoCandidate(
                        symbol=symbol,
                        market_cap_rank=100,  # Valor por defecto
                        current_price=1.0,  # Valor por defecto
                        volatility_24h=0.15,  # Valor por defecto
                        volatility_7d=0.20,  # Valor por defecto
                        adx=20.0,  # Valor por defecto
                        sentiment_score=0.5,  # Valor por defecto
                        sentiment_ma7=0.5,  # Valor por defecto
                        volume_24h=1000000,  # Valor por defecto
                        volume_change_24h=0.0,  # Valor por defecto
                        price_change_24h=0.0,  # Valor por defecto
                        price_change_7d=0.0,  # Valor por defecto
                        score=best_roi,  # Usar ROI como score
                        reasons=[f"Optimizado con ROI: {best_roi:.1f}%"]
                    )
                    
                    # Crear oportunidad
                    opportunity = TradingOpportunity(
                        candidate=candidate,
                        strategy_results=strategy_results,
                        recommended_strategy_name=best_strategy,
                        backtest_period_days=270,  # ~9 meses
                        final_score=min(100, max(0, best_roi * 2)),  # Escalar ROI a 0-100
                        risk_adjusted_score=min(100, max(0, best_roi * 1.5)),  # Algo más conservador
                        created_at=datetime.now(),
                        market_conditions="sideways"  # Valor por defecto
                    )
                    
                    opportunities.append(opportunity)
            
            # Ordenar por ROI y tomar las mejores
            opportunities.sort(key=lambda x: x.roi_percentage, reverse=True)
            top_opportunities = opportunities[:5]  # Top 5 para análisis cualitativo
            
            logger.info(f"📊 Extraídas {len(top_opportunities)} oportunidades de {len(opportunities)} totales")
            return top_opportunities
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo oportunidades: {e}")
            return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado del sistema.
        
        Returns:
            Estado de todos los componentes
        """
        return {
            'application': 'Trading AI',
            'version': '1.0.0',
            'architecture': 'Clean Architecture',
            'components': {
                'domain': {
                    'strategies': ['GridTradingStrategy', 'DCAStrategy', 'BTDStrategy'],
                    'entities': ['CryptoCandidate', 'OptimizationResult']
                },
                'application': {
                    'scanner_service': self.scanner_service is not None,
                    'optimizer_service': self.optimizer_service is not None
                },
                'infrastructure': {
                    'market_data_provider': self.market_data_provider is not None,
                    'sentiment_data_provider': self.sentiment_data_provider is not None,
                    'backtesting_service': self.backtesting_service is not None
                }
            },
            'configuration_loaded': self.settings is not None,
            'timestamp': datetime.now().isoformat()
        }


def create_argument_parser() -> argparse.ArgumentParser:
    """Crea el parser de argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Trading AI - Sistema Inteligente de Trading con Arquitectura Limpia',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

# Análisis completo (Scanner + Optimización)
python app/main.py

# Solo scanner de mercado
python app/main.py --scanner-only

# Solo optimización con símbolos específicos
python app/main.py --optimize-only --symbols BTC/USDT ETH/USDT SOL/USDT

# Análisis completo con símbolos específicos
python app/main.py --symbols BTC/USDT ETH/USDT SOL/USDT --trials 200

# Estado del sistema
python app/main.py --status
        """)
    
    parser.add_argument(
        '--scanner-only',
        action='store_true',
        help='Ejecutar solo el scanner de mercado'
    )
    
    parser.add_argument(
        '--optimize-only',
        action='store_true',
        help='Ejecutar solo optimización (requiere --symbols)'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='*',
        help='Símbolos específicos para analizar'
    )
    
    parser.add_argument(
        '--trials',
        type=int,
        default=150,
        help='Número de trials por símbolo en optimización (default: 150)'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Mostrar estado del sistema'
    )
    
    parser.add_argument(
        '--futures',
        action='store_true',
        help='Ejecutar pipeline específico para futuros'
    )
    
    return parser


def main():
    """Función principal del sistema."""
    print("🤖 Trading AI - Sistema Inteligente con Arquitectura Limpia")
    print("=" * 70)
    
    # Parsear argumentos
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Inicializar aplicación
        app = TradingAIApplication()
        
        # Ejecutar según argumentos
        if args.status:
            # Mostrar estado del sistema
            status = app.get_system_status()
            print("\n📊 ESTADO DEL SISTEMA:")
            print("=" * 40)
            print(f"Aplicación: {status['application']} v{status['version']}")
            print(f"Arquitectura: {status['architecture']}")
            print(f"Configuración: {'✅' if status['configuration_loaded'] else '❌'}")
            print(f"Timestamp: {status['timestamp']}")
            
            print(f"\n🏗️ COMPONENTES:")
            for layer, components in status['components'].items():
                print(f"  {layer.title()}:")
                if isinstance(components, dict):
                    for name, available in components.items():
                        icon = '✅' if available else '❌'
                        print(f"    {icon} {name}")
                elif isinstance(components, list):
                    for component in components:
                        print(f"    ✅ {component}")
        
        elif args.scanner_only:
            # Solo scanner
            candidates = app.run_scanner_only()
            print(f"\n✅ Scanner completado: {len(candidates)} candidatos encontrados")
        
        elif args.optimize_only:
            # Solo optimización
            if not args.symbols:
                print("❌ Error: --optimize-only requiere --symbols")
                return
            
            results = app.run_optimization_only(args.symbols, args.trials)
            print(f"\n✅ Optimización completada para {len(args.symbols)} símbolos")
        
        elif args.futures:
            # Pipeline específico para futuros
            results = app.run_futures_pipeline(args.symbols, args.trials)
            
            if results['success']:
                print(f"\n✅ Pipeline de futuros exitoso!")
                print(f"⚡ Símbolos analizados: {len(results['symbols_analyzed'])}")
                print(f"🎯 Oportunidades rentables: {results['futures_optimization_results'].get('profitable_count', 0)}")
                print(f"⏱️  Tiempo total: {results['total_time_seconds']:.1f} segundos")
                
                if results.get('best_futures_opportunity'):
                    best = results['best_futures_opportunity']
                    print(f"🏆 Mejor oportunidad: {best.get('symbol', 'N/A')}")
                    print(f"   💰 Retorno esperado: {best.get('roi', 0):.2f}%")
                    print(f"   ⚡ Leverage: {best.get('leverage', 1)}x")
                    print(f"   🚨 Sin liquidación: {not best.get('was_liquidated', True)}")
            else:
                print(f"\n❌ Error en pipeline de futuros: {results['error']}")
        
        else:
            # Análisis completo
            results = app.run_full_analysis(args.symbols, args.trials)
            
            if results['success']:
                print(f"\n✅ Análisis completo exitoso!")
                print(f"🎯 Símbolos analizados: {len(results['symbols_analyzed'])}")
                print(f"⏱️  Tiempo total: {results['total_time_seconds']:.1f} segundos")
            else:
                print(f"\n❌ Error en análisis: {results['error']}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Aplicación interrumpida por el usuario")
    except Exception as e:
        logger.error(f"❌ Error crítico: {e}")
        print(f"\n❌ Error crítico: {e}")


if __name__ == "__main__":
    main() 