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
            from app.infrastructure.services import BacktestingService
            
            # Inicializar proveedores reales
            self.market_data_provider = MarketDataProvider()
            self.sentiment_data_provider = SentimentDataProvider()
            self.backtesting_service = BacktestingService()
            self.data_service = self.market_data_provider  # Por simplicidad, usar MarketDataProvider como data_service
            
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
        Ejecuta solo optimización bayesiana.
        
        Args:
            symbols: Lista de símbolos a optimizar
            n_trials: Número de trials por símbolo
            
        Returns:
            Resultados de optimización con formato estandarizado
        """
        logger.info(f"🧠 Ejecutando optimización bayesiana para {len(symbols)} símbolos...")
        
        try:
            if self.optimizer_service:
                raw_results = self.optimizer_service.optimize_portfolio(
                    symbols=symbols,
                    n_trials_per_symbol=n_trials
                )
                
                # Convertir a formato estandarizado
                formatted_results = self._format_optimization_results(raw_results)
                
                logger.info("✅ Optimización completada")
                return {
                    'success': True,
                    'results': formatted_results,
                    'symbols_processed': len(symbols),
                    'total_optimizations': sum(len(r) for r in raw_results.values())
                }
            else:
                logger.warning("⚠️ Optimizer service no disponible (modo mock)")
                return {'success': False, 'error': 'Optimizer service not available'}
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
        
        for symbol, optimization_results in raw_results.items():
            formatted[symbol] = {}
            
            for opt_result in optimization_results:
                strategy_name = opt_result.strategy
                
                # Considerar exitosa si tiene best_value > 0 y best_params (independientemente de trials_completed)
                is_successful = (opt_result.best_value > 0 and 
                               bool(opt_result.best_params) and
                               opt_result.get_roi() > 0)
                
                if is_successful:
                    formatted[symbol][strategy_name] = {
                        'success': True,
                        'roi': opt_result.get_roi(),
                        'sharpe_ratio': opt_result.get_sharpe_ratio(),
                        'max_drawdown': opt_result.get_max_drawdown(),
                        'win_rate': opt_result.get_win_rate(),
                        'total_trades': getattr(opt_result, 'best_trial', {}).get('user_attrs', {}).get('total_trades', 0),
                        'best_params': opt_result.best_params,
                        'score': opt_result.best_value,
                        'iterations': opt_result.trials_completed,
                        'duration': opt_result.optimization_time,
                        'confidence': 0.8  # Valor por defecto
                    }
                else:
                    formatted[symbol][strategy_name] = {
                        'success': False,
                        'error': f'Optimization failed: value={opt_result.best_value}, roi={opt_result.get_roi()}'
                    }
        
        return formatted
    
    def run_full_analysis(self, 
                         force_symbols: Optional[List[str]] = None,
                         n_trials: int = 150) -> Dict[str, Any]:
        """
        Ejecuta análisis completo: Scanner + Optimización + Análisis Cualitativo + Selección Semanal + Notificaciones.
        
        Args:
            force_symbols: Símbolos específicos (omite scanner si se proporciona)
            n_trials: Número de trials por símbolo
            
        Returns:
            Resultados completos del análisis
        """
        logger.info("🚀 Ejecutando análisis completo del pipeline...")
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
            
            # Fase 2: Optimización bayesiana
            logger.info("📊 Iniciando optimización bayesiana...")
            optimization_results = self.run_optimization_only(symbols, n_trials)
            
            if not optimization_results.get('success'):
                logger.error("❌ Error en optimización bayesiana")
                return {'success': False, 'error': 'Optimization failed'}
            
            # Fase 3: Análisis cualitativo
            logger.info("🧠 Iniciando análisis cualitativo...")
            qualitative_results = self._run_qualitative_analysis(optimization_results)
            
            if not qualitative_results:
                logger.error("❌ No se obtuvieron resultados del análisis cualitativo")
                return {'success': False, 'error': 'Qualitative analysis failed'}
            
            # Fase 4: Selección de cartera semanal
            logger.info("🎯 Iniciando selección de cartera semanal...")
            weekly_results = self._run_weekly_selection(qualitative_results)
            
            # Fase 5: Resultados finales
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            results = {
                'success': True,
                'symbols_analyzed': symbols,
                'total_symbols': len(symbols),
                'optimization_results': optimization_results,
                'qualitative_results': len(qualitative_results),
                'weekly_results': weekly_results,
                'total_time_seconds': total_time,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
            
            logger.info(f"✅ Pipeline completo terminado en {total_time:.1f} segundos")
            logger.info(f"📈 Recomendaciones semanales generadas: {weekly_results.get('total_recommendations', 0)}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en pipeline completo: {e}")
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
        Ejecuta selección de cartera semanal y notificaciones.
        
        Args:
            qualitative_results: Resultados del análisis cualitativo
            
        Returns:
            Resultados de la selección semanal
        """
        try:
            from app.application.use_cases.orchestration.daily_recommendation_service import WeeklyRecommendationService
            
            # Ejecutar selección semanal completa
            weekly_service = WeeklyRecommendationService()
            weekly_results = weekly_service.process_weekly_recommendations(
                qualitative_results=qualitative_results,
                version_pipeline="weekly_v1.0"
            )
            
            logger.info(f"🎯 Selección semanal completada: {weekly_results.get('success', False)}")
            return weekly_results
            
        except Exception as e:
            logger.error(f"❌ Error en selección semanal: {e}")
            return {'success': False, 'error': str(e)}
    
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
            for symbol, strategies in optimization_data.items():
                logger.info(f"📊 {symbol}: {list(strategies.keys())}")
            
            for symbol, symbol_data in optimization_data.items():
                logger.info(f"🔍 Procesando símbolo: {symbol}")
                
                # Agrupar estrategias por símbolo
                strategy_results = {}
                best_strategy = None
                best_roi = -999
                
                for strategy_name, strategy_data in symbol_data.items():
                    logger.info(f"📊 Estrategia {strategy_name}: success={strategy_data.get('success')}, roi={strategy_data.get('roi', 0)}")
                    if strategy_data.get('success') and strategy_data.get('roi', 0) > 0:
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
                            best_strategy = strategy_name
                
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