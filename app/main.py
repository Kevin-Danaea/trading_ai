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
            from app.infrastructure.config.settings import settings
            self.settings = settings
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
            
            # Inicializar servicios reales con inyección de dependencias
            self.scanner_service = CryptoScannerService(
                market_data_provider=self.market_data_provider,
                sentiment_data_provider=self.sentiment_data_provider
            )
            
            self.optimizer_service = BayesianOptimizerService(
                backtesting_service=self.backtesting_service,
                data_service=self.data_service
            )
            
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
            Resultados de optimización
        """
        logger.info(f"🧠 Ejecutando optimización bayesiana para {len(symbols)} símbolos...")
        
        try:
            if self.optimizer_service:
                results = self.optimizer_service.optimize_portfolio(
                    symbols=symbols,
                    n_trials_per_symbol=n_trials
                )
                logger.info("✅ Optimización completada")
                return results
            else:
                logger.warning("⚠️ Optimizer service no disponible (modo mock)")
                return {}
        except Exception as e:
            logger.error(f"❌ Error en optimización: {e}")
            return {}
    
    def run_full_analysis(self, 
                         force_symbols: Optional[List[str]] = None,
                         n_trials: int = 150) -> Dict[str, Any]:
        """
        Ejecuta análisis completo: Scanner + Optimización.
        
        Args:
            force_symbols: Símbolos específicos (omite scanner si se proporciona)
            n_trials: Número de trials por símbolo
            
        Returns:
            Resultados completos del análisis
        """
        logger.info("🚀 Ejecutando análisis completo...")
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
            optimization_results = self.run_optimization_only(symbols, n_trials)
            
            # Fase 3: Resultados finales
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            results = {
                'success': True,
                'symbols_analyzed': symbols,
                'total_symbols': len(symbols),
                'optimization_results': optimization_results,
                'total_time_seconds': total_time,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
            
            logger.info(f"✅ Análisis completo terminado en {total_time:.1f} segundos")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en análisis completo: {e}")
            return {'success': False, 'error': str(e)}
    
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