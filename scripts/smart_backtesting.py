#!/usr/bin/env python3
"""
Smart Backtesting - Sistema Inteligente de Backtesting
=====================================================

Script principal que integra el scanner inteligente con el backtesting optimizado.
Este sistema actúa como "cerebro" que:

1. 🔍 USA EL SCANNER para filtrar las mejores 5-10 monedas del Top 100
2. 🚀 EJECUTA BACKTESTING solo en esas monedas seleccionadas
3. ⚡ OPTIMIZA el tiempo de procesamiento de 22 horas a ~2 horas
4. 📊 GENERA reportes comparativos de las mejores estrategias

Flujo de trabajo:
Scanner → Filtrar Candidatos → Backtesting Optimizado → Reporte Final
"""

import os
import sys
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import time
import logging

# Agregar el directorio padre al path para importar módulos del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanner import CryptoScanner, CryptoCandidate
from find_optimal_parameters import main as run_backtest, UNIVERSO_MONEDAS
from optimizer import BayesianOptimizer, OptimizationResult
from shared.config.settings import settings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmartBacktestingOrchestrator:
    """
    Orquestador inteligente que combina scanner + backtesting optimizado.
    """
    
    def __init__(self, 
                 scanner_config: Optional[Dict[str, Any]] = None,
                 use_bayesian_optimizer: bool = True,
                 optimization_window_months: int = 9):
        """
        Inicializa el orquestador inteligente.
        
        Args:
            scanner_config: Configuración del scanner (opcional)
            use_bayesian_optimizer: Si usar optimizador bayesiano (recomendado)
            optimization_window_months: Ventana de datos para optimización
        """
        # Configuración por defecto del scanner
        default_config = {
            'top_n': 100,
            'min_volume_usdt': 10_000_000,  # $10M mínimo
            'max_candidates': 10
        }
        
        if scanner_config:
            default_config.update(scanner_config)
        
        self.scanner_config = default_config
        self.scanner = CryptoScanner()  # Configuración fija para producción
        
        # Nuevo: Optimizador Bayesiano
        self.use_bayesian_optimizer = use_bayesian_optimizer
        if use_bayesian_optimizer:
            self.optimizer = BayesianOptimizer(optimization_window_months=optimization_window_months)
            logger.info("🧠 Optimizador Bayesiano activado (Master Chef mode)")
        else:
            self.optimizer = None
            logger.info("⚡ Modo tradicional (sin optimizador bayesiano)")
        
        logger.info("🧠 SmartBacktestingOrchestrator inicializado")
        logger.info(f"🔍 Scanner: Top {default_config['top_n']}, Vol mín: ${default_config['min_volume_usdt']/1_000_000:.0f}M")
        logger.info(f"🎯 Máximo candidatos: {default_config['max_candidates']}")
        logger.info(f"🔬 Ventana optimización: {optimization_window_months} meses")
    
    def run_smart_analysis(self, 
                          force_symbols: Optional[List[str]] = None,
                          skip_scanner: bool = False,
                          dry_run: bool = False,
                          trials_per_symbol: int = 150) -> Dict[str, Any]:
        """
        Ejecuta el análisis completo: Scanner + Backtesting optimizado.
        
        Args:
            force_symbols: Lista de símbolos específicos (omite scanner)
            skip_scanner: Si True, usa universo predefinido
            dry_run: Si True, solo muestra qué se va a hacer sin ejecutar
            
        Returns:
            Resultados del análisis completo
        """
        inicio_total = time.time()
        
        logger.info("🚀 INICIANDO ANÁLISIS INTELIGENTE DE TRADING")
        logger.info("=" * 70)
        
        try:
            # FASE 1: Selección inteligente de monedas
            if force_symbols:
                selected_symbols = force_symbols
                logger.info(f"💡 Usando símbolos específicos: {selected_symbols}")
                selection_method = "Símbolos específicos"
                
            elif skip_scanner:
                selected_symbols = UNIVERSO_MONEDAS[:self.scanner_config['max_candidates']]
                logger.info(f"⚡ Omitiendo scanner, usando universo predefinido")
                selection_method = "Universo predefinido"
                
            else:
                logger.info("🔍 FASE 1: Ejecutando scanner inteligente...")
                candidates = self.scanner.scan_market()  # Siempre devuelve 10 candidatos
                
                if not candidates:
                    logger.error("❌ Scanner no encontró candidatos válidos")
                    return {'success': False, 'error': 'No hay candidatos válidos'}
                
                selected_symbols = [c.symbol for c in candidates]
                selection_method = "Scanner inteligente"
                
                # Guardar resultados del scanner
                scanner_filename = self.scanner.export_candidates_to_json(candidates)
                logger.info(f"💾 Resultados del scanner guardados en: {scanner_filename}")
            
            # Mostrar selección final
            logger.info(f"\n🎯 MONEDAS SELECCIONADAS ({selection_method}):")
            logger.info(f"📊 Total: {len(selected_symbols)} monedas")
            for i, symbol in enumerate(selected_symbols, 1):
                logger.info(f"   {i}. {symbol}")
            
            # Estimación de tiempo
            estimated_time_hours = len(selected_symbols) * 0.2  # ~12 min por moneda
            logger.info(f"\n⏱️  Tiempo estimado de backtesting: {estimated_time_hours:.1f} horas")
            logger.info(f"⚡ Optimización vs universo completo: {(len(UNIVERSO_MONEDAS) / len(selected_symbols)):.1f}x más rápido")
            
            if dry_run:
                logger.info("\n🔬 DRY RUN: No se ejecutará backtesting real")
                return {
                    'success': True,
                    'dry_run': True,
                    'selected_symbols': selected_symbols,
                    'selection_method': selection_method,
                    'estimated_time_hours': estimated_time_hours
                }
            
            # FASE 2: Optimización Inteligente
            if self.use_bayesian_optimizer and self.optimizer:
                logger.info(f"\n🧠 FASE 2: Ejecutando optimización bayesiana (Master Chef)...")
                logger.info(f"🎯 Optimizando {len(selected_symbols)} monedas con Optuna...")
                
                # Ejecutar optimización bayesiana
                optimization_start = time.time()
                optimization_results = self.optimizer.optimize_portfolio(
                    symbols=selected_symbols,
                    n_trials_per_symbol=trials_per_symbol  # Trials inteligentes vs 1500 tradicionales
                )
                optimization_time = time.time() - optimization_start
                backtesting_time = optimization_time
                
                # Extraer mejores configuraciones
                best_configs = self.optimizer.get_best_configuration(optimization_results)
                
                # Guardar configuraciones optimizadas
                self._save_optimized_configurations(best_configs)
                
            else:
                logger.info(f"\n🚀 FASE 2: Ejecutando backtesting tradicional...")
                logger.info(f"🎯 Procesando {len(selected_symbols)} monedas seleccionadas...")
                
                # Ejecutar backtesting tradicional con las monedas seleccionadas
                backtesting_start = time.time()
                run_backtest(selected_symbols)
                backtesting_time = time.time() - backtesting_start
            
            # FASE 3: Resultados y métricas
            total_time = time.time() - inicio_total
            
            results = {
                'success': True,
                'selection_method': selection_method,
                'selected_symbols': selected_symbols,
                'total_symbols': len(selected_symbols),
                'total_time_seconds': total_time,
                'total_time_hours': total_time / 3600,
                'backtesting_time_seconds': backtesting_time,
                'backtesting_time_hours': backtesting_time / 3600,
                'estimated_vs_actual': estimated_time_hours / (total_time / 3600) if total_time > 0 else 0,
                'optimization_factor': len(UNIVERSO_MONEDAS) / len(selected_symbols),
                'timestamp': datetime.now().isoformat()
            }
            
            # Reporte final
            self._generate_final_report(results)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en análisis inteligente: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_final_report(self, results: Dict[str, Any]):
        """Genera reporte final del análisis."""
        logger.info("\n" + "=" * 70)
        logger.info("🎉 ANÁLISIS INTELIGENTE COMPLETADO")
        logger.info("=" * 70)
        
        logger.info(f"🧠 Método de selección: {results['selection_method']}")
        logger.info(f"📊 Monedas analizadas: {results['total_symbols']}")
        logger.info(f"⏱️  Tiempo total: {results['total_time_hours']:.2f} horas")
        logger.info(f"🚀 Tiempo backtesting: {results['backtesting_time_hours']:.2f} horas")
        logger.info(f"⚡ Factor de optimización: {results['optimization_factor']:.1f}x más rápido")
        
        # Comparación con método tradicional
        traditional_time = len(UNIVERSO_MONEDAS) * (results['backtesting_time_hours'] / results['total_symbols'])
        time_saved = traditional_time - results['backtesting_time_hours']
        
        logger.info(f"\n💡 OPTIMIZACIÓN LOGRADA:")
        logger.info(f"   Tiempo tradicional estimado: {traditional_time:.1f} horas")
        logger.info(f"   Tiempo actual: {results['backtesting_time_hours']:.1f} horas") 
        logger.info(f"   Tiempo ahorrado: {time_saved:.1f} horas ({time_saved*60:.0f} minutos)")
        logger.info(f"   Eficiencia: {(time_saved/traditional_time)*100:.1f}% más rápido")
        
        logger.info(f"\n🎯 MONEDAS SELECCIONADAS:")
        for i, symbol in enumerate(results['selected_symbols'], 1):
            logger.info(f"   {i}. {symbol}")
        
        logger.info(f"\n💾 Los resultados han sido guardados en BigQuery")
        logger.info(f"🏆 ANÁLISIS INTELIGENTE FINALIZADO EXITOSAMENTE")
    
    def _save_optimized_configurations(self, best_configs: Dict[str, Dict[str, Any]]):
        """
        Guarda las configuraciones optimizadas en archivos para referencia futura.
        
        Args:
            best_configs: Mejores configuraciones por símbolo
        """
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimized_configs_{timestamp}.json"
        
        # Preparar datos para serialización
        serializable_configs = {}
        for symbol, config in best_configs.items():
            serializable_configs[symbol] = {
                'strategy': config['strategy'],
                'params': config['params'],
                'metrics': config['metrics'],
                'optimization_stats': config['optimization_stats']
            }
        
        # Guardar archivo
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'optimization_method': 'bayesian_optuna',
                'total_symbols': len(best_configs),
                'configurations': serializable_configs
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Configuraciones optimizadas guardadas en: {filename}")
        
        # Log de resumen de configuraciones
        logger.info("\n📊 RESUMEN DE CONFIGURACIONES OPTIMIZADAS:")
        for symbol, config in best_configs.items():
            logger.info(f"   {symbol}: {config['strategy'].upper()} - ROI: {config['metrics']['roi']:.2f}%")
    
    def get_scanner_recommendations(self, save_to_file: bool = True) -> List[CryptoCandidate]:
        """
        Obtiene recomendaciones del scanner sin ejecutar backtesting.
        
        Args:
            save_to_file: Si guardar resultados en archivo JSON
            
        Returns:
            Lista de candidatos recomendados
        """
        logger.info("🔍 Obteniendo recomendaciones del scanner...")
        
        candidates = self.scanner.scan_market()  # Siempre devuelve 10 candidatos
        
        if save_to_file and candidates:
            filename = self.scanner.export_candidates_to_json(candidates)
            logger.info(f"💾 Recomendaciones guardadas en: {filename}")
        
        return candidates
    
    def compare_selection_methods(self) -> Dict[str, Any]:
        """
        Compara diferentes métodos de selección de monedas.
        
        Returns:
            Comparación de métodos
        """
        logger.info("📊 Comparando métodos de selección...")
        
        # Método 1: Scanner inteligente
        scanner_candidates = self.scanner.scan_market()  # Siempre devuelve 10 candidatos
        scanner_symbols = [c.symbol for c in scanner_candidates]
        
        # Método 2: Top volumen
        top_volume_symbols = UNIVERSO_MONEDAS[:10]
        
        # Método 3: Aleatorio
        import random
        random_symbols = random.sample(UNIVERSO_MONEDAS, 10)
        
        comparison = {
            'scanner_intelligent': {
                'symbols': scanner_symbols,
                'method': 'Scanner con indicadores técnicos y sentimiento',
                'avg_score': sum(c.score for c in scanner_candidates) / len(scanner_candidates) if scanner_candidates else 0
            },
            'top_volume': {
                'symbols': top_volume_symbols,
                'method': 'Top 10 por volumen/market cap',
                'avg_score': 0
            },
            'random': {
                'symbols': random_symbols,
                'method': 'Selección aleatoria',
                'avg_score': 0
            }
        }
        
        logger.info("📈 Métodos comparados:")
        for method, data in comparison.items():
            logger.info(f"   {method}: {len(data['symbols'])} monedas")
            if data['avg_score'] > 0:
                logger.info(f"      Score promedio: {data['avg_score']:.1f}")
        
        return comparison


def main():
    """Función principal con interfaz de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Sistema inteligente de backtesting con scanner automático',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

# Análisis completo con optimizador bayesiano (RECOMENDADO)
python scripts/smart_backtesting.py

# Análisis con optimizador bayesiano personalizado
python scripts/smart_backtesting.py --trials-per-symbol 200 --optimization-window 12

# Solo recomendaciones del scanner
python scripts/smart_backtesting.py --scanner-only

# Usar monedas específicas con optimización bayesiana
python scripts/smart_backtesting.py --symbols BTC/USDT ETH/USDT SOL/USDT

# Backtesting tradicional (sin optimizador bayesiano)
python scripts/smart_backtesting.py --disable-bayesian

# Omitir scanner, usar universo predefinido
python scripts/smart_backtesting.py --skip-scanner

# Simulación (ver qué se haría sin ejecutar)
python scripts/smart_backtesting.py --dry-run

# Configurar scanner personalizado
python scripts/smart_backtesting.py --top-n 50 --min-volume 5000000 --max-candidates 8

# Master Chef completo: Scanner + Optimización Bayesiana
python scripts/smart_backtesting.py --trials-per-symbol 300 --optimization-window 6
        """)
    
    # Argumentos principales
    parser.add_argument(
        '--scanner-only',
        action='store_true',
        help='Solo ejecutar scanner sin backtesting'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='*',
        help='Símbolos específicos para analizar (omite scanner)'
    )
    
    parser.add_argument(
        '--skip-scanner',
        action='store_true',
        help='Omitir scanner, usar universo predefinido'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulación: mostrar qué se haría sin ejecutar backtesting'
    )
    
    # Configuración del scanner
    parser.add_argument(
        '--top-n',
        type=int,
        default=100,
        help='Número de top monedas a analizar (default: 100)'
    )
    
    parser.add_argument(
        '--min-volume',
        type=float,
        default=10_000_000,
        help='Volumen mínimo en USDT (default: 10M)'
    )
    
    parser.add_argument(
        '--max-candidates',
        type=int,
        default=10,
        help='Máximo número de candidatos seleccionados (default: 10)'
    )
    
    parser.add_argument(
        '--compare-methods',
        action='store_true',
        help='Comparar diferentes métodos de selección'
    )
    
    # Optimización Bayesiana
    parser.add_argument(
        '--disable-bayesian',
        action='store_true',
        help='Desactivar optimizador bayesiano (usar backtesting tradicional)'
    )
    
    parser.add_argument(
        '--optimization-window',
        type=int,
        default=9,
        help='Ventana de datos históricos para optimización en meses (default: 9)'
    )
    
    parser.add_argument(
        '--trials-per-symbol',
        type=int,
        default=150,
        help='Número de trials por símbolo en optimización bayesiana (default: 150)'
    )
    
    args = parser.parse_args()
    
    # Configurar orquestador
    scanner_config = {
        'top_n': args.top_n,
        'min_volume_usdt': args.min_volume,
        'max_candidates': args.max_candidates
    }
    
    orchestrator = SmartBacktestingOrchestrator(
        scanner_config=scanner_config,
        use_bayesian_optimizer=not args.disable_bayesian,
        optimization_window_months=args.optimization_window
    )
    
    try:
        if args.scanner_only:
            # Solo scanner
            candidates = orchestrator.get_scanner_recommendations()
            if candidates:
                print(f"\n🎯 RECOMENDACIONES PARA BACKTESTING:")
                symbols = [c.symbol for c in candidates]
                print(f"python scripts/find_optimal_parameters.py --monedas {' '.join(symbols)}")
        
        elif args.compare_methods:
            # Comparar métodos
            comparison = orchestrator.compare_selection_methods()
            print("\n📊 COMPARACIÓN DE MÉTODOS DE SELECCIÓN:")
            for method, data in comparison.items():
                print(f"\n{method.upper()}:")
                print(f"  Método: {data['method']}")
                print(f"  Monedas: {', '.join(data['symbols'][:5])}...")
                if data['avg_score'] > 0:
                    print(f"  Score: {data['avg_score']:.1f}")
        
        else:
            # Análisis completo
            results = orchestrator.run_smart_analysis(
                force_symbols=args.symbols,
                skip_scanner=args.skip_scanner,
                dry_run=args.dry_run,
                trials_per_symbol=args.trials_per_symbol
            )
            
            if results['success']:
                print(f"\n✅ Análisis completado exitosamente!")
                if not results.get('dry_run', False):
                    print(f"⏱️  Tiempo total: {results['total_time_hours']:.2f} horas")
                    print(f"⚡ Optimización: {results['optimization_factor']:.1f}x más rápido")
            else:
                print(f"\n❌ Error: {results['error']}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Análisis interrumpido por el usuario")
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        print(f"\n❌ Error inesperado: {e}")


if __name__ == "__main__":
    main() 