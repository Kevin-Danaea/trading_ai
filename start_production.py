#!/usr/bin/env python3
"""
Trading AI - Entrada de Producción
==================================

Script de entrada simple para producción que usa el app/main.py directamente.
Ejecuta el sistema completo de trading con scanner, optimización y backtesting.

Uso:
    python start_production.py
    python start_production.py --verbose
    python start_production.py --symbols BTC/USDT ETH/USDT
"""

import sys
import os
import argparse
import subprocess
from datetime import datetime

def main():
    """Función principal para producción."""
    parser = argparse.ArgumentParser(description='Trading AI - Producción')
    parser.add_argument('--verbose', '-v', action='store_true', help='Logs detallados')
    parser.add_argument('--symbols', '-s', nargs='*', help='Símbolos específicos a analizar')
    parser.add_argument('--trials', '-t', type=int, default=50, help='Número de trials por símbolo')
    parser.add_argument('--scanner-only', action='store_true', help='Solo ejecutar scanner')
    parser.add_argument('--status', action='store_true', help='Estado del sistema')
    
    args = parser.parse_args()
    
    # Usar el mismo ejecutable de Python que está ejecutando este script
    python_executable = sys.executable
    
    # Construir comando para app/main.py
    cmd = [python_executable, 'app/main.py']
    
    if args.status:
        cmd.append('--status')
    elif args.scanner_only:
        cmd.append('--scanner-only')
    else:
        # Análisis completo
        if args.symbols:
            cmd.extend(['--symbols'] + args.symbols)
        if args.trials:
            cmd.extend(['--trials', str(args.trials)])
    
    print(f"🚀 Ejecutando: {' '.join(cmd)}")
    print(f"📅 Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    try:
        # Ejecutar el comando
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        print("\n" + "=" * 60)
        print("✅ Proceso completado exitosamente")
        print(f"📅 Finalizado: {datetime.now().isoformat()}")
        
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error en el proceso: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⏹️  Proceso interrumpido por el usuario")
        return 1
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 