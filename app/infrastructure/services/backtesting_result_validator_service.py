"""
BacktestingResultValidatorService - Validación Robusta de Resultados de Backtesting
==================================================================================

Valida resultados de backtesting para detectar:
- Resultados sospechosos o imposibles
- Curve-fitting y overfitting
- Inconsistencias entre parámetros y resultados
- Datos corruptos o anómalos

Proporciona métricas de calidad y alertas automáticas.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from datetime import datetime
import numpy as np
import pandas as pd
import json
import os

class ValidationSeverity(Enum):
    """Niveles de severidad de validación."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ValidationType(Enum):
    """Tipos de validación."""
    METRICS_VALIDATION = "metrics_validation"
    CONSISTENCY_CHECK = "consistency_check"
    OVERFITTING_DETECTION = "overfitting_detection"
    DATA_QUALITY = "data_quality"
    PARAMETER_VALIDATION = "parameter_validation"

class BacktestingResultValidatorService:
    """
    Servicio de validación robusta de resultados de backtesting.
    
    Detecta resultados sospechosos, imposibles o corruptos y proporciona
    métricas de calidad del backtesting.
    """
    
    def __init__(self, strategy_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el validador de resultados de backtesting.
        
        Args:
            strategy_name: Nombre de la estrategia
            config: Configuración del validador
        """
        self.strategy_name = strategy_name
        self.config = config or {}
        
        # Configurar logging
        self.logger = logging.getLogger(f"backtesting_validator.{strategy_name}")
        
        # Límites de validación
        self.limits = {
            'max_return_percent': 1000.0,  # 1000% máximo
            'min_return_percent': -99.0,   # -99% mínimo
            'max_sharpe_ratio': 10.0,      # Sharpe máximo realista
            'min_sharpe_ratio': -5.0,      # Sharpe mínimo
            'max_drawdown_percent': 95.0,  # Drawdown máximo
            'min_trades': 1,               # Mínimo de operaciones
            'max_trades_per_day': 50,      # Máximo operaciones por día
            'min_win_rate': 0.0,           # Win rate mínimo
            'max_win_rate': 1.0,           # Win rate máximo
            'min_profit_factor': 0.0,      # Profit factor mínimo
            'max_profit_factor': 100.0,    # Profit factor máximo
            'min_avg_trade': -1000.0,      # Trade promedio mínimo
            'max_avg_trade': 10000.0,      # Trade promedio máximo
        }
        
        # Actualizar límites con configuración
        if 'limits' in self.config:
            self.limits.update(self.config['limits'])
        
        # Métricas de calidad
        self.quality_metrics = {
            'overall_score': 0.0,
            'data_quality_score': 0.0,
            'consistency_score': 0.0,
            'realism_score': 0.0,
            'validation_issues': []
        }
        
        self.logger.info(f"🔍 BacktestingResultValidator inicializado para {strategy_name}")
    
    def validate_backtest_results(self, results: Dict[str, Any], 
                                parameters: Dict[str, Any],
                                data_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida resultados completos de backtesting.
        
        Args:
            results: Resultados del backtesting
            parameters: Parámetros usados en el backtesting
            data_info: Información sobre los datos usados
            
        Returns:
            Dict con resultados de validación y métricas de calidad
        """
        try:
            self.logger.info(f"🔍 Iniciando validación de resultados para {self.strategy_name}")
            
            validation_results = {
                'is_valid': True,
                'overall_score': 0.0,
                'issues': [],
                'warnings': [],
                'critical_issues': [],
                'quality_metrics': {},
                'recommendations': []
            }
            
            # 1. Validar métricas básicas
            metrics_validation = self._validate_metrics(results)
            validation_results['issues'].extend(metrics_validation['issues'])
            validation_results['warnings'].extend(metrics_validation['warnings'])
            validation_results['critical_issues'].extend(metrics_validation['critical_issues'])
            
            # 2. Validar consistencia
            consistency_check = self._validate_consistency(results, parameters)
            validation_results['issues'].extend(consistency_check['issues'])
            validation_results['warnings'].extend(consistency_check['warnings'])
            
            # 3. Detectar overfitting
            overfitting_check = self._detect_overfitting(results, parameters, data_info)
            validation_results['issues'].extend(overfitting_check['issues'])
            validation_results['warnings'].extend(overfitting_check['warnings'])
            
            # 4. Validar calidad de datos
            data_quality_check = self._validate_data_quality(results, data_info)
            validation_results['issues'].extend(data_quality_check['issues'])
            validation_results['warnings'].extend(data_quality_check['warnings'])
            
            # 5. Calcular métricas de calidad
            quality_metrics = self._calculate_quality_metrics(
                results, parameters, data_info,
                validation_results['issues'],
                validation_results['warnings'],
                validation_results['critical_issues']
            )
            validation_results['quality_metrics'] = quality_metrics
            validation_results['overall_score'] = quality_metrics['overall_score']
            
            # 6. Generar recomendaciones
            validation_results['recommendations'] = self._generate_recommendations(
                validation_results, results, parameters
            )
            
            # 7. Determinar si es válido
            validation_results['is_valid'] = (
                len(validation_results['critical_issues']) == 0 and
                quality_metrics['overall_score'] >= 0.6
            )
            
            # Logging de resultados
            if validation_results['is_valid']:
                self.logger.info(f"✅ Resultados de {self.strategy_name} validados correctamente (Score: {quality_metrics['overall_score']:.2f})")
            else:
                self.logger.warning(f"⚠️ Resultados de {self.strategy_name} tienen problemas (Score: {quality_metrics['overall_score']:.2f})")
                for issue in validation_results['critical_issues']:
                    self.logger.error(f"🚨 CRÍTICO: {issue}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Error en validación de resultados: {str(e)}")
            return {
                'is_valid': False,
                'overall_score': 0.0,
                'issues': [f"Error en validación: {str(e)}"],
                'warnings': [],
                'critical_issues': [f"Error crítico en validación: {str(e)}"],
                'quality_metrics': {'overall_score': 0.0},
                'recommendations': ["Revisar configuración del validador"]
            }
    
    def _validate_metrics(self, results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Valida métricas básicas del backtesting."""
        issues = []
        warnings = []
        critical_issues = []
        
        try:
            # Validar retorno
            if 'Return [%]' in results:
                return_pct = results['Return [%]']
                if return_pct > self.limits['max_return_percent']:
                    critical_issues.append(f"Retorno imposible: {return_pct:.2f}% > {self.limits['max_return_percent']}%")
                elif return_pct < self.limits['min_return_percent']:
                    warnings.append(f"Retorno muy bajo: {return_pct:.2f}%")
            
            # Validar Sharpe Ratio
            if 'Sharpe Ratio' in results:
                sharpe = results['Sharpe Ratio']
                if not np.isnan(sharpe):
                    if sharpe > self.limits['max_sharpe_ratio']:
                        warnings.append(f"Sharpe Ratio sospechoso: {sharpe:.2f} > {self.limits['max_sharpe_ratio']}")
                    elif sharpe < self.limits['min_sharpe_ratio']:
                        warnings.append(f"Sharpe Ratio muy bajo: {sharpe:.2f}")
            
            # Validar Drawdown
            if 'Max. Drawdown [%]' in results:
                drawdown = abs(results['Max. Drawdown [%]'])
                if drawdown > self.limits['max_drawdown_percent']:
                    critical_issues.append(f"Drawdown imposible: {drawdown:.2f}% > {self.limits['max_drawdown_percent']}%")
            
            # Validar número de operaciones
            if '# Trades' in results:
                trades = results['# Trades']
                if trades < self.limits['min_trades']:
                    warnings.append(f"Muy pocas operaciones: {trades} < {self.limits['min_trades']}")
                
                # Validar operaciones por día si hay datos de tiempo
                if 'Duration' in results and trades > 0:
                    duration_days = results['Duration'].days if hasattr(results['Duration'], 'days') else 1
                    trades_per_day = trades / max(duration_days, 1)
                    if trades_per_day > self.limits['max_trades_per_day']:
                        warnings.append(f"Demasiadas operaciones por día: {trades_per_day:.1f} > {self.limits['max_trades_per_day']}")
            
            # Validar Win Rate
            if 'Win Rate [%]' in results:
                win_rate = results['Win Rate [%]'] / 100.0
                if win_rate < self.limits['min_win_rate'] or win_rate > self.limits['max_win_rate']:
                    critical_issues.append(f"Win Rate inválido: {win_rate:.2f}")
            
            # Validar Profit Factor
            if 'Profit Factor' in results:
                profit_factor = results['Profit Factor']
                if not np.isnan(profit_factor):
                    if profit_factor < self.limits['min_profit_factor'] or profit_factor > self.limits['max_profit_factor']:
                        warnings.append(f"Profit Factor sospechoso: {profit_factor:.2f}")
            
            # Validar Average Trade
            if 'Avg. Trade [%]' in results:
                avg_trade = results['Avg. Trade [%]']
                if avg_trade < self.limits['min_avg_trade'] or avg_trade > self.limits['max_avg_trade']:
                    warnings.append(f"Average Trade sospechoso: {avg_trade:.2f}%")
            
        except Exception as e:
            critical_issues.append(f"Error validando métricas: {str(e)}")
        
        return {
            'issues': issues,
            'warnings': warnings,
            'critical_issues': critical_issues
        }
    
    def _validate_consistency(self, results: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, List[str]]:
        """Valida consistencia entre parámetros y resultados."""
        issues = []
        warnings = []
        
        try:
            # Validar que el número de operaciones sea consistente con la estrategia
            if '# Trades' in results and 'strategy_type' in parameters:
                trades = results['# Trades']
                strategy_type = parameters['strategy_type']
                
                if strategy_type == 'grid' and trades < 5:
                    warnings.append(f"Grid strategy con pocas operaciones: {trades}")
                elif strategy_type == 'dca' and trades < 2:
                    warnings.append(f"DCA strategy con muy pocas operaciones: {trades}")
            
            # Validar que el retorno sea consistente con el drawdown
            if 'Return [%]' in results and 'Max. Drawdown [%]' in results:
                return_pct = results['Return [%]']
                drawdown = abs(results['Max. Drawdown [%]'])
                
                # Si el retorno es muy alto pero el drawdown es muy bajo, es sospechoso
                if return_pct > 50 and drawdown < 5:
                    warnings.append(f"Retorno alto ({return_pct:.1f}%) con drawdown muy bajo ({drawdown:.1f}%) - posible overfitting")
                
                # Si el drawdown es mayor que el retorno, es problemático
                if drawdown > return_pct and return_pct > 0:
                    warnings.append(f"Drawdown ({drawdown:.1f}%) mayor que retorno ({return_pct:.1f}%)")
            
            # Validar consistencia de Sharpe Ratio
            if 'Sharpe Ratio' in results and 'Return [%]' in results and 'Max. Drawdown [%]' in results:
                sharpe = results['Sharpe Ratio']
                return_pct = results['Return [%]']
                drawdown = abs(results['Max. Drawdown [%]'])
                
                if not np.isnan(sharpe) and drawdown > 0:
                    # Calcular Sharpe aproximado para validar
                    expected_sharpe = return_pct / drawdown
                    if abs(sharpe - expected_sharpe) > 2.0:
                        warnings.append(f"Sharpe Ratio inconsistente: {sharpe:.2f} vs esperado ~{expected_sharpe:.2f}")
            
        except Exception as e:
            issues.append(f"Error validando consistencia: {str(e)}")
        
        return {
            'issues': issues,
            'warnings': warnings,
            'critical_issues': []
        }
    
    def _detect_overfitting(self, results: Dict[str, Any], parameters: Dict[str, Any], 
                           data_info: Dict[str, Any]) -> Dict[str, List[str]]:
        """Detecta posibles casos de overfitting."""
        issues = []
        warnings = []
        
        try:
            # Detectar retorno demasiado alto
            if 'Return [%]' in results:
                return_pct = results['Return [%]']
                if return_pct > 200:  # Más de 200% es sospechoso
                    warnings.append(f"Retorno muy alto ({return_pct:.1f}%) - posible overfitting")
                
                if return_pct > 500:  # Más de 500% es crítico
                    issues.append(f"Retorno extremadamente alto ({return_pct:.1f}%) - muy probable overfitting")
            
            # Detectar Sharpe Ratio demasiado alto
            if 'Sharpe Ratio' in results:
                sharpe = results['Sharpe Ratio']
                if not np.isnan(sharpe) and sharpe > 5.0:
                    warnings.append(f"Sharpe Ratio muy alto ({sharpe:.2f}) - posible overfitting")
                
                if not np.isnan(sharpe) and sharpe > 8.0:
                    issues.append(f"Sharpe Ratio extremadamente alto ({sharpe:.2f}) - muy probable overfitting")
            
            # Detectar win rate perfecto o muy alto
            if 'Win Rate [%]' in results:
                win_rate = results['Win Rate [%]']
                if win_rate > 90:
                    warnings.append(f"Win Rate muy alto ({win_rate:.1f}%) - posible overfitting")
                
                if win_rate > 95:
                    issues.append(f"Win Rate casi perfecto ({win_rate:.1f}%) - muy probable overfitting")
            
            # Detectar profit factor extremo
            if 'Profit Factor' in results:
                profit_factor = results['Profit Factor']
                if not np.isnan(profit_factor) and profit_factor > 10:
                    warnings.append(f"Profit Factor muy alto ({profit_factor:.2f}) - posible overfitting")
                
                if not np.isnan(profit_factor) and profit_factor > 20:
                    issues.append(f"Profit Factor extremo ({profit_factor:.2f}) - muy probable overfitting")
            
            # Validar contra información de datos
            if 'data_points' in data_info and '# Trades' in results:
                data_points = data_info['data_points']
                trades = results['# Trades']
                
                # Si hay muy pocas operaciones en muchos datos, es sospechoso
                if data_points > 1000 and trades < 10:
                    warnings.append(f"Muy pocas operaciones ({trades}) en muchos datos ({data_points}) - posible overfitting")
                
                # Si hay demasiadas operaciones, también es sospechoso
                if trades > data_points * 0.1:  # Más del 10% de los datos son operaciones
                    warnings.append(f"Demasiadas operaciones ({trades}) para los datos ({data_points}) - posible overfitting")
            
        except Exception as e:
            issues.append(f"Error detectando overfitting: {str(e)}")
        
        return {
            'issues': issues,
            'warnings': warnings,
            'critical_issues': []
        }
    
    def _validate_data_quality(self, results: Dict[str, Any], data_info: Dict[str, Any]) -> Dict[str, List[str]]:
        """Valida la calidad de los datos usados en el backtesting."""
        issues = []
        warnings = []
        
        try:
            # Validar que hay suficientes datos
            if 'data_points' in data_info:
                data_points = data_info['data_points']
                if data_points < 100:
                    warnings.append(f"Muy pocos datos: {data_points} puntos")
                elif data_points < 500:
                    warnings.append(f"Pocos datos: {data_points} puntos")
            
            # Validar período de datos
            if 'start_date' in data_info and 'end_date' in data_info:
                start_date = data_info['start_date']
                end_date = data_info['end_date']
                
                if hasattr(start_date, 'date') and hasattr(end_date, 'date'):
                    duration_days = (end_date.date() - start_date.date()).days
                    if duration_days < 30:
                        warnings.append(f"Período muy corto: {duration_days} días")
                    elif duration_days < 90:
                        warnings.append(f"Período corto: {duration_days} días")
            
            # Validar que no hay gaps grandes en los datos
            if 'data_gaps' in data_info:
                gaps = data_info['data_gaps']
                if gaps > 10:
                    warnings.append(f"Muchos gaps en datos: {gaps}")
            
            # Validar que los datos no son sintéticos
            if 'data_source' in data_info:
                source = data_info['data_source']
                if 'synthetic' in source.lower() or 'generated' in source.lower():
                    warnings.append(f"Datos sintéticos detectados: {source}")
            
        except Exception as e:
            issues.append(f"Error validando calidad de datos: {str(e)}")
        
        return {
            'issues': issues,
            'warnings': warnings,
            'critical_issues': []
        }
    
    def _calculate_quality_metrics(self, results: Dict[str, Any], parameters: Dict[str, Any],
                                 data_info: Dict[str, Any], issues: List[str], warnings: List[str],
                                 critical_issues: List[str]) -> Dict[str, float]:
        """Calcula métricas de calidad del backtesting."""
        try:
            # Score base
            base_score = 1.0
            
            # Penalizar por issues
            issue_penalty = len(issues) * 0.05
            warning_penalty = len(warnings) * 0.02
            critical_penalty = len(critical_issues) * 0.2
            
            # Penalizar por métricas sospechosas
            metric_penalty = 0.0
            
            if 'Return [%]' in results:
                return_pct = results['Return [%]']
                if return_pct > 100:
                    metric_penalty += 0.1
                if return_pct > 200:
                    metric_penalty += 0.2
            
            if 'Sharpe Ratio' in results:
                sharpe = results['Sharpe Ratio']
                if not np.isnan(sharpe) and sharpe > 5:
                    metric_penalty += 0.1
                if not np.isnan(sharpe) and sharpe > 8:
                    metric_penalty += 0.2
            
            if 'Win Rate [%]' in results:
                win_rate = results['Win Rate [%]']
                if win_rate > 90:
                    metric_penalty += 0.1
                if win_rate > 95:
                    metric_penalty += 0.2
            
            # Calcular score final
            final_score = max(0.0, base_score - issue_penalty - warning_penalty - critical_penalty - metric_penalty)
            
            return {
                'overall_score': final_score,
                'data_quality_score': 0.8 if len(data_info) > 0 else 0.5,
                'consistency_score': 0.9 if len(issues) == 0 else 0.6,
                'realism_score': 0.8 if metric_penalty < 0.1 else 0.5,
                'validation_issues': len(issues) + len(warnings) + len(critical_issues)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculando métricas de calidad: {str(e)}")
            return {
                'overall_score': 0.0,
                'data_quality_score': 0.0,
                'consistency_score': 0.0,
                'realism_score': 0.0,
                'validation_issues': 999
            }
    
    def _generate_recommendations(self, validation_results: Dict[str, Any], 
                                results: Dict[str, Any], parameters: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones basadas en los resultados de validación."""
        recommendations = []
        
        try:
            score = validation_results['overall_score']
            
            if score < 0.5:
                recommendations.append("Revisar completamente la estrategia - score muy bajo")
            elif score < 0.7:
                recommendations.append("Considerar ajustar parámetros - score bajo")
            
            if len(validation_results['critical_issues']) > 0:
                recommendations.append("Resolver issues críticos antes de usar en producción")
            
            if len(validation_results['warnings']) > 5:
                recommendations.append("Muchas advertencias - considerar validación adicional")
            
            # Recomendaciones específicas basadas en métricas
            if 'Return [%]' in results and results['Return [%]'] > 200:
                recommendations.append("Retorno muy alto - considerar validación con datos out-of-sample")
            
            if 'Sharpe Ratio' in results and not np.isnan(results['Sharpe Ratio']) and results['Sharpe Ratio'] > 5:
                recommendations.append("Sharpe Ratio muy alto - verificar robustez de la estrategia")
            
            if 'Win Rate [%]' in results and results['Win Rate [%]'] > 90:
                recommendations.append("Win Rate muy alto - posible overfitting")
            
            if len(recommendations) == 0:
                recommendations.append("Resultados parecen válidos - proceder con precaución")
            
        except Exception as e:
            recommendations.append(f"Error generando recomendaciones: {str(e)}")
        
        return recommendations
    
    def save_validation_report(self, validation_results: Dict[str, Any], 
                             results: Dict[str, Any], parameters: Dict[str, Any],
                             data_info: Dict[str, Any]) -> str:
        """Guarda un reporte detallado de validación."""
        try:
            # Crear directorio si no existe
            os.makedirs('logs/validation_reports', exist_ok=True)
            
            # Generar nombre de archivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/validation_reports/validation_report_{self.strategy_name}_{timestamp}.json"
            
            # Crear reporte completo
            report = {
                'strategy_name': self.strategy_name,
                'timestamp': datetime.now().isoformat(),
                'validation_results': validation_results,
                'backtest_results': results,
                'parameters': parameters,
                'data_info': data_info,
                'validator_config': self.config
            }
            
            # Guardar reporte
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"📄 Reporte de validación guardado: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error guardando reporte de validación: {str(e)}")
            return "" 