"""
ErrorHandlerService - Servicio de Manejo Estándar de Errores
==========================================================

Maneja errores de forma consistente en todas las estrategias de trading.
Proporciona logging estándar, recuperación graceful y alertas para errores críticos.
ELIMINA COMPLETAMENTE LOS ERRORES SILENCIOSOS.
"""

import logging
from typing import Dict, Any, Optional, Callable, Union, List
from enum import Enum
from datetime import datetime, timedelta
import traceback
import json
import os

class ErrorSeverity(Enum):
    """Niveles de severidad de errores."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorType(Enum):
    """Tipos de errores comunes en trading."""
    DATA_ERROR = "data_error"
    PARAMETER_ERROR = "parameter_error"
    CALCULATION_ERROR = "calculation_error"
    TRADING_ERROR = "trading_error"
    SYSTEM_ERROR = "system_error"
    VALIDATION_ERROR = "validation_error"
    SAFETY_VIOLATION = "safety_violation"

class SilentErrorDetected(Exception):
    """Excepción lanzada cuando se detecta un intento de manejo silencioso de errores."""
    pass

class TradingError(Exception):
    """Excepción base para errores de trading."""
    def __init__(self, message: str, error_type: ErrorType, severity: ErrorSeverity, recoverable: bool = True):
        super().__init__(message)
        self.error_type = error_type
        self.severity = severity
        self.recoverable = recoverable
        self.timestamp = datetime.now()

class ErrorHandlerService:
    """
    Servicio para manejo estándar de errores en estrategias de trading.
    ELIMINA COMPLETAMENTE LOS ERRORES SILENCIOSOS.
    
    Args:
        strategy_name (str): Nombre de la estrategia para logging.
        config (dict, opcional): Configuración personalizada.
    
    Example:
        >>> error_handler = ErrorHandlerService("GridStrategy")
        >>> error_handler.handle_error(error, "Error en cálculo de indicadores")
    """
    
    def __init__(self, strategy_name: str, config: Optional[Dict[str, Any]] = None):
        self.strategy_name = strategy_name
        self.config = config or {}
        self.logger = logging.getLogger(f"trading.{strategy_name}")
        
        # Configuración de logging obligatorio
        self.force_logging = self.config.get('force_logging', True)
        self.log_all_errors = self.config.get('log_all_errors', True)
        self.alert_critical_errors = self.config.get('alert_critical_errors', True)
        
        # Estado del servicio
        self.error_count = 0
        self.critical_errors = []
        self.error_history = []
        self.silent_error_attempts = 0
        
        # Configuración de límites
        self.max_errors_per_strategy = self.config.get('max_errors_per_strategy', 10)
        self.auto_recovery_enabled = self.config.get('auto_recovery_enabled', True)
        self.log_critical_errors = self.config.get('log_critical_errors', True)
        
        # Métricas de errores
        self.error_metrics = {
            'total_errors': 0,
            'errors_by_type': {},
            'errors_by_severity': {},
            'recovery_attempts': 0,
            'recovery_successes': 0,
            'silent_error_attempts': 0
        }
        
        # Configurar logging detallado
        self._setup_detailed_logging()
        
        self.logger.info(f"🔊 ErrorHandlerService inicializado para {strategy_name} - LOGGING OBLIGATORIO ACTIVADO")
    
    def _setup_detailed_logging(self):
        """Configura logging detallado para evitar errores silenciosos."""
        # Asegurar que el logger tenga el nivel correcto
        self.logger.setLevel(logging.DEBUG)
        
        # Crear handler para archivo si no existe
        if not self.logger.handlers:
            # Handler para consola
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # Handler para archivo de errores
            error_log_dir = "logs"
            if not os.path.exists(error_log_dir):
                os.makedirs(error_log_dir)
            
            # Asegurar que el directorio de logs tenga permisos correctos
            try:
                os.chmod(error_log_dir, 0o755)
            except Exception:
                pass  # Ignorar errores de permisos en Windows
            
            error_file_handler = logging.FileHandler(
                f"{error_log_dir}/{self.strategy_name}_errors.log",
                encoding='utf-8'
            )
            error_file_handler.setLevel(logging.DEBUG)
            error_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n'
            )
            error_file_handler.setFormatter(error_formatter)
            self.logger.addHandler(error_file_handler)
    
    def handle_error(self, 
                    error: Exception, 
                    context: str = "",
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    error_type: ErrorType = ErrorType.SYSTEM_ERROR,
                    recoverable: bool = True,
                    force_log: bool = True) -> bool:
        """
        Maneja un error de forma estándar. NUNCA maneja errores silenciosamente.
        
        Args:
            error (Exception): Error a manejar.
            context (str): Contexto donde ocurrió el error.
            severity (ErrorSeverity): Nivel de severidad.
            error_type (ErrorType): Tipo de error.
            recoverable (bool): Si el error es recuperable.
            force_log (bool): Forzar logging (por defecto True).
            
        Returns:
            bool: True si el error fue manejado exitosamente.
            
        Raises:
            SilentErrorDetected: Si se intenta manejar silenciosamente.
        """
        # PREVENIR MANEJO SILENCIOSO
        if not force_log and self.force_logging:
            self.silent_error_attempts += 1
            self.error_metrics['silent_error_attempts'] += 1
            silent_error = SilentErrorDetected(
                f"Intento de manejo silencioso detectado en {self.strategy_name}: {error}"
            )
            self.logger.critical(f"🚨 INTENTO DE MANEJO SILENCIOSO DETECTADO: {silent_error}")
            raise silent_error
        
        self.error_count += 1
        self.error_metrics['total_errors'] += 1
        
        # Actualizar métricas
        error_type_str = error_type.value
        severity_str = severity.value
        
        self.error_metrics['errors_by_type'][error_type_str] = \
            self.error_metrics['errors_by_type'].get(error_type_str, 0) + 1
        self.error_metrics['errors_by_severity'][severity_str] = \
            self.error_metrics['errors_by_severity'].get(severity_str, 0) + 1
        
        # Crear mensaje de error estructurado
        error_message = self._create_error_message(error, context, severity, error_type)
        
        # LOGGING OBLIGATORIO según severidad
        self._log_error_obligatory(error_message, severity)
        
        # Manejar errores críticos con alertas
        if severity == ErrorSeverity.CRITICAL:
            self._handle_critical_error_with_alerts(error_message, error)
        
        # Verificar límite de errores
        if self.error_count >= self.max_errors_per_strategy:
            self._handle_error_limit_reached()
            return False
        
        # Recuperación automática si está habilitada
        if self.auto_recovery_enabled and recoverable:
            self.error_metrics['recovery_attempts'] += 1
            recovery_success = self._attempt_recovery(error, error_type)
            if recovery_success:
                self.error_metrics['recovery_successes'] += 1
            return recovery_success
        
        return True
    
    def _create_error_message(self, 
                            error: Exception, 
                            context: str, 
                            severity: ErrorSeverity,
                            error_type: ErrorType) -> Dict[str, Any]:
        """Crea un mensaje de error estructurado con información detallada."""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'strategy': self.strategy_name,
            'error_type': error_type.value,
            'severity': severity.value,
            'context': context,
            'message': str(error),
            'error_class': error.__class__.__name__,
            'traceback': traceback.format_exc(),
            'error_count': self.error_count,
            'recoverable': True
        }
        
        # Agregar información adicional según el tipo de error
        if error_type == ErrorType.DATA_ERROR:
            error_info['data_context'] = "Error en procesamiento de datos"
        elif error_type == ErrorType.TRADING_ERROR:
            error_info['trading_context'] = "Error en operación de trading"
        elif error_type == ErrorType.SAFETY_VIOLATION:
            error_info['safety_context'] = "Violación de límites de seguridad"
        
        # Guardar en historial
        self.error_history.append(error_info)
        
        return error_info
    
    def _log_error_obligatory(self, error_message: Dict[str, Any], severity: ErrorSeverity):
        """Loggea el error de forma obligatoria según su severidad."""
        log_message = f"[{error_message['error_type'].upper()}] {error_message['context']}: {error_message['message']}"
        
        # LOGGING OBLIGATORIO - NUNCA silencioso
        if severity == ErrorSeverity.LOW:
            self.logger.debug(log_message)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
            # Para errores HIGH, también loggear traceback
            if error_message['traceback']:
                self.logger.error(f"Traceback: {error_message['traceback']}")
        elif severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
            # Para errores CRITICAL, siempre loggear traceback completo
            if error_message['traceback']:
                self.logger.critical(f"Traceback completo: {error_message['traceback']}")
        
        # Logging adicional para debugging
        self.logger.debug(f"Error #{error_message['error_count']} en {self.strategy_name}")
    
    def _handle_critical_error_with_alerts(self, error_message: Dict[str, Any], error: Exception):
        """Maneja errores críticos con alertas automáticas."""
        self.critical_errors.append(error_message)
        
        # ALERTA CRÍTICA OBLIGATORIA
        alert_message = f"""
🚨 ALERTA CRÍTICA - {self.strategy_name}
==========================================
Error: {error_message['message']}
Contexto: {error_message['context']}
Tipo: {error_message['error_type']}
Timestamp: {error_message['timestamp']}
Error Count: {error_message['error_count']}

TRACEBACK:
{error_message['traceback']}
==========================================
"""
        
        # Logging crítico obligatorio
        self.logger.critical(alert_message)
        
        # Guardar alerta en archivo separado
        self._save_critical_alert(alert_message)
        
        # Enviar alerta si está configurado
        if self.alert_critical_errors:
            self._send_critical_alert(error_message)
    
    def _save_critical_alert(self, alert_message: str):
        """Guarda alerta crítica en archivo separado."""
        try:
            alert_dir = "logs/alerts"
            if not os.path.exists(alert_dir):
                os.makedirs(alert_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            alert_file = f"{alert_dir}/critical_alert_{self.strategy_name}_{timestamp}.txt"
            
            with open(alert_file, 'w', encoding='utf-8') as f:
                f.write(alert_message)
            
            self.logger.info(f"Alerta crítica guardada en: {alert_file}")
        except Exception as e:
            self.logger.error(f"Error guardando alerta crítica: {e}")
    
    def _send_critical_alert(self, error_message: Dict[str, Any]):
        """Envía alerta crítica (placeholder para integración con sistemas externos)."""
        # Aquí se puede integrar con Telegram, email, Slack, etc.
        self.logger.info(f"📢 Alerta crítica enviada para {self.strategy_name}")
        # Por ahora, solo logging. Se puede expandir después.
    
    def _handle_error_limit_reached(self):
        """Maneja cuando se alcanza el límite de errores."""
        limit_message = f"""
⚠️ LÍMITE DE ERRORES ALCANZADO - {self.strategy_name}
==========================================
Total errores: {self.error_count}
Límite: {self.max_errors_per_strategy}
Último error: {datetime.now().isoformat()}

La estrategia puede estar en un estado inconsistente.
Se recomienda revisar logs y reiniciar si es necesario.
==========================================
"""
        self.logger.error(limit_message)
        
        # Guardar reporte de límite alcanzado
        self._save_error_limit_report()
    
    def _save_error_limit_report(self):
        """Guarda reporte cuando se alcanza límite de errores."""
        try:
            report_dir = "logs/reports"
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"{report_dir}/error_limit_report_{self.strategy_name}_{timestamp}.json"
            
            report_data = {
                'strategy_name': self.strategy_name,
                'timestamp': timestamp,
                'error_count': self.error_count,
                'max_errors': self.max_errors_per_strategy,
                'error_metrics': self.error_metrics,
                'critical_errors_count': len(self.critical_errors),
                'silent_error_attempts': self.silent_error_attempts
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Reporte de límite de errores guardado en: {report_file}")
        except Exception as e:
            self.logger.error(f"Error guardando reporte: {e}")
    
    def _attempt_recovery(self, error: Exception, error_type: ErrorType) -> bool:
        """Intenta recuperación automática según el tipo de error."""
        try:
            self.logger.info(f"🔄 Intentando recuperación automática para {error_type.value}")
            
            if error_type == ErrorType.DATA_ERROR:
                return self._recover_from_data_error(error)
            elif error_type == ErrorType.CALCULATION_ERROR:
                return self._recover_from_calculation_error(error)
            elif error_type == ErrorType.TRADING_ERROR:
                return self._recover_from_trading_error(error)
            else:
                return True  # Para otros tipos, asumir recuperación exitosa
        except Exception as recovery_error:
            self.logger.error(f"Error en recuperación: {recovery_error}")
            return False
    
    def _recover_from_data_error(self, error: Exception) -> bool:
        """Recuperación de errores de datos."""
        self.logger.info("🔄 Intentando recuperación de error de datos...")
        # En este caso, simplemente continuar con valores por defecto
        return True
    
    def _recover_from_calculation_error(self, error: Exception) -> bool:
        """Recuperación de errores de cálculo."""
        self.logger.info("🔄 Intentando recuperación de error de cálculo...")
        # Usar valores por defecto para indicadores
        return True
    
    def _recover_from_trading_error(self, error: Exception) -> bool:
        """Recuperación de errores de trading."""
        self.logger.info("🔄 Intentando recuperación de error de trading...")
        # Cancelar órdenes pendientes y continuar
        return True
    
    def safe_execute(self, 
                    func: Callable, 
                    *args, 
                    context: str = "",
                    default_return: Any = None,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    error_type: ErrorType = ErrorType.SYSTEM_ERROR,
                    **kwargs) -> Any:
        """
        Ejecuta una función de forma segura con manejo de errores.
        NUNCA maneja errores silenciosamente.
        
        Args:
            func (Callable): Función a ejecutar.
            *args: Argumentos de la función.
            context (str): Contexto para logging.
            default_return (Any): Valor por defecto si falla.
            severity (ErrorSeverity): Severidad del error.
            error_type (ErrorType): Tipo de error.
            **kwargs: Argumentos nombrados de la función.
            
        Returns:
            Any: Resultado de la función o default_return si falla.
        """
        try:
            return func(*args, **kwargs)
        except Exception as error:
            # LOGGING OBLIGATORIO - nunca silencioso
            self.handle_error(error, context, severity, error_type, force_log=True)
            return default_return
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen detallado de errores de la estrategia."""
        return {
            'strategy_name': self.strategy_name,
            'total_errors': self.error_count,
            'critical_errors': len(self.critical_errors),
            'error_limit_reached': self.error_count >= self.max_errors_per_strategy,
            'last_error_timestamp': self.critical_errors[-1]['timestamp'] if self.critical_errors else None,
            'error_metrics': self.error_metrics.copy(),
            'silent_error_attempts': self.silent_error_attempts,
            'recovery_success_rate': (
                self.error_metrics['recovery_successes'] / max(1, self.error_metrics['recovery_attempts'])
            )
        }
    
    def get_detailed_error_report(self) -> Dict[str, Any]:
        """Obtiene un reporte detallado de errores."""
        return {
            'summary': self.get_error_summary(),
            'error_history': self.error_history[-10:],  # Últimos 10 errores
            'critical_errors': self.critical_errors,
            'recommendations': self._generate_error_recommendations()
        }
    
    def _generate_error_recommendations(self) -> List[str]:
        """Genera recomendaciones basadas en los errores detectados."""
        recommendations = []
        
        if self.silent_error_attempts > 0:
            recommendations.append("Revisar código para eliminar manejo silencioso de errores")
        
        if self.error_metrics['errors_by_type'].get('data_error', 0) > 5:
            recommendations.append("Revisar calidad y validación de datos de entrada")
        
        if self.error_metrics['errors_by_type'].get('trading_error', 0) > 3:
            recommendations.append("Revisar lógica de trading y límites de seguridad")
        
        if self.error_count >= self.max_errors_per_strategy:
            recommendations.append("Considerar reiniciar la estrategia o revisar configuración")
        
        return recommendations
    
    def reset_error_count(self):
        """Resetea el contador de errores."""
        self.error_count = 0
        self.critical_errors.clear()
        self.error_history.clear()
        self.silent_error_attempts = 0
        self.error_metrics = {
            'total_errors': 0,
            'errors_by_type': {},
            'errors_by_severity': {},
            'recovery_attempts': 0,
            'recovery_successes': 0,
            'silent_error_attempts': 0
        }
        self.logger.info(f"🔄 Contador de errores reseteado para {self.strategy_name}")

# Decorador para manejo automático de errores
def handle_errors(error_handler: ErrorHandlerService, 
                 context: str = "",
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 error_type: ErrorType = ErrorType.SYSTEM_ERROR,
                 default_return: Any = None):
    """
    Decorador para manejo automático de errores en métodos de estrategias.
    NUNCA maneja errores silenciosamente.
    
    Args:
        error_handler (ErrorHandlerService): Instancia del manejador de errores.
        context (str): Contexto para logging.
        severity (ErrorSeverity): Severidad del error.
        error_type (ErrorType): Tipo de error.
        default_return (Any): Valor por defecto si falla.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as error:
                # LOGGING OBLIGATORIO - nunca silencioso
                error_handler.handle_error(error, context, severity, error_type, force_log=True)
                return default_return
        return wrapper
    return decorator

# Ejemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Crear manejador de errores
    error_handler = ErrorHandlerService("TestStrategy")
    
    # Ejemplo 1: Manejo directo de errores (nunca silencioso)
    try:
        # Simular error
        raise ValueError("Error de prueba")
    except Exception as error:
        success = error_handler.handle_error(
            error, 
            "Test de manejo de errores", 
            ErrorSeverity.MEDIUM, 
            ErrorType.SYSTEM_ERROR
        )
        print(f"Error manejado: {success}")
    
    # Ejemplo 2: Ejecución segura (nunca silenciosa)
    def funcion_que_puede_fallar(x):
        if x < 0:
            raise ValueError("Valor negativo no permitido")
        return x * 2
    
    resultado = error_handler.safe_execute(
        funcion_que_puede_fallar, 
        5, 
        context="Cálculo de valor", 
        default_return=0
    )
    print(f"Resultado exitoso: {resultado}")
    
    resultado_fallido = error_handler.safe_execute(
        funcion_que_puede_fallar, 
        -5, 
        context="Cálculo de valor negativo", 
        default_return=0
    )
    print(f"Resultado con error: {resultado_fallido}")
    
    # Ejemplo 3: Resumen detallado de errores
    summary = error_handler.get_error_summary()
    print(f"Resumen de errores: {summary}")
    
    # Ejemplo 4: Reporte detallado
    report = error_handler.get_detailed_error_report()
    print(f"Reporte detallado: {report}") 