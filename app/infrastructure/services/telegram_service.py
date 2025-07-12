"""
Telegram Service - Servicio de Telegram
========================================

Servicio de infraestructura que maneja el envío de mensajes y reportes
al canal de Telegram para notificaciones de trading.
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from telegram import Bot
from telegram.error import TelegramError
from telegram.constants import ParseMode

from app.infrastructure.config.settings import settings
from app.domain.entities.daily_recommendation import RecomendacionDiaria

logger = logging.getLogger(__name__)


class TelegramService:
    """
    Servicio de Telegram para envío de reportes de trading.
    
    Maneja:
    - Envío de mensajes formateados al canal
    - Gestión de errores de conexión
    - Formateo de mensajes con emojis y markdown
    - Soporte para múltiples mensajes por recomendación
    - Sesión persistente del bot para mejor rendimiento
    """
    
    def __init__(self):
        """Inicializa el servicio de Telegram."""
        self.bot = None
        self.chat_id = None
        self._session = None
        self._initialize_bot()
    
    def _initialize_bot(self):
        """Inicializa el bot de Telegram con sesión persistente."""
        try:
            bot_token = getattr(settings, 'TELEGRAM_BOT_TOKEN', None)
            chat_id = getattr(settings, 'TELEGRAM_CHAT_ID', None)
            
            if not bot_token:
                logger.warning("⚠️ TELEGRAM_BOT_TOKEN no configurado")
                return
            
            if not chat_id:
                logger.warning("⚠️ TELEGRAM_CHAT_ID no configurado")
                return
            
            # Crear bot con sesión persistente
            self.bot = Bot(token=bot_token)
            self.chat_id = chat_id
            logger.info("✅ Bot de Telegram inicializado correctamente con sesión persistente")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando bot de Telegram: {e}")
            self.bot = None
    
    async def send_message(self, message: str, parse_mode: ParseMode = ParseMode.MARKDOWN) -> bool:
        """
        Envía un mensaje al canal de Telegram.
        
        Args:
            message: Mensaje a enviar
            parse_mode: Modo de parsing (Markdown, HTML, etc.)
            
        Returns:
            True si el mensaje se envió exitosamente, False en caso contrario
        """
        try:
            if not self.bot or not self.chat_id:
                logger.warning("⚠️ Bot de Telegram no configurado")
                return False
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
            
            logger.info("✅ Mensaje enviado a Telegram exitosamente")
            return True
            
        except TelegramError as e:
            logger.error(f"❌ Error de Telegram enviando mensaje: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error enviando mensaje: {e}")
            return False
    
    def send_message_sync(self, message: str, parse_mode: ParseMode = ParseMode.MARKDOWN) -> bool:
        """
        Versión síncrona del envío de mensajes con bot persistente.
        
        Args:
            message: Mensaje a enviar
            parse_mode: Modo de parsing
            
        Returns:
            True si el mensaje se envió exitosamente, False en caso contrario
        """
        try:
            # Verificar si hay un loop corriendo
            try:
                loop = asyncio.get_running_loop()
                # Hay un loop corriendo, usar run_coroutine_threadsafe
                future = asyncio.run_coroutine_threadsafe(
                    self.send_message(message, parse_mode), 
                    loop
                )
                return future.result(timeout=30)  # Timeout de 30 segundos
            except RuntimeError:
                # No hay loop corriendo, crear uno nuevo
                return asyncio.run(self.send_message(message, parse_mode))
                
        except Exception as e:
            logger.error(f"❌ Error en send_message_sync: {e}")
            return False
    
    def format_recommendation_message(self, recommendation: RecomendacionDiaria) -> str:
        """
        Formatea una recomendación en un mensaje de Telegram.
        
        Args:
            recommendation: Recomendación a formatear
            
        Returns:
            Mensaje formateado con markdown
        """
        try:
            telegram_data = recommendation.get_telegram_data()
            
            # Emojis para diferentes elementos
            strategy_emoji = {
                'grid': '📊',
                'dca': '📈',
                'btd': '💰'
            }
            
            risk_emoji = {
                'BAJO': '🟢',
                'MEDIO': '🟡',
                'ALTO': '🔴'
            }
            
            category_emoji = {
                'PREMIUM': '⭐',
                'AGGRESSIVE': '🚀',
                'CONSERVATIVE': '🛡️',
                'BALANCED': '⚖️'
            }
            
            recomendacion_emoji = {
                'FUERTE_COMPRA': '💪',
                'COMPRA': '✅',
                'NEUTRAL_POSITIVO': '👍',
                'NEUTRAL': '😐'
            }
            
            # Construir mensaje
            message = f"""
🤖 *RECOMENDACIÓN DIARIA DE TRADING*

💎 *{telegram_data['simbolo']}*
{strategy_emoji.get(recommendation.estrategia_recomendada, '📊')} *Estrategia:* {telegram_data['estrategia_final']}
{telegram_data['consenso']}

{recomendacion_emoji.get(telegram_data['recomendacion'], '👍')} *Recomendación:* {telegram_data['recomendacion']}
{category_emoji.get(telegram_data['categoria'], '⚖️')} *Categoría:* {telegram_data['categoria']}

📈 *MÉTRICAS DE RENDIMIENTO*
• ROI: {telegram_data['roi']}
• Win Rate: {telegram_data['win_rate']}
• Max Drawdown: {telegram_data['drawdown']}
• Riesgo: {risk_emoji.get(telegram_data['nivel_riesgo'], '🟡')} {telegram_data['nivel_riesgo']}

🧠 *ANÁLISIS GEMINI AI*
• {telegram_data['razon_gemini']}

💪 *FORTALEZAS*
• {telegram_data['fortalezas']}

⚠️ *RIESGOS*
• {telegram_data['riesgos']}

⚙️ *PARÁMETROS OPTIMIZADOS*
{self._format_parameters(telegram_data['parametros'])}

───────────────────────
📅 {recommendation.fecha.strftime('%Y-%m-%d %H:%M')}
"""
            
            return message.strip()
            
        except Exception as e:
            logger.error(f"❌ Error formateando mensaje: {e}")
            return f"❌ Error formateando recomendación para {recommendation.simbolo}"
    
    def _format_parameters(self, params: Dict[str, Any]) -> str:
        """
        Formatea los parámetros optimizados para mostrar en Telegram.
        
        Args:
            params: Diccionario de parámetros
            
        Returns:
            String formateado con los parámetros
        """
        try:
            if not params:
                return "• No hay parámetros disponibles"
            
            formatted_params = []
            for key, value in params.items():
                if isinstance(value, float):
                    formatted_params.append(f"• {key}: {value:.4f}")
                else:
                    formatted_params.append(f"• {key}: {value}")
            
            return "\n".join(formatted_params)
            
        except Exception as e:
            logger.error(f"❌ Error formateando parámetros: {e}")
            return "• Error formateando parámetros"
    
    def format_daily_summary(self, recommendations: List[RecomendacionDiaria]) -> str:
        """
        Formatea un resumen diario de todas las recomendaciones.
        
        Args:
            recommendations: Lista de recomendaciones del día
            
        Returns:
            Mensaje resumen formateado
        """
        try:
            if not recommendations:
                return "📊 *RESUMEN DIARIO*\n\nNo hay recomendaciones para hoy."
            
            fecha = recommendations[0].fecha.strftime('%Y-%m-%d')
            
            # Estadísticas generales
            total = len(recommendations)
            consenso_count = sum(1 for r in recommendations if r.consenso_estrategia)
            consenso_rate = (consenso_count / total) * 100 if total > 0 else 0
            
            # Distribución por estrategia
            estrategias = {}
            for r in recommendations:
                estrategia = r.estrategia_gemini
                estrategias[estrategia] = estrategias.get(estrategia, 0) + 1
            
            # Distribución por riesgo
            riesgos = {}
            for r in recommendations:
                riesgo = r.nivel_riesgo
                riesgos[riesgo] = riesgos.get(riesgo, 0) + 1
            
            # Mejores oportunidades
            top_opportunities = sorted(recommendations, key=lambda x: x.score_final, reverse=True)[:3]
            
            message = f"""
📊 *RESUMEN DIARIO DE RECOMENDACIONES*
📅 {fecha}

📈 *ESTADÍSTICAS GENERALES*
• Total recomendaciones: {total}
• Consenso Quant-AI: {consenso_count}/{total} ({consenso_rate:.1f}%)
• ROI promedio: {sum(r.roi_porcentaje for r in recommendations)/total:.1f}%

🎯 *DISTRIBUCIÓN POR ESTRATEGIA*
{self._format_distribution(estrategias, {'grid': '📊', 'dca': '📈', 'btd': '💰'})}

⚡ *DISTRIBUCIÓN POR RIESGO*
{self._format_distribution(riesgos, {'BAJO': '🟢', 'MEDIO': '🟡', 'ALTO': '🔴'})}

🏆 *TOP 3 OPORTUNIDADES*
{self._format_top_opportunities(top_opportunities)}

🤖 Reporte generado por Trading AI
"""
            
            return message.strip()
            
        except Exception as e:
            logger.error(f"❌ Error formateando resumen diario: {e}")
            return "❌ Error generando resumen diario"
    
    def _format_distribution(self, distribution: Dict[str, int], emojis: Dict[str, str]) -> str:
        """Formatea una distribución con emojis."""
        lines = []
        for item, count in distribution.items():
            emoji = emojis.get(item, '•')
            lines.append(f"{emoji} {item}: {count}")
        return "\n".join(lines) if lines else "• No hay datos"
    
    def _format_top_opportunities(self, opportunities: List[RecomendacionDiaria]) -> str:
        """Formatea las top oportunidades."""
        lines = []
        for i, opp in enumerate(opportunities, 1):
            consensus = "✅" if opp.consenso_estrategia else "⚠️"
            lines.append(f"{i}. {opp.simbolo} - {opp.roi_porcentaje:.1f}% {consensus}")
        return "\n".join(lines) if lines else "• No hay oportunidades"
    
    def send_daily_report(self, recommendations: List[RecomendacionDiaria]) -> Dict[str, Any]:
        """
        Envía el reporte diario completo a Telegram con bot persistente.
        
        Args:
            recommendations: Lista de recomendaciones del día
            
        Returns:
            Diccionario con estadísticas del envío
        """
        try:
            if not recommendations:
                logger.warning("⚠️ No hay recomendaciones para enviar")
                return {
                    'total_messages': 0,
                    'sent_successfully': 0,
                    'errors': 0,
                    'success_rate': 0
                }
            
            if not self.bot or not self.chat_id:
                logger.error("❌ Bot de Telegram no está configurado")
                return {
                    'total_messages': 0,
                    'sent_successfully': 0,
                    'errors': 1,
                    'success_rate': 0
                }
            
            logger.info(f"🚀 Iniciando envío de reporte diario con {len(recommendations)} recomendaciones")
            
            sent_successfully = 0
            errors = 0
            
            # 1. Enviar resumen diario con reintentos
            logger.info("📊 Enviando resumen diario...")
            summary = self.format_daily_summary(recommendations)
            if self._send_with_retry(summary, max_retries=5):
                sent_successfully += 1
                logger.info("✅ Resumen diario enviado exitosamente")
            else:
                errors += 1
                logger.error("❌ Error enviando resumen diario después de 5 intentos")
            
            # Pausa después del resumen
            time.sleep(1)
            
            # 2. Enviar cada recomendación individual con reintentos
            for i, recommendation in enumerate(recommendations, 1):
                logger.info(f"📱 Enviando recomendación {i}/{len(recommendations)}: {recommendation.simbolo}")
                
                try:
                    message = self.format_recommendation_message(recommendation)
                    
                    if self._send_with_retry(message, max_retries=5):
                        sent_successfully += 1
                        logger.info(f"✅ Recomendación {i} enviada exitosamente")
                    else:
                        errors += 1
                        logger.error(f"❌ Error enviando recomendación {i} después de 5 intentos")
                    
                    # Pausa entre mensajes para evitar rate limits
                    time.sleep(0.8)
                    
                except Exception as e:
                    errors += 1
                    logger.error(f"❌ Error procesando recomendación {i}: {e}")
                    continue
            
            total_messages = len(recommendations) + 1  # +1 por el resumen
            success_rate = (sent_successfully / total_messages) * 100 if total_messages > 0 else 0
            
            logger.info(f"📊 Reporte Telegram completado: {sent_successfully}/{total_messages} mensajes ({success_rate:.1f}%)")
            
            if success_rate < 100:
                logger.warning(f"⚠️ Telegram: Solo {success_rate:.1f}% de mensajes enviados")
            else:
                logger.info("🎉 ¡Todos los mensajes enviados exitosamente!")
            
            return {
                'total_messages': total_messages,
                'sent_successfully': sent_successfully,
                'errors': errors,
                'success_rate': success_rate
            }
            
        except Exception as e:
            logger.error(f"❌ Error crítico enviando reporte diario: {e}")
            return {
                'total_messages': 0,
                'sent_successfully': 0,
                'errors': 1,
                'success_rate': 0
            }
    
    def _send_with_retry(self, message: str, max_retries: int = 5) -> bool:
        """
        Envía un mensaje con reintentos automáticos mejorados.
        
        Args:
            message: Mensaje a enviar
            max_retries: Número máximo de reintentos
            
        Returns:
            True si el mensaje se envió exitosamente, False en caso contrario
        """
        for attempt in range(max_retries):
            try:
                if self.send_message_sync(message):
                    return True
                else:
                    logger.warning(f"⚠️ Intento {attempt + 1}/{max_retries} falló")
                    if attempt < max_retries - 1:
                        # Pausa progresiva: 1s, 2s, 3s, 4s
                        sleep_time = attempt + 1
                        logger.info(f"⏳ Esperando {sleep_time}s antes del siguiente intento...")
                        time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"❌ Error en intento {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    # Pausa más larga si hay error: 2s, 4s, 6s, 8s
                    sleep_time = (attempt + 1) * 2
                    logger.info(f"⏳ Esperando {sleep_time}s antes del siguiente intento...")
                    time.sleep(sleep_time)
        
        return False
    
    def close(self):
        """
        Cierra limpiamente la conexión del bot de Telegram.
        """
        try:
            if self.bot:
                logger.info("🔌 Cerrando conexión del bot de Telegram...")
                # El bot de python-telegram-bot se cierra automáticamente
                self.bot = None
                self.chat_id = None
                logger.info("✅ Conexión del bot de Telegram cerrada")
        except Exception as e:
            logger.error(f"❌ Error cerrando bot de Telegram: {e}")
    
    def __del__(self):
        """Destructor para cerrar limpiamente el bot."""
        self.close()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Verifica el estado del servicio de Telegram.
        
        Returns:
            Diccionario con estado de salud
        """
        try:
            if not self.bot or not self.chat_id:
                return {
                    'status': 'error',
                    'message': 'Bot de Telegram no configurado',
                    'configured': False
                }
            
            # Intentar enviar mensaje de prueba
            test_message = "🤖 Test de conectividad - " + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            success = self.send_message_sync(test_message)
            
            if success:
                return {
                    'status': 'ok',
                    'message': 'Telegram operativo',
                    'configured': True,
                    'chat_id': self.chat_id,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Error enviando mensaje de prueba',
                    'configured': True,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error en health check: {str(e)}',
                'configured': bool(self.bot),
                'timestamp': datetime.now().isoformat()
            } 