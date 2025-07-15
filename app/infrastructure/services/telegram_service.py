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
    
    async def send_message(self, message: str, parse_mode: ParseMode = ParseMode.MARKDOWN_V2) -> bool:
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
    
    def send_message_sync(self, message: str, parse_mode: ParseMode = ParseMode.MARKDOWN_V2) -> bool:
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
            
            # Determinar direccionalidad
            direction_info = self._get_direction_info(recommendation)
            
            # Determinar si es futuros
            is_futures = getattr(recommendation, 'es_futuros', False)
            
            # Construir mensaje
            message = f"""
🤖 *RECOMENDACIÓN SEMANAL DE TRADING*

💎 *{telegram_data['simbolo']}*
{strategy_emoji.get(recommendation.estrategia_recomendada, '📊')} *Estrategia:* {telegram_data['estrategia_final']} {direction_info['emoji']} {direction_info['text']}
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

💰 *RECOMENDACIONES DE CAPITAL*
{self._format_capital_recommendations(is_futures)}

{self._format_futures_info(recommendation) if is_futures else ''}

───────────────────────
📅 {recommendation.fecha.strftime('%Y-%m-%d %H:%M')}
"""
            
            return self._escape_markdown_v2(message.strip())
            
        except Exception as e:
            logger.error(f"❌ Error formateando mensaje: {e}")
            return f"❌ Error formateando recomendación para {recommendation.simbolo}"
    
    def _get_direction_info(self, recommendation: RecomendacionDiaria) -> Dict[str, str]:
        """
        Obtiene información de direccionalidad de la estrategia.
        
        Args:
            recommendation: Recomendación
            
        Returns:
            Diccionario con emoji y texto de dirección
        """
        # Obtener dirección del análisis cualitativo si está disponible
        direction = getattr(recommendation, 'direccion', 'long')
        
        if direction.lower() == 'short':
            return {
                'emoji': '📉',
                'text': '(SHORT)'
            }
        else:
            return {
                'emoji': '📈',
                'text': '(LONG)'
            }
    
    def _format_capital_recommendations(self, is_futures: bool) -> str:
        """
        Formatea las recomendaciones de capital.
        
        Args:
            is_futures: Si es para futuros o spot
            
        Returns:
            Texto formateado con recomendaciones de capital
        """
        if is_futures:
            return """• 💸 *Bajo Capital:* $50-100 (x3-x5)
• 💰 *Medio Capital:* $200-500 (x3-x5)  
• 🏦 *Alto Capital:* $1000+ (x3-x5)
• ⚡ *Recomendado:* Capital medio con x3-x5"""
        else:
            return """• 💸 *Bajo Capital:* $50-200
• 💰 *Medio Capital:* $500-1500
• 🏦 *Alto Capital:* $2000+
• ⚡ *Recomendado:* Capital medio $500-1000"""
    
    def _format_futures_info(self, recommendation: RecomendacionDiaria) -> str:
        """
        Formatea información específica de futuros.
        
        Args:
            recommendation: Recomendación
            
        Returns:
            Texto formateado con información de futuros
        """
        # Obtener información de futuros del análisis cualitativo
        optimal_leverage = getattr(recommendation, 'apalancamiento_optimo', 'x3')
        futures_risk = getattr(recommendation, 'riesgo_futuros', 'medium')
        
        risk_emoji = {
            'low': '🟢',
            'medium': '🟡',
            'high': '🔴',
            'extreme': '⚫'
        }
        
        return f"""
🚀 *INFORMACIÓN DE FUTUROS*
• 📊 *Apalancamiento Óptimo:* {optimal_leverage}
• ⚠️ *Riesgo Futuros:* {risk_emoji.get(futures_risk, '🟡')} {futures_risk.upper()}
• 🎯 *Recomendación:* Usar stop loss estricto
• ⏰ *Timeframe:* Monitoreo activo requerido"""

    def _escape_markdown_v2(self, text: str) -> str:
        """Escapa caracteres especiales para Markdown V2."""
        escape_chars = '_*[]()~`>#+-=|{}.!'
        escaped_text = ''
        for char in text:
            if char in escape_chars:
                escaped_text += '\\' + char
            else:
                escaped_text += char
        return escaped_text
    
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
{self._format_distribution(estrategias, {'grid': '📊', 'dca': '📈', 'btd': '💰', 'FuturesGrid': '⚡'})}

⚡ *DISTRIBUCIÓN POR RIESGO*
{self._format_distribution(riesgos, {'BAJO': '🟢', 'MEDIO': '🟡', 'ALTO': '🔴'})}

🏆 *TOP 3 OPORTUNIDADES*
{self._format_top_opportunities(top_opportunities)}

🤖 Reporte generado por Trading AI
"""
            
            return self._escape_markdown_v2(message.strip())
            
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
    
    def send_weekly_report(self, recommendations: List[RecomendacionDiaria], weekly_selection=None) -> Dict[str, Any]:
        """
        Envía el reporte semanal completo a Telegram con formato especializado.
        
        Args:
            recommendations: Lista de recomendaciones semanales
            weekly_selection: Selección semanal (opcional)
            
        Returns:
            Diccionario con estadísticas del envío
        """
        try:
            if not recommendations:
                logger.warning("⚠️ No hay recomendaciones semanales para enviar")
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
            
            logger.info(f"🚀 Iniciando envío de reporte semanal con {len(recommendations)} recomendaciones")
            
            sent_successfully = 0
            errors = 0
            
            # 1. Enviar resumen semanal
            logger.info("📊 Enviando resumen semanal...")
            weekly_summary = self.format_weekly_summary(recommendations, weekly_selection)
            if self._send_with_retry(weekly_summary, max_retries=5):
                sent_successfully += 1
                logger.info("✅ Resumen semanal enviado exitosamente")
            else:
                errors += 1
                logger.error("❌ Error enviando resumen semanal después de 5 intentos")
            
            # Pausa después del resumen
            time.sleep(1)
            
            # 2. Enviar cada recomendación con categoría
            for i, recommendation in enumerate(recommendations, 1):
                logger.info(f"📱 Enviando recomendación semanal {i}/{len(recommendations)}: {recommendation.simbolo}")
                
                try:
                    message = self.format_weekly_recommendation_message(recommendation)
                    
                    if self._send_with_retry(message, max_retries=5):
                        sent_successfully += 1
                        logger.info(f"✅ Recomendación semanal {i} enviada exitosamente")
                    else:
                        errors += 1
                        logger.error(f"❌ Error enviando recomendación semanal {i} después de 5 intentos")
                    
                    # Pausa entre mensajes
                    time.sleep(0.8)
                    
                except Exception as e:
                    errors += 1
                    logger.error(f"❌ Error procesando recomendación semanal {i}: {e}")
                    continue
            
            total_messages = len(recommendations) + 1  # +1 por el resumen
            success_rate = (sent_successfully / total_messages) * 100 if total_messages > 0 else 0
            
            logger.info(f"📊 Reporte semanal Telegram completado: {sent_successfully}/{total_messages} mensajes ({success_rate:.1f}%)")
            
            return {
                'total_messages': total_messages,
                'sent_successfully': sent_successfully,
                'errors': errors,
                'success_rate': success_rate
            }
            
        except Exception as e:
            logger.error(f"❌ Error crítico enviando reporte semanal: {e}")
            return {
                'total_messages': 0,
                'sent_successfully': 0,
                'errors': 1,
                'success_rate': 0
            }
    
    def format_weekly_summary(self, recommendations: List[RecomendacionDiaria], weekly_selection=None) -> str:
        """
        Formatea el resumen semanal de la cartera.
        
        Args:
            recommendations: Lista de recomendaciones semanales
            weekly_selection: Selección semanal
            
        Returns:
            Mensaje resumen formateado
        """
        try:
            if not recommendations:
                return "📊 *CARTERA SEMANAL*\n\nNo hay recomendaciones para esta semana."
            
            fecha = recommendations[0].fecha.strftime('%Y-%m-%d')
            
            # Categorizar recomendaciones
            spot_recs = [r for r in recommendations if not r.categoria.endswith('_FUTURES')]
            futures_recs = [r for r in recommendations if r.categoria.endswith('_FUTURES')]
            
            # Estadísticas generales
            total = len(recommendations)
            avg_roi = sum(r.roi_porcentaje for r in recommendations) / total if total > 0 else 0
            avg_confidence = sum(r.score_confianza_gemini for r in recommendations) / total if total > 0 else 0
            
            message = f"""
📊 *CARTERA SEMANAL DE TRADING*
📅 Semana del {fecha}

🎯 *COMPOSICIÓN DE CARTERA*
• Total seleccionadas: {total}/5
• Spot: {len(spot_recs)} recomendaciones
• Futuros: {len(futures_recs)} recomendaciones

📈 *MÉTRICAS GENERALES*
• ROI promedio: {avg_roi:.1f}%
• Confianza promedio: {avg_confidence:.1f}%
• Período válido: 7 días

🔸 *SPOT TRADING*
{self._format_spot_summary(spot_recs)}

🔸 *FUTUROS TRADING*
{self._format_futures_summary(futures_recs)}

🔸 *RESUMEN DETALLADO FUTUROS*
{self._format_detailed_futures_summary(futures_recs)}

⚠️ *RECORDATORIO*
Esta cartera está diseñada para ejecutarse durante toda la semana. Cada estrategia ha sido validada por análisis cuantitativo y cualitativo (Gemini AI).

🤖 Cartera generada por Trading AI v2.0
"""
            
            return self._escape_markdown_v2(message.strip())
            
        except Exception as e:
            logger.error(f"❌ Error formateando resumen semanal: {e}")
            return "❌ Error generando resumen semanal"
    
    def format_weekly_recommendation_message(self, recommendation: RecomendacionDiaria) -> str:
        """
        Formatea una recomendación semanal con categoría específica.
        
        Args:
            recommendation: Recomendación semanal
            
        Returns:
            Mensaje formateado
        """
        try:
            # Emojis por categoría
            category_emoji = {
                'GRID_SPOT': '📊',
                'DCA_SPOT': '📈',
                'BTD_SPOT': '💰',
                'GRID_FUTURES': '🎯',
                'DCA_FUTURES': '🚀'
            }
            
            category_name = {
                'GRID_SPOT': 'GRID TRADING (SPOT)',
                'DCA_SPOT': 'DCA TRADING (SPOT)',
                'BTD_SPOT': 'BUY THE DIP (SPOT)',
                'GRID_FUTURES': 'GRID TRADING (FUTUROS)',
                'DCA_FUTURES': 'DCA TRADING (FUTUROS)'
            }
            
            risk_emoji = {
                'BAJO': '🟢',
                'MEDIO': '🟡',
                'ALTO': '🔴'
            }
            
            emoji = category_emoji.get(recommendation.categoria, '📊')
            cat_name = category_name.get(recommendation.categoria, recommendation.categoria)
            
            message = f"""
🤖 *RECOMENDACIÓN SEMANAL*

{emoji} *{recommendation.simbolo}*
📋 *Categoría:* {cat_name}
⏰ *Válida por:* 7 días

📊 *MÉTRICAS DE RENDIMIENTO*
• ROI: {recommendation.roi_porcentaje:.1f}%
• Sharpe Ratio: {recommendation.sharpe_ratio:.2f}
• Win Rate: {recommendation.win_rate_porcentaje:.1f}%
• Max Drawdown: {recommendation.max_drawdown_porcentaje:.1f}%
• Riesgo: {risk_emoji.get(recommendation.nivel_riesgo, '🟡')} {recommendation.nivel_riesgo}

🧠 *ANÁLISIS GEMINI AI*
• {recommendation.razon_gemini}

💪 *FORTALEZAS*
• {recommendation.fortalezas_gemini}

⚠️ *RIESGOS*
• {recommendation.riesgos_gemini}

⚙️ *PARÁMETROS OPTIMIZADOS*
{self._format_parameters(recommendation.parametros_optimizados)}

───────────────────────
📅 {recommendation.fecha.strftime('%Y-%m-%d %H:%M')}
🔄 Ejecutar durante toda la semana
"""
            
            return self._escape_markdown_v2(message.strip())
            
        except Exception as e:
            logger.error(f"❌ Error formateando mensaje semanal: {e}")
            return f"❌ Error formateando recomendación semanal para {recommendation.simbolo}"
    
    def _format_spot_summary(self, spot_recs: List[RecomendacionDiaria]) -> str:
        """Formatea resumen de recomendaciones spot."""
        if not spot_recs:
            return "• No hay recomendaciones spot"
        
        lines = []
        for rec in spot_recs:
            strategy_type = rec.categoria.replace('_SPOT', '')
            lines.append(f"• {rec.simbolo} ({strategy_type}) - ROI: {rec.roi_porcentaje:.1f}%")
        
        return "\n".join(lines)
    
    def _format_futures_summary(self, futures_recs: List[RecomendacionDiaria]) -> str:
        """Formatea resumen de recomendaciones futuros."""
        if not futures_recs:
            return "• No hay recomendaciones futuros"
        
        lines = []
        for rec in futures_recs:
            strategy_type = rec.categoria.replace('_FUTURES', '')
            lines.append(f"• {rec.simbolo} ({strategy_type}) - ROI: {rec.roi_porcentaje:.1f}%")
        
        return "\n".join(lines)
    
    def _format_detailed_futures_summary(self, futures_recs: List[RecomendacionDiaria]) -> str:
        """Formatea resumen detallado de recomendaciones futuros con fortalezas y riesgos."""
        if not futures_recs:
            return "• No hay recomendaciones futuros para análisis detallado"
        
        detailed_lines = []
        for rec in futures_recs:
            strategy_type = rec.categoria.replace('_FUTURES', '')
            
            # Obtener información específica de futuros
            optimal_leverage = getattr(rec, 'apalancamiento_optimo', 'x3')
            futures_risk = getattr(rec, 'riesgo_futuros', 'medium')
            direction = getattr(rec, 'direccion', 'long')
            
            risk_emoji = {
                'low': '🟢',
                'medium': '🟡',
                'high': '🔴',
                'extreme': '⚫'
            }
            
            direction_emoji = '📈' if direction.lower() == 'long' else '📉'
            
            detailed_lines.append(f"""
🚀 *{rec.simbolo} - {strategy_type.upper()} FUTUROS*
• ROI: {rec.roi_porcentaje:.1f}% | Sharpe: {rec.sharpe_ratio:.2f}
• Apalancamiento: {optimal_leverage} | Riesgo: {risk_emoji.get(futures_risk, '🟡')} {futures_risk.upper()}
• Dirección: {direction_emoji} {direction.upper()}
• Fortalezas: {rec.fortalezas_gemini}
• Riesgos: {rec.riesgos_gemini}
""")
        
        return "\n".join(detailed_lines)
    
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