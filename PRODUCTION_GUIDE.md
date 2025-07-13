# Guía de Producción - Trading AI

## Archivos de Entrada para Producción

### 1. `start_production.py` - Entrada Principal ⭐
**Recomendado para producción en servidores**

```bash
# Estado del sistema
python start_production.py --status

# Scanner completo (Top 150 → Top 10 candidatos)
python start_production.py --scanner-only

# Análisis completo (Scanner + Optimización + Backtesting)
python start_production.py

# Análisis con símbolos específicos
python start_production.py --symbols BTC/USDT ETH/USDT SOL/USDT

# Análisis con más trials (mayor precisión)
python start_production.py --trials 100

# Análisis completo con configuración personalizada
python start_production.py --symbols PEPE/USDT ENA/USDT --trials 50
```

**Características:**
- ✅ Usa el `app/main.py` directamente (sistema completo)
- ✅ Scanner real de 150 monedas → Top 10 candidatos
- ✅ Optimización bayesiana con backtesting real
- ✅ Datos históricos de 9 meses
- ✅ Métricas reales de ROI, Sharpe, Drawdown
- ✅ Sin problemas de encoding
- ✅ Ideal para cron jobs y servidores

### 2. `app/main.py` - Sistema Completo Directo
**Para uso avanzado y desarrollo**

```bash
# Estado del sistema
python app/main.py --status

# Solo scanner
python app/main.py --scanner-only

# Solo optimización con símbolos específicos
python app/main.py --optimize-only --symbols BTC/USDT ETH/USDT

# Análisis completo
python app/main.py

# Análisis completo con configuración personalizada
python app/main.py --symbols BTC/USDT ETH/USDT --trials 200
```

### 3. `run_weekly_recommendations.py` - Para Desarrollo
**Para testing y debugging del sistema de recomendaciones semanales**

```bash
# Ejecutar sistema de recomendaciones semanales
python run_weekly_recommendations.py

# Ver estado detallado
python run_weekly_recommendations.py --status

# Con logs detallados
python run_weekly_recommendations.py --verbose
```

## Resultados del Sistema

### Scanner Real
- **Input**: Top 150 criptomonedas por volumen
- **Output**: Top 10 candidatos con scores reales
- **Métricas**: Volatilidad, ADX, Sentimiento, Volumen
- **Ejemplo**: PEPE/USDT (100.0/100), ENA/USDT (100.0/100)

### Optimización Bayesiana
- **Estrategias**: GRID, DCA, BTD
- **Trials**: 10-200 por símbolo (configurable)
- **Backtesting**: 9 meses de datos históricos
- **Métricas**: ROI, Sharpe, Drawdown, Win Rate

### Resultados Reales Obtenidos
```
🏆 TOP 5 MEJORES RESULTADOS:
1. ENA/USDT (GRID) - Valor: 40.20, ROI: 18.87%
2. ENA/USDT (DCA) - Valor: 34.72, ROI: 17.60%
3. PEPE/USDT (BTD) - Valor: 18.79, ROI: 6.28%
4. PEPE/USDT (DCA) - Valor: 11.50, ROI: 5.78%
5. PEPE/USDT (GRID) - Valor: 9.45, ROI: 8.37%
```

## Configuración para Producción

### Variables de Entorno Requeridas

```env
# Base de datos
DATABASE_URL=postgresql://user:password@localhost:5432/trading_ai

# APIs
BINANCE_API_KEY=tu_api_key_binance
BINANCE_API_SECRET=tu_api_secret_binance
GEMINI_API_KEY=tu_api_key_gemini

# Telegram
TELEGRAM_BOT_TOKEN=tu_bot_token
TELEGRAM_CHAT_ID=tu_chat_id

# Configuración opcional
DEFAULT_COMMISSION=0.001
DEFAULT_INITIAL_CAPITAL=1000.0
LOG_LEVEL=INFO
```

### Automatización con Cron

```bash
# Editar crontab
crontab -e

# Scanner diario a las 9:00 AM
0 9 * * * cd /path/to/trading_ai && python start_production.py --scanner-only >> /var/log/trading_scanner.log 2>&1

# Análisis completo semanal los lunes a las 10:00 AM
0 10 * * 1 cd /path/to/trading_ai && python start_production.py --trials 100 >> /var/log/trading_analysis.log 2>&1

# Análisis rápido de símbolos específicos diario
0 15 * * * cd /path/to/trading_ai && python start_production.py --symbols BTC/USDT ETH/USDT --trials 20 >> /var/log/trading_quick.log 2>&1
```

## Arquitectura del Sistema

### Pipeline Completo en Producción
1. **Scanner** (150 monedas) → Top 10 candidatos con scores reales
2. **Optimizer** → Optimización bayesiana con backtesting de 9 meses
3. **Backtesting** → Validación con datos históricos reales
4. **Ranking** → Selección de mejores estrategias por ROI
5. **Recomendaciones** → Output listo para trading

### Tecnologías Utilizadas
- **Binance API** → Datos de mercado en tiempo real
- **Google BigQuery** → Datos históricos de sentimiento
- **PostgreSQL** → Almacenamiento de resultados
- **Optuna** → Optimización bayesiana
- **Backtesting.py** → Backtesting moderno
- **Gemini AI** → Análisis cualitativo

## Monitoreo y Logs

### Verificación del Sistema
```bash
# Estado completo del sistema
python start_production.py --status

# Verificar logs
tail -f /var/log/trading_analysis.log

# Test rápido
python start_production.py --symbols BTC/USDT --trials 5
```

### Métricas de Rendimiento
- **Scanner**: ~2 minutos para 150 monedas
- **Optimización**: ~1 minuto por símbolo con 50 trials
- **Backtesting**: ~0.5 segundos por estrategia
- **Total**: ~5-10 minutos para análisis completo

## Despliegue en Droplet

### Preparación del Servidor
```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependencias
sudo apt install python3 python3-pip python3-venv postgresql-client git -y

# Clonar repositorio
git clone <tu-repo> /opt/trading_ai
cd /opt/trading_ai

# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Configuración de Producción
```bash
# Configurar variables de entorno
cp .env.example .env
nano .env

# Probar sistema
python start_production.py --status
python start_production.py --scanner-only
```

### Script de Automatización
```bash
# Crear wrapper script
cat > /opt/trading_ai/run_production.sh << 'EOF'
#!/bin/bash
cd /opt/trading_ai
source .venv/bin/activate
python start_production.py "$@"
EOF

chmod +x /opt/trading_ai/run_production.sh

# Usar en cron
0 9 * * * /opt/trading_ai/run_production.sh --scanner-only
0 10 * * 1 /opt/trading_ai/run_production.sh --trials 100
```

## Troubleshooting

### Problemas Comunes

1. **Error de módulos Python**
   - Verificar que estás en el entorno virtual: `source .venv/bin/activate`
   - Reinstalar dependencias: `pip install -r requirements.txt`

2. **Error de conexión a Binance**
   - Verificar API keys en `.env`
   - Comprobar límites de rate limiting

3. **Error de base de datos**
   - Verificar `DATABASE_URL` en `.env`
   - Comprobar conexión: `psql $DATABASE_URL`

4. **Rendimiento lento**
   - Reducir `--trials` para análisis más rápido
   - Usar `--symbols` para análisis específico

### Comandos de Debugging
```bash
# Test básico
python start_production.py --status

# Test con un símbolo
python start_production.py --symbols BTC/USDT --trials 5

# Logs detallados
python app/main.py --symbols BTC/USDT --trials 10
```

## Notas Importantes

1. **Datos Reales**: Todo el sistema usa datos reales de Binance, sin mocks
2. **Backtesting**: Validación con 9 meses de datos históricos
3. **Optimización**: Bayesian optimization con Optuna para máxima precisión
4. **Escalabilidad**: Sistema optimizado para manejar 150+ monedas
5. **Producción**: Listo para despliegue en servidores con cron jobs

## Próximos Pasos

1. **Integrar Recomendaciones Semanales**: Conectar con el sistema de cartera semanal
2. **Dashboard Web**: Interfaz para monitoreo en tiempo real
3. **Alertas Avanzadas**: Notificaciones por email/SMS
4. **Análisis de Futuros**: Integración completa con trading de futuros
5. **Machine Learning**: Predicción de tendencias con modelos avanzados 