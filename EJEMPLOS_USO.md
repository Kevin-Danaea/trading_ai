# 🎯 Ejemplos de Uso - Sistema Inteligente de Trading

Esta guía contiene ejemplos prácticos paso a paso del nuevo sistema inteligente de backtesting.

## 🚀 Inicio Rápido (5 minutos)

### 1. Análisis Express (Recomendado para empezar)

```bash
# Obtener las mejores 5 monedas del mercado
python scripts/smart_backtesting.py --scanner-only --max-candidates 5

# Output esperado:
# 🏆 MEJORES 5 CANDIDATOS:
# 1. BTC/USDT - Score: 87.3/100
# 2. ETH/USDT - Score: 82.1/100  
# 3. SOL/USDT - Score: 79.5/100
# 4. AVAX/USDT - Score: 76.8/100
# 5. LINK/USDT - Score: 74.2/100
#
# 🎯 RECOMENDACIONES PARA BACKTESTING:
# python scripts/find_optimal_parameters.py --monedas BTC/USDT ETH/USDT SOL/USDT AVAX/USDT LINK/USDT
```

### 2. Ejecutar Backtesting con Recomendaciones

```bash
# Copiar y pegar el comando de arriba
python scripts/find_optimal_parameters.py --monedas BTC/USDT ETH/USDT SOL/USDT AVAX/USDT LINK/USDT

# ⏱️ Tiempo estimado: ~1 hora (vs 22 horas del método tradicional)
```

---

## 📊 Casos de Uso Específicos

### 👤 Trader Principiante: "Solo quiero las mejores monedas"

```bash
# Configuración simple y segura
python scripts/smart_backtesting.py \
  --scanner-only \
  --top-n 30 \
  --min-volume 20000000 \
  --max-candidates 3

# Por qué esta configuración:
# - Top 30: Monedas establecidas (menos riesgo)
# - $20M volumen: Excelente liquidez
# - 3 candidatos: Análisis rápido y enfocado
```

**Output esperado:**
```
🔍 CryptoScanner inicializado
📊 Analizará Top 30 monedas
💰 Volumen mínimo: $20,000,000 USDT

🏆 MEJORES 3 CANDIDATOS:
1. BTC/USDT - Score: 89.2/100
   💰 Precio: $43,250.00
   📊 Vol 7d: 2.3%
   📈 ADX: 18.5
   😊 Sentimiento: 0.15
   🎯 Razones: Volatilidad óptima, Mercado lateral ideal

2. ETH/USDT - Score: 85.7/100
   💰 Precio: $2,680.50
   📊 Vol 7d: 2.8%
   📈 ADX: 22.1
   😊 Sentimiento: 0.22
   🎯 Razones: Sentimiento muy positivo, Volumen alto

3. SOL/USDT - Score: 82.4/100
   💰 Precio: $98.75
   📊 Vol 7d: 3.4%
   📈 ADX: 19.8
   😊 Sentimiento: 0.08
   🎯 Razones: Volatilidad óptima, Volumen alto
```

### 🧑‍💻 Trader Experimentado: "Quiero análisis profundo"

```bash
# Análisis completo automático (recomendado)
python scripts/smart_backtesting.py

# O configuración personalizada
python scripts/smart_backtesting.py \
  --top-n 100 \
  --min-volume 5000000 \
  --max-candidates 12
```

**Flujo completo:**
```
🚀 INICIANDO ANÁLISIS INTELIGENTE DE TRADING
======================================================================

🔍 FASE 1: Ejecutando scanner inteligente...
📈 Obteniendo Top 100 criptomonedas...
✅ Top 100 monedas obtenidas
🥇 Top 5: ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'MATIC/USDT']

📊 Analizando BTC/USDT (1/100)...
   ✅ Score: 87.3/100
📊 Analizando ETH/USDT (2/100)...
   ✅ Score: 82.1/100
[...continúa...]

🎯 MONEDAS SELECCIONADAS (Scanner inteligente):
📊 Total: 10 monedas
   1. BTC/USDT
   2. ETH/USDT
   3. SOL/USDT
   4. AVAX/USDT
   5. MATIC/USDT
   6. LINK/USDT
   7. DOT/USDT
   8. ADA/USDT
   9. UNI/USDT
   10. AAVE/USDT

⏱️  Tiempo estimado de backtesting: 2.0 horas
⚡ Optimización vs universo completo: 2.0x más rápido

🚀 FASE 2: Ejecutando backtesting optimizado...
🎯 Procesando 10 monedas seleccionadas...

[...proceso de backtesting...]

🎉 ANÁLISIS INTELIGENTE COMPLETADO
======================================================================
🧠 Método de selección: Scanner inteligente
📊 Monedas analizadas: 10
⏱️  Tiempo total: 2.15 horas
🚀 Tiempo backtesting: 1.98 horas
⚡ Factor de optimización: 2.0x más rápido

💡 OPTIMIZACIÓN LOGRADA:
   Tiempo tradicional estimado: 4.0 horas
   Tiempo actual: 1.98 horas
   Tiempo ahorrado: 2.02 horas (121 minutos)
   Eficiencia: 50.5% más rápido
```

### 🔬 Investigador/Analista: "Quiero comparar métodos"

```bash
# Comparar diferentes estrategias de selección
python scripts/smart_backtesting.py --compare-methods
```

**Output:**
```
📊 Comparando métodos de selección...
📈 Métodos comparados:
   scanner_intelligent: 10 monedas
      Score promedio: 78.5
   top_volume: 10 monedas
      Score promedio: 0
   random: 10 monedas
      Score promedio: 0

📊 COMPARACIÓN DE MÉTODOS DE SELECCIÓN:

SCANNER_INTELLIGENT:
  Método: Scanner con indicadores técnicos y sentimiento
  Monedas: BTC/USDT, ETH/USDT, SOL/USDT, AVAX/USDT, MATIC/USDT...
  Score: 78.5

TOP_VOLUME:
  Método: Top 10 por volumen/market cap
  Monedas: BTC/USDT, ETH/USDT, USDT/USDT, USDC/USDT, BNB/USDT...

RANDOM:
  Método: Selección aleatoria
  Monedas: DOGE/USDT, CRV/USDT, NEAR/USDT, COMP/USDT, YFI/USDT...
```

---

## 🎨 Casos de Uso Avanzados

### 🎯 Enfoque en Monedas Específicas

```bash
# Analizar solo las "Big 3"
python scripts/smart_backtesting.py --symbols BTC/USDT ETH/USDT SOL/USDT

# Analizar DeFi tokens específicos
python scripts/smart_backtesting.py --symbols UNI/USDT AAVE/USDT COMP/USDT MKR/USDT

# Analizar Layer 1s
python scripts/smart_backtesting.py --symbols ETH/USDT SOL/USDT AVAX/USDT DOT/USDT ADA/USDT
```

### 🔍 Simulación Sin Ejecutar (Dry Run)

```bash
# Ver qué se va a hacer sin ejecutar realmente
python scripts/smart_backtesting.py --dry-run

# Resultado:
# 🔬 DRY RUN: No se ejecutará backtesting real
# ✅ Análisis completado exitosamente!
# Selected symbols: ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', ...]
# Estimated time: 2.0 hours
```

### ⚡ Scanner Rápido para Trading Diario

```bash
# Configuración ultra-rápida para traders diarios
python scripts/smart_backtesting.py \
  --scanner-only \
  --top-n 20 \
  --min-volume 50000000 \
  --max-candidates 3

# ⏱️ Se ejecuta en ~2 minutos
# 💡 Perfecto para análisis matutino antes de operar
```

---

## 📈 Interpretando Resultados del Scanner

### Puntuaciones Típicas

| Score | Interpretación | Acción Recomendada |
|-------|----------------|-------------------|
| 90-100 | **Excelente** - Condiciones ideales | ✅ Operar con confianza |
| 80-89 | **Muy Bueno** - Condiciones favorables | ✅ Buena oportunidad |
| 70-79 | **Bueno** - Condiciones aceptables | ⚠️ Operar con precaución |
| 60-69 | **Regular** - Condiciones mixtas | ⚠️ Analizar más a fondo |
| <60 | **Evitar** - Condiciones desfavorables | ❌ No recomendado |

### Ejemplo de Análisis Detallado

```json
{
  "symbol": "ETH/USDT",
  "score": 85.7,
  "current_price": 2680.50,
  "volatility_7d": 0.028,  // 2.8% volatilidad
  "adx": 22.1,            // Mercado lateral
  "sentiment_score": 0.22, // Sentimiento positivo
  "volume_24h": 2840000000, // $2.84B volumen
  "reasons": [
    "Sentimiento muy positivo: 0.22",
    "Volumen alto: $2840M",
    "Volatilidad óptima: 2.8%",
    "Mercado lateral moderado (ADX: 22.1)"
  ]
}
```

**Interpretación:**
- ✅ **Score 85.7**: Muy buena oportunidad
- ✅ **Volatilidad 2.8%**: Perfecta para grid trading
- ✅ **ADX 22.1**: Mercado lateral, ideal para grids
- ✅ **Sentimiento 0.22**: Ambiente positivo
- ✅ **Volumen $2.84B**: Excelente liquidez

---

## 🛠️ Troubleshooting y Optimización

### Problemas Comunes

**1. "No se encontraron candidatos válidos"**
```bash
# Solución: Reducir criterios de filtrado
python scripts/smart_backtesting.py \
  --min-volume 1000000 \
  --top-n 50
```

**2. "Timeout en APIs"**
```bash
# Solución: Reducir número de monedas analizadas
python scripts/smart_backtesting.py \
  --top-n 30 \
  --max-candidates 5
```

**3. "Scanner muy lento"**
```bash
# Configuración rápida
python scripts/smart_backtesting.py \
  --scanner-only \
  --top-n 20 \
  --max-candidates 3
```

### Optimización de Performance

**Para análisis rápido:**
```bash
python scripts/smart_backtesting.py \
  --top-n 30 \
  --min-volume 20000000 \
  --max-candidates 5
```

**Para análisis exhaustivo:**
```bash
python scripts/smart_backtesting.py \
  --top-n 100 \
  --min-volume 1000000 \
  --max-candidates 15
```

---

## 📝 Flujos de Trabajo Recomendados

### 🌅 Rutina Diaria (5 minutos)

```bash
#!/bin/bash
# daily_scan.sh

echo "🔍 Escaneo diario de oportunidades"
python scripts/smart_backtesting.py \
  --scanner-only \
  --top-n 50 \
  --max-candidates 5 \
  > daily_scan.log

echo "✅ Escaneo completado. Ver daily_scan.log"
```

### 📊 Análisis Semanal (2 horas)

```bash
#!/bin/bash
# weekly_analysis.sh

echo "📈 Análisis semanal completo"
python scripts/smart_backtesting.py \
  --top-n 100 \
  --max-candidates 10 \
  > weekly_analysis.log

echo "✅ Análisis semanal completado"
```

### 🔬 Investigación Mensual (4 horas)

```bash
#!/bin/bash
# monthly_research.sh

echo "🔬 Investigación mensual exhaustiva"

# 1. Comparar métodos
python scripts/smart_backtesting.py --compare-methods

# 2. Análisis completo
python scripts/smart_backtesting.py --top-n 100 --max-candidates 20

# 3. Análisis específico de sectores
python scripts/smart_backtesting.py --symbols BTC/USDT ETH/USDT  # Store of value
python scripts/smart_backtesting.py --symbols UNI/USDT AAVE/USDT COMP/USDT  # DeFi
python scripts/smart_backtesting.py --symbols SOL/USDT AVAX/USDT DOT/USDT  # Layer 1

echo "✅ Investigación mensual completada"
```

---

## 💡 Tips y Mejores Prácticas

### ✅ Recomendaciones

1. **Empezar simple**: Usa `--scanner-only` primero
2. **Validar resultados**: Compara con análisis manual
3. **Configurar alertas**: Automatiza escaneos diarios
4. **Guardar histórico**: Mantén logs de resultados
5. **Iterar configuración**: Ajusta parámetros según experiencia

### ⚠️ Precauciones

1. **No operar ciegamente**: El scanner es una herramienta, no una decisión
2. **Validar configuración**: Revisa APIs y permisos
3. **Monitorear recursos**: El escaneo consume APIs
4. **Diversificar análisis**: No dependas solo del scanner
5. **Mantener actualizado**: Revisa regularmente parámetros

---

## 🎓 Próximos Pasos

Una vez domines estos ejemplos:

1. 📚 **Personalizar criterios** de puntuación en `scanner.py`
2. 🔧 **Ajustar parámetros** de backtesting según tu estilo
3. 📊 **Integrar con herramientas** de visualización
4. 🤖 **Automatizar** con cron jobs o schedulers
5. 📈 **Expandir a otros** timeframes y estrategias

¡El sistema está diseñado para evolucionar contigo! 🚀 