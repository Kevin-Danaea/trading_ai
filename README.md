# 🤖 Trading AI - Sistema Inteligente de Trading Automatizado

## 🚀 NUEVO: Sistema Inteligente con Scanner

Sistema revolucionario de trading automatizado que combina **scanner inteligente** + **backtesting optimizado** para reducir tiempo de análisis de **22 horas a ~2 horas** (91% más rápido).

### ⚡ Características Principales

- 🔍 **Scanner Inteligente**: Analiza Top 100 criptomonedas y selecciona las mejores 5-10
- 📊 **Backtesting Optimizado**: Procesa solo monedas seleccionadas
- 🧠 **IA para Selección**: Volatilidad + ADX + Sentimiento + Volumen
- ⏱️ **Ultra Rápido**: De 22h → 2h (optimización 91%)
- 📈 **Escalable**: Maneja 100+ monedas dinámicamente

## 🎯 Inicio Rápido (5 minutos)

### 1. Instalación Express

```bash
# Clonar e instalar
git clone <tu-repositorio>
cd trading_ai
pip install -r requirements.txt

# Configurar variables
cp .env.example .env
# Editar .env con tus APIs
```

### 2. Análisis Instantáneo

```bash
# 🔥 Nuevo sistema inteligente (recomendado)
python scripts/smart_backtesting.py --scanner-only

# Output: Mejores 5-10 monedas + comando para backtesting
# ⏱️ Se ejecuta en ~3 minutos
```

### 3. Backtesting Completo

```bash
# Copiar y ejecutar el comando del paso anterior
python scripts/find_optimal_parameters.py --monedas BTC/USDT ETH/USDT SOL/USDT

# ⏱️ Tiempo: ~1 hora (vs 22h método tradicional)
```

## 🧠 Cómo Funciona el Sistema Inteligente

```
🔍 SCANNER INTELIGENTE
├── 📈 Obtiene Top 100 criptomonedas (Binance)
├── 📊 Calcula indicadores técnicos
│   ├── Volatilidad (2-6% óptima para grid)
│   ├── ADX (<25 mercado lateral ideal)
│   └── Momentum (estabilidad)
├── 😊 Analiza sentimiento
│   ├── BigQuery (datos históricos)
│   └── DATABASE_URL (datos recientes)
├── 🎯 Sistema de puntuación 0-100
└── ✨ Selecciona mejores 5-10 candidatos

⬇️

🚀 BACKTESTING OPTIMIZADO
├── Procesa solo monedas seleccionadas
├── Mantiene misma calidad de análisis
├── Reduce tiempo 91%
└── Guarda resultados en BigQuery
```

## 📊 Comparativa de Rendimiento

| Aspecto | Método Tradicional | Sistema Inteligente | Mejora |
|---------|-------------------|-------------------|-------|
| **Tiempo** | ~22 horas | ~2 horas | **91% más rápido** |
| **Monedas** | 20 fijas | Top 100 → 5-10 mejores | **Selección dinámica** |
| **Precisión** | Buena | Excelente | **Filtros avanzados** |
| **Escalabilidad** | Limitada | Alta | **Maneja 100+ monedas** |
| **Automation** | Manual | Automático | **IA para selección** |

## 🎯 Scripts Principales

### `smart_backtesting.py` - Orquestador Principal ⭐

El cerebro del sistema que integra todo:

```bash
# Análisis completo automático
python scripts/smart_backtesting.py

# Solo scanner (rápido)
python scripts/smart_backtesting.py --scanner-only

# Monedas específicas
python scripts/smart_backtesting.py --symbols BTC/USDT ETH/USDT SOL/USDT

# Simulación (ver plan sin ejecutar)
python scripts/smart_backtesting.py --dry-run
```

### `scanner.py` - Escáner Inteligente 🔍

Analiza Top 100 y selecciona los mejores:

```bash
# Ejecutar scanner independiente
python scripts/scanner.py

# Personalizar configuración
python scripts/smart_backtesting.py --top-n 50 --max-candidates 8
```

### `find_optimal_parameters.py` - Backtesting Mejorado 📊

Ahora acepta parámetros dinámicos:

```bash
# Método tradicional (lento)
python scripts/find_optimal_parameters.py

# Método optimizado (rápido)
python scripts/find_optimal_parameters.py --monedas BTC/USDT ETH/USDT SOL/USDT

# Top N del universo
python scripts/find_optimal_parameters.py --top 5
```

## 🎨 Casos de Uso Típicos

### 👤 Trader Principiante

```bash
# Configuración simple y segura
python scripts/smart_backtesting.py \
  --scanner-only \
  --top-n 30 \
  --max-candidates 3
```

### 🧑‍💻 Trader Experimentado

```bash
# Análisis completo automático
python scripts/smart_backtesting.py
```

### 🔬 Investigador/Analista

```bash
# Comparar métodos de selección
python scripts/smart_backtesting.py --compare-methods
```

### ⚡ Trading Diario

```bash
# Escaneo rápido matutino (2 minutos)
python scripts/smart_backtesting.py \
  --scanner-only \
  --top-n 20 \
  --max-candidates 3
```

## 🎯 Sistema de Puntuación del Scanner

### Criterios (Total 100 puntos)

- **Volatilidad (30 pts)**: 2-6% diaria óptima para grid trading
- **ADX (25 pts)**: <25 mercado lateral ideal  
- **Sentimiento (20 pts)**: Positivo mejor que negativo
- **Volumen (15 pts)**: >$10M liquidez mínima
- **Momentum (10 pts)**: Estabilidad vs extremos

### Interpretación de Scores

| Score | Calidad | Acción |
|-------|---------|--------|
| 90-100 | Excelente | ✅ Operar con confianza |
| 80-89 | Muy Bueno | ✅ Buena oportunidad |
| 70-79 | Bueno | ⚠️ Operar con precaución |
| 60-69 | Regular | ⚠️ Analizar más |
| <60 | Evitar | ❌ No recomendado |

## 🔧 Configuración

### Variables de Entorno Esenciales

```bash
# APIs de Trading
BINANCE_API_KEY=tu_api_key
BINANCE_API_SECRET=tu_api_secret

# Datos de Sentimiento (opcional)
GOOGLE_CLOUD_PROJECT_ID=tu_proyecto_bigquery
DATABASE_URL=postgresql://usuario:pass@host:5432/db
```

### Configuración del Scanner

```python
scanner_config = {
    'top_n': 100,                # Top N monedas a analizar
    'min_volume_usdt': 10_000_000,  # Volumen mínimo $10M
    'max_candidates': 10         # Máximo candidatos
}
```

## 📈 Estrategias Soportadas

### Grid Trading (Principal)
- **Nueva Metodología Grid Step**: Niveles × Paso% = Rango total
- **Filtros Avanzados**: ADX + Volatilidad + Sentimiento
- **Optimización Masiva**: Miles de combinaciones

### DCA (Dollar Cost Averaging)
- **Buy The Dip**: Compra en caídas con SMA
- **Take Profit**: Venta automática en ganancias
- **Trend Following**: Solo opera en tendencias alcistas

### BTD Short (Buy The Dip Short)
- **Sell The Rip**: Venta en subidas desde mínimos
- **Short Covering**: Recompra en caídas
- **Bear Market**: Optimizado para mercados bajistas

## 📁 Estructura del Proyecto

```
trading_ai/
├── 🆕 scripts/
│   ├── smart_backtesting.py    # Orquestador principal
│   ├── scanner.py              # Scanner inteligente
│   ├── find_optimal_parameters.py  # Backtesting mejorado
│   ├── backtesting_engine.py   # Motor de simulación
│   └── data_collector.py       # Recolector de datos
├── shared/config/
│   └── settings.py             # Configuraciones
├── 🆕 EJEMPLOS_USO.md          # Guía detallada
├── requirements.txt
└── README.md
```

## 🚀 Migración desde Sistema Anterior

Si ya usabas el sistema:

```bash
# Antes (lento)
python scripts/find_optimal_parameters.py

# Ahora (rápido) 
python scripts/smart_backtesting.py
```

**100% compatible** - Solo añade inteligencia y optimización.

## 📊 Dependencias

### Esenciales
```bash
pandas>=2.0.0          # Manipulación de datos
numpy>=1.24.0           # Cálculos numéricos  
ccxt>=4.0.0             # APIs de exchanges
python-dotenv>=1.0.0    # Variables de entorno
```

### Análisis (Opcionales)
```bash
pandas-gbq>=0.19.0      # BigQuery integration
matplotlib>=3.7.0       # Visualizaciones
seaborn>=0.12.0         # Gráficos estadísticos
```

## 🎓 Recursos de Aprendizaje

- 📖 **[EJEMPLOS_USO.md](EJEMPLOS_USO.md)**: Guía paso a paso con casos reales
- 📝 **[scripts/README.md](scripts/README.md)**: Documentación técnica detallada
- 🔧 **Inline docs**: Código completamente documentado

## 🏆 Beneficios Clave

### ⚡ Eficiencia
- **91% más rápido** que método tradicional
- **Selección automática** de mejores oportunidades
- **Escalabilidad** para 100+ monedas

### 🧠 Inteligencia
- **Filtros avanzados** con indicadores técnicos
- **Análisis de sentimiento** en tiempo real
- **Sistema de puntuación** objetivo

### 🔧 Flexibilidad  
- **Múltiples modos** de operación
- **Configuración personalizable**
- **Compatible** con sistema anterior

## 💡 Tips para Empezar

1. **Comienza simple**: `python scripts/smart_backtesting.py --scanner-only`
2. **Valida resultados**: Compara con análisis manual
3. **Itera configuración**: Ajusta según tu experiencia
4. **Automatiza**: Programa escaneos diarios
5. **Documenta**: Mantén registro de mejores configuraciones

## ⚠️ Disclaimer

Este sistema es una herramienta de análisis, no asesoría financiera. Siempre:
- ✅ Valida resultados manualmente
- ✅ Diversifica tu portfolio  
- ✅ Gestiona el riesgo apropiadamente
- ✅ Testa en papel antes de usar capital real

---

> 🔥 **¡Prueba el nuevo sistema!** Ejecuta `python scripts/smart_backtesting.py --scanner-only` y ve la magia en acción en solo 3 minutos.

## 📞 Soporte

¿Dudas o problemas? Revisa:
- 📖 [EJEMPLOS_USO.md](EJEMPLOS_USO.md) para casos específicos
- 📝 [scripts/README.md](scripts/README.md) para detalles técnicos
- 🔧 `python scripts/smart_backtesting.py --help` para todas las opciones 