# 🤖 Trading AI - Sistema de Grid Trading Automatizado

Sistema avanzado de trading automatizado con estrategias de grid trading, análisis de sentimientos y optimización masiva de parámetros.

## 🚀 Instalación Rápida

### 1. Clonar e Instalar Dependencias

```bash
# Clonar el repositorio
git clone <tu-repositorio>
cd trading_ai

# Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar Variables de Entorno

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar .env con tus credenciales
# nano .env  # o usa tu editor favorito
```

### 3. Verificar Instalación

```bash
# Ejecutar análisis de ejemplo
python scripts/analisis_backtest.py

# Verificar data collector
python scripts/data_collector.py
```

## 📦 Dependencias Principales

### Esenciales
- **pandas** - Manipulación de datos financieros
- **numpy** - Cálculos numéricos
- **ccxt** - APIs de exchanges de crypto
- **python-dotenv** - Manejo de variables de entorno

### APIs Externas
- **pandas-gbq + google-cloud-bigquery** - Integración con BigQuery
- **matplotlib + seaborn** - Visualizaciones y análisis

### Opcionales (para desarrollo)
- **plotly** - Gráficos interactivos
- **scipy + scikit-learn** - Análisis científico avanzado

## 🔧 Configuración

### Variables de Entorno Requeridas

Edita tu archivo `.env` con estas configuraciones:

```bash
# APIs de Trading
BINANCE_API_KEY=tu_api_key
BINANCE_API_SECRET=tu_api_secret

# Google Cloud (opcional)
GOOGLE_CLOUD_PROJECT_ID=tu_proyecto
GOOGLE_APPLICATION_CREDENTIALS=credenciales.json
```

### Permisos de API Binance

Tu API Key de Binance necesita estos permisos:
- ✅ **Spot & Margin Trading** (para datos históricos)
- ❌ **Futures Trading** (no necesario)
- ❌ **Withdrawals** (no necesario por seguridad)

## 📊 Scripts Disponibles

### `find_optimal_parameters.py`
Optimización masiva con nueva metodología Grid Step:
```bash
python scripts/find_optimal_parameters.py
```

### `backtesting_engine.py`
Motor de simulación de estrategias:
```bash
python scripts/backtesting_engine.py
```

### `analisis_backtest.py`
Análisis de resultados con visualizaciones:
```bash
python scripts/analisis_backtest.py
```

### `data_collector.py`
Recolección de datos históricos:
```bash
python scripts/data_collector.py
```

## 🎯 Metodología Grid Step

El sistema usa una nueva metodología profesional:

- **Niveles**: [20, 30, 50, 80, 100, 150]
- **Pasos**: [0.4%, 0.6%, 0.8%, 1.0%, 1.5%]
- **Rango Calculado**: `niveles × (paso / 100)`

### Ejemplo
- 50 niveles × 0.8% paso = 0.4% rango total
- Grid más granular y profesional

## 🔍 Estructura del Proyecto

```
trading_ai/
├── scripts/                    # Scripts principales
│   ├── find_optimal_parameters.py  # Optimización masiva
│   ├── backtesting_engine.py      # Motor de simulación
│   ├── analisis_backtest.py       # Análisis de resultados
│   └── data_collector.py          # Recolección de datos
├── shared/                     # Configuraciones compartidas
│   └── config/
│       └── settings.py         # Configuraciones centralizadas
├── requirements.txt            # Dependencias del proyecto
├── .env.example               # Ejemplo de configuración
└── README.md                  # Este archivo
```

## ⚠️ Notas Importantes

### Seguridad
- **NUNCA** subas tu archivo `.env` a git
- Usa permisos mínimos en tus API keys
- Mantén tus credenciales seguras

### Desarrollo
- El proyecto está optimizado para Python 3.8+
- Se recomienda usar un entorno virtual
- Todas las dependencias tienen versiones especificadas

### Soporte
- BigQuery es opcional (solo para datos de sentimiento)
- El sistema puede funcionar solo con datos de Binance
- Logs detallados para debugging

## 🚀 Próximos Pasos

1. **Configurar APIs** - Obtén credenciales de Binance
2. **Ejecutar Optimización** - Usa `find_optimal_parameters.py`
3. **Analizar Resultados** - Revisa con `analisis_backtest.py`
4. **Iterar y Mejorar** - Ajusta parámetros según resultados

---

*Proyecto desarrollado para optimización de estrategias de grid trading con IA* 