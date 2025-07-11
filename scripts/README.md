# Scripts del Oráculo Bot

Este directorio contiene scripts utilitarios para el procesamiento de datos históricos y otras tareas de mantenimiento.

## Procesador de Datos Históricos de Reddit (`procesador_historico.py`)

Script diseñado para procesar archivos .zst de Reddit de manera eficiente, con análisis de sentimientos usando Google Gemini.

### Características Principales

- **Streaming**: Procesa archivos .zst sin cargar todo en memoria
- **Filtrado Inteligente**: Solo procesa subreddits relevantes para cripto
- **Análisis de Sentimientos**: Usa Google Gemini para análisis completo
- **Inserción Optimizada**: Batching para máxima eficiencia en base de datos
- **Monitoreo en Tiempo Real**: Feedback constante del progreso

### Dependencias Requeridas

```bash
pip install pandas sqlalchemy psycopg2-binary zstandard python-dotenv google-generativeai
```

### Configuración

Asegúrate de que tu archivo `.env` contenga:

```bash
DATABASE_URL=tu_url_de_neon_postgres
GOOGLE_API_KEY=tu_api_key_de_gemini
```

### Uso Básico

```bash
# Procesar un archivo .zst
python scripts/procesador_historico.py RS_2023-01.zst

# Procesar con configuración personalizada
python scripts/procesador_historico.py RS_2023-01.zst --batch-size 500 --api-delay 0.3
```

### Parámetros Disponibles

- `archivo_zst`: Ruta al archivo .zst de Reddit (requerido)
- `--batch-size`: Tamaño del lote para inserción en BD (default: 1000)
- `--api-delay`: Delay entre llamadas a Gemini en segundos (default: 0.2)

### Subreddits Procesados

El script filtra y procesa posts de los siguientes subreddits:

- CryptoCurrency
- bitcoin
- ethtrader
- BitcoinMarkets
- ethereum
- dogecoin
- CryptoMarkets
- altcoin
- defi
- NFT
- CryptoNews
- binance

### Datos Extraídos

Para cada post relevante, el script extrae:

- **Información básica**: título, URL, fecha de publicación, subreddit
- **Análisis de sentimientos**:
  - `sentiment_score`: Puntuación entre -1.0 (muy negativo) y 1.0 (muy positivo)
  - `primary_emotion`: Euforia, Optimismo, Neutral, Incertidumbre, Miedo
  - `news_category`: Regulación, Tecnología/Adopción, Mercado/Trading, Seguridad, Macroeconomía

### Optimizaciones de Rendimiento

1. **Streaming**: No carga el archivo completo en memoria
2. **Batching**: Inserta en lotes de 1000 registros por defecto
3. **Rate Limiting**: Respeta límites de API de Gemini (0.2s por defecto)
4. **Filtrado Temprano**: Descarta posts irrelevantes antes del análisis

### Monitoreo

El script proporciona feedback en tiempo real:

```
🔮 PROCESADOR DE DATOS HISTÓRICOS DE REDDIT
==================================================
📁 Archivo: RS_2023-01.zst
📦 Tamaño de lote: 1000
⏱️  Delay API: 0.2s
🎯 Subreddits objetivo: CryptoCurrency, bitcoin, ethtrader...
==================================================

🔗 Conectando a la base de datos...
✅ Conexión a BD establecida exitosamente
✅ Cliente Gemini configurado exitosamente
🚀 Iniciando procesamiento del archivo RS_2023-01.zst...

📊 Registros leídos: 100,000 | Filtrados: 5,432 | Procesados: 1,000 | Llamadas API: 1,000 | Lotes insertados: 1
📦 Lote de 1000 registros insertado en la base de datos...

======================================================================
📈 RESUMEN FINAL DEL PROCESAMIENTO
======================================================================
⏱️  Tiempo total: 125.3 segundos
📄 Registros leídos: 500,000
🔍 Registros filtrados: 12,845
✅ Registros procesados: 12,845
🤖 Llamadas a API Gemini: 12,845
📦 Lotes insertados en BD: 13
❌ Errores encontrados: 23
🚀 Velocidad: 3,992.0 registros/segundo
======================================================================
✅ Procesamiento del archivo completado exitosamente
```

### Manejo de Errores

El script está diseñado para ser resiliente:

- **Errores de decodificación**: Se saltan líneas corruptas
- **JSON inválido**: Se ignoran líneas que no son JSON válido
- **Errores de API**: Se usan valores por defecto para análisis fallidos
- **Errores de BD**: Se hace rollback automático de transacciones fallidas

### Archivos de Reddit

Los archivos .zst de Reddit están disponibles en:
- [Reddit Data Dumps](https://academictorrents.com/browse.php?search=reddit)
- [Pushshift Archives](https://files.pushshift.io/reddit/)

### Consejos de Rendimiento

1. **Para archivos grandes**: Usa un SSD para mejor I/O
2. **Rate limiting**: Ajusta `--api-delay` según tu cuota de Gemini
3. **Memoria**: El script usa muy poca memoria gracias al streaming
4. **Red**: Una conexión estable mejora las llamadas a la API

### Troubleshooting

**Error: "GOOGLE_API_KEY no configurada"**
- Verifica que el archivo `.env` contenga la API key de Gemini

**Error: "Error conectando a la base de datos"**
- Verifica que `DATABASE_URL` esté correctamente configurada en `.env`

**Proceso muy lento**
- Reduce `--api-delay` si tienes cuota suficiente en Gemini
- Aumenta `--batch-size` para menos transacciones de BD

**Muchos errores de análisis**
- Revisa tu cuota de API de Gemini
- Verifica la conectividad a internet 