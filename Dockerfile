FROM python:3.11-slim

# Metadatos
LABEL maintainer="Trading AI Team"
LABEL description="Sistema inteligente de trading con arquitectura limpia"
LABEL version="1.0.0"

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY app/ ./app/
COPY .env.example .env

# Crear usuario no-root para seguridad
RUN groupadd -r trading && useradd -r -g trading trading
RUN chown -R trading:trading /app
USER trading

# Puerto por defecto (para futuras APIs)
EXPOSE 8000

# Comando por defecto
CMD ["python", "app/main.py", "--status"] 