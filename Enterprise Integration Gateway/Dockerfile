FROM python:3.12-slim

LABEL maintainer="Enterprise Integration Gateway"
LABEL description="Main application service"

WORKDIR /app

# Install system dependencies (includes librdkafka for confluent-kafka)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    librdkafka-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
