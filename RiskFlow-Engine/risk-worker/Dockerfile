FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY risk_engine ./risk_engine
CMD ["python", "-m", "risk_engine.worker"]
