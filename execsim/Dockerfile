FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN mkdir -p src/execsim && \
    touch src/execsim/__init__.py && \
    pip install --no-cache-dir ".[dev]"

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "execsim.main:app", "--host", "0.0.0.0", "--port", "8000"]
