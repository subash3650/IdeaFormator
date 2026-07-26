FROM python:3.13-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir "fastapi>=0.115.0" "uvicorn[standard]>=0.34.0" "pydantic>=2.0.0" "PyYAML>=6.0" "httpx>=0.27.0" "pyjwt>=2.8.0" "python-multipart>=0.0.9"

COPY . .

ENV PYTHONPATH=/app
ENV API_ENVIRONMENT=production

EXPOSE 8000

CMD ["uvicorn", "phase5.api.__main__:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
