# API + web UI + BiomedBERT + corrector (CPU). The LLM runs in its own container (llama.cpp).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    HF_HOME=/data/hf_cache \
    DSPY_CACHEDIR=/data/hf_cache/dspy \
    MEDEC_OUTPUT_DIR=/data/outputs \
    MEDEC_PROJECT_DIR=/app \
    TOKENIZERS_PARALLELISM=false

# Optional extra CA (e.g. a corporate proxy that inspects TLS). Empty on the servers.
COPY docker/certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates

WORKDIR /app
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
RUN pip install --no-deps .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=4)"
CMD ["uvicorn", "medec.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
