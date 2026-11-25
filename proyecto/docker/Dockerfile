# Demo de SECOPEDIA: API, interfaz web y los cuatro modelos (CPU).
# Los modelos no van en la imagen: se descargan de Hugging Face al primer arranque
# y quedan en el volumen "modelos".
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    HF_HOME=/modelos \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/app/src

# Certificado adicional opcional (redes corporativas que inspeccionan TLS). Vacío en el servidor.
COPY docker/certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates

WORKDIR /app
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# El código se usa desde /app/src porque busca configs/ y reports/ dos carpetas más arriba.
COPY configs/ configs/
COPY reports/metrics/ reports/metrics/
COPY src/ src/

RUN useradd --create-home --uid 10001 secopedia && mkdir /modelos && chown secopedia /modelos
USER secopedia
VOLUME /modelos
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/estado', timeout=4)"
# Un solo proceso: cada worker cargaría su propia copia de los modelos.
CMD ["uvicorn", "secopedia.web.servidor:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
