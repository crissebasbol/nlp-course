#!/usr/bin/env bash
# Levanta en este computador los mismos contenedores del servidor, en http://localhost:8080
#   deploy/local.sh            construir y levantar
#   deploy/local.sh detener    detener
set -euo pipefail
cd "$(dirname "$0")/.."
export PUERTO_HTTP="${PUERTO_HTTP:-8080}" PUERTO_HTTPS="${PUERTO_HTTPS:-8443}" SECOPEDIA_DIRECCION=":80"

if [ "${1:-}" = detener ]; then docker compose down; exit; fi

# Redes corporativas que inspeccionan TLS: el certificado se copia al contexto y nunca se sube a git.
mkdir -p docker/certs
if [ -n "${CERTIFICADO_EXTRA:-}" ]; then cp "$CERTIFICADO_EXTRA" docker/certs/extra.crt; fi

docker compose up -d --build
printf 'Esperando a la demo (la primera vez descarga unos 500 MB de modelos)'
for _ in $(seq 1 180); do
  if curl -fs "http://localhost:$PUERTO_HTTP/api/estado" >/dev/null; then
    echo; echo "Interfaz: http://localhost:$PUERTO_HTTP"; exit 0
  fi
  printf '.'; sleep 5
done
echo; docker compose logs --tail 40 app; exit 1
