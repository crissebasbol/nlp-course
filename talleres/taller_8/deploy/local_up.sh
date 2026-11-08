#!/usr/bin/env bash
# Runs the same two containers as the servers, on this computer.
#   deploy/local_up.sh                 llm + app in Docker (CPU, like the servers)
#   deploy/local_up.sh --native-llm    only app in Docker; the LLM is llama-server on the host (port 8001)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p docker/certs  # optional extra CAs for the build (folder is git-ignored)

# Corporate TLS inspection (e.g. Zscaler): the CA is copied into the build context, never committed.
for ca in "$HOME/.certs/zscaler-root.pem"; do
  [ -f "$ca" ] && cp "$ca" docker/certs/corporate-root.crt
done

for path in outputs/artifacts outputs/splits/test.parquet; do  # BERT: local or from the Hugging Face Hub
  [ -e "$path" ] || { echo "Falta $path: corre primero scripts/run_local_experiments.sh" >&2; exit 1; }
done

if [ "${1:-}" = "--native-llm" ]; then
  curl -sf http://127.0.0.1:8001/health >/dev/null || { echo "No hay llama-server en el puerto 8001" >&2; exit 1; }
  LLM_API_BASE=http://host.docker.internal:8001/v1 docker compose up -d --build --no-deps app
else
  [ -f outputs/models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf ] || { echo "Falta el GGUF en outputs/models/gguf" >&2; exit 1; }
  docker compose up -d --build
  echo "Esperando al LLM..."
  until curl -sf http://localhost:8080/health >/dev/null; do sleep 5; done
fi
echo "Esperando a la app..."
until curl -sf http://localhost:8000/api/health >/dev/null; do sleep 5; done
curl -s http://localhost:8000/api/health; echo
echo "Interfaz: http://localhost:8000"
