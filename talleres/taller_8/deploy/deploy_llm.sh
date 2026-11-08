#!/usr/bin/env bash
# Machine 1: downloads Qwen2.5-7B-Instruct Q4_K_M (4.7 GB) and starts llama.cpp server.
source "$(dirname "$0")/common.sh"
HOST="$M1_HOST"

log "Copiando docker-compose.yml a $HOST"
push "$HOST" docker-compose.yml

log "Descargando el modelo GGUF si no existe (4.7 GB)"
remote "$HOST" "bash -s" <<REMOTE
set -e
mkdir -p $REMOTE_DIR/models/gguf
cd $REMOTE_DIR/models/gguf
if [ ! -s $GGUF_FILE ] || [ -f $GGUF_FILE.part ]; then
  free_kb=\$(df -Pk . | awk 'NR==2{print \$4}')
  [ "\$free_kb" -gt 5500000 ] || { echo "Disco insuficiente: se necesitan ~5.5 GB libres"; exit 1; }
  touch $GGUF_FILE.part
  curl -L --fail -C - -o $GGUF_FILE "https://huggingface.co/$GGUF_REPO/resolve/main/$GGUF_FILE"
  rm -f $GGUF_FILE.part
fi
ls -lh $GGUF_FILE
REMOTE

log "Levantando el contenedor llm (puerto $LLM_PORT)"
remote "$HOST" "$(compose_cmd) pull llm && GGUF_DIR=./models/gguf LLM_PORT=$LLM_PORT LLM_THREADS=$LLM_THREADS LLM_CTX=$LLM_CTX \$C up -d llm"

log "Esperando a que cargue el modelo"
wait_http "$HOST" "http://localhost:$LLM_PORT/health" 900
echo "LLM listo en http://$HOST:$LLM_PORT/v1 (solo red interna)"
