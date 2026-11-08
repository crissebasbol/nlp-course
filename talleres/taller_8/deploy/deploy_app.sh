#!/usr/bin/env bash
# Machine 2: copies the code and the light artifacts, builds the image and starts the app.
# The fine-tuned BiomedBERT is not copied: the app downloads it from the Hugging Face Hub on the first start.
source "$(dirname "$0")/common.sh"
HOST="$M2_HOST"
cd "$PROJECT_DIR"
mkdir -p docker/certs  # optional extra CAs for the build (folder is git-ignored)

ARTIFACTS=(
  outputs/models/dspy_detector.json
  outputs/artifacts
  outputs/splits/test.parquet
  outputs/splits/train.parquet
)
for path in "${ARTIFACTS[@]}"; do
  [ -e "$path" ] || { echo "Falta $path: corre primero scripts/run_local_experiments.sh" >&2; exit 1; }
done
REPORTS=()
[ -d outputs/reports ] && REPORTS=(outputs/reports)

log "Copiando código y artefactos a $HOST (pocos MB; BiomedBERT se descarga de Hugging Face en el servidor)"
push "$HOST" docker-compose.yml docker src configs pyproject.toml requirements-serve.txt .dockerignore \
  "${ARTIFACTS[@]}" ${REPORTS[@]+"${REPORTS[@]}"}

log "Construyendo la imagen y levantando el contenedor app (puerto $APP_PORT)"
set_remote_env "$HOST" LLM_API_BASE="http://$M1_HOST:$LLM_PORT/v1" APP_PORT="$APP_PORT" OUTPUTS_DIR=./outputs
remote "$HOST" "$(compose_cmd) build app && \$C up -d --no-deps app"

log "Liberando espacio de builds anteriores"
remote "$HOST" "$DOCKER image prune -f >/dev/null; $DOCKER builder prune -f >/dev/null || true; df -h ~ | tail -1"

log "Esperando a la app (descarga el modelo de embeddings la primera vez)"
wait_http "$HOST" "http://localhost:$APP_PORT/api/health" 600
echo "Interfaz: http://$HOST:$APP_PORT"
