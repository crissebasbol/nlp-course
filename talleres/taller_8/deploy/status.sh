#!/usr/bin/env bash
# Containers, memory and health of both machines.
source "$(dirname "$0")/common.sh"

log "Máquina 1 ($M1_HOST): LLM"
remote "$M1_HOST" "$DOCKER ps --filter name=llm --format '{{.Names}}  {{.Status}}'; free -m | awk 'NR==2{print \"RAM disponible: \"\$7\" MB\"}'; curl -sf http://localhost:$LLM_PORT/health && echo" || true

log "Máquina 2 ($M2_HOST): app"
remote "$M2_HOST" "$DOCKER ps --filter name=app --format '{{.Names}}  {{.Status}}'; free -m | awk 'NR==2{print \"RAM disponible: \"\$7\" MB\"}'; curl -sf http://localhost:$APP_PORT/api/health && echo" || true

echo
echo "Interfaz: http://$M2_HOST:$APP_PORT"
