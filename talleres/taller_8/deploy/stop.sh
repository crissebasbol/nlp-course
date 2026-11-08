#!/usr/bin/env bash
# Usage: deploy/stop.sh llm|app|all
source "$(dirname "$0")/common.sh"
target=${1:-all}
if [ "$target" = llm ] || [ "$target" = all ]; then
  log "Deteniendo llm en $M1_HOST"; remote "$M1_HOST" "$(compose_cmd) stop llm"
fi
if [ "$target" = app ] || [ "$target" = all ]; then
  log "Deteniendo app en $M2_HOST"; remote "$M2_HOST" "$(compose_cmd) stop app"
fi
