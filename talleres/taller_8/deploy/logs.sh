#!/usr/bin/env bash
# Usage: deploy/logs.sh llm|app [lines]
source "$(dirname "$0")/common.sh"
service=${1:?"Uso: deploy/logs.sh llm|app [líneas]"}
lines=${2:-100}
host=$([ "$service" = llm ] && echo "$M1_HOST" || echo "$M2_HOST")
remote "$host" "$(compose_cmd) logs --tail $lines $service"
