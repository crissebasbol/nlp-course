#!/usr/bin/env bash
# Registros del servidor en vivo (Ctrl+C para salir):  deploy/registros.sh [app|proxy]
source "$(dirname "$0")/comun.sh"
compose_remoto logs -f --tail 100 "${1:-}"
