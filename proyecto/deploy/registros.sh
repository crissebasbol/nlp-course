#!/usr/bin/env bash
# Registros en vivo (Ctrl+C para salir):  deploy/registros.sh [app|proxy]
source "$(dirname "$0")/comun.sh"
compose logs -f --tail 100 "$@"
