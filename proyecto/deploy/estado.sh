#!/usr/bin/env bash
# Estado de los contenedores, memoria y disco.
source "$(dirname "$0")/comun.sh"
compose ps
"${DOCKER[@]}" stats --no-stream
echo
if command -v free >/dev/null; then free -h; fi
df -h / | tail -1
