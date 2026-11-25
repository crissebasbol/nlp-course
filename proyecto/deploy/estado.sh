#!/usr/bin/env bash
# Estado de los contenedores, memoria y disco del servidor.
source "$(dirname "$0")/comun.sh"
compose_remoto ps
remoto 'docker stats --no-stream; echo; free -h; echo; df -h / | tail -1'
