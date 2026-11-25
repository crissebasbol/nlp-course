#!/usr/bin/env bash
# Funciones compartidas por los scripts de despliegue. Se incluye con "source", no se ejecuta.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

titulo() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# Recién instalado Docker, el usuario aún no está en su grupo hasta volver a iniciar sesión.
DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  if command -v sudo >/dev/null && sudo -n docker info >/dev/null 2>&1; then
    DOCKER=(sudo docker)
  else
    echo "Docker no está disponible. En Ubuntu: deploy/preparar_servidor.sh" >&2
    exit 1
  fi
fi

compose() { "${DOCKER[@]}" compose "$@"; }
