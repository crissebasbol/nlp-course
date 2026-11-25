#!/usr/bin/env bash
# Funciones compartidas por los scripts de despliegue. Se incluye con "source", no se ejecuta.
set -euo pipefail

CARPETA_DEPLOY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARPETA_PROYECTO="$(dirname "$CARPETA_DEPLOY")"
ARCHIVO_SERVIDOR="${ARCHIVO_SERVIDOR:-$CARPETA_DEPLOY/servidor.env}"
if [ ! -f "$ARCHIVO_SERVIDOR" ]; then
  echo "Falta deploy/servidor.env: cp deploy/servidor.env.example deploy/servidor.env y edítalo." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ARCHIVO_SERVIDOR"
: "${SSH_HOST:?}" "${SSH_USUARIO:?}"
SSH_PUERTO="${SSH_PUERTO:-22}"
CARPETA_REMOTA="${CARPETA_REMOTA:-secopedia}"
SECOPEDIA_DIRECCION="${SECOPEDIA_DIRECCION:-:80}"
SECOPEDIA_PRECARGAR="${SECOPEDIA_PRECARGAR-regresion_logistica,svm_lineal,red_neuronal,beto}"
MEMORIA_APP="${MEMORIA_APP:-3g}"
HILOS="${HILOS:-2}"

OPCIONES_SSH=(-p "$SSH_PUERTO" -o ConnectTimeout=15 -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new)
if [ -n "${SSH_LLAVE:-}" ]; then
  [ -f "$SSH_LLAVE" ] || { echo "No existe la llave $SSH_LLAVE" >&2; exit 1; }
  OPCIONES_SSH+=(-i "$SSH_LLAVE" -o IdentitiesOnly=yes)
fi

titulo() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

remoto() { ssh "${OPCIONES_SSH[@]}" "$SSH_USUARIO@$SSH_HOST" "$@"; }

compose_remoto() { remoto "cd $CARPETA_REMOTA && docker compose $*"; }

url_publica() {
  case "$SECOPEDIA_DIRECCION" in
    :80) echo "http://$SSH_HOST" ;;
    *) echo "https://$SECOPEDIA_DIRECCION" ;;
  esac
}
