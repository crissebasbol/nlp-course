#!/usr/bin/env bash
# Shared helpers for the deploy scripts. Source it, do not run it.
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$DEPLOY_DIR")"
if [ ! -f "$DEPLOY_DIR/hosts.env" ]; then
  echo "Falta deploy/hosts.env: cp deploy/hosts.env.example deploy/hosts.env y edítalo." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$DEPLOY_DIR/hosts.env"
: "${SSH_USER:?}" "${M1_HOST:?}" "${M2_HOST:?}"
SSH_PORT="${SSH_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-medec}"
APP_PORT="${APP_PORT:-8000}"
LLM_PORT="${LLM_PORT:-8080}"
LLM_THREADS="${LLM_THREADS:-4}"
LLM_CTX="${LLM_CTX:-8192}"
DOCKER="${DOCKER:-docker}"
GGUF_REPO="bartowski/Qwen2.5-7B-Instruct-GGUF"
GGUF_FILE="Qwen2.5-7B-Instruct-Q4_K_M.gguf"

# One SSH connection per host is reused (a single password prompt if there is no key).
mkdir -p "$HOME/.ssh/cm"
SSH_OPTS=(-p "$SSH_PORT" -o ConnectTimeout=15 -o ControlMaster=auto
          -o "ControlPath=$HOME/.ssh/cm/%r@%h:%p" -o ControlPersist=10m)

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

remote() {  # remote <host> <command...>
  local host=$1; shift
  ssh "${SSH_OPTS[@]}" "$SSH_USER@$host" "$@"
}

push() {  # push <host> <paths relative to the project...>: copies with tar over ssh
  local host=$1; shift
  tar -C "$PROJECT_DIR" --exclude='__pycache__' --exclude='*.crt' -czf - "$@" \
    | remote "$host" "mkdir -p $REMOTE_DIR && tar -xzf - -C $REMOTE_DIR"
}

set_remote_env() {  # set_remote_env <host> KEY=VALUE...
  # Writes the deploy values into $REMOTE_DIR/.env on the server, which docker compose always reads
  # (also with "sudo docker", which drops variables passed before the command). Other lines of that
  # file, such as ANTHROPIC_API_KEY added by hand, are kept.
  local host=$1; shift
  local keys="" lines="" kv
  for kv in "$@"; do keys="$keys|${kv%%=*}"; lines="$lines$kv\\n"; done
  remote "$host" "mkdir -p $REMOTE_DIR && cd $REMOTE_DIR && touch .env && \
    { grep -vE '^(${keys#|})=' .env || true; } > .env.tmp && printf '$lines' >> .env.tmp && mv .env.tmp .env && chmod 600 .env"
}

compose_cmd() {  # docker compose v2 or legacy docker-compose, detected on the server
  echo "if $DOCKER compose version >/dev/null 2>&1; then C='$DOCKER compose'; else C='${DOCKER/docker/docker-compose}'; fi; cd $REMOTE_DIR && \$C"
}

wait_http() {  # wait_http <host> <url seen from the host> <timeout seconds>
  local host=$1 url=$2 timeout=${3:-600} waited=0
  until remote "$host" "curl -sf '$url' >/dev/null"; do
    if [ "$waited" -ge "$timeout" ]; then
      echo "No respondió $url en ${timeout}s" >&2
      return 1
    fi
    sleep 10; waited=$((waited + 10)); printf '.'
  done
  echo " listo"
}
