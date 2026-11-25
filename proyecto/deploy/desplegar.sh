#!/usr/bin/env bash
# Despliega o actualiza la demo en la máquina donde se ejecuta.
#   deploy/desplegar.sh             git pull, construir y levantar
#   deploy/desplegar.sh --sin-pull  usa el código tal como está
source "$(dirname "$0")/comun.sh"

# Todo va en una función: git pull puede reemplazar este archivo mientras se ejecuta.
principal() {
  if [ "${1:-}" != "--sin-pull" ]; then
    titulo "Actualizando el código"
    git pull --ff-only
    # Se continúa con la versión recién descargada de este script.
    exec "$PWD/deploy/desplegar.sh" --sin-pull
  fi

  [ -f .env ] || { cp .env.example .env; echo "Se creó .env con los valores por defecto"; }

  titulo "Construyendo la imagen (la primera vez tarda varios minutos)"
  compose build app

  titulo "Levantando los contenedores"
  compose up -d --remove-orphans
  # El Caddyfile se monta como archivo: se recrea el proxy para que tome los cambios.
  compose up -d --force-recreate --no-deps proxy

  titulo "Esperando a la demo (la primera vez descarga unos 500 MB de modelos)"
  local espera=0
  until [ "$("${DOCKER[@]}" inspect -f '{{.State.Health.Status}}' "$(compose ps -q app)" 2>/dev/null)" = healthy ]; do
    if [ "$espera" -ge 900 ]; then
      echo "La demo no respondió en 15 minutos. Últimos registros:" >&2
      compose logs --tail 40 app >&2
      exit 1
    fi
    sleep 10; espera=$((espera + 10)); printf '.'
  done
  echo " lista"

  titulo "Liberando espacio de imágenes anteriores"
  "${DOCKER[@]}" image prune -f >/dev/null
  df -h / | tail -1

  local puerto
  puerto="$(compose port proxy 80 | sed 's/.*://')"
  echo
  echo "Demo lista en el puerto $puerto. Prueba: deploy/probar.sh http://localhost:$puerto"
}

principal "$@"
exit
