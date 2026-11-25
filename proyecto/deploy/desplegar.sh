#!/usr/bin/env bash
# Copia el código, construye la imagen en el servidor y levanta la demo. También sirve para actualizar.
source "$(dirname "$0")/comun.sh"
cd "$CARPETA_PROYECTO"
mkdir -p docker/certs

remoto 'docker compose version >/dev/null 2>&1' || {
  echo "El servidor no tiene Docker. Ejecuta primero deploy/preparar_servidor.sh" >&2; exit 1; }

titulo "Copiando el código a $SSH_HOST"
# Se borra el código anterior (no los volúmenes) para que no queden archivos viejos.
# COPYFILE_DISABLE evita los archivos ._* que agrega macOS.
COPYFILE_DISABLE=1 tar --exclude='__pycache__' --exclude='*.egg-info' --exclude='*.parquet' --exclude='*.crt' \
    --exclude='.DS_Store' -czf - \
    docker docker-compose.yml .dockerignore requirements-serve.txt configs src reports/metrics \
  | remoto "mkdir -p $CARPETA_REMOTA && cd $CARPETA_REMOTA && rm -rf docker configs src reports && tar --warning=no-unknown-keyword -xzf -"

remoto "cat > $CARPETA_REMOTA/.env" <<ENV
SECOPEDIA_DIRECCION=$SECOPEDIA_DIRECCION
SECOPEDIA_PRECARGAR=$SECOPEDIA_PRECARGAR
MEMORIA_APP=$MEMORIA_APP
HILOS=$HILOS
ENV

titulo "Construyendo la imagen (la primera vez tarda varios minutos)"
compose_remoto build app

titulo "Levantando los contenedores"
compose_remoto up -d --remove-orphans
# El Caddyfile se monta como archivo: se recrea el proxy para que tome los cambios.
compose_remoto up -d --force-recreate --no-deps proxy

titulo "Esperando a la demo (la primera vez descarga unos 500 MB de modelos)"
espera=0
until [ "$(remoto "docker inspect -f '{{.State.Health.Status}}' \$(cd $CARPETA_REMOTA && docker compose ps -q app)" 2>/dev/null)" = healthy ]; do
  if [ "$espera" -ge 900 ]; then
    echo "La demo no respondió en 15 minutos. Últimos registros:" >&2
    compose_remoto logs --tail 40 app >&2
    exit 1
  fi
  sleep 10; espera=$((espera + 10)); printf '.'
done
echo " lista"

titulo "Liberando espacio de imágenes anteriores"
remoto 'docker image prune -f >/dev/null; docker builder prune -f --keep-storage 3GB >/dev/null 2>&1 || true; df -h / | tail -1'

echo
echo "Interfaz: $(url_publica)"
echo "API:      $(url_publica)/docs"
echo "Prueba:   deploy/probar.sh $(url_publica)"
