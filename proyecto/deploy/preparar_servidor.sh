#!/usr/bin/env bash
# Se ejecuta una sola vez en un servidor Ubuntu 24.04 recién creado, con un usuario que tenga sudo.
# Instala Docker con Compose y crea 2 GB de swap. Repetirlo no daña nada.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# En una máquina nueva, cloud-init puede estar usando apt todavía.
if command -v cloud-init >/dev/null; then sudo cloud-init status --wait >/dev/null || true; fi
sudo apt-get -o DPkg::Lock::Timeout=300 update -q
sudo apt-get -o DPkg::Lock::Timeout=300 install -yq docker.io docker-compose-v2 docker-buildx git curl
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

# Swap: margen para construir la imagen y para picos de memoria. Si no se puede, se sigue sin él.
activar_swap() {
  sudo swapon --show | grep -q /swapfile && return
  [ -f /swapfile ] || { sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null; }
  sudo swapon /swapfile || return
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
}
activar_swap || echo "Aviso: no se pudo activar el swap"

docker --version && sudo docker compose version && free -h && df -h / | tail -1
echo "Servidor listo. Sigue: deploy/desplegar.sh"
