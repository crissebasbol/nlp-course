#!/usr/bin/env bash
# Checks both machines: Docker, Compose, disk, RAM, CPU, internet access and used ports.
source "$(dirname "$0")/common.sh"

for host in "$M1_HOST" "$M2_HOST"; do
  log "Máquina $host"
  remote "$host" "bash -s" <<REMOTE
echo "Hostname: \$(hostname)"
echo "CPU: \$(nproc) núcleos | AVX2: \$(grep -q avx2 /proc/cpuinfo && echo sí || echo no)"
free -m | awk 'NR==2{print "RAM disponible: "\$7" MB de "\$2" MB"}'
df -h ~ | awk 'NR==2{print "Disco libre en home: "\$4" de "\$2}'
$DOCKER --version 2>/dev/null || echo "Docker: NO encontrado"
($DOCKER compose version 2>/dev/null || docker-compose --version 2>/dev/null) || echo "Compose: NO encontrado"
$DOCKER ps >/dev/null 2>&1 && echo "Permisos Docker: ok" || echo "Permisos Docker: falla (prueba DOCKER=\"sudo docker\")"
curl -sI --max-time 10 https://huggingface.co | head -1 | sed 's/^/Internet (huggingface.co): /' || echo "Internet: sin acceso"
echo "Puertos escuchando en 8000-8090:"; (ss -ltn 2>/dev/null || netstat -ltn) | awk '{print \$4}' | grep -E ':80[0-9]{2}$' || echo "  ninguno"
REMOTE
done
