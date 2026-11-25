#!/usr/bin/env bash
# Prueba una demo en marcha: estado, los cuatro modelos y una auditoría de la muestra.
#   deploy/probar.sh http://localhost        (por defecto)
#   deploy/probar.sh https://mi-dominio.com
set -euo pipefail
URL="${1:-http://localhost}"; URL="${URL%/}"

pedir() { curl -fsS --max-time 300 "$@"; }

pedir "$URL/" | grep -q SECOPEDIA && echo "ok  interfaz"
pedir "$URL/api/estado" | grep -q '"disponible":true' && echo "ok  /api/estado"
for modelo in regresion_logistica svm_lineal red_neuronal beto; do
  respuesta=$(pedir -X POST "$URL/api/predecir" -H 'Content-Type: application/json' \
    -d "{\"texto\":\"compra de computadores portatiles\",\"modelo\":\"$modelo\"}" -w ' %{time_total}')
  echo "$respuesta" | grep -q '"sugerencias":\[{"segmento":"43"' || { echo "FALLO $modelo: $respuesta" >&2; exit 1; }
  echo "ok  $modelo sugiere 43 (${respuesta##* } s)"
done
for modelo in regresion_logistica beto; do
  tiempo=$(pedir -o /dev/null -w '%{time_total}' -F usar_muestra=true -F "modelo=$modelo" "$URL/api/auditar")
  echo "ok  auditar 300 contratos con $modelo ($tiempo s)"
done
echo "Todo funciona en $URL"
