#!/usr/bin/env bash
# Detiene la demo. Conserva los modelos descargados y los certificados. No apaga la máquina.
source "$(dirname "$0")/comun.sh"
compose down
