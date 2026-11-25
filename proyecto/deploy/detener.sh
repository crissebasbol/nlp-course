#!/usr/bin/env bash
# Detiene la demo. Conserva los modelos descargados y los certificados.
# No apaga la instancia: eso se hace en la consola de AWS.
source "$(dirname "$0")/comun.sh"
compose_remoto down
