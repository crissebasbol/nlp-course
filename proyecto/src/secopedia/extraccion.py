"""Etapa 1: extracción de contratos de SECOP II desde la API Socrata de datos.gov.co.

Solo se piden las columnas necesarias para la tarea (texto, código UNSPSC, tipo y
fecha), de modo que nunca se descargan columnas con datos personales. El filtro se
aplica en el servidor para no traer los ~6 millones de contratos.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

log = logging.getLogger(__name__)


def construir_filtro(tipos_excluidos: list[str], fecha_minima: str) -> str:
    """Arma la cláusula SoQL $where con los tipos excluidos y la fecha mínima."""
    condiciones = [f"tipo_de_contrato != '{tipo}'" for tipo in tipos_excluidos]
    condiciones.append(f"fecha_de_firma >= '{fecha_minima}'")
    return " AND ".join(condiciones)


def _pedir(url: str, parametros: dict[str, Any], reintentos: int, pausa: float, timeout: int) -> list[dict]:
    """Hace una petición GET con reintentos y espera creciente."""
    for intento in range(1, reintentos + 1):
        try:
            respuesta = requests.get(url, params=parametros, timeout=timeout)
            respuesta.raise_for_status()
            return respuesta.json()
        except (requests.RequestException, ValueError) as error:
            if intento == reintentos:
                raise
            espera = pausa * intento
            log.warning("Fallo en intento %d (%s). Reintentando en %.0f s", intento, error, espera)
            time.sleep(espera)
    return []


def contar_registros(config: dict[str, Any]) -> int:
    """Cuenta cuántos contratos cumplen el filtro, sin descargarlos."""
    filtro = construir_filtro(config["tipos_excluidos"], config["fecha_minima"])
    datos = _pedir(
        config["url"],
        {"$select": "count(*)", "$where": filtro},
        config["reintentos"],
        config["pausa_segundos"],
        config["timeout_segundos"],
    )
    return int(datos[0]["count"])


def descargar_contratos(config: dict[str, Any], carpeta_paginas: Path, limite: int | None = None) -> pd.DataFrame:
    """Descarga los contratos página por página y guarda cada página como punto de control.

    Si el proceso se interrumpe, al volver a ejecutarlo se reutilizan las páginas ya
    guardadas. `limite` permite bajar solo una muestra para pruebas.
    """
    carpeta_paginas.mkdir(parents=True, exist_ok=True)
    filtro = construir_filtro(config["tipos_excluidos"], config["fecha_minima"])
    total = contar_registros(config)
    if limite is not None:
        total = min(total, limite)
    tamano = min(config["tamano_pagina"], total)
    log.info("Contratos a descargar: %d en páginas de %d", total, tamano)

    paginas = []
    for desplazamiento in tqdm(range(0, total, tamano), desc="Páginas"):
        archivo = carpeta_paginas / f"pagina_{desplazamiento:08d}.parquet"
        if archivo.exists():
            paginas.append(pd.read_parquet(archivo))
            continue
        filas = _pedir(
            config["url"],
            {
                "$select": ",".join(config["columnas"]),
                "$where": filtro,
                "$order": ":id",
                "$limit": min(tamano, total - desplazamiento),
                "$offset": desplazamiento,
            },
            config["reintentos"],
            config["pausa_segundos"],
            config["timeout_segundos"],
        )
        pagina = pd.DataFrame(filas, columns=config["columnas"])
        pagina.to_parquet(archivo, index=False)
        paginas.append(pagina)

    contratos = pd.concat(paginas, ignore_index=True)
    contratos["fecha_de_firma"] = pd.to_datetime(contratos["fecha_de_firma"], errors="coerce")
    contratos.attrs["fecha_descarga"] = datetime.now(timezone.utc).isoformat()
    return contratos
