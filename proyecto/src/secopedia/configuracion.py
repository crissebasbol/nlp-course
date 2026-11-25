"""Carga de la configuración y utilidades de rutas compartidas."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_CONFIG = RAIZ_PROYECTO / "configs" / "config.yaml"


def cargar_config(ruta: Path | str = RUTA_CONFIG) -> dict[str, Any]:
    """Lee el archivo YAML de configuración."""
    with open(ruta, encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


def ruta(relativa: str) -> Path:
    """Convierte una ruta relativa a proyecto/ en absoluta y crea su carpeta padre."""
    absoluta = RAIZ_PROYECTO / relativa
    absoluta.parent.mkdir(parents=True, exist_ok=True)
    return absoluta


def local_o_hub(relativa: str, en_hub: str) -> Path:
    """Ruta del archivo en proyecto/ si existe; si no, lo descarga del repositorio de Hugging Face.

    Así la demo funciona recién clonado el repositorio, sin descargar datos ni entrenar.
    """
    local = RAIZ_PROYECTO / relativa
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=cargar_config()["rutas"]["hub"], filename=en_hub))


def fijar_semilla(semilla: int) -> None:
    """Fija la semilla de las librerías de azar para que los resultados se repitan."""
    random.seed(semilla)
    np.random.seed(semilla)
