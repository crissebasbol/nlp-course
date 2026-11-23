"""Etapa 2: limpieza, filtrado y deduplicación de contratos."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


def clave_texto(texto: str) -> str:
    """Normalización mínima usada solo para detectar duplicados.

    Pasa a minúsculas, quita tildes y signos y colapsa espacios.
    """
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9ñ ]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


@dataclass
class ReporteLimpieza:
    """Conteo de registros eliminados en cada paso, para documentar la etapa."""

    pasos: list[tuple[str, int, int]] = field(default_factory=list)

    def registrar(self, paso: str, antes: int, despues: int) -> None:
        self.pasos.append((paso, antes, despues))

    def como_tabla(self) -> pd.DataFrame:
        tabla = pd.DataFrame(self.pasos, columns=["paso", "antes", "despues"])
        tabla["eliminados"] = tabla["antes"] - tabla["despues"]
        tabla["pct_eliminado"] = (100 * tabla["eliminados"] / tabla["antes"]).round(2)
        return tabla


def limpiar_contratos(
    contratos: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, ReporteLimpieza]:
    """Aplica los filtros de calidad y devuelve (limpios, conflictos, reporte).

    `conflictos` son textos idénticos registrados con segmentos distintos: se guardan
    aparte como evidencia del ruido en las etiquetas.
    """
    reporte = ReporteLimpieza()
    datos = contratos.copy()

    # 1. Texto vacío o sin contenido útil.
    antes = len(datos)
    datos["objeto_del_contrato"] = datos["objeto_del_contrato"].fillna("").astype(str).str.strip()
    datos["clave_texto"] = datos["objeto_del_contrato"].map(clave_texto)
    vacios = set(config["textos_vacios"])
    num_palabras = datos["clave_texto"].str.split().str.len().fillna(0)
    datos = datos[~datos["clave_texto"].isin(vacios) & (num_palabras >= config["min_palabras"])]
    reporte.registrar(f"texto vacío o con menos de {config['min_palabras']} palabras", antes, len(datos))

    # 2. Código UNSPSC inválido (UNSPECIFIED, PE..., vacío).
    antes = len(datos)
    codigos = datos["codigo_de_categoria_principal"].fillna("").astype(str).str.strip()
    datos = datos[codigos.str.match(config["patron_codigo"])]
    reporte.registrar("código UNSPSC inválido", antes, len(datos))

    # 2b. Códigos "por defecto" que no describen el contenido (p. ej. 801116, personal temporal).
    prefijos = tuple(config.get("prefijos_codigo_excluidos", []))
    if prefijos:
        antes = len(datos)
        datos = datos[~datos["codigo_de_categoria_principal"].str.startswith(prefijos)]
        reporte.registrar(f"código por defecto ({', '.join(prefijos)})", antes, len(datos))

    # 3. Fecha de firma inválida.
    antes = len(datos)
    datos = datos[datos["fecha_de_firma"].notna()]
    reporte.registrar("fecha de firma inválida", antes, len(datos))

    # 4. Duplicados exactos de (texto, código). Se conserva el más antiguo.
    antes = len(datos)
    datos = datos.sort_values("fecha_de_firma").drop_duplicates(
        subset=["clave_texto", "codigo_de_categoria_principal"], keep="first"
    )
    reporte.registrar("duplicados de texto y código", antes, len(datos))

    # 5. Mismo texto con segmentos distintos: se guarda como evidencia y se deja una sola fila
    #    con el segmento más frecuente para ese texto (si hay empate, el más antiguo).
    datos["segmento"] = datos["codigo_de_categoria_principal"].str[3:5]
    segmentos_por_texto = datos.groupby("clave_texto")["segmento"].nunique()
    textos_conflicto = segmentos_por_texto[segmentos_por_texto > 1].index
    conflictos = datos[datos["clave_texto"].isin(textos_conflicto)].sort_values("clave_texto")

    antes = len(datos)
    frecuencia = datos.groupby(["clave_texto", "segmento"])["segmento"].transform("size")
    datos = (
        datos.assign(_frecuencia=frecuencia)
        .sort_values(["_frecuencia", "fecha_de_firma"], ascending=[False, True])
        .drop_duplicates(subset="clave_texto", keep="first")
        .drop(columns="_frecuencia")
        .sort_values("fecha_de_firma")
    )
    reporte.registrar("mismo texto con varios códigos (se deja uno)", antes, len(datos))

    return datos.reset_index(drop=True), conflictos.reset_index(drop=True), reporte
