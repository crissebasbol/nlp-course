"""Etapa 4: normalización del texto del objeto del contrato.

Pasos, en orden:
1. Minúsculas y eliminación de tildes ("ADQUISICIÓN" -> "adquisicion").
2. Separadores raros (";", "|", "/", "_" y guiones sueltos) a espacios.
3. Años -> <AÑO> y demás números -> <NUM>.
4. Municipios y departamentos -> <LUGAR>, usando la lista DIVIPOLA:
   - después de palabras de contexto ("municipio de", "departamento del", ...) se reemplaza
     cualquier lugar, aunque su nombre sea también una palabra común;
   - en cualquier otra posición solo se reemplazan los nombres "seguros", es decir, los que
     no son palabras comunes del dominio (por ejemplo "pasto", "pesca" o "plato" sí lo son).
5. Colapsar espacios.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

TOKEN_ANIO = "<AÑO>"
TOKEN_NUMERO = "<NUM>"
TOKEN_LUGAR = "<LUGAR>"
TOKENS_ESPECIALES = (TOKEN_ANIO, TOKEN_NUMERO, TOKEN_LUGAR)

# Palabras que anuncian que lo siguiente es un lugar.
PATRON_CONTEXTO = (
    r"(?:municipios?|municipal(?:es)?|departamentos?|departamental(?:es)?|distritos?|distrital(?:es)?"
    r"|ciudad(?:es)?|corregimientos?|veredas?|alcaldias?|gobernacion(?:es)?)"
    r"(?: especial| capital| turistico| portuario| industrial| y portuario| y cultural| biodiverso)*"
    r" (?:de |del |de la |de los |de el )?"
)

PATRON_ANIO = re.compile(r"\b(?:19[5-9]\d|20[0-4]\d)\b")
PATRON_NUMERO = re.compile(r"\b\d[\d.,]*\b|\b[a-z]*\d+[a-z\d-]*\b")
PATRON_SEPARADORES = re.compile(r"[;|/_\\]+")
PATRON_GUIONES_SUELTOS = re.compile(r"(?<![\w>])-+|-+(?![\w<])")
PATRON_ESPACIOS = re.compile(r"\s+")


def quitar_tildes(texto: str) -> str:
    """Elimina tildes y diéresis, pero conserva la ñ."""
    texto = texto.replace("ñ", "\0").replace("Ñ", "\1")
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto.replace("\0", "ñ").replace("\1", "Ñ")


def normalizar_nombre_lugar(nombre: str) -> str:
    """Forma canónica de un nombre de lugar para compararlo con el texto normalizado."""
    nombre = quitar_tildes(nombre.lower())
    nombre = re.sub(r"\(.*?\)", " ", nombre)  # "MARIQUITA (SAN SEBASTIÁN DE)" -> "mariquita"
    nombre = re.sub(r"[^a-zñ ]+", " ", nombre)
    return PATRON_ESPACIOS.sub(" ", nombre).strip()


def _patron_alternativas(nombres: set[str]) -> re.Pattern | None:
    """Une los nombres en una sola expresión regular; los más largos primero."""
    if not nombres:
        return None
    ordenados = sorted(nombres, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in ordenados) + r")\b")


@dataclass
class Normalizador:
    """Normaliza textos de contratos. `lugares` son todos los nombres DIVIPOLA y
    `lugares_ambiguos` los que también son palabras comunes (solo se reemplazan con contexto)."""

    lugares: set[str] = field(default_factory=set)
    lugares_ambiguos: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._patron_libres = _patron_alternativas(self.lugares - self.lugares_ambiguos)
        self._patron_contexto = None
        if self.lugares:
            nombres = "|".join(re.escape(n) for n in sorted(self.lugares, key=len, reverse=True))
            self._patron_contexto = re.compile(r"\b(" + PATRON_CONTEXTO + r")(?:" + nombres + r")\b")

    @classmethod
    def desde_archivo(cls, ruta: Path | str) -> "Normalizador":
        """Carga el léxico de lugares generado por scripts/construir_lugares.py."""
        with open(ruta, encoding="utf-8") as archivo:
            lexico = json.load(archivo)
        return cls(set(lexico["lugares"]), set(lexico["ambiguos"]))

    def normalizar(self, texto: str) -> str:
        texto = quitar_tildes(str(texto).lower())
        texto = PATRON_SEPARADORES.sub(" ", texto)
        texto = PATRON_ANIO.sub(f" {TOKEN_ANIO} ", texto)
        texto = PATRON_NUMERO.sub(f" {TOKEN_NUMERO} ", texto)
        texto = PATRON_GUIONES_SUELTOS.sub(" ", texto)
        texto = PATRON_ESPACIOS.sub(" ", texto)
        if self._patron_contexto is not None:
            texto = self._patron_contexto.sub(lambda m: f"{m.group(1)}{TOKEN_LUGAR}", texto)
        if self._patron_libres is not None:
            texto = self._patron_libres.sub(TOKEN_LUGAR, texto)
        # Lugares compuestos que quedan seguidos ("<LUGAR> <LUGAR>") se unen en uno solo.
        texto = re.sub(rf"(?:{TOKEN_LUGAR}[ ,]*)+", f"{TOKEN_LUGAR} ", texto)
        return PATRON_ESPACIOS.sub(" ", texto).strip()

    def __call__(self, texto: str) -> str:
        return self.normalizar(texto)
