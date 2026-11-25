"""Etapa 5: tokenización, eliminación de stopwords y lematización con spaCy.

Se aplica al texto ya normalizado (etapa 4). Solo la usan los modelos clásicos
(TF-IDF y embeddings); BETO usa su propio tokenizador de subpalabras.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import spacy
from spacy.language import Language

from secopedia.normalizacion import TOKENS_ESPECIALES, quitar_tildes

MODELO_SPACY = "es_core_news_sm"

# Palabras que aparecen en casi todas las clases y no ayudan a distinguir el segmento.
STOPWORDS_DOMINIO = {
    "contratar", "contratacion", "contrato", "contratista", "adquisicion", "adquirir", "compra",
    "compraventa", "prestar", "prestacion", "servicio", "entidad", "municipio", "municipal",
    "departamento", "departamental", "distrito", "distrital", "objeto", "vigencia", "necesidad",
    "realizar", "requerir", "garantizar", "cumplimiento", "desarrollo", "marco", "acuerdo",
    "conformidad", "segun", "especificacion", "tecnico", "anexo", "estudio", "previo", "no",
    "proceso", "presente", "ejecutar", "ejecucion", "fin", "dar", "parte", "cargo",
}


def cargar_spacy(modelo: str = MODELO_SPACY) -> Language:
    """Carga spaCy sin los componentes que no se usan (parser y NER), para ir más rápido.

    Los tokens especiales (<AÑO>, <LUGAR>, <NUM>) se registran como excepciones del
    tokenizador para que no se partan en "<", "AÑO", ">".
    """
    nlp = spacy.load(modelo, disable=["parser", "ner"])
    for especial in TOKENS_ESPECIALES:
        nlp.tokenizer.add_special_case(especial, [{"ORTH": especial}])
    return nlp


def _es_util(token, stopwords: set[str]) -> bool:
    if token.text in TOKENS_ESPECIALES or token.is_punct or token.is_space or token.like_num:
        return False
    lema = token.lemma_.lower()
    # Solo lemas alfabéticos: descarta restos como "s.a" o "no." de nombres y abreviaturas.
    return len(lema) > 1 and lema.isalpha() and not token.is_stop and lema not in stopwords and token.lower_ not in stopwords


def lematizar(
    textos: Iterable[str],
    nlp: Language,
    stopwords_extra: set[str] = STOPWORDS_DOMINIO,
    tamano_lote: int = 1000,
    procesos: int = 1,
) -> Iterator[list[str]]:
    """Devuelve, para cada texto, la lista de lemas útiles."""
    for doc in nlp.pipe(textos, batch_size=tamano_lote, n_process=procesos):
        yield [quitar_tilde_lema(t.lemma_.lower()) for t in doc if _es_util(t, stopwords_extra)]


def quitar_tilde_lema(lema: str) -> str:
    """spaCy a veces devuelve lemas con tilde ("tecnológico"); se quitan para unificar."""
    return quitar_tildes(lema)
