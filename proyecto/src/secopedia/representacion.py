"""Etapa 6: representación vectorial del texto.

- TF-IDF con unigramas y bigramas sobre los lemas (etapa 5).
- Embeddings fastText para la CNN; ver `modelos/red_neuronal.py`.
- BETO usa su propio tokenizador; ver `modelos/beto.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def crear_tfidf(min_df: int = 3, max_df: float = 0.5, max_features: int | None = 100_000) -> TfidfVectorizer:
    """Vectorizador TF-IDF sobre lemas ya separados por espacios.

    `sublinear_tf` atenúa las palabras repetidas en un mismo contrato.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=True,
        token_pattern=r"[^ ]+",
    )


def unir_lemas(lemas: pd.Series) -> pd.Series:
    """Convierte la lista de lemas de cada contrato en un texto separado por espacios."""
    return lemas.map(lambda lista: " ".join(lista))


def terminos_principales(vectorizador: TfidfVectorizer, texto: str, n: int = 5) -> list[tuple[str, float]]:
    """Términos con mayor peso TF-IDF en un texto (para mostrar la etapa en el documento)."""
    vector = vectorizador.transform([texto]).toarray()[0]
    vocabulario = vectorizador.get_feature_names_out()
    indices = np.argsort(vector)[::-1][:n]
    return [(vocabulario[i], round(float(vector[i]), 3)) for i in indices if vector[i] > 0]
