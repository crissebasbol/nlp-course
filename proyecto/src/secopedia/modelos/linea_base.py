"""Modelo 7a: línea base TF-IDF + Regresión Logística o SVM lineal."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from secopedia.representacion import crear_tfidf


def crear_regresion_logistica(c: float = 4.0, semilla: int = 42) -> Pipeline:
    return Pipeline([
        ("tfidf", crear_tfidf()),
        ("clasificador", LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=semilla)),
    ])


def crear_svm(c: float = 0.5, semilla: int = 42) -> Pipeline:
    """SVM lineal calibrado: la calibración convierte sus puntajes en probabilidades para el top 3."""
    svm = LinearSVC(C=c, class_weight="balanced", random_state=semilla)
    return Pipeline([
        ("tfidf", crear_tfidf()),
        ("clasificador", CalibratedClassifierCV(svm, cv=3, method="sigmoid")),
    ])


def terminos_por_clase(modelo: Pipeline, n: int = 10) -> dict[str, list[str]]:
    """Términos con mayor peso en cada clase (solo para regresión logística): interpretabilidad."""
    vocabulario = modelo.named_steps["tfidf"].get_feature_names_out()
    clasificador = modelo.named_steps["clasificador"]
    return {
        clase: [vocabulario[i] for i in pesos.argsort()[::-1][:n]]
        for clase, pesos in zip(clasificador.classes_, clasificador.coef_)
    }
