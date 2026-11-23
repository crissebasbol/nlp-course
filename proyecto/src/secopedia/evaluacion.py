"""Etapa 8: métricas de evaluación y matriz de confusión."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def exactitud_top_k(y_real: np.ndarray, probabilidades: np.ndarray, clases: list[str], k: int = 3) -> float:
    """Proporción de contratos cuya clase real está entre las k más probables."""
    indice = {clase: i for i, clase in enumerate(clases)}
    reales = np.array([indice[y] for y in y_real])
    top_k = np.argsort(probabilidades, axis=1)[:, ::-1][:, :k]
    return float((top_k == reales[:, None]).any(axis=1).mean())


def calcular_metricas(y_real, probabilidades: np.ndarray, clases: list[str]) -> dict[str, float]:
    """Accuracy, F1 macro (métrica principal), F1 ponderado y top-3 accuracy."""
    y_real = np.asarray(y_real)
    y_pred = np.asarray(clases)[probabilidades.argmax(axis=1)]
    return {
        "accuracy": round(float(accuracy_score(y_real, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_real, y_pred, average="macro", labels=clases, zero_division=0)), 4),
        "f1_ponderado": round(float(f1_score(y_real, y_pred, average="weighted", labels=clases, zero_division=0)), 4),
        "top3_accuracy": round(exactitud_top_k(y_real, probabilidades, clases, k=3), 4),
        "n": int(len(y_real)),
    }


def reporte_por_clase(y_real, y_pred, clases: list[str]) -> pd.DataFrame:
    reporte = classification_report(y_real, y_pred, labels=clases, output_dict=True, zero_division=0)
    return pd.DataFrame(reporte).T.loc[clases].round(3)


def graficar_matriz_confusion(y_real, y_pred, clases: list[str], titulo: str, destino: Path | None = None):
    """Matriz de confusión normalizada por fila (recall de cada segmento en la diagonal)."""
    matriz = confusion_matrix(y_real, y_pred, labels=clases, normalize="true")
    fig, eje = plt.subplots(figsize=(13, 11))
    sns.heatmap(matriz, xticklabels=clases, yticklabels=clases, cmap="Blues", vmin=0, vmax=1, ax=eje,
                cbar_kws={"label": "proporción de la clase real"})
    eje.set_xlabel("Segmento predicho")
    eje.set_ylabel("Segmento real")
    eje.set_title(titulo)
    plt.tight_layout()
    if destino is not None:
        fig.savefig(destino, dpi=150)
    return fig
