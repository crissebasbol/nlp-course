"""Modelo 7c: fine-tuning de BETO (BERT en español) para clasificar segmentos UNSPSC.

Recibe el texto normalizado de la etapa 4 (no los lemas): BETO tiene su propio tokenizador
de subpalabras (WordPiece) y aprovecha el contexto completo de la oración.
Se entrena localmente con scripts/entrenar_beto.py (GPU CUDA, Apple Silicon MPS o CPU).
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd

MODELO_BASE = "dccuchile/bert-base-spanish-wwm-uncased"
LONGITUD_MAXIMA = 128


def crear_dataset(datos: pd.DataFrame, tokenizador, clase_a_id: dict[str, int]):
    """Convierte un DataFrame (texto_normalizado, etiqueta) en un Dataset tokenizado de Hugging Face."""
    from datasets import Dataset

    tabla = pd.DataFrame({
        "texto": datos["texto_normalizado"].tolist(),
        "label": datos["etiqueta"].map(clase_a_id).tolist(),
    })
    dataset = Dataset.from_pandas(tabla, preserve_index=False)
    return dataset.map(
        lambda lote: tokenizador(lote["texto"], truncation=True, max_length=LONGITUD_MAXIMA),
        batched=True,
        remove_columns=["texto"],
    )


def entrenar_beto(
    entrenamiento: pd.DataFrame,
    validacion: pd.DataFrame,
    clases: list[str],
    carpeta_salida: Path | str,
    epocas: int = 3,
    tasa_aprendizaje: float = 3e-5,
    tamano_lote: int = 32,
    semilla: int = 42,
):
    """Ajusta BETO y devuelve (trainer, tokenizador). Guarda el mejor modelo según F1 macro en validación."""
    import torch
    from sklearn.metrics import f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments, set_seed)

    set_seed(semilla)
    clase_a_id = {clase: i for i, clase in enumerate(clases)}
    tokenizador = AutoTokenizer.from_pretrained(MODELO_BASE)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        MODELO_BASE,
        num_labels=len(clases),
        id2label={i: c for c, i in clase_a_id.items()},
        label2id=clase_a_id,
    )

    def metricas(prediccion):
        logits, reales = prediccion
        predichos = logits.argmax(axis=1)
        top3 = np.argsort(logits, axis=1)[:, ::-1][:, :3]
        return {
            "f1_macro": f1_score(reales, predichos, average="macro"),
            "accuracy": float((predichos == reales).mean()),
            "top3_accuracy": float((top3 == reales[:, None]).any(axis=1).mean()),
        }

    argumentos = TrainingArguments(
        output_dir=str(carpeta_salida),
        num_train_epochs=epocas,
        learning_rate=tasa_aprendizaje,
        per_device_train_batch_size=tamano_lote,
        per_device_eval_batch_size=tamano_lote * 2,
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,  # sin el estado del optimizador: ~0,45 GB por checkpoint en vez de ~1,3 GB
        # Lotes con textos de longitud parecida: menos relleno, ~1,6x más rápido (medido en un M4 Pro).
        group_by_length=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),
        logging_steps=200,
        report_to="none",
        seed=semilla,
    )
    trainer = Trainer(
        model=modelo,
        args=argumentos,
        train_dataset=crear_dataset(entrenamiento, tokenizador, clase_a_id),
        eval_dataset=crear_dataset(validacion, tokenizador, clase_a_id),
        data_collator=DataCollatorWithPadding(tokenizador),
        compute_metrics=metricas,
    )
    trainer.train()
    return trainer, tokenizador


def probabilidades_beto(ruta_o_repo: str, datos: pd.DataFrame, clases: list[str]) -> np.ndarray:
    """Probabilidades por clase, en el mismo orden de `datos`, con el modelo guardado.

    No se usa `Trainer.predict`: con `group_by_length=True` el Trainer también agrupa por
    longitud al predecir y las filas salen en otro orden que los contratos.
    """
    clasificador = ClasificadorBeto(ruta_o_repo)
    if list(clasificador.classes_) != list(clases):
        raise ValueError("Las clases del modelo no coinciden con clases.json")
    return clasificador.predict_proba(datos["texto_normalizado"].tolist())


class ClasificadorBeto:
    """Adaptador con la misma interfaz que los modelos de scikit-learn (`classes_`, `predict_proba`),
    para usar BETO en `Predictor` y en la app. Recibe textos ya normalizados."""

    def __init__(self, ruta_o_repo: str, dispositivo: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizador = AutoTokenizer.from_pretrained(ruta_o_repo)
        self.modelo = AutoModelForSequenceClassification.from_pretrained(ruta_o_repo).to(self.dispositivo).eval()
        # El tokenizador rápido no admite varios hilos a la vez ("Already borrowed").
        self._candado = threading.Lock()
        etiquetas = self.modelo.config.id2label
        self.classes_ = np.array([etiquetas[i] for i in range(len(etiquetas))])

    def predict_proba(self, textos, tamano_lote: int = 64) -> np.ndarray:
        """Procesa los textos ordenados por longitud (menos relleno) y devuelve las filas
        en el orden original."""
        import torch

        textos = list(textos)
        orden = np.argsort([len(t) for t in textos], kind="stable")
        salida = np.zeros((len(textos), len(self.classes_)), dtype=np.float32)
        with torch.no_grad():
            for inicio in range(0, len(textos), tamano_lote):
                indices = orden[inicio:inicio + tamano_lote]
                with self._candado:
                    lote = self.tokenizador([textos[i] for i in indices], truncation=True,
                                            max_length=LONGITUD_MAXIMA, padding=True, return_tensors="pt")
                logits = self.modelo(**lote.to(self.dispositivo)).logits
                salida[indices] = torch.softmax(logits, dim=1).cpu().numpy()
        return salida
