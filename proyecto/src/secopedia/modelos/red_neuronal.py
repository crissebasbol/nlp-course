"""Modelo 7b: red neuronal convolucional (CNN 1D) sobre embeddings fastText.

1. Se entrenan embeddings fastText propios (gensim) sobre los lemas de contratos que NO
   están en validación ni prueba. fastText arma cada vector con subpalabras, así que también
   representa variantes y errores de ortografía.
2. La CNN recibe la secuencia de vectores de los lemas de un contrato. Filtros de ancho 2, 3
   y 4 detectan frases cortas ("malla vial", "poliza manejo"); un max-pooling se queda con la
   activación más fuerte de cada filtro y una capa lineal da la probabilidad de cada segmento.
3. La capa de embeddings se inicializa con fastText y se ajusta durante el entrenamiento.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

RELLENO, DESCONOCIDA = 0, 1
LONGITUD_MAXIMA = 40  # el 99% de los contratos tiene 35 lemas o menos


def entrenar_fasttext(documentos: list[list[str]], dimension: int = 100, epocas: int = 10, semilla: int = 42):
    """Embeddings fastText (skip-gram con subpalabras de 3 a 6 caracteres)."""
    from gensim.models import FastText

    return FastText(
        sentences=documentos, vector_size=dimension, window=5, min_count=2, sg=1,
        min_n=3, max_n=6, epochs=epocas, workers=4, seed=semilla,
    )


def construir_vocabulario(documentos: list[list[str]], min_frecuencia: int = 2) -> dict[str, int]:
    """Índice de cada lema; 0 es relleno y 1 es palabra desconocida."""
    from collections import Counter

    conteo = Counter(lema for documento in documentos for lema in documento)
    palabras = sorted(p for p, n in conteo.items() if n >= min_frecuencia)
    return {palabra: i + 2 for i, palabra in enumerate(palabras)}


def matriz_embeddings(vocabulario: dict[str, int], fasttext) -> np.ndarray:
    """Pesos iniciales de la capa de embeddings a partir de fastText."""
    dimension = fasttext.wv.vector_size
    matriz = np.zeros((len(vocabulario) + 2, dimension), dtype=np.float32)
    for palabra, indice in vocabulario.items():
        matriz[indice] = fasttext.wv[palabra]
    return matriz


def codificar(documentos, vocabulario: dict[str, int], longitud: int = LONGITUD_MAXIMA) -> np.ndarray:
    """Convierte listas de lemas en una matriz de índices con relleno a la derecha."""
    salida = np.full((len(documentos), longitud), RELLENO, dtype=np.int64)
    for fila, documento in enumerate(documentos):
        indices = [vocabulario.get(lema, DESCONOCIDA) for lema in documento[:longitud]]
        salida[fila, :len(indices)] = indices
    return salida


class RedConvolucional(nn.Module):
    def __init__(self, embeddings: np.ndarray, num_clases: int, filtros: int = 128,
                 anchos: tuple[int, ...] = (2, 3, 4), abandono: float = 0.4):
        super().__init__()
        self.hiperparametros = {"filtros": filtros, "anchos": list(anchos), "abandono": abandono}
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embeddings), freeze=False, padding_idx=RELLENO)
        dimension = embeddings.shape[1]
        self.convoluciones = nn.ModuleList(nn.Conv1d(dimension, filtros, ancho, padding=ancho - 1) for ancho in anchos)
        self.abandono = nn.Dropout(abandono)
        self.salida = nn.Linear(filtros * len(anchos), num_clases)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        x = self.embedding(indices).transpose(1, 2)  # (lote, dimension, longitud)
        mascara = (indices != RELLENO).unsqueeze(1)  # para que el relleno no gane el max-pooling
        agrupados = []
        for conv in self.convoluciones:
            activacion = torch.relu(conv(x))[:, :, : indices.shape[1]]
            activacion = activacion.masked_fill(~mascara, 0.0)
            agrupados.append(activacion.max(dim=2).values)
        return self.salida(self.abandono(torch.cat(agrupados, dim=1)))


def dispositivo() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _probabilidades(modelo: nn.Module, x: np.ndarray, dispositivo_: str, tamano_lote: int = 1024) -> np.ndarray:
    modelo.eval()
    salidas = []
    with torch.no_grad():
        for inicio in range(0, len(x), tamano_lote):
            lote = torch.from_numpy(x[inicio:inicio + tamano_lote]).to(dispositivo_)
            salidas.append(torch.softmax(modelo(lote), dim=1).cpu().numpy())
    return np.vstack(salidas)


def entrenar_cnn(x_entrenamiento, y_entrenamiento, x_validacion, y_validacion, embeddings: np.ndarray,
                 num_clases: int, epocas: int = 15, paciencia: int = 3, tamano_lote: int = 256,
                 tasa_aprendizaje: float = 1e-3, semilla: int = 42):
    """Entrena con Adam y parada temprana según F1 macro en validación. Devuelve (modelo, historial)."""
    torch.manual_seed(semilla)
    dispositivo_ = dispositivo()
    modelo = RedConvolucional(embeddings, num_clases).to(dispositivo_)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)
    perdida = nn.CrossEntropyLoss()
    generador = np.random.default_rng(semilla)

    mejor_f1, mejor_estado, sin_mejora, historial = -1.0, None, 0, []
    for epoca in range(1, epocas + 1):
        inicio = time.time()
        modelo.train()
        orden = generador.permutation(len(x_entrenamiento))
        perdida_total = 0.0
        for desde in range(0, len(orden), tamano_lote):
            indices = orden[desde:desde + tamano_lote]
            lote = torch.from_numpy(x_entrenamiento[indices]).to(dispositivo_)
            etiquetas = torch.from_numpy(y_entrenamiento[indices]).to(dispositivo_)
            optimizador.zero_grad()
            valor = perdida(modelo(lote), etiquetas)
            valor.backward()
            optimizador.step()
            perdida_total += valor.item() * len(indices)

        predichos = _probabilidades(modelo, x_validacion, dispositivo_).argmax(axis=1)
        f1 = f1_score(y_validacion, predichos, average="macro")
        historial.append({"epoca": epoca, "perdida": round(perdida_total / len(orden), 4),
                          "f1_macro_validacion": round(float(f1), 4), "segundos": round(time.time() - inicio, 1)})
        print(historial[-1])
        if f1 > mejor_f1:
            mejor_f1, mejor_estado, sin_mejora = f1, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    modelo.load_state_dict(mejor_estado)
    return modelo.cpu(), historial


class ClasificadorCNN:
    """Interfaz tipo scikit-learn (`classes_`, `predict_proba`) para usar la CNN en `Predictor`.

    `predict_proba` recibe textos de lemas separados por espacios (salida de la etapa 5).
    """

    def __init__(self, modelo: RedConvolucional, vocabulario: dict[str, int], clases: list[str]):
        self.modelo = modelo
        self.vocabulario = vocabulario
        self.classes_ = np.array(clases)
        self.dispositivo = dispositivo()
        self.modelo.to(self.dispositivo)

    def predict_proba(self, textos) -> np.ndarray:
        documentos = [str(t).split() for t in textos]
        return _probabilidades(self.modelo, codificar(documentos, self.vocabulario), self.dispositivo)

    def guardar(self, ruta: Path | str) -> None:
        self.modelo.cpu()
        torch.save({
            "estado": self.modelo.state_dict(),
            "vocabulario": self.vocabulario,
            "clases": list(self.classes_),
            "dimension": self.modelo.embedding.embedding_dim,
            "filas_embedding": self.modelo.embedding.num_embeddings,
            "hiperparametros": self.modelo.hiperparametros,
        }, ruta)
        self.modelo.to(self.dispositivo)

    @classmethod
    def cargar(cls, ruta: Path | str) -> "ClasificadorCNN":
        datos = torch.load(ruta, map_location="cpu", weights_only=False)
        embeddings = np.zeros((datos["filas_embedding"], datos["dimension"]), dtype=np.float32)
        # Archivos guardados antes de registrar la arquitectura usan los valores por defecto.
        hiperparametros = datos.get("hiperparametros", {})
        if "anchos" in hiperparametros:
            hiperparametros = {**hiperparametros, "anchos": tuple(hiperparametros["anchos"])}
        modelo = RedConvolucional(embeddings, len(datos["clases"]), **hiperparametros)
        modelo.load_state_dict(datos["estado"])
        return cls(modelo, datos["vocabulario"], datos["clases"])
