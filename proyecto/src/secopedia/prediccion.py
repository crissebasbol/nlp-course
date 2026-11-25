"""Etapa 9: inferencia. Aplica exactamente las mismas etapas 4 a 6 del entrenamiento.

Uso:
    from secopedia.prediccion import Predictor
    predictor = Predictor.linea_base()
    predictor.predecir("compra de computadores portátiles para la secretaría de educación")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from secopedia.configuracion import RAIZ_PROYECTO, cargar_config, local_o_hub
from secopedia.normalizacion import Normalizador
from secopedia.representacion import unir_lemas
from secopedia.segmentos import nombre_segmento
from secopedia.tokenizacion import cargar_spacy, lematizar


class AjusteDistribucion:
    """Corrige el cambio de distribución entre entrenamiento y uso real.

    Los modelos se entrenaron con un tope de contratos por clase (clases casi balanceadas), pero
    en la realidad unos segmentos son mucho más frecuentes que otros. Por la regla de Bayes, la
    probabilidad corregida es proporcional a p(clase | texto) * p_real(clase) / p_entrenamiento(clase).
    En la prueba temporal (2026) esto sube el F1 macro de BETO de 0,556 a 0,591.
    """

    def __init__(self, clases: list[str], p_real: list[float], p_entrenamiento: list[float]):
        real = np.asarray(p_real, dtype=float)
        entrenamiento = np.asarray(p_entrenamiento, dtype=float)
        self.clases = list(clases)
        self.factores = np.divide(real, entrenamiento, out=np.ones_like(real), where=entrenamiento > 0)

    @classmethod
    def desde_archivo(cls, ruta: Path | str) -> "AjusteDistribucion":
        import json

        with open(ruta, encoding="utf-8") as archivo:
            datos = json.load(archivo)
        return cls(datos["clases"], datos["p_real"], datos["p_entrenamiento"])

    def aplicar(self, probabilidades: np.ndarray, clases_modelo: list[str]) -> np.ndarray:
        """Ajusta y renormaliza. Reordena los factores según el orden de clases del modelo."""
        posicion = {clase: i for i, clase in enumerate(self.clases)}
        factores = np.array([self.factores[posicion[c]] for c in clases_modelo])
        ajustadas = probabilidades * factores
        return ajustadas / ajustadas.sum(axis=1, keepdims=True)


def _ajuste_configurado() -> AjusteDistribucion | None:
    config = cargar_config()
    ruta = RAIZ_PROYECTO / config["rutas"]["ajuste_distribucion"]
    if config.get("prediccion", {}).get("ajustar_distribucion", False) and ruta.exists():
        return AjusteDistribucion.desde_archivo(ruta)
    return None


def _normalizador() -> Normalizador:
    return Normalizador.desde_archivo(local_o_hub(cargar_config()["rutas"]["lugares"], "recursos/lugares.json"))


@dataclass
class Sugerencia:
    segmento: str
    nombre: str
    probabilidad: float


class Predictor:
    """Envuelve un modelo entrenado con su preprocesamiento."""

    def __init__(self, modelo, normalizador: Normalizador, usa_lemas: bool = True,
                 ajuste: AjusteDistribucion | None = None):
        self.modelo = modelo
        self.ajuste = ajuste
        self.normalizador = normalizador
        self.usa_lemas = usa_lemas
        self.nlp = cargar_spacy() if usa_lemas else None
        self.clases = list(modelo.classes_)

    @classmethod
    def linea_base(cls, nombre: str = "regresion_logistica") -> "Predictor":
        carpeta = cargar_config()["rutas"]["modelos"]
        ruta_modelo = local_o_hub(f"{carpeta}/{nombre}.joblib", f"clasicos/{nombre}.joblib")
        return cls(joblib.load(ruta_modelo), _normalizador(), ajuste=_ajuste_configurado())

    @classmethod
    def red_neuronal(cls) -> "Predictor":
        """CNN sobre embeddings fastText (modelo 7b); usa los mismos lemas que la línea base."""
        from secopedia.modelos.red_neuronal import ClasificadorCNN

        carpeta = cargar_config()["rutas"]["modelos"]
        ruta_modelo = local_o_hub(f"{carpeta}/red_neuronal/cnn.pt", "clasicos/cnn.pt")
        return cls(ClasificadorCNN.cargar(ruta_modelo), _normalizador(), ajuste=_ajuste_configurado())

    @classmethod
    def beto(cls, ruta_o_repo: str) -> "Predictor":
        """BETO afinado, desde una carpeta local o un repositorio de Hugging Face Hub."""
        from secopedia.modelos.beto import ClasificadorBeto

        return cls(ClasificadorBeto(ruta_o_repo), _normalizador(), usa_lemas=False, ajuste=_ajuste_configurado())

    def _preparar(self, textos: list[str]) -> pd.Series:
        normalizados = [self.normalizador(t) for t in textos]
        if not self.usa_lemas:
            return pd.Series(normalizados)
        return unir_lemas(pd.Series(list(lematizar(normalizados, self.nlp))))

    def probabilidades(self, textos: list[str]) -> np.ndarray:
        probabilidades = self.modelo.predict_proba(self._preparar(textos))
        if self.ajuste is not None:
            probabilidades = self.ajuste.aplicar(probabilidades, self.clases)
        return probabilidades

    def predecir(self, texto: str, k: int = 3) -> list[Sugerencia]:
        """Las k categorías más probables para un objeto de contrato."""
        probabilidades = self.probabilidades([texto])[0]
        mejores = np.argsort(probabilidades)[::-1][:k]
        return [
            Sugerencia(self.clases[i], nombre_segmento(self.clases[i]), float(probabilidades[i]))
            for i in mejores
        ]

    def auditar(self, contratos: pd.DataFrame, umbral: float = 0.8,
                columna_texto: str = "objeto_del_contrato",
                columna_codigo: str = "codigo_de_categoria_principal") -> pd.DataFrame:
        """Marca los contratos donde el modelo está muy seguro de un segmento distinto al registrado.

        Devuelve todos los contratos con el segmento registrado, el sugerido, su probabilidad
        y la columna `sospechoso`.
        """
        probabilidades = self.probabilidades(contratos[columna_texto].fillna("").astype(str).tolist())
        indice = probabilidades.argmax(axis=1)
        resultado = contratos.copy()
        segmento = resultado[columna_codigo].astype(str).str.replace("V1.", "", regex=False).str[:2]
        resultado["segmento_registrado"] = segmento.where(segmento.isin(self.clases), "otros")
        resultado["segmento_sugerido"] = np.asarray(self.clases)[indice]
        resultado["nombre_sugerido"] = resultado["segmento_sugerido"].map(nombre_segmento)
        resultado["probabilidad"] = probabilidades.max(axis=1).round(3)
        # Probabilidad que el modelo le da al segmento que la entidad registró.
        posicion = {clase: i for i, clase in enumerate(self.clases)}
        columnas = resultado["segmento_registrado"].map(posicion)
        conocida = columnas.notna().to_numpy()
        prob_registrado = np.zeros(len(resultado))
        prob_registrado[conocida] = probabilidades[np.flatnonzero(conocida), columnas[conocida].astype(int)]
        resultado["prob_registrado"] = prob_registrado.round(3)
        resultado["sospechoso"] = (
            (resultado["segmento_sugerido"] != resultado["segmento_registrado"])
            & (resultado["probabilidad"] >= umbral)
        )
        return resultado.sort_values(["sospechoso", "probabilidad"], ascending=False)
