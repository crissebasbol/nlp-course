"""Pruebas de la etapa 8 (métricas) y 9 (predicción) con un modelo pequeño entrenado al vuelo."""

import numpy as np
import pandas as pd
import pytest

from secopedia.evaluacion import calcular_metricas, exactitud_top_k
from secopedia.modelos.linea_base import crear_regresion_logistica
from secopedia.normalizacion import Normalizador
from secopedia.prediccion import Predictor


def test_exactitud_top_k():
    clases = ["a", "b", "c"]
    probabilidades = np.array([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7], [0.5, 0.4, 0.1]])
    assert exactitud_top_k(np.array(["b", "c", "c"]), probabilidades, clases, k=1) == pytest.approx(1 / 3)
    assert exactitud_top_k(np.array(["b", "c", "c"]), probabilidades, clases, k=2) == pytest.approx(2 / 3)
    metricas = calcular_metricas(["a", "c", "a"], probabilidades, clases)
    assert metricas["accuracy"] == 1.0 and metricas["top3_accuracy"] == 1.0


@pytest.fixture(scope="module")
def predictor():
    textos = ["computador portatil equipo", "impresora equipo computo", "llanta vehiculo bateria",
              "combustible gasolina vehiculo", "silla escritorio mueble", "mueble archivador oficina"] * 5
    etiquetas = ["43", "43", "25", "15", "56", "56"] * 5
    modelo = crear_regresion_logistica()
    modelo.set_params(tfidf__min_df=1, tfidf__max_df=1.0)
    modelo.fit(textos, etiquetas)
    return Predictor(modelo, Normalizador({"yumbo"}, set()))


def test_predecir_devuelve_top3_ordenado(predictor):
    sugerencias = predictor.predecir("Compra de computadores portátiles para el municipio de Yumbo")
    assert len(sugerencias) == 3
    assert sugerencias[0].segmento == "43"
    probabilidades = [s.probabilidad for s in sugerencias]
    assert probabilidades == sorted(probabilidades, reverse=True)


def test_auditar_marca_codigo_inconsistente(predictor):
    contratos = pd.DataFrame({
        "objeto_del_contrato": ["adquisicion de sillas y escritorios", "compra de computadores portatiles"],
        "codigo_de_categoria_principal": ["V1.72153606", "V1.43211500"],
    })
    resultado = predictor.auditar(contratos, umbral=0.3).set_index("codigo_de_categoria_principal")
    assert bool(resultado.loc["V1.72153606", "sospechoso"]) is True
    assert bool(resultado.loc["V1.43211500", "sospechoso"]) is False


def test_ajuste_distribucion_favorece_clases_frecuentes():
    from secopedia.prediccion import AjusteDistribucion

    ajuste = AjusteDistribucion(["a", "b"], p_real=[0.8, 0.2], p_entrenamiento=[0.5, 0.5])
    probabilidades = np.array([[0.4, 0.6]])
    # El orden de clases del modelo puede ser distinto al del archivo.
    ajustadas = ajuste.aplicar(probabilidades, ["a", "b"])
    assert np.allclose(ajustadas.sum(axis=1), 1)
    assert ajustadas[0, 0] > ajustadas[0, 1]  # 0,4*1,6 = 0,64 frente a 0,6*0,4 = 0,24
    invertidas = ajuste.aplicar(probabilidades[:, ::-1], ["b", "a"])
    assert np.allclose(invertidas[:, ::-1], ajustadas)
