"""Pruebas de la CNN (modelo 7b) con datos de juguete."""

import numpy as np
import torch

from secopedia.modelos.red_neuronal import (DESCONOCIDA, RELLENO, ClasificadorCNN, RedConvolucional, codificar,
                                            construir_vocabulario)


def test_vocabulario_y_codificacion():
    documentos = [["equipo", "computo"], ["equipo", "silla"], ["silla"]]
    vocabulario = construir_vocabulario(documentos, min_frecuencia=2)
    assert set(vocabulario) == {"equipo", "silla"} and min(vocabulario.values()) == 2
    x = codificar([["equipo", "nuevo"]], vocabulario, longitud=4)
    assert x.tolist() == [[vocabulario["equipo"], DESCONOCIDA, RELLENO, RELLENO]]


def test_cnn_forma_salida_y_guardado(tmp_path):
    embeddings = np.random.default_rng(0).normal(size=(6, 8)).astype(np.float32)
    modelo = RedConvolucional(embeddings, num_clases=3, filtros=4)
    salida = modelo(torch.tensor([[2, 3, 0, 0], [4, 5, 2, 0]]))
    assert salida.shape == (2, 3)

    clasificador = ClasificadorCNN(modelo, {"a": 2, "b": 3, "c": 4, "d": 5}, ["x", "y", "z"])
    probabilidades = clasificador.predict_proba(["a b", "c d a"])
    assert np.allclose(probabilidades.sum(axis=1), 1, atol=1e-5)
    clasificador.guardar(tmp_path / "cnn.pt")
    cargado = ClasificadorCNN.cargar(tmp_path / "cnn.pt")
    assert np.allclose(cargado.predict_proba(["a b", "c d a"]), probabilidades, atol=1e-5)
