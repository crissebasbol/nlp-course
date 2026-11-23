"""Pruebas de la API de la demo (requieren los modelos de la línea base entrenados)."""

import pytest
from fastapi.testclient import TestClient

from secopedia.web.servidor import MODELOS, MUESTRA_AUDITORIA, app

pytestmark = pytest.mark.skipif(not MODELOS["regresion_logistica"][1].exists(),
                                reason="falta entrenar la línea base (scripts/entrenar_linea_base.py)")
cliente = TestClient(app)


def test_inicio_y_estado():
    assert cliente.get("/").status_code == 200
    estado = cliente.get("/api/estado").json()
    assert {m["id"] for m in estado["modelos"]} == {"regresion_logistica", "svm_lineal", "red_neuronal", "beto"}


def test_predecir():
    respuesta = cliente.post("/api/predecir", json={"texto": "compra de computadores portatiles"})
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert len(datos["sugerencias"]) == 3
    assert datos["sugerencias"][0]["segmento"] == "43"


def test_predecir_modelo_desconocido():
    assert cliente.post("/api/predecir", json={"texto": "compra de sillas", "modelo": "otro"}).status_code == 404


def test_auditar_csv():
    csv = ("objeto_del_contrato,codigo_de_categoria_principal\n"
           "adquisicion de mobiliario sillas y escritorios para las sedes,V1.72153606\n"
           "compra de computadores portatiles,V1.43211500\n")
    respuesta = cliente.post("/api/auditar", files={"archivo": ("c.csv", csv, "text/csv")}, data={"umbral": "0.6"})
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["total"] == 2 and datos["sospechosos"] >= 1


def test_auditar_sin_columnas():
    respuesta = cliente.post("/api/auditar", files={"archivo": ("c.csv", "a,b\n1,2\n", "text/csv")})
    assert respuesta.status_code == 422


@pytest.mark.skipif(not MUESTRA_AUDITORIA.exists(), reason="sin muestra")
def test_auditar_muestra():
    datos = cliente.post("/api/auditar", data={"usar_muestra": "true"}).json()
    assert datos["total"] == 300
