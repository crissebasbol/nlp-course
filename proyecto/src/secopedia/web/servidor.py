"""Demo web de SECOPEDIA (etapa 9): API REST con FastAPI y una interfaz web sencilla.

Ejecutar desde proyecto/:
    uvicorn secopedia.web.servidor:app --port 8000

- Interfaz:       http://localhost:8000
- Documentación:  http://localhost:8000/docs
"""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from secopedia.configuracion import RAIZ_PROYECTO, cargar_config
from secopedia.prediccion import Predictor
from secopedia.segmentos import nombre_segmento

CARPETA_WEB = Path(__file__).parent
CARPETA_ESTATICOS = CARPETA_WEB / "estaticos"
MUESTRA_AUDITORIA = CARPETA_WEB / "muestra_auditoria.csv"
MAXIMO_CONTRATOS_AUDITORIA = 5000

config = cargar_config()
CARPETA_MODELOS = RAIZ_PROYECTO / config["rutas"]["modelos"]
CARPETA_METRICAS = RAIZ_PROYECTO / config["rutas"]["metricas"]

# Modelos que puede usar la demo: nombre visible y archivo local.
MODELOS = {
    "regresion_logistica": ("Regresión logística (TF-IDF)", CARPETA_MODELOS / "regresion_logistica.joblib"),
    "svm_lineal": ("SVM lineal (TF-IDF)", CARPETA_MODELOS / "svm_lineal.joblib"),
    "red_neuronal": ("Red neuronal CNN (fastText)", CARPETA_MODELOS / "red_neuronal" / "cnn.pt"),
    "beto": ("BETO (BERT en español)", CARPETA_MODELOS / "beto_secopedia" / "config.json"),
}
# Los modelos que no estén entrenados localmente se descargan de Hugging Face Hub.
HUB = config["rutas"].get("hub")


def modelo_disponible(clave: str) -> bool:
    return MODELOS[clave][1].exists() or bool(HUB)
EJEMPLOS = [
    "Compra de computadores portátiles para la secretaría de educación",
    "Suministro de combustible para los vehículos de la alcaldía",
    "Adquisición de mobiliario para las sedes de la Registraduría",
    "Mantenimiento y reparación de la malla vial del municipio",
    "Póliza de seguro de responsabilidad civil para los servidores públicos",
    "Suministro de refrigerios para el evento del día de la mujer",
]

_predictores: dict[str, Predictor] = {}
_candado = threading.Lock()


def obtener_predictor(modelo: str) -> Predictor:
    """Carga cada modelo la primera vez que se pide y lo deja en memoria."""
    if modelo not in MODELOS:
        raise HTTPException(404, f"Modelo desconocido: {modelo}")
    if not modelo_disponible(modelo):
        raise HTTPException(503, f"El modelo {modelo} no está entrenado todavía")
    with _candado:
        if modelo not in _predictores:
            if modelo == "red_neuronal":
                _predictores[modelo] = Predictor.red_neuronal()
            elif modelo == "beto":
                local = CARPETA_MODELOS / "beto_secopedia"
                _predictores[modelo] = Predictor.beto(str(local) if MODELOS["beto"][1].exists() else HUB)
            else:
                _predictores[modelo] = Predictor.linea_base(modelo)
        return _predictores[modelo]


app = FastAPI(
    title="SECOPEDIA",
    description="Sugerencia automática de segmentos UNSPSC para contratos públicos de SECOP II.",
    version="0.1.0",
)


class SolicitudPrediccion(BaseModel):
    texto: str = Field(..., min_length=3, description="Objeto del contrato, en lenguaje natural")
    modelo: str = Field("regresion_logistica", description="regresion_logistica, svm_lineal, red_neuronal o beto")
    k: int = Field(3, ge=1, le=10, description="Cuántas categorías devolver")


@app.get("/", include_in_schema=False)
def inicio() -> FileResponse:
    return FileResponse(CARPETA_ESTATICOS / "index.html")


@app.get("/api/estado")
def estado() -> dict:
    """Modelos disponibles y ejemplos para la interfaz."""
    return {
        "modelos": [
            {"id": clave, "nombre": nombre, "disponible": modelo_disponible(clave)}
            for clave, (nombre, _) in MODELOS.items()
        ],
        "ejemplos": EJEMPLOS,
        "hay_muestra_auditoria": MUESTRA_AUDITORIA.exists(),
    }


@app.post("/api/predecir")
def predecir(solicitud: SolicitudPrediccion) -> dict:
    """Las k categorías UNSPSC más probables para un objeto de contrato."""
    predictor = obtener_predictor(solicitud.modelo)
    sugerencias = predictor.predecir(solicitud.texto, k=solicitud.k)
    return {
        "modelo": solicitud.modelo,
        "texto_normalizado": predictor.normalizador(solicitud.texto),
        "entrada_modelo": predictor._preparar([solicitud.texto]).iat[0],
        "sugerencias": [s.__dict__ for s in sugerencias],
    }


@app.post("/api/auditar")
async def auditar(
    archivo: UploadFile | None = File(None, description="CSV con objeto_del_contrato y codigo_de_categoria_principal"),
    usar_muestra: bool = Form(False, description="Usar la muestra incluida de contratos de 2026"),
    modelo: str = Form("regresion_logistica"),
    umbral: float = Form(0.8, ge=0.5, le=0.99),
) -> dict:
    """Marca los contratos donde el modelo está muy seguro de un segmento distinto al registrado."""
    if archivo is not None and archivo.filename:
        contenido = await archivo.read()
        try:
            contratos = pd.read_csv(io.BytesIO(contenido), dtype=str)
        except Exception as error:  # CSV mal formado
            raise HTTPException(422, f"No se pudo leer el CSV: {error}") from error
    elif usar_muestra and MUESTRA_AUDITORIA.exists():
        contratos = pd.read_csv(MUESTRA_AUDITORIA, dtype=str)
    else:
        raise HTTPException(422, "Envía un archivo CSV o marca usar_muestra")

    faltantes = {"objeto_del_contrato", "codigo_de_categoria_principal"} - set(contratos.columns)
    if faltantes:
        raise HTTPException(422, f"Faltan columnas en el CSV: {sorted(faltantes)}")
    if len(contratos) > MAXIMO_CONTRATOS_AUDITORIA:
        raise HTTPException(413, f"Máximo {MAXIMO_CONTRATOS_AUDITORIA} contratos por solicitud")

    resultado = obtener_predictor(modelo).auditar(contratos, umbral=umbral)
    resultado["nombre_registrado"] = resultado["segmento_registrado"].map(nombre_segmento)
    columnas = [c for c in ["id_contrato", "objeto_del_contrato", "codigo_de_categoria_principal",
                            "segmento_registrado", "nombre_registrado", "segmento_sugerido", "nombre_sugerido",
                            "probabilidad", "prob_registrado", "sospechoso"] if c in resultado.columns]
    filas = json.loads(resultado[columnas].to_json(orient="records", force_ascii=False))
    return {"total": len(resultado), "sospechosos": int(resultado["sospechoso"].sum()), "contratos": filas}


@app.get("/api/metricas")
def metricas() -> dict:
    """Métricas guardadas de los modelos entrenados (prueba y prueba temporal)."""
    salida = {}
    for nombre in ["linea_base", "red_neuronal", "beto"]:
        ruta = CARPETA_METRICAS / f"{nombre}.json"
        if ruta.exists():
            salida[nombre] = json.loads(ruta.read_text(encoding="utf-8"))
    return salida


app.mount("/estaticos", StaticFiles(directory=CARPETA_ESTATICOS), name="estaticos")
