"""Entrena y evalúa la línea base (modelo 7a): TF-IDF + Regresión Logística y SVM lineal.

Ajusta el hiperparámetro C en validación, evalúa el mejor en prueba y prueba temporal,
y guarda modelo, métricas y matriz de confusión.

Uso:
    python scripts/entrenar_linea_base.py
"""

import json
import time

import joblib
import pandas as pd

from secopedia.configuracion import cargar_config, fijar_semilla, ruta
from secopedia.evaluacion import calcular_metricas, graficar_matriz_confusion, reporte_por_clase
from secopedia.modelos.linea_base import crear_regresion_logistica, crear_svm, terminos_por_clase
from secopedia.representacion import unir_lemas

BUSQUEDA = {
    "regresion_logistica": (crear_regresion_logistica, [1.0, 4.0, 16.0]),
    "svm_lineal": (crear_svm, [0.1, 0.5, 1.0]),
}


def main() -> None:
    config = cargar_config()
    semilla = config["semilla"]
    fijar_semilla(semilla)
    carpeta = ruta(config["rutas"]["procesados"] + "/x").parent
    metricas_dir = ruta(config["rutas"]["metricas"] + "/x").parent
    figuras_dir = ruta(config["rutas"]["figuras"] + "/x").parent
    modelos_dir = ruta(config["rutas"]["modelos"] + "/x").parent
    clases = json.loads((metricas_dir / "clases.json").read_text(encoding="utf-8"))

    datos = {n: pd.read_parquet(carpeta / f"{n}.parquet") for n in ["entrenamiento", "validacion", "prueba", "prueba_temporal"]}
    x = {n: unir_lemas(df["lemas"]) for n, df in datos.items()}
    y = {n: df["etiqueta"] for n, df in datos.items()}

    resultados = []
    mejores = {}
    for nombre, (constructor, valores_c) in BUSQUEDA.items():
        for c in valores_c:
            inicio = time.time()
            modelo = constructor(c=c, semilla=semilla).fit(x["entrenamiento"], y["entrenamiento"])
            segundos = time.time() - inicio
            assert list(modelo.classes_) == clases
            metricas = calcular_metricas(y["validacion"], modelo.predict_proba(x["validacion"]), clases)
            resultados.append({"modelo": nombre, "C": c, "segundos_entrenamiento": round(segundos, 1), **metricas})
            print(resultados[-1])
            if nombre not in mejores or metricas["f1_macro"] > mejores[nombre][1]:
                mejores[nombre] = (modelo, metricas["f1_macro"], c)

    pd.DataFrame(resultados).to_csv(metricas_dir / "linea_base_validacion.csv", index=False)

    resumen = {}
    for nombre, (modelo, _, c) in mejores.items():
        resumen[nombre] = {"C": c}
        for particion in ["prueba", "prueba_temporal"]:
            inicio = time.time()
            probabilidades = modelo.predict_proba(x[particion])
            milisegundos = 1000 * (time.time() - inicio) / len(x[particion])
            resumen[nombre][particion] = {**calcular_metricas(y[particion], probabilidades, clases),
                                          "ms_por_contrato": round(milisegundos, 3)}
        predichos = modelo.predict(x["prueba"])
        reporte_por_clase(y["prueba"], predichos, clases).to_csv(metricas_dir / f"{nombre}_reporte_clases.csv")
        graficar_matriz_confusion(y["prueba"], predichos, clases, f"Matriz de confusión: {nombre} (prueba)",
                                  figuras_dir / f"matriz_confusion_{nombre}.png")
        joblib.dump(modelo, modelos_dir / f"{nombre}.joblib")
        print(nombre, json.dumps(resumen[nombre], indent=1))

    terminos = terminos_por_clase(mejores["regresion_logistica"][0])
    (metricas_dir / "terminos_por_clase.json").write_text(json.dumps(terminos, ensure_ascii=False, indent=1), encoding="utf-8")
    (metricas_dir / "linea_base.json").write_text(json.dumps(resumen, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
