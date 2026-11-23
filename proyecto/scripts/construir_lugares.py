"""Construye el léxico de lugares (municipios y departamentos DIVIPOLA) para la etapa 4.

Descarga la lista DIVIPOLA de datos.gov.co (dataset gdxc-w37w) y marca como "ambiguos"
los nombres de lugar que también funcionan como palabras comunes en los contratos
(por ejemplo "pasto", "pesca" o "plato"). Un nombre se considera ambiguo cuando, en el
corpus limpio, aparece muchas más veces sin palabra de contexto ("municipio de ...") que
con ella. Los ambiguos solo se reemplazan por <LUGAR> cuando van después de esas palabras.

Uso:
    python scripts/construir_lugares.py
"""

import json
import re

import pandas as pd
import requests

from secopedia.configuracion import cargar_config, ruta
from secopedia.normalizacion import PATRON_CONTEXTO, normalizar_nombre_lugar, quitar_tildes

URL_DIVIPOLA = "https://www.datos.gov.co/resource/gdxc-w37w.json"
LONGITUD_MINIMA = 4
# Proporción máxima de apariciones "sin contexto" frente a "con contexto" para un nombre seguro.
RAZON_MAXIMA = 3.0
# Nombres que se marcan ambiguos a mano: palabras muy comunes que el criterio automático
# podría no detectar por ser poco frecuentes en el corpus.
AMBIGUOS_MANUALES = {
    "meta", "cesar", "union", "la union", "la paz", "paz", "victoria", "la victoria", "fundacion",
    "el banco", "la mesa", "plato", "pesca", "pasto", "retiro", "turbo", "jardin", "campamento",
    "guaduas", "purificacion", "dolores", "ovejas", "mani", "corrales", "cienaga", "capilla",
    "la capilla", "floresta", "pradera", "la cumbre", "remolino", "calamar", "concordia",
    "candelaria", "esperanza", "la esperanza", "providencia", "granada", "florida", "palestina",
    "armenia", "albania", "libano", "ginebra", "venecia", "toledo", "trinidad", "belen", "betania",
    "argelia", "san jose", "san pedro", "santa maria", "colombia", "contratacion", "uribe",
    "bello", "prado", "honda", "toro", "tame", "une", "tena", "sora", "iza", "toca", "paya", "gama",
    "pore", "neira", "medina", "salamina", "segovia", "zaragoza", "carolina", "concepcion",
    "maceo", "olaya", "giraldo", "restrepo", "suarez", "linares", "caicedo", "ricaurte", "balboa",
    "obando", "trujillo", "sevilla", "andalucia", "marsella", "versalles", "palmira", "miraflores",
    "buenavista", "puerto rico", "el tambo", "tambo", "el carmen", "el rosal", "la ceja",
    "la estrella", "el peñol", "peñol", "la plata", "la merced", "aguadas", "remedios", "sabana",
}


def descargar_divipola() -> pd.DataFrame:
    respuesta = requests.get(URL_DIVIPOLA, params={"$limit": 5000}, timeout=120)
    respuesta.raise_for_status()
    return pd.DataFrame(respuesta.json())


def nombres_de_departamentos(divipola: pd.DataFrame) -> set[str]:
    nombres = set(divipola["dpto"]) | {"bogota", "valle", "valle del cauca", "san andres"}
    return {normalizar_nombre_lugar(n) for n in nombres}


def nombres_de_lugares(divipola: pd.DataFrame) -> set[str]:
    nombres = set(divipola["nom_mpio"]) | set(divipola["dpto"])
    nombres |= {"bogota", "bogota d c", "valle", "valle del cauca", "san andres", "cartagena de indias"}
    canonicos = {normalizar_nombre_lugar(n) for n in nombres}
    return {n for n in canonicos if len(n) >= LONGITUD_MINIMA}


def detectar_ambiguos(lugares: set[str], textos: pd.Series) -> pd.DataFrame:
    """Cuenta para cada nombre las apariciones con y sin palabra de contexto."""
    corpus = "\n".join(textos.map(lambda t: quitar_tildes(str(t).lower())))
    filas = []
    for nombre in sorted(lugares):
        total = len(re.findall(rf"\b{re.escape(nombre)}\b", corpus))
        con_contexto = len(re.findall(rf"\b{PATRON_CONTEXTO}{re.escape(nombre)}\b", corpus))
        filas.append((nombre, total, con_contexto))
    conteo = pd.DataFrame(filas, columns=["nombre", "total", "con_contexto"])
    conteo["sin_contexto"] = conteo["total"] - conteo["con_contexto"]
    conteo["razon"] = conteo["sin_contexto"] / (conteo["con_contexto"] + 1)
    return conteo


def main() -> None:
    config = cargar_config()
    divipola = descargar_divipola()
    lugares = nombres_de_lugares(divipola)
    print(f"Nombres de lugares DIVIPOLA: {len(lugares):,}")

    # Una muestra del corpus basta para estimar en qué contextos aparece cada nombre.
    textos = pd.read_parquet(ruta(config["rutas"]["limpios"]), columns=["objeto_del_contrato"])
    textos = textos["objeto_del_contrato"].sample(n=min(100_000, len(textos)), random_state=config["semilla"])
    conteo = detectar_ambiguos(lugares, textos)

    # Los departamentos suelen ir después del municipio ("caldas antioquia"), sin palabra de
    # contexto, pero siempre son lugares: solo son ambiguos si están en la lista manual.
    departamentos = nombres_de_departamentos(divipola)
    automaticos = set(conteo.loc[(conteo["razon"] > RAZON_MAXIMA) & (conteo["total"] >= 20), "nombre"])
    automaticos -= departamentos
    ambiguos = (automaticos | AMBIGUOS_MANUALES) & lugares
    print(f"Ambiguos automáticos: {len(automaticos)} | manuales: {len(AMBIGUOS_MANUALES & lugares)}")
    print(conteo[conteo["nombre"].isin(automaticos)].sort_values("total", ascending=False).head(30).to_string(index=False))

    destino = ruta("data/interim/lugares.json")
    with open(destino, "w", encoding="utf-8") as archivo:
        json.dump({"lugares": sorted(lugares), "ambiguos": sorted(ambiguos)}, archivo, ensure_ascii=False, indent=1)
    conteo.to_csv(ruta(config["rutas"]["metricas"] + "/lugares_conteo.csv"), index=False)
    print(f"Guardado {destino}")


if __name__ == "__main__":
    main()
