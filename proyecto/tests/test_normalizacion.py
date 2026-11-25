"""Pruebas de la etapa 4 con ejemplos reales de la tabla de muestra."""

from secopedia.normalizacion import Normalizador, normalizar_nombre_lugar, quitar_tildes

LUGARES = {"yumbo", "sotaquira", "boyaca", "caldas", "antioquia", "pasto", "zipaquira", "villa de leyva"}
AMBIGUOS = {"pasto", "caldas"}
normalizador = Normalizador(LUGARES, AMBIGUOS)


def test_quitar_tildes_conserva_enie():
    assert quitar_tildes("adquisición de señales") == "adquisicion de señales"


def test_nombre_lugar_canonico():
    assert normalizar_nombre_lugar("MARIQUITA (SAN SEBASTIÁN DE)") == "mariquita"


def test_ejemplo_cafeteria_yumbo():
    texto = "COMPRA DE PRODUCTOS DE CAFETERIA PARA LA ENTIDAD MUNICIPIO DE YUMBO VIGENCIA 2023"
    assert normalizador(texto) == "compra de productos de cafeteria para la entidad municipio de <LUGAR> vigencia <AÑO>"


def test_lugares_seguidos_se_unen():
    texto = "POLIZA PARA LA PERSONERIA MUNICIPAL DE SOTAQUIRA BOYACA"
    assert normalizador(texto) == "poliza para la personeria municipal de <LUGAR>"


def test_lugar_ambiguo_solo_con_contexto():
    assert normalizador("corte de pasto en parques") == "corte de pasto en parques"
    assert normalizador("municipio de Pasto") == "municipio de <LUGAR>"
    assert normalizador("medicina veterinaria en el municipio de Caldas; Antioquia") == (
        "medicina veterinaria en el municipio de <LUGAR>"
    )


def test_numeros_y_separadores():
    texto = "Convenio No. 045-2023; suministro de 300 llantas"
    assert normalizador(texto) == "convenio no. <NUM> <AÑO> suministro de <NUM> llantas"
