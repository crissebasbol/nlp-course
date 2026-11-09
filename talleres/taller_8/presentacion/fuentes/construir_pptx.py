"""Construye presentacion_taller8.pptx y notas_orador.md (desde talleres/taller_8).

Uso: python presentacion/fuentes/construir_pptx.py
Los GIFs se generan antes con generar_gifs.py.
"""
from __future__ import annotations

import os
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from notas import NOTAS, PREGUNTAS

BASE = Path(__file__).resolve().parents[1]  # presentacion/
GIF = BASE / "gifs"

FONT = "Calibri"
NAVY = RGBColor(0x1E, 0x3A, 0x6E)
INK = RGBColor(0x1F, 0x29, 0x37)
MUTED = RGBColor(0x6B, 0x72, 0x80)
LINE = RGBColor(0xD1, 0xD5, 0xDB)
SOFT = RGBColor(0xF3, 0xF4, 0xF6)
GREEN = RGBColor(0x05, 0x96, 0x69)
GREEN_L = RGBColor(0xD1, 0xFA, 0xE5)
RED = RGBColor(0xDC, 0x26, 0x26)
RED_L = RGBColor(0xFE, 0xE2, 0xE2)
BLUE = RGBColor(0x25, 0x63, 0xEB)
ORANGE = RGBColor(0xD9, 0x77, 0x06)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
SW = 13.333


def text(slide, x, y, w, h, paras, size=24, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         spacing=1.0, space_after=6):
    """paras: str | lista de párrafos; cada párrafo es str o lista de (texto, {opciones})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    if isinstance(paras, str):
        paras = [paras]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.line_spacing = spacing
        para.space_after = Pt(space_after)
        runs = [(p, {})] if isinstance(p, str) else p
        for s, o in runs:
            r = para.add_run()
            r.text = s
            r.font.name = FONT
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", bold)
            r.font.color.rgb = o.get("color", color)
    return tb


def rect(slide, x, y, w, h, fill=None, line=None, lw=2, radius=True, dash=False):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                                 Inches(x), Inches(y), Inches(w), Inches(h))
    if radius:
        shp.adjustments[0] = 0.08
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(lw)
        if dash:
            shp.line.dash_style = MSO_LINE.DASH
    shp.shadow.inherit = False
    shp.text_frame.text = ""
    return shp


def gif(slide, name, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    src = GIF / name
    if os.environ.get("PREVIEW_DIR"):  # revisión: usa el cuadro final del GIF en vez del primero
        src = Path(os.environ["PREVIEW_DIR"]) / name.replace(".gif", ".png")
    return slide.shapes.add_picture(str(src), Inches(x), Inches(y), **kw)


def gif_centered(slide, name, y, w=None, h=None):
    pic = gif(slide, name, 0, y, w, h)
    pic.left = int((prs.slide_width - pic.width) / 2)
    return pic


def placeholder(slide, x, y, w, h, title, sub=None):
    rect(slide, x, y, w, h, fill=SOFT, line=MUTED, lw=2.5, dash=True)
    paras = [[(title, {"bold": True, "size": 28, "color": MUTED})]]
    if sub:
        paras.append([(sub, {"size": 18, "color": MUTED})])
    text(slide, x + 0.3, y, w - 0.6, h, paras, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def header(slide, title, kicker=None):
    if kicker:
        text(slide, 0.6, 0.28, 12, 0.4, kicker.upper(), size=14, bold=True, color=GREEN)
    text(slide, 0.6, 0.55 if kicker else 0.4, 12.1, 0.8, title, size=36, bold=True, color=NAVY)


def footer(slide, n):
    text(slide, 0.6, 7.05, 9, 0.35, "Detección y corrección de errores médicos · PLN · Pontificia Universidad Javeriana",
         size=12, color=MUTED)
    text(slide, SW - 1.6, 7.05, 1.0, 0.35, str(n), size=12, color=MUTED, align=PP_ALIGN.RIGHT)


def notes(slide, i):
    n = NOTAS[i]
    slide.notes_slide.notes_text_frame.text = f"[{n['tiempo']}]  " + "\n\n".join(n["texto"])


def new_slide(i, title=None, kicker=None, foot=True):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE
    if title:
        header(s, title, kicker)
    if foot:
        footer(s, i + 1)
    notes(s, i)
    return s


# 1. Título -------------------------------------------------------------------
s = new_slide(0, foot=False)
rect(s, 0, 0, 0.35, 7.5, fill=NAVY, radius=False)
text(s, 0.95, 0.75, 9.0, 0.45, "PROCESAMIENTO DE LENGUAJE NATURAL · TALLER 8", size=16, bold=True, color=GREEN)
text(s, 0.95, 1.35, 9.2, 2.6, "Detección y corrección de errores médicos en notas clínicas", size=50, bold=True,
     color=NAVY, spacing=0.95)
text(s, 0.95, 4.05, 9.0, 0.6, "MEDEC / MEDIQA-CORR 2024 · BiomedBERT + LLM", size=26, color=INK)
text(s, 0.95, 5.55, 9.0, 1.2, ["[Nombres del equipo]", "Pontificia Universidad Javeriana · noviembre de 2026"],
     size=20, color=MUTED)
placeholder(s, 10.15, 1.0, 2.6, 3.1, "Escudo PUJ", "insertar imagen")

# 2. El problema --------------------------------------------------------------
s = new_slide(1, "Encontrar y corregir el error médico", "El problema")
items = [("A", "¿Hay error?", "Accuracy", RED), ("B", "¿En qué oración?", "Accuracy", RED),
         ("C", "¿Cómo corregirla?", "ROUGE-1 · BERTScore", GREEN)]
for k, (l, q, m, c) in enumerate(items):
    y = 1.65 + k * 1.2
    circ = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.6), Inches(y), Inches(0.7), Inches(0.7))
    circ.fill.solid(); circ.fill.fore_color.rgb = c; circ.line.fill.background(); circ.shadow.inherit = False
    circ.text_frame.text = ""
    text(s, 0.6, y + 0.08, 0.7, 0.55, l, size=26, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    text(s, 1.45, y - 0.02, 3.3, 0.5, q, size=26, bold=True)
    text(s, 1.45, y + 0.45, 3.3, 0.4, m, size=18, color=MUTED)
text(s, 0.6, 5.35, 4.1, 1.4, [[("Cada texto: correcto o con ", {}), ("un solo error", {"bold": True})],
                              "2.189 textos · 56 % con error"], size=20, color=INK)
gif(s, "01_problema.gif", 4.85, 1.35, h=5.55)

# 3. Diseño original ----------------------------------------------------------
s = new_slide(2, "BiomedBERT + Qwen con DSPy, combinados", "El diseño original · taller 7")
gif_centered(s, "02_diseno_original.gif", 1.55, w=12.1)
text(s, 0.6, 6.45, 12.1, 0.5, "Todo con modelos abiertos: Hugging Face + llama.cpp", size=20, color=MUTED,
     align=PP_ALIGN.CENTER)

# 4. El giro ------------------------------------------------------------------
s = new_slide(3, "Implementado tal cual…", "Primer resultado · test, 327 textos")
text(s, 0.6, 2.0, 4.2, 1.3, [[("A = 0,57", {"size": 72, "bold": True, "color": RED})]])
text(s, 0.6, 3.45, 4.0, 1.5, "igual que decir siempre «hay error»", size=28, bold=True)
text(s, 0.6, 5.0, 4.0, 1.0, "¿Por qué? Antes de agregar más piezas, medimos.", size=20, color=MUTED)
gif(s, "03_giro.gif", 4.95, 1.35, h=5.55)

# 5. Diagnóstico --------------------------------------------------------------
s = new_slide(4, "BERT sabe DÓNDE, pero no SI", "Diagnóstico")
gif_centered(s, "04_diagnostico.gif", 1.35, w=12.1)
stats = [("Dónde: 95 a 98 %", BLUE), ("Si: AUC 0,48 (azar)", RED), ("Qwen 7B: AUC 0,50 a 0,60", ORANGE)]
for k, (st, c) in enumerate(stats):
    x = 0.6 + k * 4.1
    rect(s, x, 6.35, 3.9, 0.62, fill=WHITE, line=c, lw=2.5)
    text(s, x, 6.35, 3.9, 0.62, st, size=22, bold=True, color=c, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# 6. Pruebas y fuga -----------------------------------------------------------
s = new_slide(5, "Lo que probamos… y una fuga de datos", "Diagnóstico")
bul = [
    [("Qwen verifica la oración de BERT", {"bold": True})], [("no mejora (A 0,55)", {"color": MUTED, "size": 20})],
    [("BERT + 10.150 preguntas MedQA", {"bold": True})], [("no aprende: pérdida = ln 2", {"color": MUTED, "size": 20})],
    [("«Hoja de respuestas» sin LLM", {"bold": True, "color": RED})],
    [("A 0,84: los textos vienen de MedQA", {"color": RED, "size": 20})],
]
text(s, 0.6, 1.75, 4.2, 4.8, bul, size=23, space_after=4)
gif(s, "05_pruebas_y_fuga.gif", 4.95, 1.35, h=5.55)

# 7. Pipeline final -----------------------------------------------------------
s = new_slide(6, "BERT elige la oración, Claude Haiku decide y corrige", "El pipeline final")
gif_centered(s, "06_pipeline_final.gif", 1.5, w=12.1)
text(s, 0.6, 6.4, 12.1, 0.5, "Cada modelo hace lo que sabe: BERT el «dónde», el LLM el «si» · DSPy permite cambiar de proveedor",
     size=20, color=MUTED, align=PP_ALIGN.CENTER)

# 8. Resultados ---------------------------------------------------------------
s = new_slide(7, "BERT + Haiku supera a Haiku solo", "Resultados finales · test, 327 textos")
gif_centered(s, "07_resultados.gif", 1.45, h=5.5)

# 9-10. Demos -----------------------------------------------------------------
s = new_slide(8, "Un texto con error", "Demo")
gif_centered(s, "08_demo_con_error.gif", 1.4, h=5.55)
s = new_slide(9, "Un texto correcto", "Demo")
gif_centered(s, "09_demo_sin_error.gif", 1.4, h=5.55)

# 11. Despliegue --------------------------------------------------------------
s = new_slide(10, "Docker en dos servidores sin GPU", "Despliegue")
gif_centered(s, "10_despliegue.gif", 1.4, w=11.5)
text(s, 0.6, 6.55, 12.1, 0.45,
     [[("Modelo: ", {"bold": True}), ("huggingface.co/crissebasbol/medec-biomedbert-detector", {}),
       ("     Guía: ", {"bold": True}), ("DESPLIEGUE.md", {})]], size=18, color=INK, align=PP_ALIGN.CENTER)

# 12. Limitaciones y lecciones ------------------------------------------------
s = new_slide(11, "Limitaciones y lecciones")
cards = [("Memorización", "Claude pudo ver MedQA al entrenarse", ORANGE),
         ("Privacidad", "Los textos salen a una API externa: no apto para notas reales", RED),
         ("DSPy compilado", "No mejoró ni con Qwen ni con Haiku", BLUE)]
for k, (t_, d_, c) in enumerate(cards):
    x = 0.6 + k * 4.1
    rect(s, x, 1.6, 3.9, 2.6, fill=SOFT, line=c, lw=3)
    text(s, x + 0.25, 1.85, 3.4, 0.6, t_, size=28, bold=True, color=c)
    text(s, x + 0.25, 2.55, 3.4, 1.5, d_, size=22, color=INK)
rect(s, 0.6, 4.6, 12.1, 2.1, fill=GREEN_L, line=GREEN, lw=3)
text(s, 0.9, 4.6, 11.5, 2.1,
     [[("Lección principal", {"size": 20, "bold": True, "color": GREEN})],
      [("Medir y diagnosticar antes de seguir agregando complejidad", {"size": 34, "bold": True, "color": NAVY})]],
     align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# 13. Demo de la UI -----------------------------------------------------------
s = new_slide(12, "La interfaz web", "Demo en vivo")
placeholder(s, 0.6, 1.5, 12.1, 5.3, "GIF o video de la interfaz", "insertar aquí · http://<IP máquina 2>:8000")

# 14. Cierre ------------------------------------------------------------------
s = new_slide(13, foot=False)
rect(s, 0, 0, 0.35, 7.5, fill=NAVY, radius=False)
text(s, 0.95, 1.4, 11.4, 2.6,
     [[("BERT sabe ", {}), ("dónde", {"bold": True, "color": BLUE}), (" mirar; saber ", {}),
       ("si", {"bold": True, "color": GREEN}), (" está bien requiere conocimiento médico.", {})]],
     size=40, color=NAVY, spacing=1.05)
text(s, 0.95, 3.75, 11.4, 0.8, [[("Medir y diagnosticar nos llevó de A = 0,57 a ", {}),
                                   ("0,83", {"bold": True, "color": GREEN})]], size=28, color=INK)
text(s, 0.95, 5.0, 11.4, 1.0, "¿Preguntas?", size=48, bold=True, color=NAVY)
text(s, 0.95, 6.5, 11.4, 0.5, "github.com/crissebasbol/nlp-course · talleres/taller_8", size=16, color=MUTED)

out = Path(os.environ["PREVIEW_OUT"]) if os.environ.get("PREVIEW_OUT") else BASE / "presentacion_taller8.pptx"
prs.save(out)
print("pptx:", out)

# notas_orador.md ---------------------------------------------------------------
md = ["# Notas del orador: taller 8", "",
      "Detección y corrección de errores médicos en notas clínicas (MEDEC / MEDIQA-CORR 2024).", "",
      "Duración objetivo: unos 10 minutos sin contar la demo en vivo. Todas las cifras son de test (327 textos) salvo "
      "que se diga otra cosa, y salen de `README.md`, `outputs/reports/results_test.md` y "
      "`outputs/experiments/diagnostico_deteccion_A/NOTAS.md`.", ""]
for i, n in enumerate(NOTAS):
    md += [f"## {i + 1}. {n['titulo']} ({n['tiempo']})", ""]
    for p in n["texto"]:
        md += [p, ""]
md += ["## Preguntas probables", ""]
for q, a in PREGUNTAS:
    md += [f"**{q}**", "", a, ""]
(BASE / "notas_orador.md").write_text("\n".join(md), encoding="utf-8")
print("notas:", BASE / "notas_orador.md")
