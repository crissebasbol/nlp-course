"""Genera los GIFs de la presentación a partir de los archivos del repositorio.

Uso (desde talleres/taller_8):  python presentacion/fuentes/generar_gifs.py [nombre ...]

Todas las cifras y textos clínicos se leen de:
- ../taller_7/data/MEDIQA-CORR-2024-MS-TrainingData.csv  (textos y etiquetas)
- outputs/scores/bert_test.jsonl                         (p_bert por oración)
- outputs/scores/verifier_haiku_test.jsonl               (decisión y tiempo de Haiku)
- outputs/predictions/pipeline_test.jsonl                (pipeline original)
Las métricas de las gráficas son las de outputs/reports/results_test.md y de NOTAS.md.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from gif_lib import *  # noqa: F401,F403

ROOT = Path(__file__).resolve().parents[2]  # talleres/taller_8
OUT = ROOT / "presentacion" / "gifs"
OUT.mkdir(parents=True, exist_ok=True)

ROWS = {r["Text ID"]: r for r in csv.DictReader(
    open(ROOT.parent / "taller_7" / "data" / "MEDIQA-CORR-2024-MS-TrainingData.csv", encoding="utf-8"))}


def jsonl(rel):
    return {json.loads(l)["text_id"]: json.loads(l) for l in open(ROOT / rel, encoding="utf-8")}


BERT_S = jsonl("outputs/scores/bert_test.jsonl")
VERIF = jsonl("outputs/scores/verifier_haiku_test.jsonl")
PIPE = jsonl("outputs/predictions/pipeline_test.jsonl")


def sentences(tid):
    out = []
    for line in ROWS[tid]["Sentences"].split("\n"):
        line = line.strip()
        if line:
            n, _, s = line.partition(" ")
            out.append((int(n), s))
    return out


# ---------------------------------------------------------------- 1. El problema
def gif_problema():
    tid = "ms-train-90"
    W, H = 1200, 820
    sents = [s for s in sentences(tid) if s[0] in (0, 1, 2, 3, 5, 7, 8, 9)]
    err = ROWS[tid]["Error Sentence"]            # "Suspected of infection with clostridium botulinum."
    cor = ROWS[tid]["Corrected Sentence"]        # "... Streptococcus pyogenes."
    prefix = "Suspected of infection with "
    wrong = err[len(prefix):].rstrip(".")
    right = cor[len(prefix):].rstrip(".")
    f = font(25)
    fn = font(22, "SemiBold")
    T = 11.0

    def draw(d, t):
        fade = 1 - prog(t, 10.4, 0.6)
        d.rounded_rectangle((20, 16, W - 20, 606), radius=18, fill=mix(SOFT, BG, fade), outline=mix(LINE, BG, fade), width=2)
        d.text((48, 34), f"Nota clínica {tid} (se omiten las oraciones 4 y 6)", font=font(20), fill=mix(MUTED, BG, fade))
        y = 78
        for i, (n, s) in enumerate(sents):
            a = prog(t, 0.2 + i * 0.28, 0.4) * fade
            dy = (1 - a) * 12
            lines = wrap(d, s, f, W - 20 - 110 - 40)
            h = len(lines) * 33
            if n == 7:
                hp = prog(t, 3.8, 0.5) * fade
                if hp > 0:
                    d.rounded_rectangle((92, y - 6, 92 + (W - 40 - 92 - 30) * hp, y + h + 2), radius=10, fill=mix(ERR_L, BG, fade))
            d.text((52, y + dy), str(n), font=fn, fill=mix(MUTED, BG, a))
            if n == 7 and t >= 5.0:
                x = 110
                d.text((x, y), prefix, font=f, fill=mix(INK, BG, fade))
                x += text_w(d, prefix, f)
                ww = text_w(d, wrong, f)
                d.text((x, y), wrong, font=f, fill=mix(ERR, BG, fade))
                sp = prog(t, 5.0, 0.5)
                d.line([(x, y + 16), (x + ww * sp, y + 16)], fill=mix(ERR, BG, fade), width=4)
                x += ww + 12
                k = int(len(right) * clamp((t - 5.6) / 0.8))
                if k > 0:
                    part = right[:k]
                    pw = text_w(d, part, f)
                    d.rounded_rectangle((x - 6, y - 4, x + pw + 6, y + 32), radius=8, fill=mix(OK_L, BG, fade))
                    d.text((x, y), part, font=font(25, "SemiBold"), fill=mix(OK, BG, fade))
                    x += pw
                if k == len(right):
                    d.text((x + 8, y), ".", font=f, fill=mix(INK, BG, fade))
            else:
                for j, ln in enumerate(lines):
                    d.text((110, y + dy + j * 33), ln, font=f, fill=mix(INK, BG, a))
            y += h + 14

        pills = [("A", "¿Hay error?", "Sí, hay un error", ERR, ERR_L, 2.8),
                 ("B", "¿En qué oración?", "Oración 7", ERR, ERR_L, 3.8),
                 ("C", "¿Cómo corregirla?", "Streptococcus pyogenes", OK, OK_L, 5.6)]
        pw = (W - 40 - 2 * 20) / 3
        for i, (letter, q, ans, col, coll, t0) in enumerate(pills):
            x0 = 20 + i * (pw + 20)
            a = prog(t, t0, 0.45) * fade
            d.rounded_rectangle((x0, 632, x0 + pw, 800), radius=18, fill=mix(coll, BG, a),
                                outline=mix(col, mix(LINE, BG, fade), a), width=3)
            d.ellipse((x0 + 20, 652, x0 + 66, 698), fill=mix(col, mix(GRAY, BG, fade), a))
            d.text((x0 + 43, 675), letter, font=font(26, "Bold"), fill=BG, anchor="mm")
            d.text((x0 + 80, 675), q, font=font(25, "SemiBold"), fill=mix(INK, BG, fade), anchor="lm")
            d.text((x0 + 22, 748), ans, font=font(27, "Bold"), fill=mix(col, BG, a), anchor="lm")

    return render_gif(draw, (W, H), T, OUT / "01_problema.gif")


# ---------------------------------------------------------------- 2. Diseño original
def gif_diseno_original():
    W, H = 1600, 640
    T = 9.0
    N = {
        "texto": (150, 320, 230, 130, "Texto clínico", "oraciones numeradas", INK),
        "bert": (560, 150, 380, 140, "3a · BiomedBERT", "p_bert por oración", BERT),
        "qwen": (560, 490, 380, 140, "3b · Qwen2.5-7B + DSPy", "s_dspy por oración", QWEN),
        "comb": (990, 320, 330, 160, "3c · Combinación", "w·p_bert + (1 − w)·s_dspy", NAVY),
        "corr": (1380, 320, 300, 160, "4-5 · Corrector", "Qwen + ejemplos parecidos", QWEN),
    }
    E = [
        ("texto", [(265, 300), (325, 300), (325, 150), (368, 150)], 0.6),
        ("texto", [(265, 340), (325, 340), (325, 490), (368, 490)], 0.6),
        ("bert", [(752, 150), (790, 150), (790, 290), (823, 290)], 1.7),
        ("qwen", [(752, 490), (790, 490), (790, 350), (823, 350)], 1.7),
        ("comb", [(1155, 320), (1230, 320)], 2.8),
    ]
    appear = {"texto": 0.1, "bert": 1.1, "qwen": 1.1, "comb": 2.2, "corr": 3.3}

    def draw(d, t):
        fade = 1 - prog(t, 8.5, 0.5)
        for _, pts, t0 in E:
            arrow(d, pts, prog(t, t0, 0.5), color=mix(MUTED, BG, fade))
        for _, pts, t0 in E:
            if t > 4.0:
                for k in range(2):
                    ph = ((t - 4.0) / 1.6 + k * 0.5) % 1
                    dot(d, poly_point(pts, ph), r=9, color=mix(INK, BG, fade))
        for k, (cx, cy, w, h, ti, sub, col) in N.items():
            box(d, cx, cy, w, h, ti, sub, color=col, a=prog(t, appear[k], 0.5) * fade)
        a = prog(t, 4.4, 0.6) * fade
        d.text((W / 2, 610), "w y el umbral se ajustan en validación; el mismo puntaje decide A y elige la oración (B)",
               font=font(24), fill=mix(MUTED, BG, a), anchor="mm")
        d.text((1380, 430), "→ A · B · C", font=font(26, "SemiBold"), fill=mix(INK, BG, prog(t, 3.6, 0.5) * fade), anchor="mm")

    return render_gif(draw, (W, H), T, OUT / "02_diseno_original.gif")


# ---------------------------------------------------------------- barras horizontales genéricas
def hbars(path, rows, T, baseline=None, callouts=(), size=(1150, 760), label_w=400, top=70, row_h=86, gap=46,
          start=0.3, stagger=0.35):
    """rows: (etiqueta, valor, color, t_inicio o None, etiqueta_extra)."""
    W, H = size
    x0, x1 = label_w + 20, W - 110

    def X(v):
        return x0 + (x1 - x0) * v

    def draw(d, t):
        fade = 1 - prog(t, T - 0.5, 0.5)
        bottom = top + len(rows) * (row_h + gap) - gap
        for v in (0, 0.25, 0.5, 0.75, 1.0):
            d.line([(X(v), top - 20), (X(v), bottom + 10)], fill=mix(SOFT if v else LINE, BG, fade), width=2)
            d.text((X(v), bottom + 34), comma(v, 2), font=font(20), fill=mix(MUTED, BG, fade), anchor="mm")
        for i, (lab, v, col, t0, extra) in enumerate(rows):
            t0 = start + i * stagger if t0 is None else t0
            y = top + i * (row_h + gap)
            a = prog(t, t0, 0.3) * fade
            lines = wrap(d, lab, font(26, "SemiBold"), label_w - 10)
            ly = y + row_h / 2 - (len(lines) - 1) * 16
            for j, ln in enumerate(lines):
                d.text((label_w, ly + j * 32), ln, font=font(26, "SemiBold"), fill=mix(INK, BG, a), anchor="rm")
            p = prog(t, t0, 0.9)
            if p > 0:
                d.rounded_rectangle((x0, y, max(x0 + 8, X(v * p)), y + row_h), radius=10, fill=mix(col, BG, fade))
                d.text((X(v * p) + 14, y + row_h / 2), comma(v * p if p < 1 else v, 2), font=font(30, "Bold"),
                       fill=mix(INK, BG, a), anchor="lm")
                if extra and p >= 1:
                    ea = prog(t, t0 + 0.9, 0.4) * fade
                    pill(d, x0 + 20, y + row_h / 2 - 20, extra,
                         font(22, "Bold"), mix(ERR, BG, ea), mix(BG, BG, ea), h=40)
        if baseline:
            v, lab, tb = baseline
            p = prog(t, tb, 0.6)
            if p > 0:
                dashed_line(d, (X(v), top - 30), (X(v), top - 30 + (bottom - top + 40) * p), mix(ERR, BG, fade), 4)
                a = prog(t, tb + 0.5, 0.4) * fade
                d.text((X(v), top - 46), lab, font=font(22, "Bold"), fill=mix(ERR, BG, a), anchor="mm")
        for (s, tc, col) in callouts:
            a = prog(t, tc, 0.5) * fade
            if a > 0:
                fw = text_w(d, s, font(26, "Bold")) + 48
                cx = W / 2
                d.rounded_rectangle((cx - fw / 2, H - 78, cx + fw / 2, H - 18), radius=14, fill=mix(mix(col, BG, 0.12), BG, a),
                                    outline=mix(col, BG, a), width=3)
                d.text((cx, H - 48), s, font=font(26, "Bold"), fill=mix(col, BG, a), anchor="mm")

    return render_gif(draw, (W, H), T, path)


def gif_giro():
    rows = [("Siempre «hay error»", 0.560, GRAY, None, None),
            ("BiomedBERT", 0.566, BERT, None, None),
            ("Qwen2.5-7B + DSPy", 0.566, QWEN, None, None),
            ("Pipeline original", 0.566, NAVY, 1.6, None)]
    return hbars(OUT / "03_giro.gif", rows, T=10.0, baseline=(0.56, "0,56 = siempre «hay error»", 3.2),
                 callouts=[("Marca con error el 92 % de los textos", 4.6, ERR)], size=(1150, 760), top=90)


def gif_pruebas():
    rows = [("BiomedBERT solo", 0.566, BERT, 0.3, None),
            ("Qwen verifica la oración de BERT", 0.550, QWEN, 0.65, None),
            ("BERT preentrenado con MedQA", 0.563, (96, 165, 250), 1.0, None),
            ("«Hoja de respuestas» MedQA, sin LLM", 0.838, ERR, 3.2, "fuga de datos")]
    return hbars(OUT / "05_pruebas_y_fuga.gif", rows, T=11.0, baseline=(0.56, "siempre «hay error»", 2.0),
                 callouts=[("Los textos son preguntas públicas de MedQA reescritas", 5.0, ERR)],
                 size=(1150, 760), top=90, label_w=420)


# ---------------------------------------------------------------- 4. Diagnóstico: dónde vs. si
def gif_diagnostico():
    W, H = 1600, 660
    T = 11.0
    pair = [("ms-train-1994", "Con error", ERR, ERR_L), ("ms-train-1995", "Sin error", OK, OK_L)]
    f = font(21)

    def draw(d, t):
        fade = 1 - prog(t, T - 0.5, 0.5)
        for pi, (tid, lab, col, coll) in enumerate(pair):
            px = 20 + pi * 790
            pw = 770
            a0 = prog(t, 0.1 + pi * 0.2, 0.5) * fade
            d.rounded_rectangle((px, 10, px + pw, 520), radius=18, fill=mix(SOFT, BG, a0), outline=mix(LINE, BG, a0), width=2)
            pill(d, px + 20, 26, lab, font(24, "Bold"), mix(BG, BG, 0) if a0 else BG, mix(col, BG, a0), h=42)
            d.text((px + pw - 20, 47), tid, font=font(20), fill=mix(MUTED, BG, a0), anchor="rm")
            sc = BERT_S[tid]["scores"]
            k = max(range(len(sc)), key=sc.__getitem__)
            y = 92
            bx0, bw = px + 515, 140
            for (n, s) in sentences(tid):
                lines = wrap(d, s, f, 445)
                h = len(lines) * 27
                if n == k:
                    hp = prog(t, 3.0, 0.5) * fade
                    if hp > 0:
                        d.rounded_rectangle((px + 12, y - 8, px + pw - 12, y + max(h, 34) + 6), radius=10,
                                            fill=mix(coll, BG, hp), outline=mix(col, BG, hp), width=3)
                d.text((px + 26, y), str(n), font=font(20, "SemiBold"), fill=mix(MUTED, BG, a0))
                for j, ln in enumerate(lines):
                    d.text((px + 54, y + j * 27), ln, font=f, fill=mix(INK, BG, a0))
                p = prog(t, 1.2, 1.2)
                v = sc[n] * p
                cy = y + max(h, 30) / 2
                d.rounded_rectangle((bx0, cy - 12, bx0 + bw, cy + 12), radius=6, fill=mix(BG, BG, a0), outline=mix(LINE, BG, a0), width=2)
                if v > 0.005:
                    d.rounded_rectangle((bx0, cy - 12, bx0 + max(12, bw * v), cy + 12), radius=6, fill=mix(BERT, BG, fade))
                d.text((bx0 + bw + 12, cy), comma(v, 3), font=font(21, "Bold" if n == k else "Regular"),
                       fill=mix(INK if n == k else MUTED, BG, a0 * min(1, p * 3)), anchor="lm")
                y += max(h, 30) + 22
            a = prog(t, 4.0, 0.5) * fade
            d.text((px + pw / 2, 470), f"BERT elige la oración {k} con p = {comma(sc[k], 3)}",
                   font=font(26, "Bold"), fill=mix(BERT, BG, a), anchor="mm")
        a = prog(t, 5.4, 0.6) * fade
        s = "Misma oración clave, misma confianza: BERT no sabe si es correcta"
        fw = text_w(d, s, font(30, "Bold")) + 60
        d.rounded_rectangle((W / 2 - fw / 2, 560, W / 2 + fw / 2, 640), radius=16, fill=mix(QWEN_L, BG, a), outline=mix(QWEN, BG, a), width=3)
        d.text((W / 2, 600), s, font=font(30, "Bold"), fill=mix(INK, BG, a), anchor="mm")

    return render_gif(draw, (W, H), T, OUT / "04_diagnostico.gif")


# ---------------------------------------------------------------- 6. Pipeline final
def gif_pipeline_final():
    W, H = 1600, 680
    T = 11.0
    N = {
        "texto": (135, 340, 230, 150, "Texto clínico", "oraciones numeradas", INK),
        "bert": (440, 340, 280, 150, "BiomedBERT", "elige la oración k (la más sospechosa)", BERT),
        "haiku": (790, 340, 300, 170, "Claude Haiku 5.5", "¿la oración k es correcta?", HAIKU),
        "si": (1250, 130, 400, 130, "Sin error", "A = 0 · B = −1", OK),
        "corr": (1130, 540, 250, 140, "Corrección", "el mismo LLM", HAIKU),
        "qc": (1430, 540, 270, 140, "Control de calidad", "error en k + corrección", NAVY),
    }
    P = {
        "a": [(252, 340), (298, 340)],
        "b": [(580, 340), (640, 340)],
        "si": [(940, 340), (990, 340), (990, 130), (1050, 130)],
        "no": [(940, 340), (990, 340), (990, 540), (1005, 540)],
        "c": [(1255, 540), (1295, 540)],
    }
    appear = {"texto": 0.1, "bert": 0.7, "haiku": 1.4, "si": 2.2, "corr": 2.2, "qc": 2.7}
    edge_t = {"a": 0.4, "b": 1.1, "si": 1.9, "no": 1.9, "c": 2.5}

    def draw(d, t):
        fade = 1 - prog(t, T - 0.4, 0.4)
        for k, pts in P.items():
            col = OK if k == "si" else (ERR if k == "no" else MUTED)
            arrow(d, pts, prog(t, edge_t[k], 0.5), color=mix(col, BG, fade))
        la = prog(t, 2.2, 0.4) * fade
        d.text((1005, 230), "sí", font=font(28, "Bold"), fill=mix(OK, BG, la), anchor="lm")
        d.text((1005, 450), "no", font=font(28, "Bold"), fill=mix(ERR, BG, la), anchor="lm")
        # recorrido: primero un texto con error (rama "no"), luego uno correcto (rama "sí")
        runs = [(3.2, "no", "Texto con error"), (7.0, "si", "Texto correcto")]
        for t0, branch, lab in runs:
            route = [P["a"], [(298, 340), (298, 340)], P["b"], [(640, 340), (640, 340)]]
            seg_t = [0.5, 0.5, 0.5, 0.7]
            route += [P[branch]] + ([[(1005, 540)], P["c"]] if branch == "no" else [])
            seg_t += [0.7] + ([0.5, 0.4] if branch == "no" else [])
            tt = t - t0
            if 0 <= tt < sum(seg_t) + 0.8:
                pill(d, 40, 600, lab, font(24, "Bold"), BG, mix(ERR if branch == "no" else OK, BG, fade), h=44)
                acc = 0
                for pts, st in zip(route, seg_t):
                    if tt < acc + st:
                        dot(d, poly_point(pts, (tt - acc) / st) if len(pts) > 1 else pts[0], r=13, color=mix(INK, BG, fade))
                        break
                    acc += st
                else:
                    dot(d, route[-1][-1], r=13, color=mix(INK, BG, fade))
                # pulsos en las cajas por las que pasa
                ring(d, *N["bert"][:4], (tt - 0.9) / 0.6, BERT)
                ring(d, *N["haiku"][:4], (tt - 1.9) / 0.6, HAIKU)
                tgt = "si" if branch == "si" else "qc"
                ring(d, *N[tgt][:4], (tt - sum(seg_t)) / 0.7, OK if branch == "si" else NAVY)
        for k, (cx, cy, w, h, ti, sub, col) in N.items():
            box(d, cx, cy, w, h, ti, sub, color=col, a=prog(t, appear[k], 0.5) * fade, tf=28, sf=22)

    return render_gif(draw, (W, H), T, OUT / "06_pipeline_final.gif")


# ---------------------------------------------------------------- 7. Resultados finales
def gif_resultados():
    W, H = 1600, 780
    T = 12.0
    systems = [("Pipeline original (BERT + Qwen)", NAVY, [0.566, 0.547, 0.341]),
               ("BERT + Claude Haiku 5.5", HAIKU, [0.826, 0.823, 0.692]),
               ("Claude Haiku 5.5 solo", HAIKU_SOLO, [0.801, 0.771, 0.601]),
               ("Claude Opus 5.5 solo (techo)", OPUS, [0.948, 0.942, 0.799])]
    groups = ["A: ¿hay error?", "B: ¿qué oración?", "C: ROUGE-1 oficial"]
    x0, x1, y0, y1 = 90, 1150, 690, 90

    def Y(v):
        return y0 - (y0 - y1) * v

    def draw(d, t):
        fade = 1 - prog(t, T - 0.5, 0.5)
        for v in (0, 0.25, 0.5, 0.75, 1.0):
            d.line([(x0, Y(v)), (x1, Y(v))], fill=mix(LINE if v == 0 else SOFT, BG, fade), width=2)
            d.text((x0 - 14, Y(v)), comma(v, 2), font=font(20), fill=mix(MUTED, BG, fade), anchor="rm")
        gw = (x1 - x0) / 3
        bw, bg_ = 66, 12
        for gi, g in enumerate(groups):
            gx = x0 + gi * gw + (gw - (4 * bw + 3 * bg_)) / 2
            d.text((x0 + gi * gw + gw / 2, y0 + 36), g, font=font(26, "Bold"), fill=mix(INK, BG, fade), anchor="mm")
            for si, (_, col, vals) in enumerate(systems):
                t0 = 0.5 + si * 1.0
                p = prog(t, t0, 0.8)
                if p <= 0:
                    continue
                v = vals[gi]
                bx = gx + si * (bw + bg_)
                d.rounded_rectangle((bx, Y(v * p), bx + bw, y0), radius=8, fill=mix(col, BG, fade))
                d.text((bx + bw / 2, Y(v * p) - 18), comma(v, 2), font=font(21, "Bold"),
                       fill=mix(INK, BG, fade * clamp(p * 2 - 1)), anchor="mm")
        # leyenda
        for si, (lab, col, _) in enumerate(systems):
            a = prog(t, 0.5 + si * 1.0, 0.4) * fade
            ly = 110 + si * 66
            d.rounded_rectangle((1190, ly - 16, 1222, ly + 16), radius=6, fill=mix(col, BG, a))
            for j, ln in enumerate(wrap(d, lab, font(23, "SemiBold"), 330)):
                d.text((1236, ly - (12 if len(wrap(d, lab, font(23, 'SemiBold'), 330)) > 1 else 0) + j * 26), ln,
                       font=font(23, "SemiBold"), fill=mix(INK, BG, a), anchor="lm")
        a = prog(t, 5.0, 0.5) * fade
        d.rounded_rectangle((1180, 400, 1590, 530), radius=14, fill=mix(HAIKU_L, BG, a), outline=mix(HAIKU, BG, a), width=3)
        d.text((1385, 440), "BERT + Haiku > Haiku solo", font=font(25, "Bold"), fill=mix(HAIKU, BG, a), anchor="mm")
        d.text((1385, 486), "en A, B y C: el pipeline aporta", font=font(23), fill=mix(INK, BG, a), anchor="mm")
        a = prog(t, 6.4, 0.5) * fade
        d.rounded_rectangle((1180, 556, 1590, 686), radius=14, fill=mix(SOFT, BG, a), outline=mix(NAVY, BG, a), width=3)
        d.text((1385, 596), "Textos correctos reconocidos", font=font(22, "SemiBold"), fill=mix(INK, BG, a), anchor="mm")
        d.text((1385, 646), "14  →  124 de 144", font=font(32, "Bold"), fill=mix(NAVY, BG, a), anchor="mm")

    return render_gif(draw, (W, H), T, OUT / "07_resultados.gif")


# ---------------------------------------------------------------- 8-9. Demos
def demo(tid, path, T=13.5):
    W, H = 1600, 840
    sents = sentences(tid)
    sc = BERT_S[tid]["scores"]
    ver = VERIF[tid]
    k = ver["k"]
    is_ok = ver["is_correct"]
    secs = ver["seconds"]
    f = font(23)
    has_err = ROWS[tid]["Error Flag"] == "1"

    steps = ["1  BiomedBERT elige la oración", "2  Claude Haiku verifica", "3  Resultado"]
    step_t = [1.0, 3.6, 6.0]

    def draw(d, t):
        fade = 1 - prog(t, T - 0.5, 0.5)
        x = 20
        for i, s in enumerate(steps):
            on = t >= step_t[i]
            col = [BERT, HAIKU, NAVY][i]
            w = pill(d, x, 14, s, font(24, "Bold"), mix(BG if on else MUTED, BG, fade),
                     mix(col if on else SOFT, BG, fade), pad=20, h=50)
            x += w + 18
        d.text((W - 24, 39), f"{tid} · conjunto de prueba", font=font(20), fill=mix(MUTED, BG, fade), anchor="rm")
        # panel del texto
        d.rounded_rectangle((20, 84, 1080, 600), radius=18, fill=mix(SOFT, BG, fade), outline=mix(LINE, BG, fade), width=2)
        y = 110
        for (n, s) in sents:
            a = prog(t, 0.1 + n * 0.12, 0.4) * fade
            lines = wrap(d, s, f, 760)
            h = len(lines) * 30
            if n == k:
                hp = prog(t, 2.8, 0.4) * fade
                if hp > 0:
                    d.rounded_rectangle((34, y - 8, 1066, y + h + 4), radius=10, fill=mix(BERT_L, BG, hp),
                                        outline=mix(BERT, BG, hp), width=3)
            d.text((50, y), str(n), font=font(21, "SemiBold"), fill=mix(MUTED, BG, a))
            for j, ln in enumerate(lines):
                d.text((80, y + j * 30), ln, font=f, fill=mix(INK, BG, a))
            p = prog(t, 1.2, 1.3)
            cy = y + h / 2 - 2
            bx0, bw = 860, 120
            d.rounded_rectangle((bx0, cy - 11, bx0 + bw, cy + 11), radius=6, fill=mix(BG, BG, a), outline=mix(LINE, BG, a), width=2)
            v = sc[n] * p
            if v > 0.01:
                d.rounded_rectangle((bx0, cy - 11, bx0 + max(12, bw * v), cy + 11), radius=6, fill=mix(BERT, BG, fade))
            d.text((bx0 + bw + 10, cy), comma(v, 2), font=font(21, "Bold" if n == k else "Regular"),
                   fill=mix(INK if n == k else MUTED, BG, a * min(1, p * 3)), anchor="lm")
            y += h + 20
        # panel de Haiku
        a = prog(t, 3.6, 0.4) * fade
        d.rounded_rectangle((1100, 84, 1580, 600), radius=18, fill=mix(BG, BG, a), outline=mix(HAIKU, mix(LINE, BG, fade), a), width=3)
        d.text((1340, 124), "Claude Haiku 5.5", font=font(28, "Bold"), fill=mix(HAIKU, mix(MUTED, BG, fade), a), anchor="mm")
        if t >= 3.8:
            qa = prog(t, 3.8, 0.4) * fade
            d.rounded_rectangle((1124, 170, 1556, 290), radius=16, fill=mix(SOFT, BG, qa))
            d.text((1144, 200), "Pregunta (DSPy):", font=font(20), fill=mix(MUTED, BG, qa), anchor="lm")
            d.text((1144, 248), f"¿La oración {k} es correcta?", font=font(27, "Bold"), fill=mix(INK, BG, qa), anchor="lm")
        if 4.4 <= t < 5.3:
            for i in range(3):
                ph = ((t - 4.4) * 3 - i * 0.3) % 1
                dot(d, (1310 + i * 30, 350 - 8 * math.sin(ph * math.pi)), r=8, color=mix(MUTED, BG, fade))
        if t >= 5.3:
            aa = prog(t, 5.3, 0.4) * fade
            col, colL = (OK, OK_L) if is_ok else (ERR, ERR_L)
            d.rounded_rectangle((1124, 320, 1556, 470), radius=16, fill=mix(colL, BG, aa), outline=mix(col, BG, aa), width=3)
            d.text((1340, 370), "Sí, es correcta" if is_ok else "No, tiene un error", font=font(32, "Bold"),
                   fill=mix(col, BG, aa), anchor="mm")
            d.text((1340, 424), "→ sin error" if is_ok else "→ y propone la corrección", font=font(24),
                   fill=mix(INK, BG, aa), anchor="mm")
            d.text((1340, 540), f"Respuesta en {comma(secs, 1)} s", font=font(24, "SemiBold"), fill=mix(MUTED, BG, aa), anchor="mm")
        # resultado
        ra = prog(t, 6.0, 0.4) * fade
        d.rounded_rectangle((20, 620, 1580, 826), radius=18, fill=mix(BG, BG, ra), outline=mix(NAVY, mix(LINE, BG, fade), ra), width=3)
        if ra <= 0:
            return
        if not is_ok:
            orig = ROWS[tid]["Error Sentence"]
            corr = ver["corrected_sentence"]
            # parte común al inicio y al final, el resto es el cambio
            i = 0
            while i < min(len(orig), len(corr)) and orig[i] == corr[i]:
                i += 1
            i = orig.rfind(" ", 0, i) + 1
            j = 0
            while j < min(len(orig), len(corr)) - i and orig[-1 - j] == corr[-1 - j]:
                j += 1
            j = j - (orig[len(orig) - j:].find(" ") if " " in orig[len(orig) - j:] else 0)
            pre, old, new, post = orig[:i], orig[i:len(orig) - j], corr[i:len(corr) - j], orig[len(orig) - j:]
            d.text((44, 660), f"Error en la oración {k}:", font=font(24, "Bold"), fill=mix(NAVY, BG, ra), anchor="lm")
            fx = font(26)
            x, yy = 44, 712
            d.text((x, yy), pre, font=fx, fill=mix(INK, BG, ra), anchor="lm")
            x += text_w(d, pre, fx)
            ow = text_w(d, old, fx)
            d.text((x, yy), old, font=fx, fill=mix(ERR, BG, ra), anchor="lm")
            sp = prog(t, 6.4, 0.4)
            d.line([(x, yy + 2), (x + ow * sp, yy + 2)], fill=mix(ERR, BG, fade), width=4)
            x += ow + 12
            kk = int(len(new) * clamp((t - 6.9) / 0.7))
            nf = font(26, "SemiBold")
            if kk:
                part = new[:kk]
                pw = text_w(d, part, nf)
                d.rounded_rectangle((x - 6, yy - 20, x + pw + 6, yy + 20), radius=8, fill=mix(OK_L, BG, fade))
                d.text((x, yy), part, font=nf, fill=mix(OK, BG, fade), anchor="lm")
                x += pw
            if kk == len(new):
                d.text((x + 6, yy), post, font=fx, fill=mix(INK, BG, fade), anchor="lm")
            ca = prog(t, 8.2, 0.5) * fade
            ref = ROWS[tid]["Corrected Sentence"][i:len(ROWS[tid]["Corrected Sentence"]) - j]
            if ca > 0:
                check(d, 44, 760, 34, color=mix(OK, BG, ca))
                d.text((92, 778), f"Coincide con la respuesta correcta: «{ref}»", font=font(25, "SemiBold"),
                       fill=mix(INK, BG, ca), anchor="lm")
        else:
            d.text((44, 668), "Pipeline final (BERT + Haiku):", font=font(26, "Bold"), fill=mix(NAVY, BG, ra), anchor="lm")
            check(d, 470, 650, 34, color=mix(OK, BG, ra))
            d.text((516, 668), f"sin error, en {comma(secs, 1)} s", font=font(26, "Bold"), fill=mix(OK, BG, ra), anchor="lm")
            ca = prog(t, 7.4, 0.5) * fade
            if ca > 0:
                po = PIPE[tid]
                orig = dict(sents)[po["error_sentence_id"]]
                new = po["corrected_sentence"]
                old_term, new_term = "pemphigus vulgaris", "bullous pemphigoid"
                assert old_term in orig and new_term in new
                d.text((44, 728), "Pipeline original (BERT + Qwen):", font=font(26, "Bold"), fill=mix(NAVY, BG, ca), anchor="lm")
                cross(d, 500, 714, 28, color=mix(ERR, BG, ca))
                d.text((546, 728), f"falsa alarma: cambió «{old_term}» por «{new_term}»", font=font(26, "Bold"),
                       fill=mix(ERR, BG, ca), anchor="lm")
            ra2 = prog(t, 8.6, 0.5) * fade
            if ra2 > 0:
                d.text((44, 790), "Respuesta correcta: el texto no tiene error.", font=font(25, "SemiBold"),
                       fill=mix(MUTED, BG, ra2), anchor="lm")
        assert has_err == (not is_ok)

    return render_gif(draw, (W, H), T, path)


def gif_demo_error():
    return demo("ms-train-13", OUT / "08_demo_con_error.gif")


def gif_demo_correcto():
    return demo("ms-train-381", OUT / "09_demo_sin_error.gif")


# ---------------------------------------------------------------- 10. Despliegue
def gif_despliegue():
    W, H = 1600, 720
    T = 12.0

    def draw(d, t):
        fade = 1 - prog(t, T - 0.5, 0.5)
        a = lambda t0: prog(t, t0, 0.5) * fade  # noqa: E731
        # cajas
        box(d, 130, 400, 210, 140, "Usuario", "navegador", color=INK, a=a(0.1), tf=28, sf=22)
        box(d, 600, 110, 470, 120, "Hugging Face", "crissebasbol/medec-biomedbert-detector", color=QWEN, a=a(0.4), tf=28, sf=21)
        # máquina 2
        am2 = a(0.7)
        if am2 > 0:
            d.rounded_rectangle((370, 270, 830, 560), radius=20, fill=mix((239, 246, 255), BG, am2), outline=mix(BERT, BG, am2), width=4)
            d.text((600, 306), "Máquina 2 · Docker", font=font(28, "Bold"), fill=mix(BERT, BG, am2), anchor="mm")
            for i, s in enumerate(["app: FastAPI + interfaz web", "BiomedBERT en CPU", "puerto 8000 (público)"]):
                d.text((600, 368 + i * 46), s, font=font(24), fill=mix(INK, BG, am2), anchor="mm")
            pill(d, 395, 500, "4 vCPU · sin GPU", font(20, "SemiBold"), mix(MUTED, BG, am2), mix(SOFT, BG, am2), h=38)
        aan = a(1.0)
        box(d, 1300, 200, 470, 150, "API de Anthropic", "Claude Haiku 5.5 · pipeline final", color=HAIKU, a=aan, tf=28, sf=22)
        am1 = a(1.3)
        if am1 > 0:
            d.rounded_rectangle((1065, 420, 1535, 650), radius=20, fill=mix(QWEN_L, BG, am1 * 0.6), outline=mix(QWEN, BG, am1), width=4)
            d.text((1300, 456), "Máquina 1 · Docker", font=font(28, "Bold"), fill=mix(QWEN, BG, am1), anchor="mm")
            for i, s in enumerate(["llm: llama.cpp + Qwen2.5-7B Q4", "puerto 8080 (solo red interna)", "pipeline original"]):
                d.text((1300, 512 + i * 44), s, font=font(23), fill=mix(INK, BG, am1), anchor="mm")
        # flechas
        A = {
            "hf": [(600, 172), (600, 268)],
            "req": [(236, 380), (368, 380)],
            "res": [(368, 430), (236, 430)],
            "an_go": [(832, 330), (1062, 230)],
            "an_back": [(1062, 270), (832, 370)],
            "m1": [(832, 500), (1062, 530)],
        }
        arrow(d, A["hf"], prog(t, 1.6, 0.4), color=mix(MUTED, BG, fade), dashed=True)
        arrow(d, A["req"], prog(t, 1.8, 0.4), color=mix(MUTED, BG, fade))
        arrow(d, A["res"], prog(t, 1.8, 0.4), color=mix(MUTED, BG, fade))
        arrow(d, A["an_go"], prog(t, 2.0, 0.4), color=mix(HAIKU, BG, fade))
        arrow(d, A["an_back"], prog(t, 2.0, 0.4), color=mix(HAIKU, BG, fade))
        arrow(d, A["m1"], prog(t, 2.2, 0.4), color=mix(QWEN, BG, fade), dashed=True)
        la = a(2.4)
        d.text((612, 220), "descarga al arrancar", font=font(20), fill=mix(MUTED, BG, la), anchor="lm")
        # recorrido de una petición
        seq = [("hf", 3.0, 0.8), ("req", 4.2, 0.6), ("an_go", 5.4, 0.6), ("an_back", 6.3, 0.6), ("res", 7.2, 0.6)]
        for key, t0, du in seq:
            if t0 <= t < t0 + du:
                dot(d, poly_point(A[key], (t - t0) / du), r=12, color=mix(INK, BG, fade))
        ring(d, 600, 415, 460, 290, (t - 4.8) / 0.6, BERT, radius=20)
        if 4.8 <= t < 5.4:
            d.text((600, 590), "BERT elige la oración", font=font(22, "SemiBold"), fill=mix(BERT, BG, fade), anchor="mm")
        ring(d, 1300, 200, 470, 150, (t - 6.0) / 0.5, HAIKU)
        ring(d, 130, 400, 210, 140, (t - 7.8) / 0.6, OK)
        # cronómetro
        if t >= 4.2:
            ta = prog(t, 4.2, 0.3) * fade
            v = 30 * clamp((t - 4.2) / 3.6)
            s = f"≈ {int(round(v))} s por análisis completo" if v < 30 else "≈ 30 s por análisis completo"
            d.rounded_rectangle((24, 590, 404, 670), radius=16, fill=mix(OK_L, BG, ta), outline=mix(OK, BG, ta), width=3)
            d.text((214, 630), s, font=font(24, "Bold"), fill=mix(OK, BG, ta), anchor="mm")

    return render_gif(draw, (W, H), T, OUT / "10_despliegue.gif")


import math  # noqa: E402

ALL = {
    "problema": gif_problema,
    "diseno": gif_diseno_original,
    "giro": gif_giro,
    "diagnostico": gif_diagnostico,
    "pruebas": gif_pruebas,
    "final": gif_pipeline_final,
    "resultados": gif_resultados,
    "demo_error": gif_demo_error,
    "demo_ok": gif_demo_correcto,
    "despliegue": gif_despliegue,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for nm in names:
        p, n, dur = ALL[nm]()
        print(f"{p.name}: {n} cuadros, {dur:.1f} s, {p.stat().st_size / 1e6:.2f} MB")
