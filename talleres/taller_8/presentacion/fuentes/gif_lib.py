"""Utilidades de dibujo para los GIFs de la presentación (solo Pillow)."""
from __future__ import annotations

import math

import numpy as np
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path("/usr/share/fonts/opentype/inter")

# Paleta (la misma que usa la presentación)
BG = (255, 255, 255)
INK = (31, 41, 55)
MUTED = (107, 114, 128)
LINE = (209, 213, 219)
SOFT = (243, 244, 246)
NAVY = (30, 58, 110)
BERT = (37, 99, 235)
BERT_L = (219, 234, 254)
QWEN = (217, 119, 6)
QWEN_L = (254, 243, 199)
HAIKU = (5, 150, 105)
HAIKU_L = (209, 250, 229)
HAIKU_SOLO = (124, 58, 237)
OPUS = (156, 163, 175)
ERR = (220, 38, 38)
ERR_L = (254, 226, 226)
OK = (22, 163, 74)
OK_L = (220, 252, 231)
GRAY = (156, 163, 175)

_fonts: dict = {}


def font(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    key = (size, weight)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(str(FONT_DIR / f"Inter-{weight}.otf"), size)
    return _fonts[key]


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def prog(t, start, dur):
    """Progreso suavizado (0 a 1) de una fase que empieza en `start` y dura `dur`."""
    return ease((t - start) / dur) if dur > 0 else float(t >= start)


def mix(c, bg=BG, a=1.0):
    a = clamp(a)
    return tuple(int(bg[i] + (c[i] - bg[i]) * a) for i in range(3))


def comma(v, nd=2):
    return f"{v:.{nd}f}".replace(".", ",")


def text_w(d: ImageDraw.ImageDraw, s: str, f) -> float:
    return d.textlength(s, font=f)


def wrap(d, s, f, width):
    words, lines, cur = s.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if text_w(d, trial, f) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def text_center(d, cx, cy, s, f, fill):
    d.text((cx, cy), s, font=f, fill=fill, anchor="mm")


def box(d, cx, cy, w, h, title, sub=None, color=INK, a=1.0, fill=None, tf=30, sf=24, radius=18, lw=4):
    """Caja redondeada centrada con título y subtítulo; `a` controla la aparición."""
    if a <= 0:
        return
    s = 0.92 + 0.08 * a
    w2, h2 = w * s / 2, h * s / 2
    fill = fill if fill is not None else mix(color, BG, 0.10)
    d.rounded_rectangle((cx - w2, cy - h2, cx + w2, cy + h2), radius=radius,
                        fill=mix(fill, BG, a), outline=mix(color, BG, a), width=lw)
    lines_t = wrap(d, title, font(tf, "Bold"), w - 30)
    lines_s = wrap(d, sub, font(sf), w - 30) if sub else []
    lh_t, lh_s = tf * 1.25, sf * 1.3
    total = len(lines_t) * lh_t + len(lines_s) * lh_s + (8 if lines_s else 0)
    y = cy - total / 2 + lh_t / 2
    for ln in lines_t:
        text_center(d, cx, y, ln, font(tf, "Bold"), mix(color, BG, a))
        y += lh_t
    y += 8 - lh_t / 2 + lh_s / 2
    for ln in lines_s:
        text_center(d, cx, y, ln, font(sf), mix(INK, BG, a))
        y += lh_s


def poly_len(pts):
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def poly_point(pts, frac):
    total = poly_len(pts)
    target = clamp(frac) * total
    for i in range(len(pts) - 1):
        seg = math.dist(pts[i], pts[i + 1])
        if target <= seg or i == len(pts) - 2:
            r = 0 if seg == 0 else clamp(target / seg)
            return (pts[i][0] + (pts[i + 1][0] - pts[i][0]) * r,
                    pts[i][1] + (pts[i + 1][1] - pts[i][1]) * r)
        target -= seg
    return pts[-1]


def poly_partial(pts, frac):
    """Puntos de la polilínea hasta la fracción `frac` de su longitud."""
    total = poly_len(pts)
    target = clamp(frac) * total
    out = [pts[0]]
    for i in range(len(pts) - 1):
        seg = math.dist(pts[i], pts[i + 1])
        if target >= seg:
            out.append(pts[i + 1])
            target -= seg
        else:
            r = 0 if seg == 0 else target / seg
            out.append((pts[i][0] + (pts[i + 1][0] - pts[i][0]) * r,
                        pts[i][1] + (pts[i + 1][1] - pts[i][1]) * r))
            break
    return out


def arrow(d, pts, p=1.0, color=MUTED, width=5, head=16, dashed=False):
    if p <= 0:
        return
    part = poly_partial(pts, p)
    if dashed:
        for i in range(len(part) - 1):
            dashed_line(d, part[i], part[i + 1], color, width)
    else:
        d.line(part, fill=color, width=width, joint="curve")
    if p >= 0.98:
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        ang = math.atan2(y2 - y1, x2 - x1)
        a1, a2 = ang + math.radians(150), ang - math.radians(150)
        d.polygon([(x2, y2), (x2 + head * math.cos(a1), y2 + head * math.sin(a1)),
                   (x2 + head * math.cos(a2), y2 + head * math.sin(a2))], fill=color)


def dashed_line(d, a, b, color, width, dash=14, gap=10):
    L = math.dist(a, b)
    if L == 0:
        return
    n = 0.0
    while n < L:
        e = min(n + dash, L)
        p1 = (a[0] + (b[0] - a[0]) * n / L, a[1] + (b[1] - a[1]) * n / L)
        p2 = (a[0] + (b[0] - a[0]) * e / L, a[1] + (b[1] - a[1]) * e / L)
        d.line([p1, p2], fill=color, width=width)
        n += dash + gap


def dot(d, xy, r=11, color=INK):
    x, y = xy
    d.ellipse((x - r, y - r, x + r, y + r), fill=color)


def ring(d, cx, cy, w, h, t, color, radius=18):
    """Pulso: un contorno que crece y se desvanece (t de 0 a 1)."""
    if not 0 < t < 1:
        return
    g = 6 + 22 * t
    d.rounded_rectangle((cx - w / 2 - g, cy - h / 2 - g, cx + w / 2 + g, cy + h / 2 + g),
                        radius=radius + g, outline=mix(color, BG, 1 - t), width=4)


def check(d, x, y, s, color=OK, width=6):
    d.line([(x, y + s * 0.5), (x + s * 0.38, y + s * 0.85), (x + s, y + s * 0.1)],
           fill=color, width=width, joint="curve")


def cross(d, x, y, s, color=ERR, width=6):
    d.line([(x, y), (x + s, y + s)], fill=color, width=width)
    d.line([(x + s, y), (x, y + s)], fill=color, width=width)


def pill(d, x, y, s, f, fg, bg, pad=14, h=None):
    w = text_w(d, s, f) + 2 * pad
    h = h or f.size + 18
    d.rounded_rectangle((x, y, x + w, y + h), radius=h / 2, fill=bg)
    d.text((x + pad, y + h / 2), s, font=f, fill=fg, anchor="lm")
    return w


def render_gif(draw_fn, size, total, path, fps=20, colors=160):
    """Dibuja `draw_fn(d, t)` para cada cuadro y guarda un GIF en bucle con paleta común."""
    W, H = size
    frames, durs = [], []
    step = 1.0 / fps
    n = int(round(total * fps))
    last = None
    for i in range(n):
        im = Image.new("RGB", (W, H), BG)
        draw_fn(ImageDraw.Draw(im), i * step)
        b = im.tobytes()
        if last is not None and b == last:
            durs[-1] += int(1000 * step)
            continue
        frames.append(im)
        durs.append(int(1000 * step))
        last = b
    # Paleta común a partir de cuadros repartidos en el tiempo (evita parpadeo de colores)
    sample = frames[:: max(1, len(frames) // 12)] + [frames[-1]]
    sheet = Image.new("RGB", (W, H * len(sample)), BG)
    for k, f in enumerate(sample):
        sheet.paste(f, (0, k * H))
    pal = sheet.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    # el color más cercano al fondo pasa a ser blanco exacto, para que el GIF no se vea como un recuadro gris
    rgb = pal.getpalette()[: 3 * colors]
    idx = min(range(len(rgb) // 3), key=lambda i: sum((rgb[3 * i + c] - BG[c]) ** 2 for c in range(3)))
    rgb[3 * idx: 3 * idx + 3] = list(BG)
    pal.putpalette(rgb)
    # mapeo exacto al color más cercano (quantize de Pillow redondea y no respeta el blanco puro)
    pal_arr = np.array(rgb, dtype=np.int32).reshape(-1, 3)
    q = []
    for f in frames:
        arr = np.asarray(f, dtype=np.int32).reshape(-1, 3)
        key = (arr[:, 0] << 16) | (arr[:, 1] << 8) | arr[:, 2]
        uk, inv = np.unique(key, return_inverse=True)
        uniq = np.stack([uk >> 16, (uk >> 8) & 255, uk & 255], 1)
        dist = ((uniq[:, None, :] - pal_arr[None, :, :]) ** 2).sum(-1)
        idx_map = dist.argmin(1).astype(np.uint8)
        im = Image.fromarray(idx_map[inv.ravel()].reshape(f.size[1], f.size[0]))
        im.putpalette(rgb)
        q.append(im)
    q[0].save(path, save_all=True, append_images=q[1:], duration=durs, loop=0,
              optimize=False, disposal=1)
    return path, len(frames), sum(durs) / 1000
