"""Recursos gráficos institucionales (se CARGAN en CPU; se COMPONEN en GPU).

La guía permite usar bibliotecas para cargar el recurso gráfico; todas las
operaciones de mezcla, escalado y coloreado se hacen en kernels propios.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ASSETS_DIR = os.environ.get(
    "UPSGLAM_ASSETS_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets"))

# Paleta institucional (tomada del diseño data/filtroups.png; ajustar a manual de marca)
NAVY = (16, 42, 92)        # azul UPS
GOLD = (242, 183, 36)      # dorado/amarillo UPS
LIGHT_BLUE = (60, 125, 196)  # celeste de acento
WHITE = (255, 255, 255)

# Región del rostro dentro del recurso don_bosco.png (coordenadas normalizadas)
BOSCO_FACE = {"cx": 0.52, "cy": 0.40, "w": 0.51}   # medido con scripts/extract_don_bosco.py

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]


@lru_cache(maxsize=1)
def don_bosco() -> np.ndarray:
    """Recurso LA (alto, ancho, 2) uint8: canal 0 = tono, canal 1 = cobertura."""
    path = os.path.join(ASSETS_DIR, "don_bosco.png")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta scripts/extract_don_bosco.py para generarlo.")
    im = Image.open(path).convert("LA")
    return np.ascontiguousarray(np.asarray(im, dtype=np.uint8))


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


@lru_cache(maxsize=32)
def ups_text_mask(height_px: int, max_width_px: int) -> np.ndarray:
    """Máscara (h, w, 2) uint8 con 'UPS | ORGULLO SALESIANO'.
    Canal 0 = texto (blanco), canal 1 = barra separadora (dorado)."""
    size = max(8, int(height_px))
    while True:
        big, small = _font(size), _font(max(6, int(size * 0.36)))
        w_ups = big.getbbox("UPS")[2]
        w_small = max(small.getbbox("ORGULLO")[2], small.getbbox("SALESIANO")[2])
        bar_w = max(2, int(size * 0.05))
        gap = int(size * 0.28)
        total = w_ups + gap + bar_w + gap + w_small
        if total <= max_width_px or size <= 8:
            break
        size = int(size * max_width_px / total * 0.98)

    h = int(size * 1.05)
    w = total + 2
    text = Image.new("L", (w, h), 0)
    bar = Image.new("L", (w, h), 0)
    dt, db = ImageDraw.Draw(text), ImageDraw.Draw(bar)
    mid = h // 2
    dt.text((0, mid), "UPS", font=big, fill=255, anchor="lm")
    bx = w_ups + gap
    db.rectangle([bx, int(mid - size * 0.36), bx + bar_w, int(mid + size * 0.36)], fill=255)
    sx = bx + bar_w + gap
    dt.text((sx, mid - int(size * 0.02)), "ORGULLO", font=small, fill=255, anchor="ls")
    dt.text((sx, mid + int(size * 0.02)), "SALESIANO", font=small, fill=255, anchor="lt")
    return np.ascontiguousarray(np.stack([np.asarray(text), np.asarray(bar)], axis=-1))
