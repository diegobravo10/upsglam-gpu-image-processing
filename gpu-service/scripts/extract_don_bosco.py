"""Extrae la silueta de Don Bosco de data/filtroups.png y genera assets/don_bosco.png.

Preparación de recurso gráfico (se ejecuta UNA vez, en CPU, fuera del filtro).
El PNG resultante es de 2 canales (modo "LA"):
  L (tono)     : 0 = zona azul de la silueta, 255 = zona clara (rostro/cuello)
  A (cobertura): 255 = dentro de la silueta, 0 = fondo transparente
Los kernels de GPU son los que escalan, colorean y mezclan este recurso.

Uso:  python scripts/extract_don_bosco.py ../data/filtroups.png assets/don_bosco.png
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SRC_BOX = (1125, 122, 1292, 282)   # recuadro del ícono "Silueta Don Bosco"
SCALE = 4                          # se agranda para suavizar bordes
CLOSE_R = 36                       # radio del disco de cierre (px en escala x4)


def _disk_dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Dilatación con disco aproximada mediante desenfoque gaussiano + umbral."""
    b = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(r / 2))
    return np.asarray(b) > 6


def main(src: str, dst: str) -> None:
    im = Image.open(src).convert("RGB").crop(SRC_BOX)
    im = im.resize((im.width * SCALE, im.height * SCALE), Image.LANCZOS)
    # margen blanco para que el relleno desde la esquina siempre parta del fondo
    pad = CLOSE_R * 3
    canvas = Image.new("RGB", (im.width + 2 * pad, im.height + 2 * pad), (255, 255, 255))
    canvas.paste(im, (pad, pad))
    im = canvas
    rgb = np.asarray(im).astype(np.float32)
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    navy = (lum < 150).astype(np.uint8) * 255          # trazos azules

    # Cierre morfológico con elemento circular (disco) para sellar el contorno
    # del rostro: dilatar -> rellenar fondo desde las esquinas -> erosionar.
    dil = _disk_dilate(navy, CLOSE_R)
    bg = Image.fromarray(np.where(dil, 0, 255).astype(np.uint8)).copy()
    ImageDraw.floodfill(bg, (0, 0), 128)
    ImageDraw.floodfill(bg, (bg.width - 1, 0), 128)
    background = np.asarray(bg) == 128
    background = _disk_dilate(background.astype(np.uint8) * 255, CLOSE_R)  # erosión del interior
    inside = np.where(background, 0, 255).astype(np.uint8)
    inside = np.maximum(inside, navy)

    coverage = Image.fromarray(inside).filter(ImageFilter.GaussianBlur(1.5))
    tone = Image.fromarray(np.clip((lum - 90) * 2.0, 0, 255).astype(np.uint8))
    tone = tone.filter(ImageFilter.GaussianBlur(0.8))

    out = Image.merge("LA", (tone, coverage))
    bbox = coverage.point(lambda v: 255 if v > 8 else 0).getbbox()
    out = out.crop(bbox)
    out.save(dst)
    print(f"guardado {dst} tamaño={out.size}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../data/filtroups.png",
         sys.argv[2] if len(sys.argv) > 2 else "assets/don_bosco.png")
