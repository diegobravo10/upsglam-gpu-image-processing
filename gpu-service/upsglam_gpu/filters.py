"""Registro de filtros UPSGlam: parámetros + orquestación de kernels.

Cada runner recibe (rc: RunContext, img: (H,W,3) uint8, p: params validados)
y devuelve (imagen_salida, raw_float_o_None, extra_dict).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import assets
from .params import Param, ParamSet

i32, f32 = np.int32, np.float32
BORDERS = {"replicar": 0, "espejo": 1, "cero": 2}
_NULL = np.intp(0)


def _rgb(color) -> tuple:
    return tuple(f32(c) for c in color)


@dataclass(frozen=True)
class FilterSpec:
    code: str
    name: str
    category: str           # "convolución" | "transformación" | "composición" | mezcla
    official: bool          # cuenta dentro de los 6 filtros evaluados
    description: str
    params: ParamSet
    runner: Callable

    def to_dict(self) -> dict:
        return {"code": self.code, "name": self.name, "category": self.category,
                "official": self.official, "description": self.description,
                "params": [p.to_dict() for p in self.params.params]}


def _border_param() -> Param:
    return Param("borde", "choice", "replicar", "Tratamiento de bordes", choices=tuple(BORDERS))


# ---------------------------------------------------------------------------
# Convoluciones genéricas
# ---------------------------------------------------------------------------
def _convolve_rgb(rc, img, mask, border) -> tuple[np.ndarray, np.ndarray | None]:
    W, H = rc.W, rc.H
    k = mask.shape[0]
    src = rc.upload(img)
    raw = rc.alloc(W * H * 3 * 4)
    out = rc.alloc(W * H * 3)
    rc.set_mask(mask)
    rc.launch("conv2d_u8_f32", src, raw, i32(W), i32(H), i32(3), i32(k), i32(border))
    rc.launch("f32_to_u8", raw, out, i32(W), i32(H), i32(3), i32(3), f32(1), f32(0), i32(0))
    res = rc.download(out, (H, W, 3), np.uint8)
    raw_h = rc.download(raw, (H, W, 3), np.float32) if rc.want_raw else None
    return res, raw_h


def mean_mask(k: int) -> np.ndarray:
    return np.full((k, k), 1.0 / (k * k), np.float32)


def sharpen_mask(amount: float, neighbors: int) -> np.ndarray:
    a = amount
    if neighbors == 4:
        return np.array([[0, -a, 0], [-a, 1 + 4 * a, -a], [0, -a, 0]], np.float32)
    return np.array([[-a, -a, -a], [-a, 1 + 8 * a, -a], [-a, -a, -a]], np.float32)


def laplacian_mask(neighbors: int) -> np.ndarray:
    if neighbors == 4:
        return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
    return np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32)


def run_media(rc, img, p):
    out, raw = _convolve_rgb(rc, img, mean_mask(p["k"]), BORDERS[p["borde"]])
    return out, raw, {}


def run_nitidez(rc, img, p):
    out, raw = _convolve_rgb(rc, img, sharpen_mask(p["intensidad"], int(p["vecinos"])),
                             BORDERS[p["borde"]])
    return out, raw, {}


def run_laplaciano(rc, img, p):
    W, H = rc.W, rc.H
    src = rc.upload(img)
    gray = rc.alloc(W * H)
    raw = rc.alloc(W * H * 4)
    out = rc.alloc(W * H * 3)
    rc.set_mask(laplacian_mask(int(p["vecinos"])))
    rc.launch("rgb_to_gray_u8", src, gray, i32(W), i32(H))
    rc.launch("conv2d_u8_f32", gray, raw, i32(W), i32(H), i32(1), i32(3), i32(BORDERS[p["borde"]]))
    absmode, offset = (1, 0.0) if p["modo"] == "absoluto" else (0, 128.0)
    rc.launch("f32_to_u8", raw, out, i32(W), i32(H), i32(1), i32(3),
              f32(p["ganancia"]), f32(offset), i32(absmode))
    res = rc.download(out, (H, W, 3), np.uint8)
    raw_h = rc.download(raw, (H, W), np.float32) if rc.want_raw else None
    return res, raw_h, {}


# ---------------------------------------------------------------------------
# Efecto institucional UPS "Salesiano"
# ---------------------------------------------------------------------------
def _bosco_overlay(rc, buf, bx, by, bw, bh, opacity, outline_px, mode, fade_start):
    W, H = rc.W, rc.H
    asset, aw, ah = rc.engine.bosco_device()
    pad = outline_px + 2
    rx0 = max(0, int(math.floor(bx - pad)))
    ry0 = max(0, int(math.floor(by - pad)))
    rx1 = min(W, int(math.ceil(bx + bw + pad)))
    ry1 = min(H, int(math.ceil(by + bh + pad)))
    rw, rh = rx1 - rx0, ry1 - ry0
    if rw <= 0 or rh <= 0:
        return
    rc.launch("overlay_bosco", buf, i32(W), i32(H), asset, i32(aw), i32(ah),
              f32(bx), f32(by), f32(bw), f32(bh), i32(rx0), i32(ry0), i32(rw), i32(rh),
              *_rgb(assets.NAVY), *_rgb(assets.GOLD),
              f32(opacity), f32(outline_px), i32(mode), f32(fade_start), w=rw, h=rh)


def run_salesiano(rc, img, p):
    W, H = rc.W, rc.H
    S = min(W, H)
    src = rc.upload(img)
    out = rc.alloc(W * H * 3)
    rc.launch("ups_tone", src, out, i32(W), i32(H), f32(p["tono"]),
              *_rgb(assets.NAVY), *_rgb(assets.GOLD))
    if p["marco"] > 0:
        rc.launch("ups_frame", out, i32(W), i32(H), f32(p["marco"]),
                  *_rgb(assets.NAVY), *_rgb(assets.GOLD), *_rgb(assets.LIGHT_BLUE))
    extra = {}
    if p["texto"] and p["marco"] > 0:
        band_top = 0.82 * H + 0.02 * S
        text_h = max(10, int((H - band_top) * 0.62))
        mask = assets.ups_text_mask(text_h, int(W * 0.56))
        mh, mw = mask.shape[:2]
        ox = int(0.05 * W)
        oy = int(band_top + (H - band_top - mh) / 2)
        mbuf = rc.upload(mask)
        rc.launch("overlay_text2", out, i32(W), i32(H), mbuf, i32(mw), i32(mh), i32(ox), i32(oy),
                  *_rgb(assets.WHITE), *_rgb(assets.GOLD), f32(p["marco"]), w=mw, h=mh)
        extra["text_box"] = [ox, oy, mw, mh]
    if p["don_bosco"]:
        _, aw, ah = rc.engine.bosco_device()
        bh = 0.40 * H
        bw = bh * aw / ah
        if bw > 0.42 * W:
            bw = 0.42 * W
            bh = bw * ah / aw
        bx = W - bw - 0.015 * W
        by = H - bh * 0.95
        _bosco_overlay(rc, out, bx, by, bw, bh, 1.0, max(1.5, 0.007 * S), 0, 2.0)
        extra["bosco_box"] = [round(bx, 1), round(by, 1), round(bw, 1), round(bh, 1)]
    return rc.download(out, (H, W, 3), np.uint8), None, extra


# ---------------------------------------------------------------------------
# Montaje Don Bosco sobre el rostro
# ---------------------------------------------------------------------------
def estimate_face(rc, src) -> dict:
    """Detección de piel + reducción en GPU, 2 iteraciones de refinamiento."""
    W, H = rc.W, rc.H
    stats = rc.alloc(5 * 8)
    win = (0.0, 0.0, 1.0, 1.0)
    found = None
    for _ in range(2):
        rc.upload_to(stats, np.zeros(5, np.float64))
        rc.launch("skin_stats", src, i32(W), i32(H), *(f32(v) for v in win), stats)
        s = rc.download(stats, (5,), np.float64)
        n = s[0]
        if n < max(64, 0.004 * W * H):
            break
        mu, mv = s[1] / n, s[2] / n
        su = math.sqrt(max(s[3] / n - mu * mu, 1e-8))
        sv = math.sqrt(max(s[4] / n - mv * mv, 1e-8))
        found = (mu, mv, su, sv, n)
        win = (mu - 2 * su, mv - 2 * sv, mu + 2 * su, mv + 2 * sv)
    S = min(W, H)
    # Plausibilidad: un rostro es una mancha de piel compacta. Si la "piel" está
    # muy dispersa (paisajes cálidos, paredes) o cubre casi toda la imagen, no hay rostro.
    if found is not None:
        mu, mv, su, sv, n = found
        if su > 0.16 or sv > 0.20 or n > 0.6 * W * H:
            found = None
    if found is None:
        return {"method": "fallback", "cx": 0.5 * W, "cy": 0.42 * H, "w": 0.30 * S,
                "skin_pixels": 0}
    mu, mv, su, sv, n = found
    # Para una elipse uniforme de ancho a: sigma_x = a/4  =>  a ~= 4*sigma (+5% por el cuello)
    fw = min(max(4.2 * su * W, 0.12 * S), 0.85 * S)
    cy = mv * H - 0.20 * fw          # el cuello baja el centroide de piel; se compensa
    return {"method": "auto", "cx": mu * W, "cy": cy, "w": fw, "skin_pixels": int(n)}


def run_montaje(rc, img, p):
    W, H = rc.W, rc.H
    S = min(W, H)
    src = rc.upload(img)
    out = rc.alloc(W * H * 3)
    if p["rostro_x"] >= 0 and p["rostro_y"] >= 0 and p["rostro_ancho"] > 0:
        face = {"method": "manual", "cx": p["rostro_x"] * W, "cy": p["rostro_y"] * H,
                "w": p["rostro_ancho"] * W}
    else:
        face = estimate_face(rc, src)
    rc.copy_dtod(out, src, W * H * 3)

    fw = face["w"]
    fh = fw * 1.3
    if p["foco"] > 0:
        rc.launch("face_spotlight", out, i32(W), i32(H), f32(face["cx"]), f32(face["cy"]),
                  f32(fw * 0.62), f32(fh * 0.62), f32(p["foco"]),
                  *_rgb(assets.NAVY), *_rgb(assets.GOLD))
    _, aw, ah = rc.engine.bosco_device()
    ref = assets.BOSCO_FACE
    bw = fw / ref["w"] * p["escala"]
    bh = bw * ah / aw
    bx = face["cx"] - ref["cx"] * bw
    by = face["cy"] - ref["cy"] * bh
    mode = 0 if p["estilo"] == "poster" else 1
    outline = max(1.5, 0.006 * S) if p["contorno"] else 0.0
    _bosco_overlay(rc, out, bx, by, bw, bh, p["opacidad"], outline, mode, 2.0)
    face = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in face.items()}
    return rc.download(out, (H, W, 3), np.uint8), None, {"face": face}


# ---------------------------------------------------------------------------
# Cómic / pop-art
# ---------------------------------------------------------------------------
def _sobel(rc, src, with_angle=False):
    W, H = rc.W, rc.H
    gray = rc.alloc(W * H)
    mag = rc.alloc(W * H * 4)
    ang = rc.alloc(W * H * 4) if with_angle else None
    rc.launch("rgb_to_gray_u8", src, gray, i32(W), i32(H))
    rc.launch("sobel_u8", gray, mag, ang if ang is not None else _NULL, i32(W), i32(H), i32(0))
    return mag, ang


def run_comic(rc, img, p):
    W, H = rc.W, rc.H
    src = rc.upload(img)
    out = rc.alloc(W * H * 3)
    mag, _ = _sobel(rc, src)
    cell = p["punto"] if p["punto"] > 0 else max(4.0, min(W, H) / 120)
    rc.launch("comic", src, mag, out, i32(W), i32(H), i32(p["niveles"]), f32(p["umbral_borde"]),
              f32(cell), f32(p["saturacion"]), f32(p["tinta"]))
    return rc.download(out, (H, W, 3), np.uint8), None, {"dot_cell_px": round(cell, 2)}


# ---------------------------------------------------------------------------
# Extras (fuera de los 6 evaluados)
# ---------------------------------------------------------------------------
def run_glitch(rc, img, p):
    W, H = rc.W, rc.H
    src = rc.upload(img)
    out = rc.alloc(W * H * 3)
    shift = max(1, int(p["desplazamiento"] * W / 100))
    band = max(2, int(H / p["bandas"]))
    rc.launch("glitch", src, out, i32(W), i32(H), i32(shift), i32(band), f32(p["probabilidad"]),
              np.uint32(p["semilla"]), f32(p["barrido"]))
    return rc.download(out, (H, W, 3), np.uint8), None, {}


def run_neon(rc, img, p):
    W, H = rc.W, rc.H
    src = rc.upload(img)
    mag, ang = _sobel(rc, src, with_angle=True)
    mag8 = rc.alloc(W * H)
    glow = rc.alloc(W * H * 4)
    out = rc.alloc(W * H * 3)
    k = int(p["resplandor"])
    rc.set_mask(mean_mask(k))
    rc.launch("f32_to_u8", mag, mag8, i32(W), i32(H), i32(1), i32(1), f32(1), f32(0), i32(0))
    rc.launch("conv2d_u8_f32", mag8, glow, i32(W), i32(H), i32(1), i32(k), i32(0))
    rc.launch("neon", src, mag, ang, glow, out, i32(W), i32(H), f32(p["fondo"]), f32(p["ganancia"]))
    return rc.download(out, (H, W, 3), np.uint8), None, {}


# ---------------------------------------------------------------------------
# Registro
# ---------------------------------------------------------------------------
_SPECS = [
    FilterSpec("media", "Media (desenfoque)", "convolución", True,
               "Promedio de la vecindad k×k con máscara de coeficientes 1/k².",
               ParamSet((Param("k", "int", 5, "Tamaño de la máscara (impar)", 3, 15, odd=True),
                         _border_param())), run_media),
    FilterSpec("nitidez", "Nitidez", "convolución", True,
               "Realce: identidad + a·(−laplaciano). La suma de coeficientes es 1.",
               ParamSet((Param("intensidad", "float", 1.0, "Factor a del realce", 0.1, 5.0),
                         Param("vecinos", "choice", "4", "Vecindad 4 u 8", choices=("4", "8")),
                         _border_param())), run_nitidez),
    FilterSpec("laplaciano", "Laplaciano (bordes)", "convolución", True,
               "Segunda derivada sobre escala de grises. Visualización |v|·g o v·g+128.",
               ParamSet((Param("vecinos", "choice", "8", "Vecindad 4 u 8", choices=("4", "8")),
                         Param("modo", "choice", "absoluto", "Visualización",
                               choices=("absoluto", "desplazado")),
                         Param("ganancia", "float", 2.0, "Multiplicador de visualización", 0.1, 10.0),
                         _border_param())), run_laplaciano),
    FilterSpec("salesiano", "UPS Salesiano", "transformación de color + composición", True,
               "Gradación institucional azul/dorado, marco procedural UPS, texto "
               "'UPS | ORGULLO SALESIANO' y silueta de Don Bosco, todo mezclado en GPU.",
               ParamSet((Param("tono", "float", 0.8, "Intensidad del tono institucional", 0.0, 1.0),
                         Param("marco", "float", 0.9, "Opacidad del marco", 0.0, 1.0),
                         Param("don_bosco", "bool", True, "Mostrar silueta de Don Bosco"),
                         Param("texto", "bool", True, "Mostrar texto institucional"))),
               run_salesiano),
    FilterSpec("montaje_don_bosco", "Montaje Don Bosco", "composición + transformación geométrica", True,
               "Detecta el rostro (piel + reducción paralela en GPU) o usa la caja enviada "
               "por la app, y monta la silueta de Don Bosco escalada con contorno dorado y foco.",
               ParamSet((Param("rostro_x", "float", -1.0, "Centro X del rostro (0..1)", 0.0, 1.0,
                               allow_auto=True),
                         Param("rostro_y", "float", -1.0, "Centro Y del rostro (0..1)", 0.0, 1.0,
                               allow_auto=True),
                         Param("rostro_ancho", "float", -1.0, "Ancho del rostro (0..1 del ancho)",
                               0.02, 1.0, allow_auto=True),
                         Param("estilo", "choice", "poster", "Estilo de la silueta",
                               choices=("poster", "fusion")),
                         Param("opacidad", "float", 1.0, "Opacidad de la silueta", 0.0, 1.0),
                         Param("escala", "float", 1.15, "Escala relativa al rostro", 0.5, 2.0),
                         Param("contorno", "bool", True, "Contorno dorado"),
                         Param("foco", "float", 0.6, "Viñeta y halo alrededor del rostro", 0.0, 1.0))),
               run_montaje),
    FilterSpec("comic", "Cómic pop-art", "convolución (Sobel) + transformación de color", True,
               "Saturación y posterización, semitono a 45° y tinta negra en bordes Sobel.",
               ParamSet((Param("niveles", "int", 5, "Niveles de posterización por canal", 2, 16),
                         Param("umbral_borde", "float", 120.0, "Umbral de magnitud Sobel", 10.0, 1000.0),
                         Param("punto", "float", -1.0, "Celda del semitono en px (-1 auto)", 2.0, 64.0,
                               allow_auto=True),
                         Param("saturacion", "float", 1.4, "Factor de saturación", 0.0, 3.0),
                         Param("tinta", "float", 1.0, "Intensidad de la tinta", 0.0, 1.0))),
               run_comic),
    FilterSpec("glitch", "Glitch", "transformación", False,
               "Bandas desplazadas, separación RGB, líneas de barrido (extra).",
               ParamSet((Param("desplazamiento", "float", 1.2, "Separación RGB (% del ancho)", 0.0, 10.0),
                         Param("bandas", "int", 40, "Número de bandas", 4, 400),
                         Param("probabilidad", "float", 0.25, "Prob. de banda alterada", 0.0, 1.0),
                         Param("semilla", "int", 7, "Semilla", 0, 1_000_000),
                         Param("barrido", "float", 0.18, "Oscurecimiento de líneas", 0.0, 1.0))),
               run_glitch),
    FilterSpec("neon", "Neón", "convolución + transformación", False,
               "Bordes coloreados por orientación con resplandor (extra).",
               ParamSet((Param("resplandor", "int", 9, "Tamaño del resplandor (impar)", 3, 15, odd=True),
                         Param("fondo", "float", 0.25, "Brillo de la foto de fondo", 0.0, 1.0),
                         Param("ganancia", "float", 1.2, "Ganancia de bordes", 0.1, 5.0))),
               run_neon),
]

FILTERS: dict[str, FilterSpec] = {s.code: s for s in _SPECS}
OFFICIAL_FILTERS = [s.code for s in _SPECS if s.official]
