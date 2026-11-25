"""Validación del archivo REAL (firma/magic bytes + decodificación) y codificación del resultado.

Se usa Pillow solo para leer/convertir/escribir, como permite la guía. Estas funciones
son CPU y se ejecutan en el pool de hilos, nunca en el hilo de la GPU ni en el event loop.
"""
from __future__ import annotations

import io
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings
from .errors import ServiceError

SIGNATURES = [
    (b"\xff\xd8\xff", "JPEG"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
]


def sniff_format(data: bytes) -> str | None:
    for sig, fmt in SIGNATURES:
        if data.startswith(sig):
            return fmt
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


def decode_image(data: bytes, s: Settings, request_id: str | None = None) -> tuple[np.ndarray, dict]:
    if not data:
        raise ServiceError(400, "EMPTY_FILE", "El archivo está vacío", request_id)
    if len(data) > s.max_image_bytes:
        raise ServiceError(413, "FILE_TOO_LARGE",
                           f"El archivo supera el máximo de {s.max_image_mb:g} MB", request_id)
    fmt = sniff_format(data)
    if fmt is None:
        raise ServiceError(415, "UNSUPPORTED_FORMAT",
                           "El contenido no es una imagen JPEG, PNG o WEBP "
                           "(se verifica la firma real del archivo, no la extensión)", request_id)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            probe = Image.open(io.BytesIO(data))
            w, h = probe.size
            if probe.format != fmt:
                raise ServiceError(415, "UNSUPPORTED_FORMAT",
                                   f"Firma {fmt} pero contenido {probe.format}", request_id)
            if min(w, h) < s.min_side or max(w, h) > s.max_side or w * h > s.max_pixels:
                raise ServiceError(413, "IMAGE_DIMENSIONS",
                                   f"Dimensiones {w}x{h} fuera de límites: lado entre {s.min_side} y "
                                   f"{s.max_side} px, máximo {s.max_pixels} píxeles", request_id)
            probe.verify()                              # integridad sin decodificar todo
            im = Image.open(io.BytesIO(data))
            im = ImageOps.exif_transpose(im)            # fotos de celular rotadas por EXIF
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
    except ServiceError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise ServiceError(400, "CORRUPT_IMAGE", f"No se pudo decodificar la imagen: {e}", request_id)
    return np.ascontiguousarray(arr), {"format": fmt, "bytes": len(data),
                                       "width": arr.shape[1], "height": arr.shape[0]}


def encode_image(img: np.ndarray, fmt: str, s: Settings) -> tuple[bytes, str]:
    buf = io.BytesIO()
    pil = Image.fromarray(img)
    if fmt == "jpeg":
        pil.save(buf, "JPEG", quality=s.jpeg_quality)
        return buf.getvalue(), "image/jpeg"
    pil.save(buf, "PNG", compress_level=s.png_compress_level)
    return buf.getvalue(), "image/png"
