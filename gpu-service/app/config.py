"""Configuración del servicio GPU (variables de entorno con valores por defecto)."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in ("1", "true", "yes", "si", "sí", "on")
    return cast(raw)


@dataclass(frozen=True)
class Settings:
    queue_capacity: int = 16          # solicitudes en ESPERA admitidas (sin contar la que se procesa)
    max_image_mb: float = 10.0        # tamaño máximo del archivo recibido
    max_side: int = 4096              # lado máximo en píxeles
    min_side: int = 8
    max_pixels: int = 4096 * 4096     # protección contra "decompression bombs"
    cuda_device: int = 0
    block_x: int = 16
    block_y: int = 16
    sync_timeout_s: float = 60.0      # espera máxima de POST /process antes de responder 504
    result_ttl_s: int = 300           # tiempo que se conserva el resultado para GET /tasks/{id}/result
    max_records: int = 2000           # registros de tareas en memoria
    png_compress_level: int = 1       # 0-9; 1 = rápido (el resultado sigue siendo sin pérdida)
    jpeg_quality: int = 92
    log_file: str | None = None

    @property
    def max_image_bytes(self) -> int:
        return int(self.max_image_mb * 1024 * 1024)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            queue_capacity=_env("GPU_QUEUE_CAPACITY", 16, int),
            max_image_mb=_env("GPU_MAX_IMAGE_MB", 10.0, float),
            max_side=_env("GPU_MAX_SIDE_PX", 4096, int),
            max_pixels=_env("GPU_MAX_PIXELS", 4096 * 4096, int),
            cuda_device=_env("CUDA_DEVICE", 0, int),
            block_x=_env("GPU_BLOCK_X", 16, int),
            block_y=_env("GPU_BLOCK_Y", 16, int),
            sync_timeout_s=_env("GPU_SYNC_TIMEOUT_S", 60.0, float),
            result_ttl_s=_env("GPU_RESULT_TTL_S", 300, int),
            max_records=_env("GPU_MAX_RECORDS", 2000, int),
            png_compress_level=_env("GPU_PNG_COMPRESS_LEVEL", 1, int),
            jpeg_quality=_env("GPU_JPEG_QUALITY", 92, int),
            log_file=_env("GPU_LOG_FILE", None),
        )
