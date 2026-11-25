"""Motor PyCUDA de UPSGlam.

- Crea y posee el contexto CUDA (el hilo que crea el GpuEngine debe ser el que
  llama a run(); en el servicio REST será un único hilo trabajador).
- Compila TODOS los kernels propios una sola vez (un SourceModule).
- Por cada solicitud crea un RunContext que registra reservas de memoria,
  transferencias y lanzamientos (bloque, grid, hilos) y mide tiempos con
  eventos CUDA. La memoria se libera siempre en `finally`.
"""
from __future__ import annotations

import math
import os
import time
import unicodedata
import warnings
from dataclasses import dataclass, field

import numpy as np

from .errors import (GpuProcessingError, GpuUnavailableError, InvalidImageError,
                     UnknownFilterError, UpsGlamGpuError)
from .hostcompiler import ensure_host_compiler

KERNEL_DIR = os.path.join(os.path.dirname(__file__), "kernels")
KERNEL_FILES = ["common.cuh", "convolution.cu", "composite.cu", "face.cu", "effects.cu"]

MIN_SIDE = 8
MAX_SIDE = 4096


def load_kernel_source() -> str:
    parts = []
    for name in KERNEL_FILES:
        with open(os.path.join(KERNEL_DIR, name), encoding="utf-8") as f:
            parts.append(f"// ---- {name} ----\n{f.read()}")
    src = "\n".join(parts)
    # PyCUDA escribe el .cu con la codificación del sistema (cp1252 en Windows):
    # se eliminan tildes y símbolos de los comentarios para compilar en cualquier SO.
    src = unicodedata.normalize("NFKD", src).replace("−", "-")
    return src.encode("ascii", "ignore").decode("ascii")


@dataclass
class Launch:
    kernel: str
    block: tuple[int, int, int]
    grid: tuple[int, int, int]
    start: object = None
    end: object = None
    time_ms: float | None = None

    @property
    def threads(self) -> int:
        return math.prod(self.block) * math.prod(self.grid)

    def to_dict(self) -> dict:
        return {"kernel": self.kernel, "block": list(self.block), "grid": list(self.grid),
                "threads": self.threads,
                "time_ms": None if self.time_ms is None else round(self.time_ms, 4)}


@dataclass
class FilterResult:
    image: np.ndarray                  # (H, W, 3) uint8
    metrics: dict
    raw: np.ndarray | None = None      # valores float32 antes del recorte (si aplica)
    extra: dict = field(default_factory=dict)


class RunContext:
    """Estado de UNA solicitud: memoria, lanzamientos y tiempos."""

    def __init__(self, engine: "GpuEngine", width: int, height: int, block: tuple[int, int, int],
                 want_raw: bool = False):
        self.engine = engine
        self.cuda = engine.cuda
        self.W, self.H = width, height
        self.block = block
        self.want_raw = want_raw
        self.launches: list[Launch] = []
        self._allocs = []
        self._copies: list[tuple[str, object, object]] = []

    # ---------- memoria ----------
    def alloc(self, nbytes: int):
        buf = self.cuda.mem_alloc(max(1, int(nbytes)))
        self._allocs.append(buf)
        return buf

    def _timed_copy(self, kind: str, fn):
        e0, e1 = self.cuda.Event(), self.cuda.Event()
        e0.record()
        fn()
        e1.record()
        self._copies.append((kind, e0, e1))

    def upload(self, arr: np.ndarray):
        arr = np.ascontiguousarray(arr)
        buf = self.alloc(arr.nbytes)
        self._timed_copy("h2d", lambda: self.cuda.memcpy_htod(buf, arr))
        return buf

    def upload_to(self, dst, arr: np.ndarray):
        arr = np.ascontiguousarray(arr)
        self._timed_copy("h2d", lambda: self.cuda.memcpy_htod(dst, arr))

    def download(self, buf, shape, dtype) -> np.ndarray:
        out = np.empty(shape, dtype=dtype)
        self._timed_copy("d2h", lambda: self.cuda.memcpy_dtoh(out, buf))
        return out

    def copy_dtod(self, dst, src, nbytes: int):
        self._timed_copy("d2d", lambda: self.cuda.memcpy_dtod(dst, src, int(nbytes)))

    def set_mask(self, mask: np.ndarray):
        """Copia la máscara de convolución a memoria constante (c_mask)."""
        mask = np.ascontiguousarray(mask, dtype=np.float32)
        ptr, size = self.engine.mask_symbol
        if mask.nbytes > size:
            raise GpuProcessingError(f"máscara de {mask.shape} excede la memoria constante")
        self._timed_copy("h2d", lambda: self.cuda.memcpy_htod(ptr, mask))

    def free_all(self):
        for b in self._allocs:
            try:
                b.free()
            except Exception:
                pass
        self._allocs.clear()

    # ---------- lanzamientos ----------
    def grid_for(self, w: int, h: int) -> tuple[int, int, int]:
        bx, by, _ = self.block
        return (math.ceil(w / bx), math.ceil(h / by), 1)

    def launch(self, kernel: str, *args, w: int | None = None, h: int | None = None):
        """Lanza `kernel` con un hilo por píxel sobre un área w×h (por defecto la imagen)."""
        grid = self.grid_for(w if w is not None else self.W, h if h is not None else self.H)
        fn = self.engine.function(kernel)
        rec = Launch(kernel, self.block, grid, self.cuda.Event(), self.cuda.Event())
        rec.start.record()
        fn(*args, block=self.block, grid=grid)
        rec.end.record()
        self.launches.append(rec)

    # ---------- métricas ----------
    def finalize_metrics(self) -> dict:
        # Esperar a que TODOS los eventos (kernels y copias) hayan ocurrido en la GPU
        for l in self.launches:
            l.end.synchronize()
        for _, _, e1 in self._copies:
            e1.synchronize()
        for l in self.launches:
            l.time_ms = l.end.time_since(l.start)
        h2d = sum(e1.time_since(e0) for k, e0, e1 in self._copies if k == "h2d")
        d2h = sum(e1.time_since(e0) for k, e0, e1 in self._copies if k == "d2h")
        main = self.launches[0] if self.launches else None
        return {
            "width": self.W,
            "height": self.H,
            "block": list(self.block),
            "grid": list(main.grid) if main else None,
            "kernel_launches": len(self.launches),
            "launches": [l.to_dict() for l in self.launches],
            "total_threads": sum(l.threads for l in self.launches) if self.launches else None,
            "gpu_time_ms": round(sum(l.time_ms for l in self.launches), 4) if self.launches else None,
            "h2d_ms": round(h2d, 4),
            "d2h_ms": round(d2h, 4),
            "time_unit": "ms",
        }


class GpuEngine:
    def __init__(self, device_id: int = 0, block: tuple[int, int] = (16, 16)):
        self.ctx = None
        self._bosco_dev = None
        ensure_host_compiler()
        try:
            import pycuda.driver as cuda
            from pycuda.compiler import SourceModule
        except Exception as e:  # pragma: no cover - depende del entorno
            raise GpuUnavailableError(f"PyCUDA no disponible: {e}") from e
        self.cuda = cuda
        try:
            cuda.init()
            if cuda.Device.count() == 0:
                raise GpuUnavailableError("No se detectó ninguna GPU NVIDIA")
            self.device = cuda.Device(device_id)
            self.ctx = self.device.make_context()
        except GpuUnavailableError:
            raise
        except Exception as e:
            raise GpuUnavailableError(f"No se pudo crear el contexto CUDA: {e}") from e
        try:
            t0 = time.perf_counter()
            with warnings.catch_warnings():   # nvcc imprime el nombre del archivo como "warning"
                warnings.simplefilter("ignore")
                self.module = SourceModule(load_kernel_source(), no_extern_c=False)
            self.compile_s = time.perf_counter() - t0
            self.mask_symbol = self.module.get_global("c_mask")
        except Exception as e:
            self.close()
            raise GpuUnavailableError(f"Error compilando kernels CUDA: {e}") from e
        self._functions: dict[str, object] = {}
        self.default_block = self.validate_block(block)

    # ---------- info ----------
    def info(self) -> dict:
        free, total = self.cuda.mem_get_info()
        major, minor = self.device.compute_capability()
        return {
            "gpu": self.device.name(),
            "compute_capability": f"{major}.{minor}",
            "total_memory_mb": total // 2**20,
            "free_memory_mb": free // 2**20,
            "cuda_driver_version": self.cuda.get_driver_version(),
            "max_threads_per_block": self.device.get_attribute(
                self.cuda.device_attribute.MAX_THREADS_PER_BLOCK),
            "kernel_compile_s": round(self.compile_s, 3),
        }

    def function(self, name: str):
        f = self._functions.get(name)
        if f is None:
            f = self._functions[name] = self.module.get_function(name)
        return f

    @staticmethod
    def validate_block(block) -> tuple[int, int, int]:
        bx, by = int(block[0]), int(block[1])
        n = bx * by
        if bx < 1 or by < 1 or n > 1024 or (n & (n - 1)) != 0:
            raise InvalidImageError(
                f"bloque {bx}x{by} inválido: bx*by debe ser potencia de 2 y <= 1024")
        return (bx, by, 1)

    def bosco_device(self):
        """El recurso Don Bosco se sube UNA vez y queda en memoria global (solo lectura)."""
        if self._bosco_dev is None:
            from .assets import don_bosco
            arr = don_bosco()
            buf = self.cuda.mem_alloc(arr.nbytes)
            self.cuda.memcpy_htod(buf, arr)
            self._bosco_dev = (buf, arr.shape[1], arr.shape[0])
        return self._bosco_dev

    # ---------- ejecución ----------
    def run(self, filter_code: str, image: np.ndarray, params: dict | None = None,
            block: tuple[int, int] | None = None, want_raw: bool = False) -> FilterResult:
        from .filters import FILTERS  # import diferido (evita ciclo)

        spec = FILTERS.get(filter_code)
        if spec is None:
            raise UnknownFilterError(
                f"filtro desconocido '{filter_code}'; disponibles: {sorted(FILTERS)}")
        clean = spec.params.validate(params)
        img = self._check_image(image)
        H, W = img.shape[:2]
        blk = self.validate_block(block) if block else self.default_block

        rc = RunContext(self, W, H, blk, want_raw)
        t0 = time.perf_counter()
        try:
            out = spec.runner(rc, img, clean)
            metrics = rc.finalize_metrics()
        except UpsGlamGpuError:
            raise
        except Exception as e:
            partial = {"width": W, "height": H, "block": list(blk), "grid": None,
                       "total_threads": None, "gpu_time_ms": None, "status": "FAILED",
                       "launches": [l.to_dict() for l in rc.launches]}
            raise GpuProcessingError(f"fallo en GPU ({filter_code}): {e}", partial) from e
        finally:
            rc.free_all()
        metrics.update({"filter": filter_code, "params": clean, "status": "DONE",
                        "total_ms": round((time.perf_counter() - t0) * 1000, 4)})
        image_out, raw, extra = out
        return FilterResult(image_out, metrics, raw if want_raw else None, extra or {})

    @staticmethod
    def _check_image(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise InvalidImageError("la imagen debe ser un arreglo uint8")
        if image.ndim != 3 or image.shape[2] != 3:
            raise InvalidImageError(f"se esperaba (alto, ancho, 3) RGB; recibido {image.shape}")
        h, w = image.shape[:2]
        if min(h, w) < MIN_SIDE or max(h, w) > MAX_SIDE:
            raise InvalidImageError(
                f"dimensiones {w}x{h} fuera de límites [{MIN_SIDE}, {MAX_SIDE}] px por lado")
        return np.ascontiguousarray(image)

    def close(self):
        ctx = getattr(self, "ctx", None)
        if ctx is not None:
            try:
                if self._bosco_dev is not None:
                    self._bosco_dev[0].free()
            except Exception:
                pass
            ctx.pop()
            ctx.detach()
            self.ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
