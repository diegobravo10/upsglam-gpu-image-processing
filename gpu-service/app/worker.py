"""Trabajador GPU: UN hilo dueño del contexto CUDA + cola FIFO acotada.

Política de concurrencia (documentada en el README/informe):
- El servicio ACEPTA solicitudes concurrentes (FastAPI/uvicorn asíncrono).
- Decodificar/validar y codificar el resultado ocurre en el pool de hilos (CPU, en paralelo).
- La ejecución GPU se SERIALIZA en un único hilo: los kernels de distintas solicitudes no
  se ejecutan simultáneamente. Cada kernel ya ocupa toda la GPU (miles de bloques), el
  contexto CUDA de PyCUDA es por hilo y la máscara vive en memoria __constant__ compartida.
- Si hay `queue_capacity` solicitudes esperando, la siguiente se RECHAZA (503 + Retry-After)
  en vez de acumular memoria y latencia sin límite.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from upsglam_gpu import GpuEngine, GpuProcessingError, GpuUnavailableError, UpsGlamGpuError

from .config import Settings
from .logging_setup import event


@dataclass
class Job:
    request_id: str
    user_id: str | None
    filter: str
    params: dict
    image: np.ndarray
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop
    enqueued_at: float


@dataclass
class Outcome:
    ok: bool
    result: object = None          # FilterResult
    error: Exception | None = None
    queue_wait_ms: float = 0.0
    gpu_total_ms: float = 0.0
    started_at: float = 0.0


class GpuWorker:
    def __init__(self, settings: Settings, log: logging.Logger):
        self.s = settings
        self.log = log
        self._queue: queue.Queue = queue.Queue(maxsize=settings.queue_capacity)
        self._thread = threading.Thread(target=self._run, name="gpu-worker", daemon=True)
        self._ready = threading.Event()
        self.engine: GpuEngine | None = None
        self.gpu_error: str | None = None
        self.gpu_info: dict | None = None
        self.processing: str | None = None
        self.processed = 0
        self.busy_ms = 0.0
        self.started = time.time()

    # ------------------------------------------------------------ ciclo de vida
    def start(self, timeout: float = 180.0):
        self._thread.start()
        self._ready.wait(timeout)

    def stop(self, timeout: float = 30.0):
        if self._thread.is_alive():
            while True:
                try:
                    self._queue.put(None, timeout=1)
                    break
                except queue.Full:
                    continue
            self._thread.join(timeout)

    @property
    def available(self) -> bool:
        return self.engine is not None

    def queue_size(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------ API
    def submit(self, job: Job):
        """Encola sin bloquear. Lanza queue.Full si se alcanzó la capacidad."""
        self._queue.put_nowait(job)

    # ------------------------------------------------------------ hilo GPU
    def _init_engine(self):
        t0 = time.perf_counter()
        try:
            self.engine = GpuEngine(device_id=self.s.cuda_device, block=(self.s.block_x, self.s.block_y))
            self.gpu_info = self.engine.info()
            self.gpu_error = None
            event(self.log, "gpu_ready", init_s=round(time.perf_counter() - t0, 3), **self.gpu_info)
        except GpuUnavailableError as e:
            self.engine = None
            self.gpu_error = str(e)
            event(self.log, "gpu_unavailable", logging.ERROR, error=self.gpu_error,
                  hint="Verifique driver NVIDIA, `nvidia-smi` y que el contenedor tenga --gpus all")

    def _run(self):
        self._init_engine()
        self._ready.set()
        while True:
            job = self._queue.get()
            if job is None:
                break
            self._process(job)
        if self.engine is not None:
            try:
                self.engine.close()
            except Exception:
                pass

    def _process(self, job: Job):
        started = time.time()
        wait_ms = (time.perf_counter() - job.enqueued_at) * 1000
        self.processing = job.request_id
        event(self.log, "gpu_start", request_id=job.request_id, user_id=job.user_id,
              filter=job.filter, queue_wait_ms=round(wait_ms, 3), queue_size=self._queue.qsize())
        t0 = time.perf_counter()
        try:
            if self.engine is None:
                raise GpuUnavailableError(self.gpu_error or "GPU no disponible")
            res = self.engine.run(job.filter, job.image, job.params)
            out = Outcome(True, res, None, wait_ms, (time.perf_counter() - t0) * 1000, started)
            m = res.metrics
            event(self.log, "gpu_done", request_id=job.request_id, user_id=job.user_id,
                  filter=job.filter, device=(self.gpu_info or {}).get("gpu"),
                  width=m["width"], height=m["height"], block=m["block"], grid=m["grid"],
                  total_threads=m["total_threads"], gpu_time_ms=m["gpu_time_ms"],
                  kernels=[l["kernel"] for l in m["launches"]],
                  h2d_ms=m["h2d_ms"], d2h_ms=m["d2h_ms"])
        except UpsGlamGpuError as e:
            out = Outcome(False, None, e, wait_ms, (time.perf_counter() - t0) * 1000, started)
            event(self.log, "gpu_failed", logging.ERROR, request_id=job.request_id,
                  user_id=job.user_id, filter=job.filter, error_type=type(e).__name__, error=str(e))
            if isinstance(e, GpuProcessingError):
                self._check_context()
        except Exception as e:  # defensa: el hilo GPU nunca debe morir
            out = Outcome(False, None, e, wait_ms, (time.perf_counter() - t0) * 1000, started)
            event(self.log, "gpu_failed", logging.ERROR, request_id=job.request_id,
                  error_type=type(e).__name__, error=str(e))
        finally:
            self.processing = None
        self.processed += 1
        self.busy_ms += out.gpu_total_ms
        job.loop.call_soon_threadsafe(_resolve, job.future, out)

    def _check_context(self):
        """Tras un error CUDA, verifica el contexto; si quedó inutilizable lo recrea."""
        try:
            self.engine.cuda.Context.synchronize()
        except Exception as e:
            event(self.log, "gpu_context_reset", logging.WARNING, error=str(e))
            try:
                self.engine.close()
            except Exception:
                pass
            self.engine = None
            self._init_engine()


def _resolve(fut: asyncio.Future, out: Outcome):
    if not fut.done():
        fut.set_result(out)
