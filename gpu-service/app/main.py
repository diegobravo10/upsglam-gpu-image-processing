"""UPSGlam · Servicio REST de procesamiento GPU (FastAPI + PyCUDA).

Endpoints
  GET    /health                 estado de la GPU y de la cola (503 si no hay GPU)
  GET    /filters                filtros disponibles y sus parámetros
  POST   /process                procesa y ESPERA el resultado (síncrono, 200)
  POST   /tasks                  encola y responde de inmediato (asíncrono, 202)
  GET    /tasks/{id}             estado, métricas y tiempos de una solicitud
  GET    /tasks/{id}/result      imagen procesada (PNG/JPEG)
  DELETE /tasks/{id}             libera el resultado (el backend ya lo guardó en Storage)
  GET    /stats                  contadores para las pruebas de carga

Ejecutar: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
(¡un solo worker de proceso! la GPU la posee un único hilo dentro del proceso)
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from upsglam_gpu import FILTERS, InvalidParamsError, UnknownFilterError
from upsglam_gpu import GpuUnavailableError, InvalidImageError

from .config import Settings
from .errors import ServiceError
from .imageio import decode_image, encode_image
from .logging_setup import event, setup_logging
from .tasks import TaskRecord, TaskStore, empty_metrics
from .worker import GpuWorker, Job

REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
OUTPUT_FORMATS = ("png", "jpeg")


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or Settings.from_env()
    log = setup_logging(s.log_file)
    store = TaskStore(s.result_ttl_s, s.max_records)
    worker = GpuWorker(s, log)
    pending: set[asyncio.Task] = set()

    async def cleanup_loop():
        while True:
            await asyncio.sleep(30)
            n = store.purge()
            if n:
                event(log, "records_purged", count=n)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        event(log, "service_starting", settings={k: v for k, v in s.__dict__.items()})
        await run_in_threadpool(worker.start)
        cleaner = asyncio.create_task(cleanup_loop())
        yield
        cleaner.cancel()
        await run_in_threadpool(worker.stop)
        event(log, "service_stopped")

    app = FastAPI(title="UPSGlam GPU Service", version="1.0.0", lifespan=lifespan,
                  description="Procesamiento de imágenes con kernels propios PyCUDA")
    app.state.settings, app.state.store, app.state.worker = s, store, worker

    # ------------------------------------------------------------------ errores
    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError):
        return JSONResponse(exc.to_dict(), status_code=exc.status, headers=exc.headers)

    def fail_record(rec: TaskRecord, err: ServiceError, counter: str = "invalid_input"):
        """Fallo ANTES del lanzamiento: queda registrado con métricas GPU = null."""
        rec.status = "REJECTED" if err.code == "QUEUE_FULL" else "FAILED"
        rec.finished_at = time.time()
        rec.error = {"code": err.code, "message": err.message}
        rec.metrics = empty_metrics((rec.input_info or {}).get("width"),
                                    (rec.input_info or {}).get("height"))
        rec.timings["service_total"] = round((time.perf_counter() - rec._t0) * 1000, 3)
        store.count(counter)
        event(log, "request_rejected" if rec.status == "REJECTED" else "request_invalid",
              logging.WARNING, request_id=rec.request_id, user_id=rec.user_id,
              filter=rec.filter, code=err.code, message=err.message)
        err.extra.setdefault("task", rec.to_dict())

    # ------------------------------------------------------------------ envío común
    async def submit(image: UploadFile, filter_code: str, params_raw: str, request_id: str | None,
                     user_id: str | None, output_format: str) -> tuple[TaskRecord, asyncio.Task]:
        store.count("received")
        rid = request_id or uuid.uuid4().hex
        if not REQUEST_ID_RE.match(rid):
            raise ServiceError(400, "INVALID_REQUEST_ID",
                               "request_id debe tener 1-64 caracteres [A-Za-z0-9_-]")
        rec = TaskRecord(rid, user_id, filter_code, None, output_format)
        if not store.add(rec):
            raise ServiceError(409, "DUPLICATE_REQUEST", f"Ya existe una solicitud con id {rid}", rid)
        event(log, "request_received", request_id=rid, user_id=user_id, filter=filter_code,
              content_type=image.content_type, filename=image.filename)
        try:
            if not worker.available:
                raise ServiceError(503, "GPU_UNAVAILABLE",
                                   f"GPU no disponible: {worker.gpu_error}. "
                                   "El servicio no procesa en CPU.", rid)
            if output_format not in OUTPUT_FORMATS:
                raise ServiceError(422, "INVALID_PARAMS",
                                   f"output_format debe ser uno de {list(OUTPUT_FORMATS)}", rid)
            spec = FILTERS.get(filter_code)
            if spec is None:
                raise ServiceError(400, "UNKNOWN_FILTER",
                                   f"Filtro desconocido '{filter_code}'. Disponibles: {sorted(FILTERS)}",
                                   rid)
            try:
                raw = json.loads(params_raw or "{}")
            except json.JSONDecodeError as e:
                raise ServiceError(422, "INVALID_PARAMS", f"params no es JSON válido: {e}", rid)
            if not isinstance(raw, dict):
                raise ServiceError(422, "INVALID_PARAMS", "params debe ser un objeto JSON", rid)
            try:
                rec.params = spec.params.validate(raw)
            except InvalidParamsError as e:
                raise ServiceError(422, "INVALID_PARAMS", str(e), rid)

            data = await image.read(s.max_image_bytes + 1)
            t0 = time.perf_counter()
            arr, info = await run_in_threadpool(decode_image, data, s, rid)
            rec.timings["decode"] = round((time.perf_counter() - t0) * 1000, 3)
            rec.input_info = info
        except ServiceError as e:
            fail_record(rec, e)
            raise

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        job = Job(rid, user_id, filter_code, rec.params, arr, fut, loop, time.perf_counter())
        try:
            worker.submit(job)
        except queue.Full:
            err = ServiceError(503, "QUEUE_FULL",
                               f"Capacidad alcanzada: {s.queue_capacity} solicitudes en espera. "
                               "Reintente más tarde.", rid, headers={"Retry-After": "2"})
            fail_record(rec, err, "rejected_queue_full")
            raise err
        store.count("accepted")
        event(log, "request_queued", request_id=rid, user_id=user_id, filter=filter_code,
              width=info["width"], height=info["height"], queue_size=worker.queue_size())
        task = asyncio.create_task(finalize(rec, fut))
        pending.add(task)
        task.add_done_callback(pending.discard)
        return rec, task

    async def finalize(rec: TaskRecord, fut: asyncio.Future) -> TaskRecord:
        out = await fut
        rec.started_at = out.started_at
        rec.timings["queue_wait"] = round(out.queue_wait_ms, 3)
        rec.timings["gpu_total"] = round(out.gpu_total_ms, 3)
        if out.ok:
            res = out.result
            rec.metrics = {k: v for k, v in res.metrics.items()
                           if k not in ("filter", "params", "status", "total_ms")}
            rec.extra = res.extra
            t0 = time.perf_counter()
            rec.result, rec.content_type = await run_in_threadpool(
                encode_image, res.image, rec.output_format, s)
            rec.timings["encode"] = round((time.perf_counter() - t0) * 1000, 3)
            rec.status = "DONE"
            store.count("done")
        else:
            e = out.error
            code = ("GPU_UNAVAILABLE" if isinstance(e, GpuUnavailableError) else
                    "INVALID_PARAMS" if isinstance(e, (InvalidParamsError, InvalidImageError)) else
                    "UNKNOWN_FILTER" if isinstance(e, UnknownFilterError) else
                    "GPU_PROCESSING_ERROR")
            rec.error = {"code": code, "message": str(e)}
            partial = getattr(e, "metrics", None)
            rec.metrics = empty_metrics(rec.input_info["width"], rec.input_info["height"])
            if partial:   # hubo lanzamientos antes del fallo: se informan tal cual
                rec.metrics.update({k: partial.get(k) for k in rec.metrics if k in partial})
            rec.status = "FAILED"
            store.count("failed")
        rec.finished_at = time.time()
        rec.timings["service_total"] = round((time.perf_counter() - rec._t0) * 1000, 3)
        event(log, "request_finished", request_id=rec.request_id, user_id=rec.user_id,
              filter=rec.filter, status=rec.status, timings_ms=rec.timings,
              gpu_time_ms=(rec.metrics or {}).get("gpu_time_ms"),
              total_threads=(rec.metrics or {}).get("total_threads"),
              error=rec.error)
        return rec

    def http_status_for(rec: TaskRecord) -> int:
        if rec.status == "DONE":
            return 200
        code = (rec.error or {}).get("code")
        return {"GPU_UNAVAILABLE": 503, "INVALID_PARAMS": 422}.get(code, 500)

    # ------------------------------------------------------------------ endpoints
    @app.get("/health")
    async def health():
        body = {"status": "ok" if worker.available else "unhealthy",
                "gpu": worker.gpu_info, "gpu_error": worker.gpu_error,
                "queue": {"size": worker.queue_size(), "capacity": s.queue_capacity,
                          "processing": worker.processing},
                "cpu_fallback": False}
        return JSONResponse(body, status_code=200 if worker.available else 503)

    @app.get("/filters")
    async def filters():
        return {"filters": [spec.to_dict() for spec in FILTERS.values()]}

    @app.post("/process")
    async def process(image: UploadFile = File(...), filter: str = Form(...),
                      params: str = Form("{}"), request_id: str | None = Form(None),
                      user_id: str | None = Form(None), output_format: str = Form("png")):
        rec, task = await submit(image, filter, params, request_id, user_id, output_format)
        try:
            rec = await asyncio.wait_for(asyncio.shield(task), timeout=s.sync_timeout_s)
        except asyncio.TimeoutError:
            raise ServiceError(504, "TIMEOUT",
                               f"La solicitud sigue en proceso tras {s.sync_timeout_s:g} s; "
                               f"consulte GET /tasks/{rec.request_id}", rec.request_id,
                               extra={"task": rec.to_dict()})
        return JSONResponse(rec.to_dict(), status_code=http_status_for(rec))

    @app.post("/tasks", status_code=202)
    async def create_task(image: UploadFile = File(...), filter: str = Form(...),
                          params: str = Form("{}"), request_id: str | None = Form(None),
                          user_id: str | None = Form(None), output_format: str = Form("png")):
        rec, _ = await submit(image, filter, params, request_id, user_id, output_format)
        return JSONResponse(rec.to_dict(), status_code=202,
                            headers={"Location": f"/tasks/{rec.request_id}"})

    def _get(request_id: str) -> TaskRecord:
        rec = store.get(request_id)
        if rec is None:
            raise ServiceError(404, "NOT_FOUND", "Solicitud inexistente o expirada", request_id)
        return rec

    @app.get("/tasks/{request_id}")
    async def get_task(request_id: str):
        return _get(request_id).to_dict()

    @app.get("/tasks/{request_id}/result")
    async def get_result(request_id: str):
        rec = _get(request_id)
        if rec.status in ("QUEUED", "PROCESSING"):
            raise ServiceError(409, "RESULT_NOT_READY", f"Estado actual: {rec.status}", request_id,
                               headers={"Retry-After": "1"})
        if rec.result is None:
            raise ServiceError(409, "NO_RESULT", f"La solicitud terminó en {rec.status}", request_id,
                               extra={"task": rec.to_dict()})
        return Response(rec.result, media_type=rec.content_type,
                        headers={"X-Request-Id": rec.request_id,
                                 "X-User-Id": rec.user_id or "",
                                 "X-Filter": rec.filter or ""})

    @app.delete("/tasks/{request_id}", status_code=204)
    async def delete_task(request_id: str):
        rec = _get(request_id)
        if rec.status in ("QUEUED", "PROCESSING"):
            raise ServiceError(409, "TASK_ACTIVE", "No se puede eliminar una tarea activa", request_id)
        rec.result = None
        return Response(status_code=204)

    @app.get("/stats")
    async def stats():
        up = time.time() - worker.started
        return {**store.snapshot_counters(),
                "queue": {"size": worker.queue_size(), "capacity": s.queue_capacity,
                          "processing": worker.processing},
                "gpu_processed": worker.processed,
                "gpu_busy_ratio": round(worker.busy_ms / 1000 / up, 4) if up > 0 else 0,
                "uptime_s": round(up, 1)}

    return app


app = create_app()
