"""Registro de solicitudes: estado, tiempos, métricas y resultado (en memoria, con TTL).

Estados: QUEUED -> PROCESSING -> DONE | FAILED ; REJECTED (cola llena) ; FAILED también
para entradas inválidas detectadas antes del lanzamiento (métricas GPU = null).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

ACTIVE = ("QUEUED", "PROCESSING")
FINAL = ("DONE", "FAILED", "REJECTED")

GPU_METRIC_KEYS = ("width", "height", "block", "grid", "kernel_launches", "launches",
                   "total_threads", "gpu_time_ms", "h2d_ms", "d2h_ms", "time_unit")


def _iso(ts: float | None) -> str | None:
    return None if ts is None else datetime.fromtimestamp(ts, timezone.utc).isoformat(
        timespec="milliseconds")


def empty_metrics(width=None, height=None) -> dict:
    """Métricas para fallos antes del lanzamiento: NO se inventan valores."""
    m = {k: None for k in GPU_METRIC_KEYS}
    m.update({"width": width, "height": height, "time_unit": "ms"})
    return m


@dataclass
class TaskRecord:
    request_id: str
    user_id: str | None
    filter: str | None
    params: dict | None
    output_format: str = "png"
    status: str = "QUEUED"
    received_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    input_info: dict | None = None
    metrics: dict | None = None
    extra: dict | None = None
    timings: dict = field(default_factory=dict)
    error: dict | None = None
    result: bytes | None = None
    content_type: str | None = None
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def to_dict(self) -> dict:
        info = self.input_info or {}
        m = self.metrics or empty_metrics(info.get("width"), info.get("height"))
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "status": self.status,
            "filter": self.filter,
            "params": self.params,
            "input": self.input_info,
            "metrics": {**m, "status": self.status},
            "timings_ms": self.timings,
            "extra": self.extra or {},
            "received_at": _iso(self.received_at),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "result": ({"url": f"/tasks/{self.request_id}/result", "content_type": self.content_type,
                        "bytes": len(self.result)} if self.result is not None else None),
            "error": self.error,
        }


class TaskStore:
    def __init__(self, ttl_s: int, max_records: int):
        self.ttl_s = ttl_s
        self.max_records = max_records
        self._lock = threading.Lock()
        self._items: dict[str, TaskRecord] = {}
        self.counters = {"received": 0, "accepted": 0, "rejected_queue_full": 0,
                         "invalid_input": 0, "done": 0, "failed": 0}

    def add(self, rec: TaskRecord) -> bool:
        with self._lock:
            if rec.request_id in self._items:
                return False
            self._items[rec.request_id] = rec
            return True

    def get(self, request_id: str) -> TaskRecord | None:
        with self._lock:
            return self._items.get(request_id)

    def count(self, key: str):
        with self._lock:
            self.counters[key] += 1

    def snapshot_counters(self) -> dict:
        with self._lock:
            by_status: dict[str, int] = {}
            for r in self._items.values():
                by_status[r.status] = by_status.get(r.status, 0) + 1
            return {"counters": dict(self.counters), "records_by_status": by_status,
                    "records": len(self._items)}

    def purge(self) -> int:
        """Elimina registros terminados con TTL vencido y, si hay exceso, los más antiguos."""
        now = time.time()
        with self._lock:
            old = [k for k, r in self._items.items()
                   if r.status in FINAL and r.finished_at and now - r.finished_at > self.ttl_s]
            for k in old:
                del self._items[k]
            excess = len(self._items) - self.max_records
            if excess > 0:
                finished = sorted((r.finished_at or 0, k) for k, r in self._items.items()
                                  if r.status in FINAL)
                for _, k in finished[:excess]:
                    del self._items[k]
                    old.append(k)
            return len(old)
