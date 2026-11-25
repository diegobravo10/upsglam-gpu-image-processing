"""Logs estructurados (una línea JSON por evento) para correlacionar por request_id."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
             "level": record.levelname, "event": record.getMessage()}
        fields = getattr(record, "fields", None)
        if fields:
            d.update(fields)
        if record.exc_info:
            d["exception"] = self.formatException(record.exc_info)
        return json.dumps(d, ensure_ascii=False, default=str)


def setup_logging(log_file: str | None = None) -> logging.Logger:
    log = logging.getLogger("upsglam.gpu")
    if getattr(log, "_upsglam_configured", False):
        return log
    log.setLevel(logging.INFO)
    log.propagate = False
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    log.addHandler(h)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        log.addHandler(fh)
    log._upsglam_configured = True
    return log


def event(log: logging.Logger, name: str, level: int = logging.INFO, **fields):
    log.log(level, name, extra={"fields": fields})
