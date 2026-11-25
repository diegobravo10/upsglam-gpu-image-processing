"""Errores HTTP del servicio con código estable para el backend y la app."""
from __future__ import annotations


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str, request_id: str | None = None,
                 headers: dict | None = None, extra: dict | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.request_id = request_id
        self.headers = headers or {}
        self.extra = extra or {}

    def to_dict(self) -> dict:
        d = {"error": self.code, "message": self.message, "status": self.status,
             "request_id": self.request_id}
        d.update(self.extra)
        return d
