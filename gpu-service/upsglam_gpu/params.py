"""Definición y validación de parámetros de cada filtro."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import InvalidParamsError


@dataclass(frozen=True)
class Param:
    name: str
    kind: str                      # "int" | "float" | "bool" | "choice"
    default: Any
    description: str
    min: float | None = None
    max: float | None = None
    choices: tuple = ()
    odd: bool = False              # para tamaños de máscara
    allow_auto: bool = False       # acepta -1 = "automático"

    def to_dict(self) -> dict:
        d = {"name": self.name, "type": self.kind, "default": self.default,
             "description": self.description}
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.choices:
            d["choices"] = list(self.choices)
        if self.allow_auto:
            d["auto_value"] = -1
        return d

    def coerce(self, raw: Any) -> Any:
        try:
            if self.kind == "bool":
                if isinstance(raw, bool):
                    return raw
                s = str(raw).strip().lower()
                if s in ("1", "true", "si", "sí", "yes", "on"):
                    return True
                if s in ("0", "false", "no", "off"):
                    return False
                raise ValueError
            if self.kind == "choice":
                s = str(raw).strip().lower()
                if s not in self.choices:
                    raise InvalidParamsError(
                        f"'{self.name}' debe ser uno de {list(self.choices)} (recibido '{raw}')")
                return s
            if isinstance(raw, bool):
                raise ValueError
            v = int(raw) if self.kind == "int" else float(raw)
            if self.kind == "int" and float(raw) != v:
                raise ValueError
        except InvalidParamsError:
            raise
        except (TypeError, ValueError):
            raise InvalidParamsError(f"'{self.name}' debe ser de tipo {self.kind} (recibido '{raw}')")
        if self.allow_auto and v == -1:
            return v
        if v != v or (self.min is not None and v < self.min) or (self.max is not None and v > self.max):
            raise InvalidParamsError(
                f"'{self.name}'={raw} fuera de rango [{self.min}, {self.max}]")
        if self.odd and int(v) % 2 == 0:
            raise InvalidParamsError(f"'{self.name}' debe ser impar (recibido {raw})")
        return v


@dataclass(frozen=True)
class ParamSet:
    params: tuple[Param, ...] = field(default_factory=tuple)

    def validate(self, raw: dict | None) -> dict:
        raw = dict(raw or {})
        known = {p.name for p in self.params}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise InvalidParamsError(
                f"parámetros no permitidos: {unknown}; permitidos: {sorted(known)}")
        return {p.name: (p.coerce(raw[p.name]) if p.name in raw else p.default) for p in self.params}
