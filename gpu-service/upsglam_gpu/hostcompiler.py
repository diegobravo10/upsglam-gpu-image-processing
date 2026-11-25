"""Localiza el compilador host (cl.exe) que nvcc necesita en Windows.

PyCUDA compila los kernels en tiempo de ejecución con nvcc, y en Windows nvcc
requiere cl.exe (MSVC) en el PATH. En Linux/Docker se usa gcc y esto no hace nada.
Se puede forzar con la variable de entorno UPSGLAM_CL_DIR.
"""
from __future__ import annotations

import glob
import os
import shutil

_PATTERNS = [
    r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
    r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
]


def ensure_host_compiler() -> str | None:
    if os.name != "nt":
        return None
    found = shutil.which("cl")
    if found:
        return found
    candidates: list[str] = []
    env_dir = os.environ.get("UPSGLAM_CL_DIR")
    if env_dir:
        candidates.append(os.path.join(env_dir, "cl.exe"))
    for pat in _PATTERNS:
        candidates.extend(sorted(glob.glob(pat), reverse=True))
    for cl in candidates:
        if os.path.isfile(cl):
            os.environ["PATH"] = os.path.dirname(cl) + os.pathsep + os.environ.get("PATH", "")
            return cl
    return None
