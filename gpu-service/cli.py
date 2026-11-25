"""CLI para probar los filtros GPU sin el servicio REST.

Ejemplos:
  python cli.py --list
  python cli.py --filter laplaciano -i samples/input/catedral.png -o out.png -p vecinos=4
  python cli.py --all -i samples/input/catedral.png --out-dir samples/output
  python cli.py --filter media -i foto.jpg -o m.png -p k=9 --block 32x8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

from upsglam_gpu import FILTERS, GpuEngine, UpsGlamGpuError


def parse_params(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"parámetro inválido '{it}', usa clave=valor")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser(description="UPSGlam · filtros PyCUDA")
    ap.add_argument("--list", action="store_true", help="listar filtros y parámetros")
    ap.add_argument("--filter", "-f")
    ap.add_argument("--all", action="store_true", help="aplicar todos los filtros")
    ap.add_argument("--input", "-i")
    ap.add_argument("--output", "-o")
    ap.add_argument("--out-dir", default="samples/output")
    ap.add_argument("--param", "-p", action="append", help="clave=valor (repetible)")
    ap.add_argument("--block", default="16x16", help="dimensiones de bloque, ej. 16x16 o 32x8")
    a = ap.parse_args(argv)

    if a.list:
        print(json.dumps([s.to_dict() for s in FILTERS.values()], ensure_ascii=False, indent=2))
        return 0
    if not a.input or not (a.filter or a.all):
        ap.error("se requiere --input y --filter o --all")

    bx, by = (int(v) for v in a.block.lower().split("x"))
    img = load_rgb(a.input)
    codes = list(FILTERS) if a.all else [a.filter]
    params = parse_params(a.param)
    with GpuEngine(block=(bx, by)) as eng:
        print(json.dumps(eng.info(), indent=2), file=sys.stderr)
        os.makedirs(a.out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(a.input))[0]
        for code in codes:
            try:
                res = eng.run(code, img, params if not a.all else None)
            except UpsGlamGpuError as e:
                print(f"[ERROR] {code}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            out = a.output if (a.output and not a.all) else os.path.join(a.out_dir, f"{base}_{code}.png")
            Image.fromarray(res.image).save(out)
            m = dict(res.metrics)
            m.update(res.extra)
            print(json.dumps({"output": out, **m}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
