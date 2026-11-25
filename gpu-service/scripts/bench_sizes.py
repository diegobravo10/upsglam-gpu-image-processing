"""Tiempo GPU por filtro y resolución (para el informe).

Uso: python scripts/bench_sizes.py samples/input/catedral.png --reps 5
Salida: CSV en samples/output/bench_sizes.csv (mediana de `reps` corridas tras 1 calentamiento).
"""
import argparse
import csv
import os
import statistics
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from upsglam_gpu import GpuEngine, OFFICIAL_FILTERS  # noqa: E402

SIZES = [(640, 480), (1280, 960), (1920, 1440), (3840, 2880)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="samples/output/bench_sizes.csv")
    a = ap.parse_args()
    base = Image.open(a.image).convert("RGB")
    rows = []
    with GpuEngine() as eng:
        for w, h in SIZES:
            img = np.asarray(base.resize((w, h), Image.BICUBIC), dtype=np.uint8)
            for f in OFFICIAL_FILTERS:
                eng.run(f, img)                                   # calentamiento
                ms = [eng.run(f, img).metrics for _ in range(a.reps)]
                row = {"filter": f, "width": w, "height": h,
                       "total_threads": ms[0]["total_threads"],
                       "kernel_launches": ms[0]["kernel_launches"],
                       "gpu_ms_median": round(statistics.median(m["gpu_time_ms"] for m in ms), 4),
                       "h2d_ms_median": round(statistics.median(m["h2d_ms"] for m in ms), 4),
                       "d2h_ms_median": round(statistics.median(m["d2h_ms"] for m in ms), 4),
                       "total_ms_median": round(statistics.median(m["total_ms"] for m in ms), 4)}
                rows.append(row)
                print(row)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("CSV:", a.out)


if __name__ == "__main__":
    main()
