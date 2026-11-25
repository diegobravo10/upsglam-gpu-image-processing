"""Prueba de humo de concurrencia contra el servicio GPU en ejecución.

1. Referencia: procesa secuencialmente una imagen DISTINTA por usuario y guarda el hash
   de los píxeles del resultado validado.
2. Ráfaga: lanza N solicitudes simultáneas mezclando usuarios y verifica que cada
   respuesta (request_id, user_id, píxeles) corresponde a SU solicitud.
3. Informa DONE / REJECTED (503 cola llena) / errores, mediana y p95 del flujo completo
   (POST /process + GET del resultado).

Uso:
  python scripts/smoke_concurrency.py --url http://localhost:8000 --users 4 --requests 40
  python scripts/smoke_concurrency.py --requests 120 --filter media --params '{"k":9}' --size 1920x1440
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import statistics
import time
import uuid

import httpx
import numpy as np
from PIL import Image


def make_image(seed: int, w: int, h: int) -> bytes:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    img = Image.fromarray(base).resize((w, h), Image.BICUBIC)   # textura distinta por usuario
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def pixel_hash(png: bytes) -> str:
    return hashlib.sha256(np.asarray(Image.open(io.BytesIO(png))).tobytes()).hexdigest()


async def run_one(client, url, user, img, flt, params, rid=None):
    rid = rid or f"smoke-{user}-{uuid.uuid4().hex[:10]}"
    t0 = time.perf_counter()
    r = await client.post(f"{url}/process", data={"filter": flt, "params": json.dumps(params),
                                                  "user_id": user, "request_id": rid},
                          files={"image": (f"{user}.jpg", img, "image/jpeg")})
    body = r.json()
    if r.status_code != 200:
        return {"user": user, "rid": rid, "http": r.status_code, "error": body.get("error"),
                "ms": (time.perf_counter() - t0) * 1000}
    res = await client.get(f"{url}{body['result']['url']}")
    ms = (time.perf_counter() - t0) * 1000
    return {"user": user, "rid": rid, "http": 200, "resp_user": body["user_id"],
            "resp_rid": body["request_id"], "resp_hdr_rid": res.headers.get("x-request-id"),
            "hash": pixel_hash(res.content), "ms": ms, "gpu_ms": body["metrics"]["gpu_time_ms"],
            "queue_ms": body["timings_ms"].get("queue_wait")}


def pct(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))] if v else None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--requests", type=int, default=40)
    ap.add_argument("--filter", default="nitidez")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--size", default="1280x960")
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.lower().split("x"))
    params = json.loads(a.params)
    users = [f"user{i}" for i in range(a.users)]
    imgs = {u: make_image(1000 + i, w, h) for i, u in enumerate(users)}

    async with httpx.AsyncClient(timeout=120) as client:
        ref = {}
        for u in users:                                           # referencia validada
            r = await run_one(client, a.url, u, imgs[u], a.filter, params)
            assert r["http"] == 200, r
            ref[u] = r["hash"]
        assert len(set(ref.values())) == len(users), "las referencias deberían ser distintas"

        t0 = time.perf_counter()
        jobs = [run_one(client, a.url, users[i % len(users)], imgs[users[i % len(users)]],
                        a.filter, params) for i in range(a.requests)]
        results = await asyncio.gather(*jobs)
        wall = time.perf_counter() - t0
        stats = (await client.get(f"{a.url}/stats")).json()

    done = [r for r in results if r["http"] == 200]
    rejected = [r for r in results if r.get("error") == "QUEUE_FULL"]
    other = [r for r in results if r["http"] != 200 and r.get("error") != "QUEUE_FULL"]
    mismatches = [r for r in done if r["resp_user"] != r["user"] or r["resp_rid"] != r["rid"]
                  or r["resp_hdr_rid"] != r["rid"] or r["hash"] != ref[r["user"]]]
    lat = [r["ms"] for r in done]
    report = {
        "scenario": {"filter": a.filter, "params": params, "size": a.size, "users": a.users,
                     "concurrent_requests": a.requests},
        "done": len(done), "rejected_queue_full": len(rejected), "other_errors": len(other),
        "mismatches": len(mismatches),
        "wall_s": round(wall, 3),
        "completed_per_s": round(len(done) / wall, 2) if wall else None,
        "latency_ms_median": round(statistics.median(lat), 1) if lat else None,
        "latency_ms_p95": round(pct(lat, 95), 1) if lat else None,
        "gpu_ms_median": round(statistics.median(r["gpu_ms"] for r in done), 3) if done else None,
        "queue_wait_ms_max": round(max(r["queue_ms"] for r in done), 1) if done else None,
        "service_stats": stats,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if mismatches:
        print("ERROR: resultados cruzados:", mismatches[:3])
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
