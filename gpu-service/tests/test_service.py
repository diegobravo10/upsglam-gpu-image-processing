"""Pruebas del servicio REST: validación de entradas, estados, métricas, cola y recuperación."""
import io
import json
import queue
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app

SETTINGS = Settings(queue_capacity=4, max_image_mb=1.0, max_side=1024, max_pixels=1024 * 1024,
                    result_ttl_s=60)


def png_bytes(arr: np.ndarray, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt)
    return buf.getvalue()


def rand_img(seed: int, h=120, w=160) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(SETTINGS)) as c:
        yield c


def post(client, data: bytes, filter="media", params=None, endpoint="/process", **form):
    fields = {"filter": filter, "params": json.dumps(params or {}), **form}
    return client.post(endpoint, data=fields, files={"image": ("foto.png", data, "image/png")})


# ------------------------------------------------------------------ básicos
def test_health_y_filtros(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["cpu_fallback"] is False
    assert "RTX" in body["gpu"]["gpu"] or body["gpu"]["gpu"]
    codes = {f["code"] for f in client.get("/filters").json()["filters"]}
    assert {"media", "nitidez", "laplaciano", "salesiano", "montaje_don_bosco", "comic"} <= codes


def test_process_ok_metricas_y_resultado(client):
    img = rand_img(1)
    r = post(client, png_bytes(img), "media", {"k": 3}, request_id="t-ok-1", user_id="userA")
    assert r.status_code == 200, r.text
    t = r.json()
    assert t["status"] == "DONE" and t["request_id"] == "t-ok-1" and t["user_id"] == "userA"
    m = t["metrics"]
    assert m["width"] == 160 and m["height"] == 120
    assert m["block"] == [16, 16, 1] and m["grid"] == [10, 8, 1]
    assert m["total_threads"] == 2 * 10 * 8 * 256 and m["gpu_time_ms"] > 0
    assert m["time_unit"] == "ms" and m["status"] == "DONE"
    assert {"decode", "queue_wait", "gpu_total", "encode", "service_total"} <= set(t["timings_ms"])
    res = client.get(t["result"]["url"])
    assert res.status_code == 200 and res.headers["content-type"] == "image/png"
    assert res.headers["x-request-id"] == "t-ok-1" and res.headers["x-user-id"] == "userA"
    out = np.asarray(Image.open(io.BytesIO(res.content)))
    assert out.shape == img.shape
    assert client.get("/tasks/t-ok-1").json()["status"] == "DONE"


def test_jpeg_entrada_y_salida(client):
    r = post(client, png_bytes(rand_img(2), "JPEG"), "comic", output_format="jpeg")
    assert r.status_code == 200
    res = client.get(r.json()["result"]["url"])
    assert res.headers["content-type"] == "image/jpeg"


def test_tarea_asincrona(client):
    r = post(client, png_bytes(rand_img(3)), "salesiano", endpoint="/tasks", request_id="t-async")
    assert r.status_code == 202 and r.headers["location"] == "/tasks/t-async"
    for _ in range(200):
        st = client.get("/tasks/t-async").json()["status"]
        if st in ("DONE", "FAILED"):
            break
        threading.Event().wait(0.02)
    assert st == "DONE"
    assert client.get("/tasks/t-async/result").status_code == 200
    assert client.delete("/tasks/t-async").status_code == 204
    assert client.get("/tasks/t-async/result").json()["error"] == "NO_RESULT"


# ------------------------------------------------------------------ entradas inválidas
@pytest.mark.parametrize("data,code,status", [
    (b"", "EMPTY_FILE", 400),
    (b"esto es un txt renombrado a .png", "UNSUPPORTED_FORMAT", 415),
    (b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, "CORRUPT_IMAGE", 400),
    (b"\xff\xd8\xff" + b"basura" * 10, "CORRUPT_IMAGE", 400),
])
def test_archivos_invalidos(client, data, code, status):
    r = post(client, data)
    assert r.status_code == status and r.json()["error"] == code
    task = r.json()["task"]
    assert task["status"] == "FAILED" and task["metrics"]["gpu_time_ms"] is None
    assert task["metrics"]["total_threads"] is None                 # no se inventan valores


def test_imagen_excede_limites(client):
    big = np.zeros((1100, 50, 3), np.uint8)                           # lado > 1024
    r = post(client, png_bytes(big))
    assert r.status_code == 413 and r.json()["error"] == "IMAGE_DIMENSIONS"
    tiny = np.zeros((4, 4, 3), np.uint8)
    assert post(client, png_bytes(tiny)).json()["error"] == "IMAGE_DIMENSIONS"
    noise = np.random.default_rng(0).integers(0, 256, (1000, 1000, 3), dtype=np.uint8)
    r = post(client, png_bytes(noise))                                # ~3 MB > 1 MB
    assert r.status_code == 413 and r.json()["error"] == "FILE_TOO_LARGE"


def test_filtro_y_parametros_invalidos(client):
    data = png_bytes(rand_img(4))
    r = post(client, data, "sepia_magica")
    assert r.status_code == 400 and r.json()["error"] == "UNKNOWN_FILTER"
    for params in ({"k": 4}, {"k": 99}, {"radio": 2}):
        r = post(client, data, "media", params)
        assert r.status_code == 422 and r.json()["error"] == "INVALID_PARAMS", params
    r = client.post("/process", data={"filter": "media", "params": "{no json"},
                    files={"image": ("a.png", data, "image/png")})
    assert r.status_code == 422
    r = post(client, data, "media", output_format="gif")
    assert r.status_code == 422
    r = post(client, data, "media", request_id="id con espacios")
    assert r.status_code == 400 and r.json()["error"] == "INVALID_REQUEST_ID"


def test_request_id_duplicado(client):
    data = png_bytes(rand_img(5))
    assert post(client, data, request_id="dup-1").status_code == 200
    r = post(client, data, request_id="dup-1")
    assert r.status_code == 409 and r.json()["error"] == "DUPLICATE_REQUEST"


def test_tarea_inexistente(client):
    assert client.get("/tasks/no-existe").status_code == 404


# ------------------------------------------------------------------ cola y recuperación
def test_cola_llena_rechazo_controlado(client):
    worker = client.app.state.worker
    original = worker._queue
    full = queue.Queue(maxsize=1)
    full.put_nowait("ocupado")
    worker._queue = full                     # simula capacidad agotada
    try:
        r = post(client, png_bytes(rand_img(6)), request_id="t-full")
        assert r.status_code == 503 and r.json()["error"] == "QUEUE_FULL"
        assert r.headers["retry-after"] == "2"
        assert client.get("/tasks/t-full").json()["status"] == "REJECTED"
    finally:
        worker._queue = original
    r = post(client, png_bytes(rand_img(7)), request_id="t-after-full")
    assert r.status_code == 200 and r.json()["status"] == "DONE"     # se recupera
    assert client.get("/stats").json()["counters"]["rejected_queue_full"] >= 1


def test_recuperacion_tras_fallo(client):
    assert post(client, b"no soy imagen").status_code == 415
    r = post(client, png_bytes(rand_img(8)), "laplaciano")
    assert r.status_code == 200 and r.json()["status"] == "DONE"


def test_aislamiento_concurrente(client):
    """Dos usuarios con imágenes distintas en paralelo: cada uno recibe SU resultado."""
    imgs = {"userA": rand_img(10), "userB": np.full((120, 160, 3), 200, np.uint8)}
    expected = {}
    for u, im in imgs.items():                                       # referencia secuencial
        r = post(client, png_bytes(im), "nitidez", user_id=u)
        expected[u] = client.get(r.json()["result"]["url"]).content
    results, errors = [], []

    def call(u, i):
        try:
            r = post(client, png_bytes(imgs[u]), "nitidez", user_id=u, request_id=f"iso-{u}-{i}")
            body = r.json()
            content = client.get(body["result"]["url"]).content
            results.append((u, body["user_id"], body["request_id"], content))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=call, args=(u, i)) for i in range(4) for u in imgs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(results) == 8
    for u, uid, rid, content in results:
        assert uid == u and rid.startswith(f"iso-{u}-")
        assert np.array_equal(np.asarray(Image.open(io.BytesIO(content))),
                              np.asarray(Image.open(io.BytesIO(expected[u]))))


def test_sin_gpu_diagnostico_claro():
    s = Settings(cuda_device=99)                                     # dispositivo inexistente
    with TestClient(create_app(s)) as c:
        h = c.get("/health")
        assert h.status_code == 503 and h.json()["status"] == "unhealthy"
        assert h.json()["gpu_error"] and h.json()["cpu_fallback"] is False
        r = post(c, png_bytes(rand_img(1)))
        assert r.status_code == 503 and r.json()["error"] == "GPU_UNAVAILABLE"
