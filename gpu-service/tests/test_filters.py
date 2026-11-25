"""Pruebas de corrección de los filtros GPU con entradas pequeñas de resultado conocido.

Se comparan los valores float32 ANTES del recorte a 8 bits (want_raw=True) cuando
el filtro es una convolución. La referencia NumPy solo existe para las pruebas.
Ejecutar:  python -m pytest -v
"""
import math

import numpy as np
import pytest

from upsglam_gpu import (GpuEngine, InvalidImageError, InvalidParamsError, UnknownFilterError)
from upsglam_gpu import assets
from upsglam_gpu.filters import laplacian_mask, mean_mask, sharpen_mask


@pytest.fixture(scope="module")
def eng():
    e = GpuEngine()
    yield e
    e.close()


def gray_img(values: np.ndarray) -> np.ndarray:
    """Imagen RGB con los tres canales iguales (gris exacto)."""
    v = np.asarray(values, dtype=np.uint8)
    return np.repeat(v[:, :, None], 3, axis=2).copy()


def ref_conv(img2d: np.ndarray, mask: np.ndarray, border: str) -> np.ndarray:
    """Convolución de referencia (correlación, como el kernel) en float64."""
    k = mask.shape[0]
    r = k // 2
    mode = {"replicar": "edge", "espejo": "reflect", "cero": "constant"}[border]
    p = np.pad(img2d.astype(np.float64), r, mode=mode)
    out = np.zeros(img2d.shape, np.float64)
    for j in range(k):
        for i in range(k):
            out += mask[j, i] * p[j:j + img2d.shape[0], i:i + img2d.shape[1]]
    return out


# ---------------------------------------------------------------- media
def test_media_impulso(eng):
    a = np.zeros((9, 9), np.uint8)
    a[4, 4] = 90
    res = eng.run("media", gray_img(a), {"k": 3}, want_raw=True)
    raw = res.raw[..., 0]
    assert np.allclose(raw[3:6, 3:6], 10.0, atol=1e-4)      # 90/9 en la vecindad 3x3
    assert raw[0, 0] == 0 and raw[4, 6] == 0
    assert res.image[4, 4, 0] == 10


def test_media_constante_bordes(eng):
    img = gray_img(np.full((8, 8), 77))
    for borde in ("replicar", "espejo"):
        raw = eng.run("media", img, {"k": 5, "borde": borde}, want_raw=True).raw
        assert np.allclose(raw, 77.0, atol=1e-3), borde
    raw = eng.run("media", img, {"k": 3, "borde": "cero"}, want_raw=True).raw[..., 0]
    assert math.isclose(raw[0, 0], 77 * 4 / 9, rel_tol=1e-5)     # esquina: 4 de 9 vecinos


@pytest.mark.parametrize("borde", ["replicar", "espejo", "cero"])
@pytest.mark.parametrize("k", [3, 7])
def test_media_vs_referencia_dimensiones_irregulares(eng, borde, k):
    rng = np.random.default_rng(k)
    a = rng.integers(0, 256, (23, 37), dtype=np.uint8)       # no múltiplo de 16
    raw = eng.run("media", gray_img(a), {"k": k, "borde": borde}, want_raw=True).raw[..., 0]
    assert np.allclose(raw, ref_conv(a, mean_mask(k), borde), atol=1e-3)


# ---------------------------------------------------------------- nitidez
def test_nitidez_impulso_y_constante(eng):
    a = np.full((8, 8), 50, np.uint8)
    raw = eng.run("nitidez", gray_img(a), {"intensidad": 1.0}, want_raw=True).raw
    assert np.allclose(raw, 50.0, atol=1e-4)                  # suma de coeficientes = 1
    a = np.zeros((9, 9), np.uint8)
    a[4, 4] = 40
    res = eng.run("nitidez", gray_img(a), {"intensidad": 1.0, "vecinos": "4"}, want_raw=True)
    raw = res.raw[..., 0]
    assert math.isclose(raw[4, 4], 200.0, abs_tol=1e-4)       # (1+4a)*40
    assert math.isclose(raw[3, 4], -40.0, abs_tol=1e-4)       # -a*40 (antes del recorte)
    assert res.image[3, 4, 0] == 0 and res.image[4, 4, 0] == 200


def test_nitidez_vs_referencia(eng):
    rng = np.random.default_rng(3)
    a = rng.integers(0, 256, (20, 31), dtype=np.uint8)
    raw = eng.run("nitidez", gray_img(a), {"intensidad": 0.7, "vecinos": "8", "borde": "espejo"},
                  want_raw=True).raw[..., 0]
    assert np.allclose(raw, ref_conv(a, sharpen_mask(0.7, 8), "espejo"), atol=1e-3)


# ---------------------------------------------------------------- laplaciano
def test_laplaciano_impulso(eng):
    a = np.zeros((9, 9), np.uint8)
    a[4, 4] = 30
    raw = eng.run("laplaciano", gray_img(a), {"vecinos": "4"}, want_raw=True).raw
    assert raw[4, 4] == -120 and raw[3, 4] == 30 and raw[3, 3] == 0
    raw8 = eng.run("laplaciano", gray_img(a), {"vecinos": "8"}, want_raw=True).raw
    assert raw8[4, 4] == -240 and raw8[3, 3] == 30


def test_laplaciano_bordes_y_visualizacion(eng):
    img = gray_img(np.full((8, 8), 60))
    assert np.all(eng.run("laplaciano", img, {"vecinos": "4"}, want_raw=True).raw == 0)
    raw = eng.run("laplaciano", img, {"vecinos": "4", "borde": "cero"}, want_raw=True).raw
    assert raw[0, 0] == -120 and raw[0, 4] == -60 and raw[4, 4] == 0
    a = np.zeros((9, 9), np.uint8)
    a[4, 4] = 30
    out = eng.run("laplaciano", gray_img(a), {"vecinos": "4", "ganancia": 1.0}).image
    assert out[4, 4, 0] == 120 and out[3, 4, 0] == 30           # |v|
    out = eng.run("laplaciano", gray_img(a), {"vecinos": "4", "modo": "desplazado",
                                               "ganancia": 1.0}).image
    assert out[4, 4, 0] == 8 and out[3, 4, 0] == 158            # v + 128


def test_laplaciano_vs_referencia(eng):
    rng = np.random.default_rng(5)
    a = rng.integers(0, 256, (17, 29), dtype=np.uint8)
    raw = eng.run("laplaciano", gray_img(a), {"vecinos": "8", "borde": "replicar"}, want_raw=True).raw
    assert np.allclose(raw, ref_conv(a, laplacian_mask(8), "replicar"), atol=1e-3)


# ---------------------------------------------------------------- salesiano (UPS)
OFF = {"tono": 0.0, "marco": 0.0, "don_bosco": False, "texto": False}


def test_salesiano_identidad(eng):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
    assert np.array_equal(eng.run("salesiano", img, OFF).image, img)


def test_salesiano_tono_extremos(eng):
    img = np.zeros((16, 16, 3), np.uint8)
    img[8:] = 255
    out = eng.run("salesiano", img, {**OFF, "tono": 1.0}).image
    assert np.all(out[:8] == 0) and np.all(out[8:] == 255)      # soft-light conserva 0 y 1
    mid = eng.run("salesiano", np.full((16, 16, 3), 128, np.uint8), {**OFF, "tono": 1.0}).image
    assert mid[0, 0, 2] != mid[0, 0, 0]                          # gris medio queda teñido


def test_salesiano_regiones_del_marco(eng):
    H, W = 200, 300
    img = np.full((H, W, 3), 200, np.uint8)
    out = eng.run("salesiano", img, {**OFF, "marco": 1.0}).image
    assert tuple(out[0, 0]) == assets.NAVY                       # cuña superior izquierda
    assert tuple(out[100, 150]) == (200, 200, 200)               # centro intacto
    # banda inferior: azul con sombreado vertical (fórmula del kernel)
    y, x = H - 1, 0
    by = (0.82 - 0.20 * ((x + 0.5) / W) ** 1.6) * H
    shade = 1 - 0.30 * min(max((y + 0.5 - by) / (H - by + 1), 0), 1)
    exp = tuple(int(min(max(c * shade, 0), 255) + 0.5) for c in assets.NAVY)
    assert tuple(out[y, x]) == exp


def test_salesiano_completo_modifica_esquina_bosco(eng):
    img = np.full((300, 400, 3), 200, np.uint8)
    res = eng.run("salesiano", img, {"tono": 0.0, "marco": 0.0, "don_bosco": True, "texto": False})
    bx, by, bw, bh = res.extra["bosco_box"]
    assert bx > 400 * 0.5 and by > 300 * 0.5                    # esquina inferior derecha
    assert np.array_equal(res.image[:100, :150], img[:100, :150])
    assert not np.array_equal(res.image[int(by):, int(bx):], img[int(by):, int(bx):])


# ---------------------------------------------------------------- montaje Don Bosco
def synthetic_face(W=320, H=400, cx=160, cy=170, rx=60, ry=78):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (40, 70, 120)
    yy, xx = np.mgrid[0:H, 0:W]
    inside = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
    img[inside] = (224, 172, 140)                                # tono piel
    return img


def test_montaje_deteccion_automatica(eng):
    res = eng.run("montaje_don_bosco", synthetic_face(), {"foco": 0.0})
    f = res.extra["face"]
    assert f["method"] == "auto"
    assert abs(f["cx"] - 160) < 8
    assert abs(f["w"] - 120) < 0.15 * 120                        # ancho ~ 2*rx


def test_montaje_sin_rostro_usa_respaldo(eng):
    img = np.zeros((200, 200, 3), np.uint8)
    img[:] = (30, 90, 200)
    assert eng.run("montaje_don_bosco", img).extra["face"]["method"] == "fallback"


def test_montaje_manual_solo_afecta_region(eng):
    img = np.full((400, 400, 3), 128, np.uint8)
    res = eng.run("montaje_don_bosco", img, {"rostro_x": 0.25, "rostro_y": 0.25,
                                             "rostro_ancho": 0.15, "foco": 0.0})
    assert res.extra["face"]["method"] == "manual"
    assert np.array_equal(res.image[300:, 300:], img[300:, 300:])  # lejos del rostro
    assert not np.array_equal(res.image[60:140, 60:140], img[60:140, 60:140])


# ---------------------------------------------------------------- cómic
def test_comic_posterizacion_plana(eng):
    img = np.zeros((32, 32, 3), np.uint8)
    img[:, :16] = 250
    img[:, 16:] = 10
    out = eng.run("comic", img, {"niveles": 2, "saturacion": 1.0, "tinta": 0.0}).image
    assert np.all(out[:, :8] == 255)                             # claro -> 255, sin semitono
    assert np.all(out[:, 24:] == 0)                              # oscuro -> 0


def test_comic_tinta_en_borde(eng):
    img = np.zeros((32, 32, 3), np.uint8)
    img[:, :16] = 250
    out = eng.run("comic", img, {"niveles": 2, "saturacion": 1.0, "tinta": 1.0,
                                 "umbral_borde": 100}).image
    assert out[10, 15, 0] == 18 and out[10, 16, 0] == 18         # bordes en tinta


# ---------------------------------------------------------------- métricas
@pytest.mark.parametrize("block", [(16, 16), (32, 8), (8, 8)])
def test_metricas_hilos_y_grid(eng, block):
    img = np.zeros((50, 70, 3), np.uint8)
    m = eng.run("media", img, {"k": 3}, block=block).metrics
    gx, gy = math.ceil(70 / block[0]), math.ceil(50 / block[1])
    assert m["block"] == [block[0], block[1], 1] and m["grid"] == [gx, gy, 1]
    assert m["kernel_launches"] == 2
    assert m["total_threads"] == 2 * gx * gy * block[0] * block[1]
    assert m["gpu_time_ms"] > 0 and m["status"] == "DONE" and m["time_unit"] == "ms"


def test_metricas_filtros_multikernel(eng):
    m = eng.run("salesiano", np.zeros((64, 64, 3), np.uint8)).metrics
    assert m["total_threads"] == sum(l["threads"] for l in m["launches"])
    assert [l["kernel"] for l in m["launches"]][:2] == ["ups_tone", "ups_frame"]


# ---------------------------------------------------------------- entradas inválidas
def test_errores_de_entrada(eng):
    img = np.zeros((16, 16, 3), np.uint8)
    with pytest.raises(UnknownFilterError):
        eng.run("no_existe", img)
    with pytest.raises(InvalidParamsError):
        eng.run("media", img, {"k": 4})                           # par
    with pytest.raises(InvalidParamsError):
        eng.run("media", img, {"k": 17})                          # fuera de rango
    with pytest.raises(InvalidParamsError):
        eng.run("media", img, {"radio": 3})                       # no permitido
    with pytest.raises(InvalidParamsError):
        eng.run("laplaciano", img, {"vecinos": "6"})
    with pytest.raises(InvalidParamsError):
        eng.run("salesiano", img, {"tono": "mucho"})
    with pytest.raises(InvalidImageError):
        eng.run("media", np.zeros((16, 16), np.uint8))            # sin canales
    with pytest.raises(InvalidImageError):
        eng.run("media", np.zeros((4, 4, 3), np.uint8))           # demasiado pequeña
    with pytest.raises(InvalidImageError):
        eng.run("media", img, block=(3, 3))                       # bloque inválido


def test_recuperacion_tras_error(eng):
    img = np.full((16, 16, 3), 9, np.uint8)
    with pytest.raises(InvalidParamsError):
        eng.run("media", img, {"k": 2})
    assert eng.run("media", img, {"k": 3}).metrics["status"] == "DONE"


def test_todos_los_filtros_conservan_forma(eng):
    from upsglam_gpu import FILTERS
    img = np.random.default_rng(0).integers(0, 256, (45, 61, 3), dtype=np.uint8)
    for code in FILTERS:
        r = eng.run(code, img)
        assert r.image.shape == img.shape and r.image.dtype == np.uint8, code
