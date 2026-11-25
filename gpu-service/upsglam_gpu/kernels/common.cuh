// =====================================================================
// UPSGlam · utilidades de dispositivo compartidas por todos los kernels
// Imágenes: uint8 intercaladas (RGBRGB...), fila mayor, índice (y*W + x)*C
// =====================================================================
#define CLAMPF(v, lo, hi) fminf(fmaxf((v), (lo)), (hi))
#define PI_F 3.14159265358979f

// Modos de borde: 0 = replicar (clamp), 1 = espejo (reflect-101), 2 = cero
__device__ __forceinline__ int border_index(int i, int n, int mode)
{
    if (i >= 0 && i < n) return i;
    if (mode == 0) return i < 0 ? 0 : n - 1;
    if (mode == 1) {
        if (n == 1) return 0;
        while (i < 0 || i >= n) {
            if (i < 0) i = -i;
            if (i >= n) i = 2 * (n - 1) - i;
        }
        return i;
    }
    return -1;  // cero: el llamador omite la muestra
}

__device__ __forceinline__ float luminance(float r, float g, float b)
{
    return 0.299f * r + 0.587f * g + 0.114f * b;
}

__device__ __forceinline__ float smoothstepf(float e0, float e1, float x)
{
    float t = CLAMPF((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

__device__ __forceinline__ float mixf(float a, float b, float t) { return a + (b - a) * t; }

__device__ __forceinline__ unsigned char to_u8(float v)
{
    return (unsigned char)(CLAMPF(v, 0.0f, 255.0f) + 0.5f);
}

// Muestreo bilineal del canal c de una imagen uint8 de C canales.
// Las muestras fuera de la imagen valen 0 (útil para máscaras/alfa).
__device__ float sample_bilinear_zero(const unsigned char* img, int w, int h, int C, int c,
                                      float x, float y)
{
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    float fx = x - x0, fy = y - y0;
    float v[4];
    int xs[4] = {x0, x0 + 1, x0, x0 + 1};
    int ys[4] = {y0, y0, y0 + 1, y0 + 1};
    for (int k = 0; k < 4; ++k) {
        int xx = xs[k], yy = ys[k];
        v[k] = (xx >= 0 && xx < w && yy >= 0 && yy < h) ? (float)img[(yy * w + xx) * C + c] : 0.0f;
    }
    return mixf(mixf(v[0], v[1], fx), mixf(v[2], v[3], fx), fy) / 255.0f;
}

// Muestreo bilineal con replicación de borde (para colores).
__device__ float sample_bilinear_clamp(const unsigned char* img, int w, int h, int C, int c,
                                       float x, float y)
{
    x = CLAMPF(x, 0.0f, (float)(w - 1));
    y = CLAMPF(y, 0.0f, (float)(h - 1));
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    float fx = x - x0, fy = y - y0;
    float a = img[(y0 * w + x0) * C + c], b = img[(y0 * w + x1) * C + c];
    float d = img[(y1 * w + x0) * C + c], e = img[(y1 * w + x1) * C + c];
    return mixf(mixf(a, b, fx), mixf(d, e, fx), fy) / 255.0f;
}

__device__ __forceinline__ unsigned int hash_u32(unsigned int x)
{
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
