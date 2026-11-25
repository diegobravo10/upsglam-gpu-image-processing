// =====================================================================
// Convoluciones: media, nitidez, laplaciano (máscara en memoria constante)
// y Sobel (coeficientes fijos). Salida en float32 SIN recorte para poder
// validar los valores exactos antes de convertir a 8 bits.
// =====================================================================
#define MAX_MASK 31
__constant__ float c_mask[MAX_MASK * MAX_MASK];

// Un hilo = un píxel (x, y). Recorre la vecindad k×k para C canales (C<=3).
__global__ void conv2d_u8_f32(const unsigned char* __restrict__ in, float* __restrict__ out,
                              int W, int H, int C, int k, int border)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;                       // límite de acceso

    int r = k / 2;
    float acc[3] = {0.0f, 0.0f, 0.0f};
    for (int j = -r; j <= r; ++j) {
        int yy = border_index(y + j, H, border);
        for (int i = -r; i <= r; ++i) {
            int xx = border_index(x + i, W, border);
            if (yy < 0 || xx < 0) continue;             // borde cero
            float m = c_mask[(j + r) * k + (i + r)];
            const unsigned char* p = in + ((size_t)yy * W + xx) * C;
            for (int c = 0; c < C; ++c) acc[c] += m * (float)p[c];
        }
    }
    float* o = out + ((size_t)y * W + x) * C;
    for (int c = 0; c < C; ++c) o[c] = acc[c];
}

// RGB -> escala de grises (BT.601), redondeado a uint8
__global__ void rgb_to_gray_u8(const unsigned char* __restrict__ rgb, unsigned char* __restrict__ gray,
                               int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    size_t i = (size_t)y * W + x;
    gray[i] = to_u8(luminance(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]));
}

// float32 -> uint8 para visualización: v' = clamp(|v|·scale + offset) si absmode,
// si no clamp(v·scale + offset). Si Cin==1 y Cout==3 replica el canal (gris -> RGB).
__global__ void f32_to_u8(const float* __restrict__ in, unsigned char* __restrict__ out,
                          int W, int H, int Cin, int Cout, float scale, float offset, int absmode)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    size_t i = (size_t)y * W + x;
    for (int c = 0; c < Cout; ++c) {
        float v = in[i * Cin + (Cin == 1 ? 0 : c)];
        if (absmode) v = fabsf(v);
        out[i * Cout + c] = to_u8(v * scale + offset);
    }
}

// Sobel 3×3 sobre gris uint8. Gx = [-1 0 1; -2 0 2; -1 0 1], Gy = Gxᵀ.
// mag = sqrt(Gx²+Gy²) (float32); ang = atan2(Gy, Gx) si ang != NULL.
__global__ void sobel_u8(const unsigned char* __restrict__ gray, float* __restrict__ mag,
                         float* __restrict__ ang, int W, int H, int border)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const float kx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const float ky[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    float gx = 0.0f, gy = 0.0f;
    for (int j = -1; j <= 1; ++j) {
        int yy = border_index(y + j, H, border);
        for (int i = -1; i <= 1; ++i) {
            int xx = border_index(x + i, W, border);
            if (yy < 0 || xx < 0) continue;
            float v = (float)gray[(size_t)yy * W + xx];
            int t = (j + 1) * 3 + (i + 1);
            gx += kx[t] * v;
            gy += ky[t] * v;
        }
    }
    size_t o = (size_t)y * W + x;
    mag[o] = sqrtf(gx * gx + gy * gy);
    if (ang) ang[o] = atan2f(gy, gx);
}
