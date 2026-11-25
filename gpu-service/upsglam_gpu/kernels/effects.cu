// =====================================================================
// Efectos creativos: cómic (pop-art), glitch y neón.
// =====================================================================

// Cómic: saturación -> posterización -> semitono (puntos a 45°) -> tinta en bordes.
// mag = magnitud Sobel (float32) calculada previamente.
__global__ void comic(const unsigned char* __restrict__ rgb, const float* __restrict__ mag,
                      unsigned char* __restrict__ out, int W, int H,
                      int levels, float edge_thr, float dot_cell, float saturation, float ink)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    size_t p = (size_t)y * W + x, i = p * 3;
    float c[3] = {(float)rgb[i], (float)rgb[i + 1], (float)rgb[i + 2]};
    float L = luminance(c[0], c[1], c[2]);

    float q = (float)(levels - 1);
    for (int k = 0; k < 3; ++k) {
        c[k] = CLAMPF(L + (c[k] - L) * saturation, 0.0f, 255.0f);
        c[k] = rintf(c[k] / 255.0f * q) / q * 255.0f;
    }

    // semitono: rejilla rotada 45°, radio del punto crece con la oscuridad
    const float cs = 0.70710678f;
    float xr = (x + y) * cs, yr = (y - x) * cs;
    float cx = (floorf(xr / dot_cell) + 0.5f) * dot_cell;
    float cy = (floorf(yr / dot_cell) + 0.5f) * dot_cell;
    float d = sqrtf((xr - cx) * (xr - cx) + (yr - cy) * (yr - cy)) / dot_cell;
    float Ln = L / 255.0f;
    float radius = 0.55f * sqrtf(fmaxf(0.0f, 1.0f - Ln));
    float dot = 1.0f - smoothstepf(radius - 0.06f, radius + 0.06f, d);
    float shade = 1.0f - 0.35f * dot * (1.0f - smoothstepf(0.55f, 0.8f, Ln));

    float e = smoothstepf(edge_thr * 0.8f, edge_thr * 1.2f, mag[p]) * ink;
    for (int k = 0; k < 3; ++k) out[i + k] = to_u8(mixf(c[k] * shade, 18.0f, e));
}

// Glitch: bandas horizontales desplazadas al azar (hash por banda), separación
// de canales RGB, líneas de barrido e inversión ocasional de bandas.
__global__ void glitch(const unsigned char* __restrict__ in, unsigned char* __restrict__ out,
                       int W, int H, int shift, int band_h, float band_prob,
                       unsigned int seed, float scan)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    unsigned int h = hash_u32((unsigned int)(y / band_h) ^ (seed * 0x9e3779b9U));
    float rnd = (h & 0xFFFF) / 65535.0f;
    int disp = 0;
    if (rnd < band_prob) disp = (int)((((h >> 16) & 0xFF) / 255.0f * 2.0f - 1.0f) * shift * 4.0f);
    int xs = x + disp;
    int xr = min(max(xs + shift, 0), W - 1);
    int xg = min(max(xs, 0), W - 1);
    int xb = min(max(xs - shift, 0), W - 1);
    size_t row = (size_t)y * W;
    float c[3] = {(float)in[(row + xr) * 3], (float)in[(row + xg) * 3 + 1], (float)in[(row + xb) * 3 + 2]};
    bool invert = rnd < band_prob * 0.2f;
    float s = ((y / 2) % 2 == 0) ? 1.0f - scan : 1.0f;
    size_t o = (row + x) * 3;
    for (int k = 0; k < 3; ++k) out[o + k] = to_u8((invert ? 255.0f - c[k] : c[k]) * s);
}

// Neón: bordes Sobel coloreados según la orientación del gradiente (tono HSV),
// con resplandor (bordes desenfocados) sobre la foto oscurecida.
__device__ __forceinline__ void hue_to_rgb(float h, float* rgb)
{
    float r = fabsf(h * 6.0f - 3.0f) - 1.0f;
    float g = 2.0f - fabsf(h * 6.0f - 2.0f);
    float b = 2.0f - fabsf(h * 6.0f - 4.0f);
    rgb[0] = CLAMPF(r, 0.0f, 1.0f); rgb[1] = CLAMPF(g, 0.0f, 1.0f); rgb[2] = CLAMPF(b, 0.0f, 1.0f);
}

__global__ void neon(const unsigned char* __restrict__ rgb, const float* __restrict__ mag,
                     const float* __restrict__ ang, const float* __restrict__ glow,
                     unsigned char* __restrict__ out, int W, int H, float bg_dim, float gain)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    size_t p = (size_t)y * W + x, i = p * 3;
    float col[3];
    hue_to_rgb((ang[p] + PI_F) / (2.0f * PI_F), col);
    float e = CLAMPF(mag[p] / 255.0f * gain, 0.0f, 1.0f);
    float g = CLAMPF(glow[p] / 255.0f * gain, 0.0f, 1.0f);
    for (int k = 0; k < 3; ++k)
        out[i + k] = to_u8(rgb[i + k] * bg_dim + col[k] * 255.0f * (e + 0.9f * g));
}
