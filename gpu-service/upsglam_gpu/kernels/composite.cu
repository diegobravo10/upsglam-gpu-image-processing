// =====================================================================
// Efectos institucionales UPS: tono, marco procedural, texto y silueta
// de Don Bosco. Todas las mezclas se hacen aquí, en GPU.
// Colores llegan como floats 0..255 (navy = azul UPS, gold = dorado UPS).
// =====================================================================

// Gradación institucional: soft-light de la imagen con un duotono
// azul (sombras) -> dorado (luces), mezclado con el original por `amount`.
__global__ void ups_tone(const unsigned char* __restrict__ in, unsigned char* __restrict__ out,
                         int W, int H, float amount,
                         float nr, float ng, float nb, float gr, float gg, float gb)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    size_t i = ((size_t)y * W + x) * 3;
    float a[3] = {in[i] / 255.0f, in[i + 1] / 255.0f, in[i + 2] / 255.0f};
    float L = luminance(a[0], a[1], a[2]);
    float t = smoothstepf(0.05f, 0.95f, L);
    float duo[3] = {mixf(nr, gr, t) / 255.0f, mixf(ng, gg, t) / 255.0f, mixf(nb, gb, t) / 255.0f};
    for (int c = 0; c < 3; ++c) {
        float d = duo[c];
        float sl = (1.0f - 2.0f * d) * a[c] * a[c] + 2.0f * d * a[c];   // soft-light (Pegtop)
        sl = CLAMPF((sl - 0.5f) * 1.08f + 0.5f, 0.0f, 1.0f);             // leve contraste
        out[i + c] = to_u8(mixf(a[c], sl, amount) * 255.0f);
    }
}

// Marco institucional procedural (in-place):
//  · banda inferior azul delimitada por la curva by(u) = 0.82 - 0.20·u^1.6
//  · franja dorada y línea celeste sobre la curva
//  · cuña azul en la esquina superior izquierda con borde dorado
// Antialiasing con smoothstep de ±1 px. `intensity` = opacidad del marco.
__global__ void ups_frame(unsigned char* img, int W, int H, float intensity,
                          float nr, float ng, float nb, float gr, float gg, float gb,
                          float lr, float lg, float lb)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float S = fminf((float)W, (float)H);
    float u = (x + 0.5f) / W;
    float fy = y + 0.5f;

    // banda inferior
    float by = (0.82f - 0.20f * powf(u, 1.6f)) * H;
    float d = fy - by;                                  // > 0 dentro de la banda
    float gold_t = 0.016f * S, lb_t = 0.006f * S;
    float in_navy = smoothstepf(-1.0f, 1.0f, d);
    float in_gold = smoothstepf(-gold_t - 1.0f, -gold_t + 1.0f, d) * (1.0f - in_navy);
    float lb0 = -gold_t - 2.0f * lb_t, lb1 = -gold_t - lb_t;
    float in_lb = smoothstepf(lb0 - 1.0f, lb0 + 1.0f, d) * (1.0f - smoothstepf(lb1 - 1.0f, lb1 + 1.0f, d));
    float shade = 1.0f - 0.30f * CLAMPF(d / (H - by + 1.0f), 0.0f, 1.0f);

    // cuña superior izquierda
    float tu = u / 0.62f;
    float ty = tu < 1.0f ? 0.17f * (1.0f - tu) * (1.0f - tu) * H : -1e9f;
    float dt = ty - fy;                                 // > 0 dentro de la cuña
    float t_navy = smoothstepf(-1.0f, 1.0f, dt);
    float t_gold = smoothstepf(-0.8f * gold_t - 1.0f, -0.8f * gold_t + 1.0f, dt) * (1.0f - t_navy);

    size_t i = ((size_t)y * W + x) * 3;
    float navy[3] = {nr, ng, nb}, gold[3] = {gr, gg, gb}, light[3] = {lr, lg, lb};
    for (int c = 0; c < 3; ++c) {
        float v = img[i + c];
        v = mixf(v, light[c], in_lb * intensity);
        v = mixf(v, gold[c], (in_gold + t_gold) * intensity);
        v = mixf(v, navy[c] * shade, in_navy * intensity);
        v = mixf(v, navy[c], t_navy * intensity);
        img[i + c] = to_u8(v);
    }
}

// Composición de una máscara de texto de 2 canales (ch0 -> colorA, ch1 -> colorB)
// colocada en (ox, oy). El grid cubre la MÁSCARA, no la imagen completa.
__global__ void overlay_text2(unsigned char* img, int W, int H,
                              const unsigned char* __restrict__ mask, int mw, int mh, int ox, int oy,
                              float ar, float ag, float ab, float br, float bg, float bb, float opacity)
{
    int mx = blockIdx.x * blockDim.x + threadIdx.x;
    int my = blockIdx.y * blockDim.y + threadIdx.y;
    if (mx >= mw || my >= mh) return;
    int x = ox + mx, y = oy + my;
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    size_t m = ((size_t)my * mw + mx) * 2;
    float a = mask[m] / 255.0f * opacity, b = mask[m + 1] / 255.0f * opacity;
    float A[3] = {ar, ag, ab}, B[3] = {br, bg, bb};
    size_t i = ((size_t)y * W + x) * 3;
    for (int c = 0; c < 3; ++c) {
        float v = mixf((float)img[i + c], A[c], a);
        img[i + c] = to_u8(mixf(v, B[c], b));
    }
}

// Silueta de Don Bosco (recurso LA: ch0 = tono, ch1 = cobertura) escalada a la
// caja destino (bx, by, bw, bh) con muestreo bilineal, contorno dorado
// calculado en GPU (máximo de cobertura en 16 direcciones) y dos estilos:
//  mode 0 "poster": azul en zonas oscuras, dorado en zonas claras
//  mode 1 "fusion": las zonas claras conservan la luminancia de la foto teñida de dorado
// El grid cubre solo la región (rx0, ry0, rw, rh) afectada.
__global__ void overlay_bosco(unsigned char* img, int W, int H,
                              const unsigned char* __restrict__ asset, int aw, int ah,
                              float bx, float by, float bw, float bh,
                              int rx0, int ry0, int rw, int rh,
                              float nr, float ng, float nb, float gr, float gg, float gb,
                              float opacity, float outline_px, int mode, float fade_start)
{
    int lx = blockIdx.x * blockDim.x + threadIdx.x;
    int ly = blockIdx.y * blockDim.y + threadIdx.y;
    if (lx >= rw || ly >= rh) return;
    int x = rx0 + lx, y = ry0 + ly;
    if (x < 0 || y < 0 || x >= W || y >= H) return;

    float sx = (x + 0.5f - bx) * aw / bw - 0.5f;        // transformación destino -> recurso
    float sy = (y + 0.5f - by) * ah / bh - 0.5f;
    float cov = sample_bilinear_zero(asset, aw, ah, 2, 1, sx, sy);

    float outline = 0.0f;
    if (outline_px > 0.0f) {
        float ra = outline_px * aw / bw, ring = 0.0f;
        for (int k = 0; k < 16; ++k) {
            float ang = k * (2.0f * PI_F / 16.0f);
            ring = fmaxf(ring, sample_bilinear_zero(asset, aw, ah, 2, 1,
                                                    sx + ra * __cosf(ang), sy + ra * __sinf(ang)));
        }
        outline = fmaxf(ring - cov, 0.0f);
    }
    if (cov <= 0.0f && outline <= 0.0f) return;

    float fade = fade_start < 1.0f ? 1.0f - smoothstepf(fade_start, 1.0f, (sy + 0.5f) / ah) : 1.0f;
    float tone = sample_bilinear_clamp(asset, aw, ah, 2, 0, sx, sy);

    size_t i = ((size_t)y * W + x) * 3;
    float p[3] = {(float)img[i], (float)img[i + 1], (float)img[i + 2]};
    float L = luminance(p[0], p[1], p[2]) / 255.0f;
    float navy[3] = {nr, ng, nb}, gold[3] = {gr, gg, gb};
    for (int c = 0; c < 3; ++c) {
        float fig;
        if (mode == 0) {
            fig = mixf(navy[c], gold[c], tone);
        } else {
            float light = gold[c] * (0.35f + 0.85f * L);
            float dark = mixf(navy[c], p[c], 0.15f);
            fig = mixf(dark, light, tone);
        }
        float v = mixf(p[c], gold[c], outline * fade * opacity);
        img[i + c] = to_u8(mixf(v, fig, cov * fade * opacity));
    }
}

// Foco sobre el rostro: viñeta azul fuera de la elipse y halo dorado aditivo.
__global__ void face_spotlight(unsigned char* img, int W, int H,
                               float cx, float cy, float rx, float ry, float strength,
                               float nr, float ng, float nb, float gr, float gg, float gb)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float dx = (x + 0.5f - cx) / rx, dy = (y + 0.5f - cy) / ry;
    float d = sqrtf(dx * dx + dy * dy);                 // 1 = borde de la elipse
    float vig = smoothstepf(1.3f, 3.2f, d) * strength * 0.75f;
    float halo = __expf(-(d - 1.45f) * (d - 1.45f) / 0.03f) * strength * 0.45f;
    size_t i = ((size_t)y * W + x) * 3;
    float navy[3] = {nr, ng, nb}, gold[3] = {gr, gg, gb};
    for (int c = 0; c < 3; ++c) {
        float v = img[i + c];
        v = mixf(v, navy[c] * 0.55f + v * 0.25f, vig);
        v += gold[c] * halo;
        img[i + c] = to_u8(v);
    }
}
