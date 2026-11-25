// =====================================================================
// Estimación del rostro por detección de piel + reducción paralela.
// Cada hilo clasifica su píxel (reglas YCbCr + RGB). Cada bloque reduce en
// memoria compartida 5 sumas: n, Σu, Σv, Σu², Σv² (coordenadas normalizadas)
// y un solo hilo por bloque las acumula con atomicAdd (double) en global.
// El host obtiene media y desviación => centro y tamaño aproximado del rostro.
// Requisito: blockDim.x*blockDim.y potencia de 2 y <= 1024.
// =====================================================================
#define MAX_THREADS 1024

__device__ __forceinline__ bool is_skin(float r, float g, float b)
{
    float cb = 128.0f - 0.168736f * r - 0.331264f * g + 0.5f * b;
    float cr = 128.0f + 0.5f * r - 0.418688f * g - 0.081312f * b;
    bool ycc = cb >= 77.0f && cb <= 127.0f && cr >= 133.0f && cr <= 173.0f;
    float mx = fmaxf(r, fmaxf(g, b)), mn = fminf(r, fminf(g, b));
    bool rgb = r > 95.0f && g > 40.0f && b > 20.0f && (mx - mn) > 15.0f &&
               fabsf(r - g) > 15.0f && r > g && r > b;
    return ycc && rgb;
}

__global__ void skin_stats(const unsigned char* __restrict__ rgb, int W, int H,
                           float wx0, float wy0, float wx1, float wy1, double* stats)
{
    __shared__ float sh[5][MAX_THREADS];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nt = blockDim.x * blockDim.y;

    float v[5] = {0, 0, 0, 0, 0};
    if (x < W && y < H) {
        float u = (x + 0.5f) / W, w = (y + 0.5f) / H;
        size_t i = ((size_t)y * W + x) * 3;
        if (u >= wx0 && u <= wx1 && w >= wy0 && w <= wy1 &&
            is_skin((float)rgb[i], (float)rgb[i + 1], (float)rgb[i + 2])) {
            v[0] = 1.0f; v[1] = u; v[2] = w; v[3] = u * u; v[4] = w * w;
        }
    }
    for (int k = 0; k < 5; ++k) sh[k][tid] = v[k];
    __syncthreads();

    for (int s = nt / 2; s > 0; s >>= 1) {               // reducción en árbol
        if (tid < s)
            for (int k = 0; k < 5; ++k) sh[k][tid] += sh[k][tid + s];
        __syncthreads();
    }
    if (tid == 0)
        for (int k = 0; k < 5; ++k) atomicAdd(&stats[k], (double)sh[k][0]);
}
