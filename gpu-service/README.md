# UPSGlam · Servicio GPU (FastAPI + PyCUDA)

Recibe una imagen y los parámetros, ejecuta los **kernels propios** en la GPU y permite recuperar el resultado. Para cada solicitud registra el estado, los tiempos y las métricas GPU.

## Ejecutar

```bash
# Con Docker (recomendado): desde la raíz del repositorio
docker compose up -d --build gpu-service
curl http://localhost:8000/health

# Local (Windows/Linux con CUDA Toolkit + compilador host)
cd gpu-service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> Usa siempre `--workers 1`: un único proceso con un único hilo es dueño del contexto CUDA.

Para las pruebas:
```bash
python -m pytest -v                                    # 45 pruebas (filtros + servicio)
docker exec upsglam-gpu python -m pytest -q -p no:cacheprovider   # dentro del contenedor
python scripts/smoke_concurrency.py --users 4 --requests 40       # concurrencia y aislamiento
```

## Variables de entorno

| Variable | Defecto | Descripción |
|---|---|---|
| `GPU_QUEUE_CAPACITY` | 16 | Solicitudes en espera antes de rechazar con 503 |
| `GPU_MAX_IMAGE_MB` | 10 | Tamaño máximo del archivo |
| `GPU_MAX_SIDE_PX` | 4096 | Lado máximo de la imagen |
| `GPU_MAX_PIXELS` | 16777216 | Píxeles máximos (protección contra *decompression bomb*) |
| `GPU_SYNC_TIMEOUT_S` | 60 | Espera máxima de `POST /process` antes de responder 504 |
| `GPU_RESULT_TTL_S` | 300 | Tiempo que se conserva el resultado en memoria |
| `CUDA_DEVICE` | 0 | Índice de la GPU |
| `GPU_BLOCK_X` / `GPU_BLOCK_Y` | 16 / 16 | Dimensiones de bloque. `bx·by` debe ser potencia de 2 y ≤ 1024 |
| `GPU_LOG_FILE` | — | Archivo de logs JSON (en Docker: `/app/logs/gpu-service.log`) |

## Endpoints

| Método | Ruta | Respuesta |
|---|---|---|
| GET | `/health` | 200 `{status:"ok", gpu:{...}, queue:{size,capacity,processing}, cpu_fallback:false}`; **503** si no hay GPU |
| GET | `/filters` | Lista de filtros con sus parámetros (tipo, rango, valor por defecto) |
| POST | `/process` | **Síncrono.** Espera a que termine y devuelve 200 con la *tarea* (ver abajo) |
| POST | `/tasks` | **Asíncrono.** Responde 202 con la tarea en estado `QUEUED` y el header `Location: /tasks/{id}` |
| GET | `/tasks/{id}` | Estado, métricas y tiempos de la solicitud |
| GET | `/tasks/{id}/result` | La imagen (`image/png` o `image/jpeg`), con los headers `X-Request-Id`, `X-User-Id` y `X-Filter` |
| DELETE | `/tasks/{id}` | 204. Libera el resultado; el backend lo llama después de subirlo a Supabase Storage |
| GET | `/stats` | Contadores (recibidas, aceptadas, rechazadas, terminadas, fallidas), cola y uso de la GPU |

### Petición (`multipart/form-data`) para `POST /process` y `POST /tasks`

| Campo | Obligatorio | Descripción |
|---|---|---|
| `image` | sí | Archivo JPEG, PNG o WEBP. Se valida la **firma real** (magic bytes), no la extensión |
| `filter` | sí | Código del filtro: `media`, `nitidez`, `laplaciano`, `salesiano`, `montaje_don_bosco`, `comic` (más los extra `glitch` y `neon`) |
| `params` | no | Objeto JSON en texto, por ejemplo `{"k":7,"borde":"espejo"}`. Si se omite, se usan los valores por defecto |
| `request_id` | no | Id de 1 a 64 caracteres `[A-Za-z0-9_-]`. **El backend debe enviar el id de `processing_requests`.** Si se omite, se genera un UUID |
| `user_id` | no | Id del usuario de Supabase, para correlacionar en los logs |
| `output_format` | no | `png` (por defecto, sin pérdida) o `jpeg` |

```bash
curl -X POST http://localhost:8000/process \
  -F image=@foto.jpg -F filter=salesiano -F 'params={"tono":0.8,"marco":0.9}' \
  -F request_id=8f1c... -F user_id=2b7e...
curl -o resultado.png http://localhost:8000/tasks/8f1c.../result
```

### Respuesta (la "tarea")

```json
{
  "request_id": "demo-001", "user_id": "demo-user", "status": "DONE",
  "filter": "salesiano", "params": {"tono": 0.8, "marco": 0.9, "don_bosco": true, "texto": true},
  "input": {"format": "PNG", "bytes": 1301334, "width": 873, "height": 792},
  "metrics": {
    "width": 873, "height": 792, "block": [16,16,1], "grid": [55,50,1],
    "kernel_launches": 4,
    "launches": [{"kernel":"ups_tone","block":[16,16,1],"grid":[55,50,1],"threads":704000,"time_ms":0.21}, "..."],
    "total_threads": 1550336, "gpu_time_ms": 0.861, "h2d_ms": 0.4, "d2h_ms": 0.5,
    "time_unit": "ms", "status": "DONE"
  },
  "timings_ms": {"decode": 23.9, "queue_wait": 0.18, "gpu_total": 8.78, "encode": 33.8, "service_total": 71.4},
  "extra": {"bosco_box": ["..."]},
  "received_at": "2026-11-25T21:51:02.120+00:00", "started_at": "...", "finished_at": "...",
  "result": {"url": "/tasks/demo-001/result", "content_type": "image/png", "bytes": 840594},
  "error": null
}
```

**Qué mide cada tiempo:**
- `metrics.gpu_time_ms`: suma de los tiempos de **los kernels**, medidos con eventos CUDA. No incluye las transferencias, que van en `h2d_ms` y `d2h_ms`.
- `timings_ms.gpu_total`: todo lo que ocurre en el hilo GPU (memoria, transferencias, kernels).
- `queue_wait`: tiempo de espera en la cola.
- `decode` y `encode`: tiempo de CPU para leer la imagen recibida y escribir el resultado.
- `service_total`: desde que el servicio recibe la solicitud hasta que el resultado está listo.

### Estados
`QUEUED` → `PROCESSING` → `DONE` o `FAILED`. Una solicitud rechazada porque la cola estaba llena queda como `REJECTED`.

Si la solicitud falla **antes de lanzar un kernel** (entrada inválida, cola llena, GPU no disponible), `metrics.block`, `grid`, `total_threads` y `gpu_time_ms` valen **`null`**. No se inventan valores.

### Errores (siempre `{error, message, status, request_id, task?}`)

| HTTP | `error` | Causa |
|---|---|---|
| 400 | `EMPTY_FILE` | Archivo vacío |
| 400 | `CORRUPT_IMAGE` | La firma es válida pero la imagen no se puede decodificar |
| 400 | `UNKNOWN_FILTER` | El filtro no existe |
| 400 | `INVALID_REQUEST_ID` | El formato del id es inválido |
| 404 | `NOT_FOUND` | La tarea no existe o ya expiró |
| 409 | `DUPLICATE_REQUEST` | Ya existe una solicitud con ese `request_id` |
| 409 | `RESULT_NOT_READY` / `NO_RESULT` | El resultado aún no está listo, o la tarea falló o fue liberada |
| 413 | `FILE_TOO_LARGE` / `IMAGE_DIMENSIONS` | Se superó el tamaño o las dimensiones permitidas |
| 415 | `UNSUPPORTED_FORMAT` | El archivo no es JPEG, PNG ni WEBP (por ejemplo, un .txt renombrado) |
| 422 | `INVALID_PARAMS` | Parámetro desconocido, de tipo incorrecto, fuera de rango o par donde se exige impar; también `params` que no es JSON u `output_format` inválido |
| 500 | `GPU_PROCESSING_ERROR` | Fallo CUDA durante el procesamiento. El contexto se verifica y se recrea si quedó dañado |
| 503 | `QUEUE_FULL` | Capacidad alcanzada. Incluye el header `Retry-After: 2` |
| 503 | `GPU_UNAVAILABLE` | No hay GPU o driver. **El servicio nunca procesa en CPU** |
| 504 | `TIMEOUT` | Se superó `GPU_SYNC_TIMEOUT_S`. La tarea sigue en curso y se puede consultar con `GET /tasks/{id}` |

## Política de solicitudes simultáneas

```
cliente ─► uvicorn (asyncio) ─► validar + decodificar (pool de hilos, CPU, en paralelo)
                                  │
                                  ▼
                       cola FIFO acotada (capacidad N) ──llena──► 503 QUEUE_FULL
                                  │
                                  ▼
                hilo GPU único (dueño del contexto CUDA): kernels de a una solicitud
                                  │
                                  ▼
                  codificar PNG/JPEG (pool de hilos, CPU) ─► resultado disponible
```

- **Se aceptan solicitudes concurrentes**, pero **los kernels de distintas solicitudes no se ejecutan a la vez**. Hay tres razones:
  1. Cada kernel ya ocupa toda la GPU con miles de bloques.
  2. El contexto CUDA de PyCUDA está ligado al hilo que lo crea.
  3. La máscara de convolución vive en memoria `__constant__`, que comparte todo el módulo.
- La cola está **acotada**. Al llegar a la capacidad, el servicio **rechaza de forma controlada** (503 con `Retry-After`) en lugar de acumular latencia y memoria sin límite.
- Cada solicitud tiene sus propios buffers en la GPU y su propio registro, por lo que **los datos y resultados nunca se mezclan**. La prueba `test_aislamiento_concurrente` y `smoke_concurrency.py` lo verifican comparando los píxeles con una referencia.
- Después de un error CUDA, el hilo comprueba el contexto (`Context.synchronize`). Si quedó inutilizable, lo recrea y el servicio sigue atendiendo.

## Evidencia de uso real de la GPU

Cada solicitud deja estas líneas JSON en los logs (`docker logs upsglam-gpu` o `logs/gpu-service.log`), todas con el mismo `request_id`:
```
request_received → request_queued → gpu_start → gpu_done {device, grid, block, total_threads,
gpu_time_ms, kernels:["ups_tone","ups_frame","overlay_text2","overlay_bosco"]} → request_finished
```
