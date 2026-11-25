# UPSGlam · servicio GPU (FastAPI + PyCUDA)
# Imagen "devel": PyCUDA compila los kernels en tiempo de ejecución con nvcc + g++.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_ROOT=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH \
    UPSGLAM_ASSETS_DIR=/app/assets

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential fonts-dejavu-core curl \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY upsglam_gpu ./upsglam_gpu
COPY app ./app
COPY assets ./assets
COPY tests ./tests
COPY scripts ./scripts
COPY cli.py pytest.ini ./

RUN useradd --create-home --uid 10001 upsglam && mkdir -p /app/logs && chown -R upsglam /app
USER upsglam

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# UN solo proceso: la GPU la posee un único hilo trabajador dentro de él.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
