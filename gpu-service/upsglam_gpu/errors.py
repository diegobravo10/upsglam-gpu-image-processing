"""Errores del motor GPU (el servicio REST los traduce a códigos HTTP)."""


class UpsGlamGpuError(Exception):
    """Base."""


class GpuUnavailableError(UpsGlamGpuError):
    """No hay GPU NVIDIA / driver / contexto CUDA. NUNCA se cae a CPU."""


class UnknownFilterError(UpsGlamGpuError):
    """Filtro no registrado."""


class InvalidParamsError(UpsGlamGpuError):
    """Parámetros fuera de rango o de tipo incorrecto."""


class InvalidImageError(UpsGlamGpuError):
    """Imagen con forma, tipo o tamaño no admitido."""


class GpuProcessingError(UpsGlamGpuError):
    """Fallo durante la ejecución en GPU (compilación, memoria, lanzamiento)."""

    def __init__(self, message: str, metrics: dict | None = None):
        super().__init__(message)
        self.metrics = metrics
