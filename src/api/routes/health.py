from fastapi import APIRouter, Depends

from src.api.deps import get_engines_map, get_hardware_info, get_inference_queue
from src.api.schemas import HardwareSummary, HealthResponse
from src.api.worker.queue_manager import InferenceQueue
from src.hardware import HardwareInfo

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def live() -> dict:
    # Always 200 if the process can respond at all. This is what the Docker HEALTHCHECK calls -
    # it must not depend on model_files being populated, or the container would restart-loop
    # before the user has a chance to mount real models.
    return {"status": "alive"}


@router.get("/health/ready", response_model=HealthResponse)
@router.get("/health", response_model=HealthResponse)
async def ready(
    engines: dict = Depends(get_engines_map),
    queue: InferenceQueue = Depends(get_inference_queue),
    hardware: HardwareInfo = Depends(get_hardware_info),
) -> HealthResponse:
    # Always 200 as long as the worker loop is alive - a deployment with some or all
    # capabilities unconfigured is a normal, expected transitional state (this project's own
    # current state for table-cell/doc-orientation), not a failure. Restarting a correctly
    # running, intentionally-partial deployment would be worse than reporting it accurately.
    # 503 is reserved for genuine malfunction, which nothing here currently detects.
    capabilities = {name: engine.is_loaded() for name, engine in engines.items()}
    status_label = "ready" if any(capabilities.values()) else "no_capabilities_configured"

    return HealthResponse(
        status=status_label,
        capabilities=capabilities,
        queue_depth=queue.queue_depth,
        hardware=HardwareSummary(
            using_device=hardware.using_device,
            compiled_with_cuda=hardware.compiled_with_cuda,
            cuda_device_count=hardware.cuda_device_count,
            device_name=hardware.device_name,
        ),
    )
