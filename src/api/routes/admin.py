import asyncio
import functools
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from src.api.capabilities import CAPABILITY_LABELS
from src.api.deps import get_engines_map, get_inference_queue, get_model_files_dir, get_settings
from src.api.errors import ModelNotFoundError
from src.api.schemas import (
    AvailableModelsResponse,
    CapabilitiesListResponse,
    CapabilityInfo,
    ReloadCapabilityRequest,
    ReloadCapabilityResponse,
)
from src.api.worker.job import InferenceJob
from src.api.worker.queue_manager import InferenceQueue
from src.config import Settings
from src.security.auth import Principal, get_current_principal

router = APIRouter(prefix="/admin", tags=["admin"])


def _require_capability(name: str, engines: dict) -> None:
    if name not in engines:
        raise HTTPException(status_code=404, detail=f"Unknown capability {name!r}")


@router.get("/capabilities", response_model=CapabilitiesListResponse)
async def list_capabilities(
    engines: dict = Depends(get_engines_map),
    principal: Principal = Depends(get_current_principal),
) -> CapabilitiesListResponse:
    return CapabilitiesListResponse(
        capabilities=[
            CapabilityInfo(
                name=name,
                label=CAPABILITY_LABELS.get(name, name),
                loaded=engine.is_loaded(),
                config=engine.get_config(),
                problems=engine.get_load_problems(),
            )
            for name, engine in engines.items()
        ]
    )


@router.get("/capabilities/{name}/available-models", response_model=AvailableModelsResponse)
async def available_models(
    name: str,
    engines: dict = Depends(get_engines_map),
    model_files_dir: Path = Depends(get_model_files_dir),
    principal: Principal = Depends(get_current_principal),
) -> AvailableModelsResponse:
    _require_capability(name, engines)
    if not model_files_dir.is_dir():
        return AvailableModelsResponse(directories=[])
    directories = sorted(p.name for p in model_files_dir.iterdir() if p.is_dir())
    return AvailableModelsResponse(directories=directories)


@router.post("/capabilities/{name}/reload", response_model=ReloadCapabilityResponse)
async def reload_capability(
    name: str,
    body: ReloadCapabilityRequest,
    engines: dict = Depends(get_engines_map),
    queue: InferenceQueue = Depends(get_inference_queue),
    settings: Settings = Depends(get_settings),
    principal: Principal = Depends(get_current_principal),
) -> ReloadCapabilityResponse:
    _require_capability(name, engines)
    engine = engines[name]

    loop = asyncio.get_running_loop()
    job = InferenceJob(
        call=functools.partial(engine.reload, **body.config),
        future=loop.create_future(),
        timeout=settings.model_reload_timeout_seconds,
    )

    try:
        result = await queue.submit(job)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ReloadCapabilityResponse(name=name, loaded=result.loaded, problems=result.problems)
