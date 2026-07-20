import asyncio
import functools

from fastapi import APIRouter, Depends, UploadFile

from src.api.deps import get_doc_orientation_engine, get_hardware_info, get_inference_queue
from src.api.schemas import DocOrientationResponse
from src.api.upload import read_validated_image
from src.api.worker.job import InferenceJob
from src.api.worker.paddlex_engine import PaddleXModelEngine
from src.api.worker.queue_manager import InferenceQueue
from src.config import Settings, get_settings
from src.hardware import HardwareInfo
from src.security.auth import Principal, get_current_principal

router = APIRouter(prefix="/document", tags=["document"])


@router.post("/orientation", response_model=DocOrientationResponse)
async def classify_orientation(
    file: UploadFile,
    settings: Settings = Depends(get_settings),
    queue: InferenceQueue = Depends(get_inference_queue),
    hardware: HardwareInfo = Depends(get_hardware_info),
    engine: PaddleXModelEngine = Depends(get_doc_orientation_engine),
    principal: Principal = Depends(get_current_principal),
) -> DocOrientationResponse:
    image_bytes = await read_validated_image(file, settings)

    loop = asyncio.get_running_loop()
    job = InferenceJob(
        call=functools.partial(engine.run, image_bytes, batch_size=1),
        future=loop.create_future(),
        timeout=settings.inference_timeout_seconds,
    )

    results, elapsed_ms = await queue.submit(job)
    result = results[0]

    return DocOrientationResponse(
        request_id=job.request_id,
        angle=result["angle"],
        score=result["score"],
        processing_time_ms=elapsed_ms,
        device_used=hardware.paddle_device_string,
    )
