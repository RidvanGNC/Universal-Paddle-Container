import asyncio
import functools

from fastapi import APIRouter, Depends, UploadFile

from src.api.deps import get_hardware_info, get_inference_queue, get_ocr_engine
from src.api.schemas import PredictParams, PredictResponse
from src.api.upload import read_validated_image
from src.api.worker.engine import PaddleOCREngine
from src.api.worker.job import InferenceJob
from src.api.worker.queue_manager import InferenceQueue
from src.config import Settings, get_settings
from src.hardware import HardwareInfo
from src.security.auth import Principal, get_current_principal

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile,
    settings: Settings = Depends(get_settings),
    queue: InferenceQueue = Depends(get_inference_queue),
    hardware: HardwareInfo = Depends(get_hardware_info),
    engine: PaddleOCREngine = Depends(get_ocr_engine),
    principal: Principal = Depends(get_current_principal),
) -> PredictResponse:
    image_bytes = await read_validated_image(file, settings)

    loop = asyncio.get_running_loop()
    job = InferenceJob(
        call=functools.partial(engine.run, image_bytes, PredictParams()),
        future=loop.create_future(),
        timeout=settings.inference_timeout_seconds,
    )

    results, elapsed_ms = await queue.submit(job)

    return PredictResponse(
        request_id=job.request_id,
        results=results,
        processing_time_ms=elapsed_ms,
        device_used=hardware.paddle_device_string,
    )
