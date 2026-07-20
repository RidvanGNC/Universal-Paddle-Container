import asyncio
import functools
from typing import Literal

from fastapi import APIRouter, Depends, Form, UploadFile

from src.api.deps import (
    get_hardware_info,
    get_inference_queue,
    get_table_cell_wired_engine,
    get_table_cell_wireless_engine,
)
from src.api.schemas import TableCellDetectResponse
from src.api.upload import read_validated_image
from src.api.worker.job import InferenceJob
from src.api.worker.paddlex_engine import PaddleXModelEngine
from src.api.worker.queue_manager import InferenceQueue
from src.config import Settings, get_settings
from src.hardware import HardwareInfo
from src.security.auth import Principal, get_current_principal

router = APIRouter(prefix="/table", tags=["table"])


@router.post("/detect-cells", response_model=TableCellDetectResponse)
async def detect_cells(
    file: UploadFile,
    table_type: Literal["wired", "wireless"] = Form(...),
    threshold: float | None = Form(None),
    settings: Settings = Depends(get_settings),
    queue: InferenceQueue = Depends(get_inference_queue),
    hardware: HardwareInfo = Depends(get_hardware_info),
    wired_engine: PaddleXModelEngine = Depends(get_table_cell_wired_engine),
    wireless_engine: PaddleXModelEngine = Depends(get_table_cell_wireless_engine),
    principal: Principal = Depends(get_current_principal),
) -> TableCellDetectResponse:
    image_bytes = await read_validated_image(file, settings)

    engine = wired_engine if table_type == "wired" else wireless_engine
    effective_threshold = threshold if threshold is not None else settings.table_cell_detection_threshold

    loop = asyncio.get_running_loop()
    job = InferenceJob(
        call=functools.partial(engine.run, image_bytes, threshold=effective_threshold, batch_size=1),
        future=loop.create_future(),
        timeout=settings.inference_timeout_seconds,
    )

    boxes, elapsed_ms = await queue.submit(job)

    return TableCellDetectResponse(
        request_id=job.request_id,
        table_type=table_type,
        boxes=boxes,
        processing_time_ms=elapsed_ms,
        device_used=hardware.paddle_device_string,
    )
