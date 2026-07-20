from typing import Literal

from pydantic import BaseModel


class PredictParams(BaseModel):
    use_doc_orientation_classify: bool = True


class OcrTextResult(BaseModel):
    text: str
    score: float
    box: list[list[float]]


class PredictResponse(BaseModel):
    request_id: str
    results: list[OcrTextResult]
    processing_time_ms: float
    device_used: str


class TableCellBox(BaseModel):
    label: str
    score: float
    coordinate: list[float]  # [xmin, ymin, xmax, ymax]


class TableCellDetectResponse(BaseModel):
    request_id: str
    table_type: Literal["wired", "wireless"]
    boxes: list[TableCellBox]
    processing_time_ms: float
    device_used: str


class DocOrientationResponse(BaseModel):
    request_id: str
    angle: Literal["0", "90", "180", "270"]
    score: float
    processing_time_ms: float
    device_used: str


class HardwareSummary(BaseModel):
    using_device: str
    compiled_with_cuda: bool
    cuda_device_count: int
    device_name: str | None = None


class CapabilitiesMap(BaseModel):
    ocr: bool
    table_cell_wired: bool
    table_cell_wireless: bool
    doc_orientation: bool


class HealthResponse(BaseModel):
    status: Literal["ready", "no_capabilities_configured"]
    capabilities: CapabilitiesMap
    queue_depth: int
    hardware: HardwareSummary


class ErrorResponse(BaseModel):
    error_code: str
    detail: str
