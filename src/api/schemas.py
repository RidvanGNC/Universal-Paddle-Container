from typing import Literal

from pydantic import BaseModel


class PredictParams(BaseModel):
    use_doc_orientation_classify: bool = True
    use_doc_unwarping: bool = True
    use_textline_orientation: bool = True


class OcrTextResult(BaseModel):
    text: str
    score: float
    box: list[list[float]]


class OcrPageResult(BaseModel):
    page_index: int
    results: list[OcrTextResult]


class PredictResponse(BaseModel):
    request_id: str
    pages: list[OcrPageResult]
    processing_time_ms: float
    device_used: str


class DetectionBox(BaseModel):
    label: str
    score: float
    coordinate: list[float]  # [xmin, ymin, xmax, ymax]


class TableCellDetectResponse(BaseModel):
    request_id: str
    table_type: Literal["wired", "wireless"]
    boxes: list[DetectionBox]
    processing_time_ms: float
    device_used: str


class LayoutDetectResponse(BaseModel):
    request_id: str
    boxes: list[DetectionBox]
    processing_time_ms: float
    device_used: str


class TableStructureResponse(BaseModel):
    request_id: str
    bbox: list[list[float]]
    structure: list[str]
    structure_score: float
    processing_time_ms: float
    device_used: str


class FormulaRecognitionResponse(BaseModel):
    request_id: str
    rec_formula: str
    processing_time_ms: float
    device_used: str


class SealDetectionBox(BaseModel):
    score: float
    box: list[list[float]]


class SealDetectionResponse(BaseModel):
    request_id: str
    boxes: list[SealDetectionBox]
    processing_time_ms: float
    device_used: str


class DocOrientationResponse(BaseModel):
    request_id: str
    angle: Literal["0", "90", "180", "270"]
    score: float
    processing_time_ms: float
    device_used: str


class TextLineOrientationResponse(BaseModel):
    request_id: str
    # Not a strict Literal like DocOrientationResponse.angle - the real label vocabulary for
    # this model (e.g. "180" vs "180_degree") isn't confirmed against real weights yet.
    angle: str
    score: float
    processing_time_ms: float
    device_used: str


class HardwareSummary(BaseModel):
    using_device: str
    compiled_with_cuda: bool
    cuda_device_count: int
    device_name: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ready", "no_capabilities_configured"]
    capabilities: dict[str, bool]
    queue_depth: int
    hardware: HardwareSummary


class ErrorResponse(BaseModel):
    error_code: str
    detail: str


class CapabilityInfo(BaseModel):
    name: str
    label: str
    loaded: bool
    config: dict[str, str | None]
    problems: list[str]


class CapabilitiesListResponse(BaseModel):
    capabilities: list[CapabilityInfo]


class AvailableModelsResponse(BaseModel):
    directories: list[str]


class ReloadCapabilityRequest(BaseModel):
    config: dict[str, str]


class ReloadCapabilityResponse(BaseModel):
    name: str
    loaded: bool
    problems: list[str]
