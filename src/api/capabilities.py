"""Declarative registry of every PaddleX standalone capability this service exposes. This is the
single source of truth main.py's lifespan uses to construct engines, and the admin/UI layer uses
to introspect labels/config - adding a new capability means adding one entry here, not a new
hand-written constructor block."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.api.worker.engine import PaddleOCREngine
from src.api.worker.paddlex_engine import PaddleXModelEngine
from src.api.worker.result_mappers import (
    map_detection_box_result,
    map_formula_result,
    map_orientation_result,
    map_seal_detection_result,
    map_table_structure_result,
    map_unwarp_result,
)
from src.config import Settings
from src.hardware import HardwareInfo


@dataclass(frozen=True)
class PaddleXCapabilitySpec:
    name: str
    label: str
    model_dir_name_field: str
    model_name_field: str
    default_model_name: str
    result_mapper: Callable[[Any], list[Any]]
    default_predict_kwargs: dict[str, Any] = field(default_factory=dict)


PADDLEX_CAPABILITIES: list[PaddleXCapabilitySpec] = [
    PaddleXCapabilitySpec(
        "table_cell_wired",
        "Table Cell Detection (Wired)",
        "table_cell_wired_model_dir_name",
        "table_cell_wired_model_name",
        "RT-DETR-L_wired_table_cell_det",
        map_detection_box_result,
    ),
    PaddleXCapabilitySpec(
        "table_cell_wireless",
        "Table Cell Detection (Wireless)",
        "table_cell_wireless_model_dir_name",
        "table_cell_wireless_model_name",
        "RT-DETR-L_wireless_table_cell_det",
        map_detection_box_result,
    ),
    PaddleXCapabilitySpec(
        "doc_orientation",
        "Document Orientation",
        "doc_orientation_model_dir_name",
        "doc_orientation_model_name",
        "PP-LCNet_x1_0_doc_ori",
        map_orientation_result,
    ),
    PaddleXCapabilitySpec(
        "table_structure",
        "Table Structure Recognition",
        "table_structure_model_dir_name",
        "table_structure_model_name",
        "SLANet_plus",
        map_table_structure_result,
    ),
    PaddleXCapabilitySpec(
        "layout_detection",
        "Layout Detection",
        "layout_detection_model_dir_name",
        "layout_detection_model_name",
        "PP-DocLayout_plus-L",
        map_detection_box_result,
    ),
    PaddleXCapabilitySpec(
        "formula_recognition",
        "Formula Recognition",
        "formula_model_dir_name",
        "formula_model_name",
        "PP-FormulaNet_plus-M",
        map_formula_result,
    ),
    PaddleXCapabilitySpec(
        "seal_detection",
        "Seal Text Detection",
        "seal_det_model_dir_name",
        "seal_det_model_name",
        "PP-OCRv4_server_seal_det",
        map_seal_detection_result,
    ),
    PaddleXCapabilitySpec(
        "doc_unwarping",
        "Document Unwarping",
        "doc_unwarping_model_dir_name",
        "doc_unwarping_model_name",
        "UVDoc",
        map_unwarp_result,
    ),
    PaddleXCapabilitySpec(
        "textline_orientation",
        "Text-line Orientation",
        "textline_orientation_model_dir_name",
        "textline_orientation_model_name",
        "PP-LCNet_x1_0_textline_ori",
        map_orientation_result,
    ),
]

CAPABILITY_LABELS: dict[str, str] = {
    "ocr": "OCR (Detection + Recognition)",
    **{spec.name: spec.label for spec in PADDLEX_CAPABILITIES},
}


def build_engines(settings: Settings, hardware: HardwareInfo, model_files_dir: Path) -> dict[str, Any]:
    engines: dict[str, Any] = {
        "ocr": PaddleOCREngine(model_files_dir=model_files_dir, hardware=hardware, settings=settings)
    }
    for spec in PADDLEX_CAPABILITIES:
        predict_kwargs = dict(spec.default_predict_kwargs)
        if spec.name in ("table_cell_wired", "table_cell_wireless"):
            predict_kwargs.setdefault("threshold", settings.table_cell_detection_threshold)
        predict_kwargs.setdefault("batch_size", 1)

        engines[spec.name] = PaddleXModelEngine(
            capability_label=spec.label,
            model_dir_name=getattr(settings, spec.model_dir_name_field),
            model_name=getattr(settings, spec.model_name_field),
            default_model_name=spec.default_model_name,
            result_mapper=spec.result_mapper,
            model_files_dir=model_files_dir,
            hardware=hardware,
            settings=settings,
            default_predict_kwargs=predict_kwargs,
        )
    return engines
