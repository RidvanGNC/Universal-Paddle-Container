"""Declarative playground form config - one entry per testable capability, so the playground
page is a single generic form renderer instead of one hand-written form per capability."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class PlaygroundFieldSpec:
    name: str
    kind: Literal["checkbox", "select", "number"]
    default: Any = None
    options: list[str] | None = None


@dataclass(frozen=True)
class PlaygroundEndpointSpec:
    capability_name: str
    label: str
    path: str
    method: str = "POST"
    extra_fields: list[PlaygroundFieldSpec] = field(default_factory=list)
    response_kind: Literal["json", "image"] = "json"


PLAYGROUND_ENDPOINTS: list[PlaygroundEndpointSpec] = [
    PlaygroundEndpointSpec(
        "ocr",
        "OCR (Detection + Recognition)",
        "/predict",
        extra_fields=[
            PlaygroundFieldSpec("use_doc_orientation_classify", "checkbox", True),
            PlaygroundFieldSpec("use_doc_unwarping", "checkbox", True),
            PlaygroundFieldSpec("use_textline_orientation", "checkbox", True),
        ],
    ),
    PlaygroundEndpointSpec(
        "table_cell_wired",
        "Table Cell Detection",
        "/table/detect-cells",
        extra_fields=[
            PlaygroundFieldSpec("table_type", "select", "wired", options=["wired", "wireless"]),
            PlaygroundFieldSpec("threshold", "number", 0.3),
        ],
    ),
    PlaygroundEndpointSpec("table_structure", "Table Structure Recognition", "/table/structure"),
    PlaygroundEndpointSpec("layout_detection", "Layout Detection", "/layout/detect"),
    PlaygroundEndpointSpec("formula_recognition", "Formula Recognition", "/formula/recognize"),
    PlaygroundEndpointSpec("seal_detection", "Seal Text Detection", "/seal/detect"),
    PlaygroundEndpointSpec("doc_orientation", "Document Orientation", "/document/orientation"),
    PlaygroundEndpointSpec("doc_unwarping", "Document Unwarping", "/document/unwarp", response_kind="image"),
    PlaygroundEndpointSpec("textline_orientation", "Text-line Orientation", "/textline/orientation"),
]
