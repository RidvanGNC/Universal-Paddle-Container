from pathlib import Path

from fastapi import Request

from src.api.worker.engine import PaddleOCREngine
from src.api.worker.paddlex_engine import PaddleXModelEngine
from src.api.worker.queue_manager import InferenceQueue
from src.config import Settings, get_settings
from src.hardware import HardwareInfo


def get_hardware_info(request: Request) -> HardwareInfo:
    return request.app.state.hardware


def get_ocr_engine(request: Request) -> PaddleOCREngine:
    return request.app.state.engines["ocr"]


def get_table_cell_wired_engine(request: Request) -> PaddleXModelEngine:
    return request.app.state.engines["table_cell_wired"]


def get_table_cell_wireless_engine(request: Request) -> PaddleXModelEngine:
    return request.app.state.engines["table_cell_wireless"]


def get_doc_orientation_engine(request: Request) -> PaddleXModelEngine:
    return request.app.state.engines["doc_orientation"]


def engine_dependency(name: str):
    """Generic dependency factory for capabilities that only need one route each - avoids adding
    another one-line named getter per capability. Use the specific getters above where a route
    is established/reused in multiple places; use this for everything new."""

    def _get(request: Request):
        return request.app.state.engines[name]

    return _get


def get_engines_map(request: Request) -> dict:
    return request.app.state.engines


def get_inference_queue(request: Request) -> InferenceQueue:
    return request.app.state.inference_queue


def get_model_files_dir(request: Request) -> Path:
    return request.app.state.model_files_dir


__all__ = [
    "get_settings",
    "get_hardware_info",
    "get_ocr_engine",
    "get_table_cell_wired_engine",
    "get_table_cell_wireless_engine",
    "get_doc_orientation_engine",
    "engine_dependency",
    "get_engines_map",
    "get_inference_queue",
    "get_model_files_dir",
    "Settings",
]
