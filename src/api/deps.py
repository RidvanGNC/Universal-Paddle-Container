from fastapi import Request

from src.api.worker.engine import PaddleOCREngine
from src.api.worker.paddlex_engine import PaddleXModelEngine
from src.api.worker.queue_manager import InferenceQueue
from src.config import Settings, get_settings
from src.hardware import HardwareInfo


def get_hardware_info(request: Request) -> HardwareInfo:
    return request.app.state.hardware


def get_ocr_engine(request: Request) -> PaddleOCREngine:
    return request.app.state.ocr_engine


def get_table_cell_wired_engine(request: Request) -> PaddleXModelEngine:
    return request.app.state.table_cell_wired_engine


def get_table_cell_wireless_engine(request: Request) -> PaddleXModelEngine:
    return request.app.state.table_cell_wireless_engine


def get_doc_orientation_engine(request: Request) -> PaddleXModelEngine:
    return request.app.state.doc_orientation_engine


def get_engines_map(request: Request) -> dict:
    return request.app.state.engines


def get_inference_queue(request: Request) -> InferenceQueue:
    return request.app.state.inference_queue


__all__ = [
    "get_settings",
    "get_hardware_info",
    "get_ocr_engine",
    "get_table_cell_wired_engine",
    "get_table_cell_wireless_engine",
    "get_doc_orientation_engine",
    "get_engines_map",
    "get_inference_queue",
    "Settings",
]
