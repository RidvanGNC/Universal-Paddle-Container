import time
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from src.api.errors import ModelNotFoundError, ModelNotLoadedError
from src.api.schemas import TableCellBox
from src.config import Settings
from src.hardware import HardwareInfo
from src.utils.image_decoding import decode_image_to_bgr
from src.utils.paddlex_registry import is_known_model_name
from src.utils.path_finder import find_path


class PaddleXModelEngine:
    """Generic single-model engine wrapping paddlex.create_model() for standalone (non-pipeline)
    capabilities - table cell detection and standalone doc-orientation classification. Mirrors
    PaddleOCREngine's load/is_loaded/unload/degradation contract exactly; the only
    capability-specific piece is `result_mapper`, which turns one page of PaddleX's dict-like
    predict() output into a flat list of normalized items."""

    def __init__(
        self,
        capability_label: str,
        model_dir_name: str | None,
        model_name: str | None,
        default_model_name: str,
        result_mapper: Callable[[Any], list[Any]],
        model_files_dir: Path,
        hardware: HardwareInfo,
        settings: Settings,
        default_predict_kwargs: dict | None = None,
    ):
        self._capability_label = capability_label
        self._model_dir_name = model_dir_name
        self._model_name = model_name or default_model_name
        self._result_mapper = result_mapper
        self._model_files_dir = model_files_dir
        self._hardware = hardware
        self._settings = settings
        self._default_predict_kwargs = default_predict_kwargs or {}
        self._model = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def _resolve_model_dir(self) -> str | None:
        if not self._model_dir_name:
            return None
        found = find_path(name=self._model_dir_name, find_type="folder", start_path=str(self._model_files_dir))
        if found is None:
            return None
        return str((self._model_files_dir / found).resolve())

    def load(self) -> None:
        if not self._model_dir_name:
            logger.warning(
                f"[{self._capability_label}] Not configured (no model dir name) - this "
                f"capability's endpoint will return 503 until configured."
            )
            self._model = None
            return

        model_dir = self._resolve_model_dir()
        if model_dir is None:
            message = (
                f"[{self._capability_label}] Configured model directory "
                f"{self._model_dir_name!r} not found under {self._model_files_dir}."
            )
            if self._settings.strict_model_loading:
                logger.error(message)
                raise ModelNotFoundError(message)
            logger.warning(message + " - will return 503 until available.")
            self._model = None
            return

        # Catches typos and version mismatches (e.g. a model name from a newer paddlex release
        # than what's installed) with a clear message instead of a deep, unclear error from
        # inside paddlex internals - see PaddlePaddle/PaddleX#3797.
        if not is_known_model_name(self._model_name):
            message = (
                f"[{self._capability_label}] Model name {self._model_name!r} is not recognized "
                f"by the installed paddlex version - check spelling, or that paddleocr/paddlex "
                f"is new enough."
            )
            if self._settings.strict_model_loading:
                logger.error(message)
                raise ModelNotFoundError(message)
            logger.warning(message + " - will return 503 until resolved.")
            self._model = None
            return

        from paddlex import create_model

        logger.info(
            f"[{self._capability_label}] Loading {self._model_name!r} "
            f"device={self._hardware.paddle_device_string!r}"
        )
        self._model = create_model(
            model_name=self._model_name,
            model_dir=model_dir,
            device=self._hardware.paddle_device_string,
        )
        logger.info(f"[{self._capability_label}] Loaded successfully.")

    def unload(self) -> None:
        self._model = None

    def run(self, image_bytes: bytes, **predict_kwargs) -> tuple[list[Any], float]:
        if not self.is_loaded():
            raise ModelNotLoadedError(f"No {self._capability_label} model is currently loaded.")

        image = decode_image_to_bgr(image_bytes)
        kwargs = {**self._default_predict_kwargs, **predict_kwargs}

        started = time.monotonic()
        raw_results = list(self._model.predict(image, **kwargs))
        elapsed_ms = (time.monotonic() - started) * 1000

        mapped: list[Any] = []
        for page in raw_results:
            mapped.extend(self._result_mapper(page))
        return mapped, elapsed_ms


def map_table_cell_result(page) -> list[TableCellBox]:
    boxes = page.get("boxes", []) if hasattr(page, "get") else []
    return [
        TableCellBox(
            label=box.get("label", "cell"),
            score=float(box.get("score", 0.0)),
            coordinate=[float(v) for v in box.get("coordinate", [])],
        )
        for box in boxes
    ]


def map_doc_orientation_result(page) -> list[dict]:
    label_names = page.get("label_names", []) if hasattr(page, "get") else []
    scores = page.get("scores", []) if hasattr(page, "get") else []
    if not label_names:
        return []
    return [{"angle": label_names[0], "score": float(scores[0]) if scores else 0.0}]
