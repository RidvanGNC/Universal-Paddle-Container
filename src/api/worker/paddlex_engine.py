import gc
import time
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from src.api.errors import ModelNotFoundError, ModelNotLoadedError
from src.api.worker.load_result import LoadResult
from src.config import Settings
from src.hardware import HardwareInfo
from src.utils.image_decoding import decode_image_to_bgr
from src.utils.paddlex_registry import is_known_model_name
from src.utils.path_finder import find_path


class PaddleXModelEngine:
    """Generic single-model engine wrapping paddlex.create_model() for standalone (non-pipeline)
    capabilities - table cell detection, layout detection, table structure recognition, formula
    recognition, seal text detection, doc-unwarping, standalone doc/textline orientation. Mirrors
    PaddleOCREngine's load/is_loaded/unload/degradation/reload contract; the only
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
        self._load_problems: list[str] = []

    def is_loaded(self) -> bool:
        return self._model is not None

    def get_config(self) -> dict[str, str | None]:
        return {"model_dir_name": self._model_dir_name, "model_name": self._model_name}

    def get_load_problems(self) -> list[str]:
        return list(self._load_problems)

    def _resolve_model_dir(self) -> str | None:
        if not self._model_dir_name:
            return None
        found = find_path(name=self._model_dir_name, find_type="folder", start_path=str(self._model_files_dir))
        if found is None:
            return None
        return str((self._model_files_dir / found).resolve())

    def load(self) -> None:
        self._load_problems = []

        if not self._model_dir_name:
            logger.warning(
                f"[{self._capability_label}] Not configured (no model dir name) - this "
                f"capability's endpoint will return 503 until configured."
            )
            self._model = None
            return

        model_dir = self._resolve_model_dir()
        if model_dir is None:
            self._load_problems.append(
                f"model directory {self._model_dir_name!r} not found under {self._model_files_dir}"
            )

        # Catches typos and version mismatches (e.g. a model name from a newer paddlex release
        # than what's installed) with a clear message instead of a deep, unclear error from
        # inside paddlex internals - see PaddlePaddle/PaddleX#3797.
        if not is_known_model_name(self._model_name):
            self._load_problems.append(
                f"model name {self._model_name!r} is not recognized by the installed paddlex "
                f"version - check spelling, or that paddleocr/paddlex is new enough"
            )

        if self._load_problems:
            message = f"[{self._capability_label}] Cannot load: {'; '.join(self._load_problems)}."
            if self._settings.strict_model_loading:
                logger.error(message)
                raise ModelNotFoundError(message)
            logger.warning(message + " - will return 503 until resolved.")
            self._model = None
            return

        from paddlex import create_model

        # Drop any previously loaded model before constructing the new one, so a reload never
        # transiently holds two model instances resident (matters most for GPU memory).
        self._model = None
        gc.collect()

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

    def reload(self, *, model_dir_name: str | None = None, model_name: str | None = None) -> LoadResult:
        if model_dir_name is not None:
            self._model_dir_name = model_dir_name
        if model_name is not None:
            self._model_name = model_name
        self.load()
        return LoadResult(loaded=self.is_loaded(), problems=self.get_load_problems())

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
