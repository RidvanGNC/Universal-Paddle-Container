import dataclasses
import gc
import time
from dataclasses import dataclass, fields
from pathlib import Path

from loguru import logger

from src.api.errors import ModelNotFoundError, ModelNotLoadedError
from src.api.schemas import OcrTextResult, PredictParams
from src.api.worker.load_result import LoadResult
from src.config import Settings
from src.hardware import HardwareInfo
from src.utils.image_decoding import decode_image_to_bgr
from src.utils.paddlex_registry import is_known_model_name
from src.utils.path_finder import find_path


@dataclass
class OcrEngineConfig:
    det_model_dir_name: str | None = None
    det_model_name: str | None = None
    rec_model_dir_name: str | None = None
    rec_model_name: str | None = None
    doc_orientation_model_dir_name: str | None = None
    doc_orientation_model_name: str | None = None
    doc_unwarping_model_dir_name: str | None = None
    doc_unwarping_model_name: str | None = None
    textline_orientation_model_dir_name: str | None = None
    textline_orientation_model_name: str | None = None


_OCR_CONFIG_FIELDS = {f.name for f in fields(OcrEngineConfig)}


class PaddleOCREngine:
    """Owns the PaddleOCR pipeline instance. All methods that touch the pipeline are blocking
    and expected to be called via an executor - see api/worker/queue_manager.py."""

    def __init__(self, model_files_dir: Path, hardware: HardwareInfo, settings: Settings):
        self._model_files_dir = model_files_dir
        self._hardware = hardware
        self._settings = settings
        self._config = OcrEngineConfig(
            det_model_dir_name=settings.det_model_dir_name,
            det_model_name=settings.det_model_name,
            rec_model_dir_name=settings.rec_model_dir_name,
            rec_model_name=settings.rec_model_name,
            doc_orientation_model_dir_name=settings.doc_orientation_model_dir_name,
            doc_orientation_model_name=settings.doc_orientation_model_name,
            doc_unwarping_model_dir_name=settings.doc_unwarping_model_dir_name,
            doc_unwarping_model_name=settings.doc_unwarping_model_name,
            textline_orientation_model_dir_name=settings.textline_orientation_model_dir_name,
            textline_orientation_model_name=settings.textline_orientation_model_name,
        )
        self._pipeline = None
        self._doc_orientation_enabled = False
        self._doc_unwarping_enabled = False
        self._textline_orientation_enabled = False
        self._load_problems: list[str] = []

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def get_config(self) -> dict[str, str | None]:
        return dataclasses.asdict(self._config)

    def get_load_problems(self) -> list[str]:
        return list(self._load_problems)

    def _resolve_slot(
        self, label: str, dir_name: str | None, model_name: str | None, *, required: bool
    ) -> tuple[str | None, list[str]]:
        """Resolves+validates one (dir_name, model_name) sub-model slot. Returns the resolved
        absolute directory (or None) and any problems found - de-duplicates what used to be
        inline logic repeated once per slot (det, rec, doc_orientation - now also doc_unwarping
        and textline_orientation, which would otherwise be a 5th copy-paste)."""
        if not dir_name:
            return None, ([f"{label}: no model directory configured"] if required else [])

        found = find_path(name=dir_name, find_type="folder", start_path=str(self._model_files_dir))
        problems = []
        resolved = None
        if found is None:
            problems.append(f"{label}: directory {dir_name!r} not found under {self._model_files_dir}")
        else:
            resolved = str((self._model_files_dir / found).resolve())

        # Catches typos and version mismatches (e.g. a PP-OCRv6 model name against a paddlex
        # build that predates PP-OCRv6) with a clear message instead of a deep, unclear error
        # from inside paddlex/paddleocr internals - see PaddlePaddle/PaddleX#3797.
        if model_name and not is_known_model_name(model_name):
            problems.append(f"{label}: model name {model_name!r} is not recognized by the installed paddlex version")

        return resolved, problems

    def load(self) -> None:
        c = self._config
        self._load_problems = []

        if not c.det_model_dir_name and not c.rec_model_dir_name:
            logger.warning(
                "No model directory names configured (det_model_dir_name/rec_model_dir_name) - "
                "skipping pipeline load. /predict will return 503 until configured."
            )
            self._pipeline = None
            return

        det_dir, det_problems = self._resolve_slot("detection", c.det_model_dir_name, c.det_model_name, required=True)
        rec_dir, rec_problems = self._resolve_slot(
            "recognition", c.rec_model_dir_name, c.rec_model_name, required=True
        )
        orient_dir, orient_problems = self._resolve_slot(
            "doc orientation", c.doc_orientation_model_dir_name, c.doc_orientation_model_name, required=False
        )
        unwarp_dir, unwarp_problems = self._resolve_slot(
            "doc unwarping", c.doc_unwarping_model_dir_name, c.doc_unwarping_model_name, required=False
        )
        textline_dir, textline_problems = self._resolve_slot(
            "textline orientation",
            c.textline_orientation_model_dir_name,
            c.textline_orientation_model_name,
            required=False,
        )

        required_problems = det_problems + rec_problems
        optional_problems = orient_problems + unwarp_problems + textline_problems
        self._load_problems = required_problems + optional_problems

        if required_problems:
            message = f"Cannot load OCR pipeline: {'; '.join(required_problems)}."
            if self._settings.strict_model_loading:
                logger.error(message)
                raise ModelNotFoundError(message)
            logger.warning(message + " - /predict will return 503 until resolved.")
            self._pipeline = None
            return

        if optional_problems:
            # A problem in an optional slot only disables that one feature - it does not take
            # down det+rec, unlike a problem in a required slot above. This matters more now
            # that there are 3 optional slots instead of 1, and a reload UI where testing a
            # deliberately-bad optional config is an expected action.
            logger.warning("Optional OCR sub-model(s) disabled due to configuration problems: " + "; ".join(
                optional_problems
            ))

        self._doc_orientation_enabled = bool(orient_dir) and not orient_problems
        self._doc_unwarping_enabled = bool(unwarp_dir) and not unwarp_problems
        self._textline_orientation_enabled = bool(textline_dir) and not textline_problems

        from paddleocr import PaddleOCR

        # Drop any previously loaded pipeline before constructing the new one, so a reload never
        # transiently holds two model instances resident (matters most for GPU memory).
        self._pipeline = None
        gc.collect()

        kwargs: dict = {
            "device": self._hardware.paddle_device_string,
            "text_detection_model_dir": det_dir,
            "text_recognition_model_dir": rec_dir,
            "use_doc_orientation_classify": self._doc_orientation_enabled,
            "use_doc_unwarping": self._doc_unwarping_enabled,
            "use_textline_orientation": self._textline_orientation_enabled,
        }
        if c.det_model_name:
            kwargs["text_detection_model_name"] = c.det_model_name
        if c.rec_model_name:
            kwargs["text_recognition_model_name"] = c.rec_model_name
        if self._doc_orientation_enabled:
            kwargs["doc_orientation_classify_model_dir"] = orient_dir
            if c.doc_orientation_model_name:
                kwargs["doc_orientation_classify_model_name"] = c.doc_orientation_model_name
        if self._doc_unwarping_enabled:
            kwargs["doc_unwarping_model_dir"] = unwarp_dir
            if c.doc_unwarping_model_name:
                kwargs["doc_unwarping_model_name"] = c.doc_unwarping_model_name
        if self._textline_orientation_enabled:
            kwargs["textline_orientation_model_dir"] = textline_dir
            if c.textline_orientation_model_name:
                kwargs["textline_orientation_model_name"] = c.textline_orientation_model_name

        logger.info(f"Loading PaddleOCR pipeline with device={kwargs['device']!r}")
        self._pipeline = PaddleOCR(**kwargs)
        logger.info("PaddleOCR pipeline loaded successfully.")

    def reload(self, **updates) -> LoadResult:
        unknown = set(updates) - _OCR_CONFIG_FIELDS
        if unknown:
            raise ValueError(f"Unknown OCR config field(s): {sorted(unknown)}")
        self._config = dataclasses.replace(self._config, **updates)
        self.load()
        return LoadResult(loaded=self.is_loaded(), problems=self.get_load_problems())

    def unload(self) -> None:
        self._pipeline = None

    def run(self, image_bytes: bytes, params: PredictParams) -> tuple[list[OcrTextResult], float]:
        if not self.is_loaded():
            raise ModelNotLoadedError("No PaddleOCR model is currently loaded.")

        image = decode_image_to_bgr(image_bytes)

        # PaddleX only builds a sub-pipeline (doc-orientation/doc-unwarping/textline-orientation)
        # when its use_* flag is True at *construction* time (see load()); requesting one
        # per-call when it wasn't constructed raises AttributeError inside paddlex. Clamp to what
        # the loaded pipeline actually supports rather than trusting the request's params.
        use_doc_orientation_classify = self._doc_orientation_enabled and params.use_doc_orientation_classify
        use_doc_unwarping = self._doc_unwarping_enabled and params.use_doc_unwarping
        use_textline_orientation = self._textline_orientation_enabled and params.use_textline_orientation

        started = time.monotonic()
        raw_results = self._pipeline.predict(
            image,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
        )
        elapsed_ms = (time.monotonic() - started) * 1000

        results: list[OcrTextResult] = []
        for page in raw_results:
            texts = page.get("rec_texts", []) if hasattr(page, "get") else []
            scores = page.get("rec_scores", []) if hasattr(page, "get") else []
            polys = page.get("rec_polys", page.get("dt_polys", [])) if hasattr(page, "get") else []
            for text, score, box in zip(texts, scores, polys):
                results.append(
                    OcrTextResult(
                        text=text,
                        score=float(score),
                        box=[[float(x), float(y)] for x, y in box],
                    )
                )

        return results, elapsed_ms
