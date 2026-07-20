import time
from pathlib import Path

from loguru import logger

from src.api.errors import ModelNotFoundError, ModelNotLoadedError
from src.api.schemas import OcrTextResult, PredictParams
from src.config import Settings
from src.hardware import HardwareInfo
from src.utils.image_decoding import decode_image_to_bgr
from src.utils.paddlex_registry import is_known_model_name
from src.utils.path_finder import find_path


class PaddleOCREngine:
    """Owns the PaddleOCR pipeline instance. All methods that touch the pipeline are blocking
    and expected to be called via an executor - see api/worker/queue_manager.py."""

    def __init__(self, model_files_dir: Path, hardware: HardwareInfo, settings: Settings):
        self._model_files_dir = model_files_dir
        self._hardware = hardware
        self._settings = settings
        self._pipeline = None
        self._doc_orientation_enabled = False

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def _resolve_model_dir(self, dir_name: str | None) -> str | None:
        if not dir_name:
            return None
        found = find_path(
            name=dir_name,
            find_type="folder",
            start_path=str(self._model_files_dir),
        )
        if found is None:
            return None
        return str((self._model_files_dir / found).resolve())

    def load(self) -> None:
        settings = self._settings

        det_dir = self._resolve_model_dir(settings.det_model_dir_name)
        rec_dir = self._resolve_model_dir(settings.rec_model_dir_name)
        doc_orientation_dir = self._resolve_model_dir(settings.doc_orientation_model_dir_name)

        problems = []
        if settings.det_model_dir_name and det_dir is None:
            problems.append(f"detection model directory {settings.det_model_dir_name!r} not found")
        if settings.rec_model_dir_name and rec_dir is None:
            problems.append(f"recognition model directory {settings.rec_model_dir_name!r} not found")
        if settings.doc_orientation_model_dir_name and doc_orientation_dir is None:
            problems.append(f"doc orientation model directory {settings.doc_orientation_model_dir_name!r} not found")

        # Catches typos and version mismatches (e.g. a PP-OCRv6 model name against a paddlex
        # build that predates PP-OCRv6) with a clear message instead of a deep, unclear error
        # from inside paddlex/paddleocr internals - see PaddlePaddle/PaddleX#3797.
        for label, model_name in (
            ("detection", settings.det_model_name),
            ("recognition", settings.rec_model_name),
            ("doc orientation", settings.doc_orientation_model_name),
        ):
            if model_name and not is_known_model_name(model_name):
                problems.append(
                    f"{label} model name {model_name!r} is not recognized by the installed "
                    f"paddlex version - check spelling, or that paddleocr/paddlex is new enough"
                )

        if problems:
            message = f"Cannot load OCR pipeline: {'; '.join(problems)}."
            if settings.strict_model_loading:
                logger.error(message)
                raise ModelNotFoundError(message)
            logger.warning(message + " - /predict will return 503 until resolved.")
            self._pipeline = None
            return

        if det_dir is None and rec_dir is None:
            logger.warning(
                "No model directory names configured (DET_MODEL_DIR_NAME/REC_MODEL_DIR_NAME) - "
                "skipping pipeline load. /predict will return 503 until configured."
            )
            self._pipeline = None
            return

        from paddleocr import PaddleOCR

        kwargs: dict = {"device": self._hardware.paddle_device_string}
        if det_dir:
            kwargs["text_detection_model_dir"] = det_dir
            if settings.det_model_name:
                kwargs["text_detection_model_name"] = settings.det_model_name
        if rec_dir:
            kwargs["text_recognition_model_dir"] = rec_dir
            if settings.rec_model_name:
                kwargs["text_recognition_model_name"] = settings.rec_model_name
        if doc_orientation_dir:
            kwargs["doc_orientation_classify_model_dir"] = doc_orientation_dir
            if settings.doc_orientation_model_name:
                kwargs["doc_orientation_classify_model_name"] = settings.doc_orientation_model_name
            kwargs["use_doc_orientation_classify"] = True
        else:
            kwargs["use_doc_orientation_classify"] = False
        self._doc_orientation_enabled = bool(doc_orientation_dir)

        # Det + rec are the only mandatory sub-models. PaddleOCR's default pipeline also wires
        # up doc-unwarping (UVDoc) and text-line orientation classification, silently
        # auto-downloading their weights from the internet on first use if left enabled - which
        # breaks the "network-isolated by default" requirement and adds an undocumented runtime
        # dependency. Disable both since we have no local weights configured for either.
        kwargs["use_doc_unwarping"] = False
        kwargs["use_textline_orientation"] = False

        logger.info(f"Loading PaddleOCR pipeline with device={kwargs['device']!r}")
        self._pipeline = PaddleOCR(**kwargs)
        logger.info("PaddleOCR pipeline loaded successfully.")

    def unload(self) -> None:
        self._pipeline = None

    def run(self, image_bytes: bytes, params: PredictParams) -> tuple[list[OcrTextResult], float]:
        if not self.is_loaded():
            raise ModelNotLoadedError("No PaddleOCR model is currently loaded.")

        image = decode_image_to_bgr(image_bytes)

        # PaddleX only builds the doc-preprocessor sub-pipeline when
        # use_doc_orientation_classify=True at *construction* time (see load()); requesting it
        # per-call when it wasn't constructed raises AttributeError inside paddlex. Clamp to what
        # the loaded pipeline actually supports rather than trusting the request's params.
        use_doc_orientation_classify = self._doc_orientation_enabled and params.use_doc_orientation_classify

        started = time.monotonic()
        raw_results = self._pipeline.predict(
            image,
            use_doc_orientation_classify=use_doc_orientation_classify,
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
