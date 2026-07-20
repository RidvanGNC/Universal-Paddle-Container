from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.schemas import ErrorResponse


class QueueFullError(RuntimeError):
    """Raised when the inference queue is at capacity."""


class InferenceTimeoutError(RuntimeError):
    """Raised when a submitted job does not complete within its timeout."""


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested but no model is loaded."""


class InvalidImageError(ValueError):
    """Raised when the uploaded file cannot be decoded/validated as an image."""


class ShuttingDownError(RuntimeError):
    """Raised for jobs still queued when the application is shutting down."""


class ModelNotFoundError(RuntimeError):
    """Raised at load() time when strict_model_loading=True and a configured model dir is missing.

    Not registered as an HTTP exception handler - it's only ever raised during startup, where it's
    meant to abort the process, not produce a response."""


_STATUS_MAP: dict[type[Exception], int] = {
    QueueFullError: status.HTTP_503_SERVICE_UNAVAILABLE,
    InferenceTimeoutError: status.HTTP_504_GATEWAY_TIMEOUT,
    ModelNotLoadedError: status.HTTP_503_SERVICE_UNAVAILABLE,
    InvalidImageError: status.HTTP_400_BAD_REQUEST,
    ShuttingDownError: status.HTTP_503_SERVICE_UNAVAILABLE,
}


def _make_handler(exc_type: type[Exception], status_code: int):
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(error_code=exc_type.__name__, detail=str(exc)).model_dump(),
        )

    return handler


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled exception while processing {request.method} {request.url.path}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(error_code="InternalServerError", detail="An unexpected error occurred").model_dump(),
    )


def register_exception_handlers(app: FastAPI) -> None:
    for exc_type, status_code in _STATUS_MAP.items():
        app.add_exception_handler(exc_type, _make_handler(exc_type, status_code))
    app.add_exception_handler(Exception, _unhandled_exception_handler)
