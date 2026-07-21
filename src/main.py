import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src.api.capabilities import build_engines
from src.api.errors import register_exception_handlers
from src.api.routes.admin import router as admin_router
from src.api.routes.document import router as document_router
from src.api.routes.formula import router as formula_router
from src.api.routes.health import router as health_router
from src.api.routes.layout import router as layout_router
from src.api.routes.predict import router as predict_router
from src.api.routes.seal import router as seal_router
from src.api.routes.table import router as table_router
from src.api.routes.textline import router as textline_router
from src.api.worker.queue_manager import InferenceQueue
from src.config import get_settings
from src.hardware import resolve_device
from src.logging_setup import configure_logging
from src.ui.routes import STATIC_DIR
from src.ui.routes import router as ui_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings)

    logger.info(f"Starting {settings.app_name} ...")

    hardware = resolve_device(settings)

    # Anchored to this file's directory (src/), not the process cwd - the cwd is the Docker
    # WORKDIR (/app), while the model_files mount point is /app/src/model_files. An absolute
    # MODEL_FILES_DIR value in settings still overrides this anchor as expected.
    model_files_dir = Path(__file__).resolve().parent / settings.model_files_dir

    engines = build_engines(settings, hardware, model_files_dir)

    loop = asyncio.get_running_loop()
    for engine in engines.values():
        await loop.run_in_executor(None, engine.load)

    inference_queue = InferenceQueue(settings=settings)
    await inference_queue.start()

    app.state.settings = settings
    app.state.hardware = hardware
    app.state.inference_queue = inference_queue
    app.state.engines = engines
    app.state.model_files_dir = model_files_dir

    logger.info("Startup complete.")
    yield

    logger.info("Shutting down ...")
    await inference_queue.stop()
    for engine in engines.values():
        engine.unload()
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    app = FastAPI(title="paddle-ocr-api", lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(table_router)
    app.include_router(document_router)
    app.include_router(layout_router)
    app.include_router(formula_router)
    app.include_router(seal_router)
    app.include_router(textline_router)
    app.include_router(admin_router)
    app.include_router(ui_router)
    app.mount("/ui/static", StaticFiles(directory=str(STATIC_DIR)), name="ui-static")
    register_exception_handlers(app)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port, workers=1)
