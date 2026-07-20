import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from loguru import logger

from src.api.errors import register_exception_handlers
from src.api.routes.document import router as document_router
from src.api.routes.health import router as health_router
from src.api.routes.predict import router as predict_router
from src.api.routes.table import router as table_router
from src.api.worker.engine import PaddleOCREngine
from src.api.worker.paddlex_engine import PaddleXModelEngine, map_doc_orientation_result, map_table_cell_result
from src.api.worker.queue_manager import InferenceQueue
from src.config import get_settings
from src.hardware import resolve_device
from src.logging_setup import configure_logging


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

    ocr_engine = PaddleOCREngine(model_files_dir=model_files_dir, hardware=hardware, settings=settings)
    table_cell_wired_engine = PaddleXModelEngine(
        capability_label="table_cell_wired",
        model_dir_name=settings.table_cell_wired_model_dir_name,
        model_name=settings.table_cell_wired_model_name,
        default_model_name="RT-DETR-L_wired_table_cell_det",
        result_mapper=map_table_cell_result,
        model_files_dir=model_files_dir,
        hardware=hardware,
        settings=settings,
        default_predict_kwargs={"threshold": settings.table_cell_detection_threshold, "batch_size": 1},
    )
    table_cell_wireless_engine = PaddleXModelEngine(
        capability_label="table_cell_wireless",
        model_dir_name=settings.table_cell_wireless_model_dir_name,
        model_name=settings.table_cell_wireless_model_name,
        default_model_name="RT-DETR-L_wireless_table_cell_det",
        result_mapper=map_table_cell_result,
        model_files_dir=model_files_dir,
        hardware=hardware,
        settings=settings,
        default_predict_kwargs={"threshold": settings.table_cell_detection_threshold, "batch_size": 1},
    )
    doc_orientation_engine = PaddleXModelEngine(
        capability_label="doc_orientation",
        model_dir_name=settings.doc_orientation_model_dir_name,
        model_name=settings.doc_orientation_model_name,
        default_model_name="PP-LCNet_x1_0_doc_ori",
        result_mapper=map_doc_orientation_result,
        model_files_dir=model_files_dir,
        hardware=hardware,
        settings=settings,
        default_predict_kwargs={"batch_size": 1},
    )

    engines = {
        "ocr": ocr_engine,
        "table_cell_wired": table_cell_wired_engine,
        "table_cell_wireless": table_cell_wireless_engine,
        "doc_orientation": doc_orientation_engine,
    }

    loop = asyncio.get_running_loop()
    for engine in engines.values():
        await loop.run_in_executor(None, engine.load)

    inference_queue = InferenceQueue(settings=settings)
    await inference_queue.start()

    app.state.settings = settings
    app.state.hardware = hardware
    app.state.inference_queue = inference_queue
    app.state.ocr_engine = ocr_engine
    app.state.table_cell_wired_engine = table_cell_wired_engine
    app.state.table_cell_wireless_engine = table_cell_wireless_engine
    app.state.doc_orientation_engine = doc_orientation_engine
    app.state.engines = engines

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
    register_exception_handlers(app)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port, workers=1)
