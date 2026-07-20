import asyncio

from loguru import logger

from src.api.errors import InferenceTimeoutError, QueueFullError, ShuttingDownError
from src.api.worker.job import InferenceJob
from src.config import Settings


class InferenceQueue:
    """Single-consumer inference pipeline: bounded asyncio.Queue + one worker loop task.

    Deliberately NOT one-worker-per-request or multi-process, and deliberately engine-agnostic -
    every job (OCR, table cell detection, doc orientation, ...) carries its own zero-arg closure
    to invoke, so exactly one Paddle inference call across ALL capabilities is ever in flight at
    a time, invoked via run_in_executor so the event loop stays responsive. This requires the
    process to run with `uvicorn --workers 1`; running multiple uvicorn worker processes would
    each load a separate set of model copies and operate disconnected queues, silently breaking
    this guarantee. Scale by running multiple containers instead.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._queue: asyncio.Queue[InferenceJob] = asyncio.Queue(maxsize=settings.inference_queue_max_size)
        self._worker_task: asyncio.Task | None = None

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        self._worker_task = asyncio.create_task(self._worker_loop(), name="inference-worker")
        logger.info("Inference worker started.")

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

        while not self._queue.empty():
            job = self._queue.get_nowait()
            if not job.future.done():
                job.future.set_exception(ShuttingDownError("Server is shutting down."))
            self._queue.task_done()
        logger.info("Inference worker stopped.")

    async def submit(self, job: InferenceJob):
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise QueueFullError(
                f"Inference queue is full (max_size={self._settings.inference_queue_max_size})."
            ) from exc

        try:
            return await asyncio.wait_for(job.future, timeout=job.timeout)
        except asyncio.TimeoutError as exc:
            job.future.cancel()
            raise InferenceTimeoutError(
                f"Inference did not complete within {job.timeout}s (request_id={job.request_id})."
            ) from exc

    async def _worker_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            job = await self._queue.get()
            try:
                if job.future.cancelled():
                    continue
                try:
                    result = await loop.run_in_executor(None, job.call)
                except Exception as exc:  # noqa: BLE001 - forward any engine failure to the caller
                    if not job.future.done():
                        job.future.set_exception(exc)
                else:
                    if not job.future.done():
                        job.future.set_result(result)
            finally:
                self._queue.task_done()
