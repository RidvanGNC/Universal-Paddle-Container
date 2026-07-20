import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class InferenceJob:
    call: Callable[[], Any]
    future: asyncio.Future
    timeout: float
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    enqueued_at: float = field(default_factory=time.monotonic)
