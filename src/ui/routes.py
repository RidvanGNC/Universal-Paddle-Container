import dataclasses
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates

from src.api.capabilities import CAPABILITY_LABELS
from src.api.deps import get_engines_map
from src.ui.capability_ui_config import PLAYGROUND_ENDPOINTS

_UI_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = _UI_DIR / "templates"
STATIC_DIR = _UI_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(prefix="/ui", tags=["ui"])


@router.get("/capabilities")
async def capabilities_dashboard(request: Request, engines: dict = Depends(get_engines_map)):
    rows = [
        {
            "name": name,
            "label": CAPABILITY_LABELS.get(name, name),
            "loaded": engine.is_loaded(),
            "config": engine.get_config(),
            "problems": engine.get_load_problems(),
        }
        for name, engine in engines.items()
    ]
    return templates.TemplateResponse(request, "capabilities.html", {"rows": rows})


@router.get("/playground")
async def playground(request: Request):
    return templates.TemplateResponse(request, "playground.html", {})


@router.get("/playground-config")
async def playground_config():
    return [dataclasses.asdict(spec) for spec in PLAYGROUND_ENDPOINTS]
