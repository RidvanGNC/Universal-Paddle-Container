from fastapi import Depends, HTTPException, Request, status
from pydantic import BaseModel

from src.config import Settings, get_settings


class Principal(BaseModel):
    id: str
    authenticated: bool


async def get_current_principal(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> Principal:
    if settings.security_mode == "none":
        return Principal(id="anonymous", authenticated=False)

    if settings.security_mode == "api_key":
        # TODO: implement when auth is actually needed - compare the X-API-Key header
        # against settings.api_key, raise 401 on missing/mismatch. Left unimplemented so a
        # misconfigured deployment fails loudly instead of silently granting access.
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="security_mode='api_key' is configured but not yet implemented",
        )

    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unknown security_mode")
