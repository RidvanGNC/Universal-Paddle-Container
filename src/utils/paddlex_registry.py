def is_known_model_name(model_name: str) -> bool:
    """Checks whether model_name is registered with the installed paddlex version. Catches
    typos and version mismatches (e.g. requesting a PP-OCRv6 model name against a paddlex build
    that predates PP-OCRv6) with a clear, fail-fast message instead of a deep, unclear error
    surfacing from inside paddlex/paddleocr internals (upstream has shipped confusing raw
    KeyErrors for this exact situation - see PaddlePaddle/PaddleX#3797)."""
    from paddlex.inference.utils.official_models import official_models

    return model_name in official_models.model_list
