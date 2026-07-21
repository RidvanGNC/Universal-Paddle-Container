"""Pure functions turning one page of PaddleX's dict-like predict() output into normalized,
typed items. Kept separate from the engine classes that call them so each mapper stays an
independently readable, testable unit - the "parts working under their own pieces" cut point
between engine mechanics (paddlex_engine.py) and capability-specific result shapes (here).

All PaddleX result pages are accessed defensively via `.get(...) if hasattr(page, "get") else`,
consistent with the pattern already established for OCR result parsing in engine.py.
"""

from src.api.schemas import DetectionBox, SealDetectionBox


def map_detection_box_result(page) -> list[DetectionBox]:
    """Table-cell detection AND layout detection share this exact output shape."""
    boxes = page.get("boxes", []) if hasattr(page, "get") else []
    return [
        DetectionBox(
            label=box.get("label", "unknown"),
            score=float(box.get("score", 0.0)),
            coordinate=[float(v) for v in box.get("coordinate", [])],
        )
        for box in boxes
    ]


def map_table_structure_result(page) -> list[dict]:
    if not hasattr(page, "get"):
        return []
    bbox = page.get("bbox", [])
    structure = page.get("structure", [])
    structure_score = page.get("structure_score", 0.0)
    return [
        {
            "bbox": [[float(v) for v in box] for box in bbox],
            "structure": list(structure),
            "structure_score": float(structure_score),
        }
    ]


def map_formula_result(page) -> list[str]:
    rec_formula = page.get("rec_formula") if hasattr(page, "get") else None
    return [rec_formula] if rec_formula is not None else []


def map_seal_detection_result(page) -> list[SealDetectionBox]:
    polys = page.get("dt_polys", []) if hasattr(page, "get") else []
    scores = page.get("dt_scores", []) if hasattr(page, "get") else []
    return [
        SealDetectionBox(score=float(score), box=[[float(x), float(y)] for x, y in poly])
        for poly, score in zip(polys, scores)
    ]


def map_unwarp_result(page):
    """Returns the raw rectified image array (BGR, matching this project's own decode
    convention) - not JSON-shaped data, unlike every other mapper here."""
    doctr_img = page.get("doctr_img") if hasattr(page, "get") else None
    return [doctr_img] if doctr_img is not None else []


def map_orientation_result(page) -> list[dict]:
    """Shared by doc-orientation and textline-orientation - both PaddleX modules yield the same
    class_ids/scores/label_names shape. The response schema layer (not this mapper) is
    responsible for how strictly the label string is typed."""
    label_names = page.get("label_names", []) if hasattr(page, "get") else []
    scores = page.get("scores", []) if hasattr(page, "get") else []
    if not label_names:
        return []
    return [{"angle": label_names[0], "score": float(scores[0]) if scores else 0.0}]
