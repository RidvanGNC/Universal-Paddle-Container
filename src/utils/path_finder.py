from loguru import logger
import os


def find_path(name: str, max_deep: int = -1, start_path: str = ".", find_type: str = "file"):
    """
    find_type: "file", "folder", veya "both"
    """
    start_path = os.path.abspath(start_path)

    for root, dirs, files in os.walk(start_path):
        if max_deep > -1:
            rel_path = os.path.relpath(root, start_path)
            current_depth = 0 if rel_path == "." else rel_path.count(os.sep) + 1
            if current_depth >= max_deep:
                dirs.clear()
                continue

        search_in = []
        if find_type in ("file", "both"):
            search_in.extend(files)
        if find_type in ("folder", "both"):
            search_in.extend(dirs)

        if name in search_in:
            found_path = os.path.join(root, name)
            logger.info(f"Path found: {found_path}")
            return os.path.relpath(found_path, start_path)

    logger.warning("Path not found.")
    return None
