#!/usr/bin/env python3
"""One-off helper: pre-converts a local PaddlePaddle model directory to include an ONNX export
(inference.onnx), so it can be used with USE_HPI=true once mounted read-only into model_files/.

Why this exists: PaddleX's High-Performance Inference (HPI) backend converts a Paddle model to
ONNX automatically the first time it's used - but it writes the converted file *into the model
directory itself*, and this project mounts model_files/ read-only inside the container on
purpose. So conversion has to happen here, against a writable copy, before the model directory
is placed under model_files/.

Setup (run once, in any Python 3.9-3.12 environment - does NOT need to be inside this project's
container):
    pip install paddlepaddle paddlex
    paddlex --install paddle2onnx -y

Usage:
    python tools/convert_to_onnx.py /path/to/model_dir [/path/to/another_model_dir ...]

Each argument should be a directory containing a Paddle inference model (inference.pdmodel/.json
+ inference.pdiparams) - i.e. exactly the kind of directory you'd otherwise drop directly under
model_files/. After conversion, copy the directory (now also containing inference.onnx) into
model_files/ as usual.
"""

import subprocess
import sys
from pathlib import Path


def convert(model_dir: Path) -> None:
    if not model_dir.is_dir():
        print(f"[skip] {model_dir}: not a directory")
        return

    onnx_path = model_dir / "inference.onnx"
    if onnx_path.exists():
        print(f"[skip] {model_dir}: inference.onnx already present")
        return

    print(f"[convert] {model_dir} ...")
    subprocess.run(
        [
            "paddlex",
            "--paddle2onnx",
            "--paddle_model_dir",
            str(model_dir),
            "--onnx_model_dir",
            str(model_dir),
        ],
        check=True,
    )
    print(f"[done] {model_dir} -> {onnx_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    for arg in sys.argv[1:]:
        convert(Path(arg))


if __name__ == "__main__":
    main()
