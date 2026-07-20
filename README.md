# paddle-container

Self-hosted, containerized PaddleX/PaddleOCR inference API. General-purpose PaddleX capability
host (not receipt/document-specific) — currently exposes OCR (text detection + recognition),
table cell detection, and document orientation classification as independent endpoints, each
independently optional. Supports two build variants — CPU and GPU (CUDA 12.6) — with runtime
hardware auto-detection, a bounded single-worker inference queue shared across every capability,
and model weights supplied separately at runtime (not baked into the image).

## Quick start

1. Copy `src/.env.example` to `src/.env` and adjust values (at minimum `PORT`).
2. Drop your PaddleX/PaddleOCR inference model directories into `src/model_files/` and set the
   matching `*_MODEL_DIR_NAME`/`*_MODEL_NAME` pair in `.env` for each capability you want enabled
   (see the table below). Every capability is independently optional — the API runs fine with
   `model_files/` empty or partially populated; unconfigured capabilities simply return 503 at
   their endpoint until their weights are supplied (see [Endpoints](#endpoints)).
3. Build and run:

   ```bash
   # CPU
   docker compose --profile cpu up --build

   # GPU (requires NVIDIA driver 560+ and the NVIDIA Container Toolkit on the host)
   docker compose --profile gpu up --build
   ```

4. Check health: `curl http://localhost:8000/health/ready`
5. Run a prediction: `curl -F file=@sample.png http://localhost:8000/predict`

## Endpoints

| Endpoint | Required config | Notes |
|---|---|---|
| `POST /predict` | `DET_MODEL_DIR_NAME`/`_NAME`, `REC_MODEL_DIR_NAME`/`_NAME` (required); `DOC_ORIENTATION_MODEL_DIR_NAME`/`_NAME` (optional, applied as pipeline preprocessing) | OCR: text detection + recognition. |
| `POST /table/detect-cells` | `TABLE_CELL_WIRED_MODEL_DIR_NAME`/`_NAME` or `TABLE_CELL_WIRELESS_MODEL_DIR_NAME`/`_NAME` | Multipart `file` + form field `table_type=wired\|wireless` (+ optional `threshold` override). Runs `RT-DETR-L_wired_table_cell_det` or `RT-DETR-L_wireless_table_cell_det`. |
| `POST /document/orientation` | `DOC_ORIENTATION_MODEL_DIR_NAME`/`_NAME` | Standalone document rotation classifier (`PP-LCNet_x1_0_doc_ori`) — same config as the optional OCR-pipeline preprocessing above, just callable directly. Returns one of `angle: "0"\|"90"\|"180"\|"270"`. |
| `GET /health/live` | — | Always 200 if the process is up. Used by the Docker `HEALTHCHECK`. |
| `GET /health/ready` (alias `/health`) | — | Always 200 once the worker loop is running; reports a `capabilities` map (`ocr`, `table_cell_wired`, `table_cell_wireless`, `doc_orientation`) so callers know which specific endpoints are currently usable. |

These endpoints are deliberately separate rather than folded into one combined call, so a caller
can compose/post-process each capability's output independently; a caller who doesn't care can
still just call multiple endpoints from client code.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `LOG_LEVEL` | `INFO` | Loguru log level |
| `MODEL_FILES_DIR` | `model_files` | Root dir searched for model subfolders (relative to `src/`) |
| `DEVICE_PREFERENCE` | `auto` | `auto` \| `cpu` \| `gpu` |
| `FAIL_FAST_ON_GPU_MISMATCH` | `true` | If `gpu` is requested but unavailable, crash instead of silently falling back to CPU |
| `STRICT_MODEL_LOADING` | `false` | If `true`, missing model dirs crash startup instead of degrading `/predict` to 503 |
| `DET_MODEL_DIR_NAME` / `DET_MODEL_NAME` | unset | Text detection model folder name + PaddleOCR model name |
| `REC_MODEL_DIR_NAME` / `REC_MODEL_NAME` | unset | Text recognition model folder name + PaddleOCR model name |
| `DOC_ORIENTATION_MODEL_DIR_NAME` / `DOC_ORIENTATION_MODEL_NAME` | unset | Optional doc orientation classifier folder name + model name; used both as OCR-pipeline preprocessing and by the standalone `/document/orientation` endpoint |
| `TABLE_CELL_WIRED_MODEL_DIR_NAME` / `TABLE_CELL_WIRED_MODEL_NAME` | unset | Optional wired-table cell detection folder name + model name (`RT-DETR-L_wired_table_cell_det`) |
| `TABLE_CELL_WIRELESS_MODEL_DIR_NAME` / `TABLE_CELL_WIRELESS_MODEL_NAME` | unset | Optional wireless-table cell detection folder name + model name (`RT-DETR-L_wireless_table_cell_det`) |
| `TABLE_CELL_DETECTION_THRESHOLD` | `0.3` | Default confidence threshold for `/table/detect-cells`, overridable per-request via the `threshold` form field |
| `INFERENCE_QUEUE_MAX_SIZE` | `50` | Max queued requests before returning 503 |
| `INFERENCE_TIMEOUT_SECONDS` | `30` | Max time a request waits for its result before 504 |
| `MAX_UPLOAD_SIZE_MB` | `10` | Max accepted upload size |
| `SECURITY_MODE` | `none` | `none` \| `api_key` (see `src/security/auth.py`) |
| `API_KEY` | unset | Required header value when `SECURITY_MODE=api_key` |

## Model scope

Within OCR (`/predict`), only text detection and text recognition are mandatory; doc-orientation
classification is optional (enabled automatically when `DOC_ORIENTATION_MODEL_DIR_NAME` is set).
Doc-unwarping (`UVDoc`) and text-line orientation classification are **always disabled** in
`engine.py` — enabling either without a locally supplied model would make PaddleOCR silently
download weights from the internet at startup, which breaks the network-isolated-by-default
deployment model. Table cell detection and standalone doc orientation are entirely optional
capabilities — the container is fully functional as an OCR-only service if their weights are
never supplied.

## Model versions and language variants (PP-OCRv5 vs PP-OCRv6)

`DET_MODEL_NAME`/`REC_MODEL_NAME` accept any model name registered by the installed `paddlex`
(pinned via `paddleocr==3.7.0` in `requirements/base.txt`, which resolves `paddlex>=3.7.0,<3.8.0`
— PP-OCRv6 support requires at least `paddleocr==3.7.0`; it wasn't registered in the previously
pinned `3.6.0`). No separate "OCR version" setting is needed: `paddleocr`'s own `ocr_version`/`lang`
constructor arguments only take effect when model dir/name are left unset (auto-selection) — since
this project always passes explicit `text_detection_model_dir`/`text_recognition_model_dir`, those
two arguments are silently ignored by upstream, and each model's own bundled config (shipped
alongside its weights) fully determines its pre/post-processing. Practically, this means:

- **PP-OCRv6** (`PP-OCRv6_{tiny,small,medium}_det` / `_rec`) — a single unified model per tier
  covering Chinese, English, Japanese, and 46 Latin-script languages. No per-language rec model
  needed; pick a tier for speed/accuracy trade-off (medium ≈ PP-OCRv5_server+, tiny/small target
  edge/mobile).
- **PP-OCRv5** (`PP-OCRv5_{server,mobile}_det` / per-language `*_PP-OCRv5_mobile_rec`) — recognition
  requires picking the rec model matching your target script: `en_`, `korean_`, `latin_`,
  `eslav_`, `cyrillic_`, `arabic_`, `devanagari_`, `th_`, `el_`, `ta_`, `te_` (Chinese/Japanese use
  `PP-OCRv5_server_rec` directly, no prefix).

Keep det and rec from the **same** version (both v5 or both v6) — mixing is not blocked at the code
level (nothing enforces it, since each sub-model is independent once explicit dirs are given) but
isn't a combination upstream tests or documents, so treat it as unsupported.

**Fail-fast on bad model names:** both `PaddleOCREngine` and `PaddleXModelEngine` validate every
configured `*_MODEL_NAME` against the installed paddlex's model registry at load time
(`src/utils/paddlex_registry.py`) before attempting to construct anything. A typo'd or
version-mismatched name (e.g. a v6 name against a paddlex build that predates v6) produces a clear
log message and a 503 at that capability's endpoint — not the confusing raw `KeyError` upstream
has been known to throw for this exact situation
([PaddlePaddle/PaddleX#3797](https://github.com/PaddlePaddle/PaddleX/issues/3797)).

## Concurrency model

A single background worker consumes a bounded `asyncio.Queue` and runs one Paddle inference call
at a time via `run_in_executor` — **shared across every capability** (OCR, table cell detection,
doc orientation), not one queue per model — keeping the event loop responsive to other requests
(uploads, health checks) while inference runs. This preserves a hard guarantee: exactly one Paddle
inference call, across the whole service, is ever in flight at once, avoiding GPU/CPU resource
contention between capabilities. **This requires `uvicorn --workers 1`** — the Dockerfiles pin
this. Running multiple uvicorn worker processes would each load a separate copy of every model
(multiplying GPU memory) and operate disconnected queues, breaking the single-worker guarantee. To
scale throughput, run multiple **containers** behind a load balancer rather than multiple workers
inside one container.

## GPU base image note

`Dockerfile.gpu` uses `nvidia/cuda:12.6.3-base-ubuntu22.04` rather than a `-cudnn-runtime-` tag,
because `paddlepaddle-gpu` wheels bundle their own CUDA/cuDNN runtime libraries via pip
dependencies — the base image only needs to supply the driver stub and NVIDIA Container Toolkit
labels. If you hit CUDA library resolution errors at runtime, switch the base image tag to
`nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04` in `Dockerfile.gpu`.

## Troubleshooting

- **Container starts but always reports `using_device: cpu` on a GPU host:** confirm
  `docker run --gpus all` (or the compose `gpu` profile) was used, and that the host has NVIDIA
  driver 560+ and `nvidia-container-toolkit` installed. Check startup logs from `hardware.py` —
  it explicitly logs `is_compiled_with_cuda` and `cuda_device_count` for diagnosis (this mirrors
  known upstream issues where PaddleX/Paddle containers silently fail to see CUDA — see
  [PaddlePaddle/Paddle#67982](https://github.com/PaddlePaddle/Paddle/issues/67982) and
  [#69218](https://github.com/PaddlePaddle/Paddle/issues/69218)).
- **`/health/ready` shows a capability as `false` in `capabilities`:** that capability's configured
  model subfolder name wasn't found under `MODEL_FILES_DIR` (or was never configured). Check the
  `find_path` log lines emitted at startup for that capability's label. Note `/health/ready` always
  returns 200 regardless — an unconfigured capability is a normal, expected state, not a failure;
  only its specific endpoint returns 503 when actually called.
- **`/predict` returns 503 "queue full":** more requests are in flight than
  `INFERENCE_QUEUE_MAX_SIZE`; the client should back off/retry.
- **`/predict` returns 504:** inference took longer than `INFERENCE_TIMEOUT_SECONDS`.
- **Any endpoint returns 500 with `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not
  support [...]` in the logs (CPU image):** an unresolved upstream PaddlePaddle 3.3.1 / PaddleX
  bug where some models crash under the default oneDNN+PIR CPU execution path (the affected models
  aren't in PaddleX's own `MKLDNN_BLOCKLIST`). Confirmed so far against `PP-OCRv5_server_det`
  (found during this project's own testing) and `PP-LCNet_x1_0_doc_ori` under `paddleocr==3.7.0`
  ([PaddlePaddle/PaddleOCR#18162](https://github.com/PaddlePaddle/PaddleOCR/issues/18162), open,
  no upstream fix yet as of this writing) — i.e. this is a model-specific gap, not something to
  assume is fixed for any particular model just because it isn't in the blocklist. `Dockerfile.cpu`
  already sets `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=False` globally to work around this for every
  model this container loads (present and future); if you've overridden it back to `True`, unset
  the override. If a *new* model hits this on CPU, this env var is the fix — no code change needed.
- **`/predict`, `/table/detect-cells`, or `/document/orientation` return 503 with `... model name
  '...' is not recognized by the installed paddlex version ...`:** either a typo in the
  corresponding `*_MODEL_NAME`, or that model name requires a newer `paddleocr`/`paddlex` than
  what's installed (see the PP-OCRv6/PP-OCRv5 section above). Check `pip show paddlex` inside the
  container against the model's introduction version.
