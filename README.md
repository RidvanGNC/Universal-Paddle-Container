# paddle-container

Self-hosted, containerized PaddleX/PaddleOCR inference API. General-purpose PaddleX capability
host (not receipt/document-specific) — currently exposes OCR (text detection + recognition),
table cell detection, table structure recognition, layout detection, formula recognition, seal
text detection, document unwarping, and document/text-line orientation classification, each as
an independent, independently-optional endpoint. Supports two build variants — CPU and GPU (CUDA
12.6) — with runtime hardware auto-detection, a bounded single-worker inference queue shared
across every capability, and model weights supplied separately at runtime (not baked into the
image). Includes a built-in admin/test UI (`/ui/capabilities`, `/ui/playground`) for viewing and
live-reloading each capability's model directory/name, and for uploading a test image and seeing
results without leaving the browser.

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
6. Open the admin/test UI: `http://localhost:8000/ui/capabilities` (view/reload each capability's
   model) and `http://localhost:8000/ui/playground` (upload a test image and see results).

## Endpoints

| Endpoint | Required config | Notes |
|---|---|---|
| `POST /predict` | `DET_MODEL_DIR_NAME`/`_NAME`, `REC_MODEL_DIR_NAME`/`_NAME` (required); `DOC_ORIENTATION_MODEL_DIR_NAME`/`_NAME`, `DOC_UNWARPING_MODEL_DIR_NAME`/`_NAME`, `TEXTLINE_ORIENTATION_MODEL_DIR_NAME`/`_NAME` (all optional, applied as pipeline preprocessing) | OCR: text detection + recognition. |
| `POST /table/detect-cells` | `TABLE_CELL_WIRED_MODEL_DIR_NAME`/`_NAME` or `TABLE_CELL_WIRELESS_MODEL_DIR_NAME`/`_NAME` | Multipart `file` + form field `table_type=wired\|wireless` (+ optional `threshold` override). Runs `RT-DETR-L_wired_table_cell_det` or `RT-DETR-L_wireless_table_cell_det`. |
| `POST /table/structure` | `TABLE_STRUCTURE_MODEL_DIR_NAME`/`_NAME` | Table structure recognition (`SLANet`/`SLANet_plus`/`SLANeXt_wired`/`SLANeXt_wireless`) — returns cell `bbox`, HTML `structure` tokens, `structure_score`. |
| `POST /layout/detect` | `LAYOUT_DETECTION_MODEL_DIR_NAME`/`_NAME` | Document region detection (title, text, table, image, formula, seal, etc. — `PP-DocLayout*`/`PicoDet*_layout_*`/`RT-DETR-H_layout_*`). Same box shape as table-cell detection. |
| `POST /formula/recognize` | `FORMULA_MODEL_DIR_NAME`/`_NAME` | Formula image → LaTeX (`UniMERNet`/`PP-FormulaNet*`/`LaTeX_OCR_rec`). |
| `POST /seal/detect` | `SEAL_DET_MODEL_DIR_NAME`/`_NAME` | Seal/stamp text region detection (`PP-OCRv4_{server,mobile}_seal_det`) — detection only; reading the seal's text is a future crop+OCR-rec combo, not implemented here. |
| `POST /document/orientation` | `DOC_ORIENTATION_MODEL_DIR_NAME`/`_NAME` | Standalone document rotation classifier (`PP-LCNet_x1_0_doc_ori`) — same config as the optional OCR-pipeline preprocessing above, just callable directly. Returns one of `angle: "0"\|"90"\|"180"\|"270"`. |
| `POST /document/unwarp` | `DOC_UNWARPING_MODEL_DIR_NAME`/`_NAME` | Document image rectification (`UVDoc`) — the one endpoint that returns a **PNG image** (`image/png`), not JSON; processing time/device are returned as `X-Processing-Time-Ms`/`X-Device-Used` headers instead. |
| `POST /textline/orientation` | `TEXTLINE_ORIENTATION_MODEL_DIR_NAME`/`_NAME` | Per-text-line rotation classifier (`PP-LCNet_{x0_25,x1_0}_textline_ori`). `angle` is a plain string, not a strict `"0"/"90"/"180"/"270"` enum like `/document/orientation` — this model's real label vocabulary (e.g. possibly `"180_degree"`) isn't confirmed against real weights yet. |
| `GET /health/live` | — | Always 200 if the process is up. Used by the Docker `HEALTHCHECK`. |
| `GET /health/ready` (alias `/health`) | — | Always 200 once the worker loop is running; reports a `capabilities: dict[str,bool]` map (one entry per capability) so callers know which specific endpoints are currently usable. |
| `GET/POST /admin/capabilities*`, `GET /ui/*` | — | See [Admin API & UI](#admin-api--ui) below. |

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
| `TABLE_STRUCTURE_MODEL_DIR_NAME` / `TABLE_STRUCTURE_MODEL_NAME` | unset | Optional table structure recognition folder name + model name (`SLANet`/`SLANet_plus`/`SLANeXt_wired`/`SLANeXt_wireless`) |
| `LAYOUT_DETECTION_MODEL_DIR_NAME` / `LAYOUT_DETECTION_MODEL_NAME` | unset | Optional layout detection folder name + model name (`PP-DocLayout*` family) |
| `FORMULA_MODEL_DIR_NAME` / `FORMULA_MODEL_NAME` | unset | Optional formula recognition folder name + model name (`PP-FormulaNet*`/`UniMERNet`/`LaTeX_OCR_rec`) |
| `SEAL_DET_MODEL_DIR_NAME` / `SEAL_DET_MODEL_NAME` | unset | Optional seal text detection folder name + model name (`PP-OCRv4_{server,mobile}_seal_det`) |
| `DOC_UNWARPING_MODEL_DIR_NAME` / `DOC_UNWARPING_MODEL_NAME` | unset | Optional doc-unwarping folder name + model name (`UVDoc`); used both as OCR-pipeline preprocessing and by the standalone `/document/unwarp` endpoint |
| `TEXTLINE_ORIENTATION_MODEL_DIR_NAME` / `TEXTLINE_ORIENTATION_MODEL_NAME` | unset | Optional text-line orientation folder name + model name (`PP-LCNet_{x0_25,x1_0}_textline_ori`); used both as OCR-pipeline preprocessing and by the standalone `/textline/orientation` endpoint |
| `INFERENCE_QUEUE_MAX_SIZE` | `50` | Max queued requests before returning 503 |
| `INFERENCE_TIMEOUT_SECONDS` | `30` | Max time a request waits for its result before 504 |
| `MODEL_RELOAD_TIMEOUT_SECONDS` | `120` | Max time an admin-triggered model reload may take before 504 — separate from `INFERENCE_TIMEOUT_SECONDS` since loading weights from disk is a different latency class than a single predict call |
| `MAX_UPLOAD_SIZE_MB` | `10` | Max accepted upload size |
| `SECURITY_MODE` | `none` | `none` \| `api_key` (see `src/security/auth.py`) |
| `API_KEY` | unset | Required header value when `SECURITY_MODE=api_key` |

## Model scope

Within OCR (`/predict`), only text detection and text recognition are mandatory. Doc-orientation
classification, doc-unwarping, and text-line orientation classification are each optional pipeline
preprocessing steps, enabled automatically (and independently of each other) whenever their
respective `*_MODEL_DIR_NAME` is configured and resolvable — none of them auto-download weights
from the internet; an unconfigured or misconfigured optional slot just disables that one feature
rather than the whole pipeline (see the behavior note below). Every other capability (table cell
detection, table structure recognition, layout detection, formula recognition, seal detection,
standalone doc-unwarping/orientation) is entirely optional — the container is fully functional as
an OCR-only service if none of their weights are ever supplied.

**Per-slot degradation, not all-or-nothing:** a problem in `doc_orientation`/`doc_unwarping`/
`textline_orientation`'s configuration only disables that one optional OCR feature; only a problem
in the *required* detection or recognition slot takes down the whole `/predict` endpoint. This
matters most when using the admin UI to test a deliberately-wrong optional-slot config — it won't
break basic OCR while you're doing so.

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

## Admin API & UI

- `GET /admin/capabilities` — every capability's label, loaded state, current config, and any
  load problems.
- `GET /admin/capabilities/{name}/available-models` — subfolders currently present under
  `MODEL_FILES_DIR` (plain directory listing — populates the UI's directory dropdown).
- `POST /admin/capabilities/{name}/reload` — body `{"config": {...}}` (e.g.
  `{"model_dir_name": "...", "model_name": "..."}`, or for `"ocr"` any of its 10 field names, e.g.
  `det_model_dir_name`) — swaps that capability's model **live**, without a container restart.

**Reload is a real hot-swap, not a request-time parameter tweak**: it's submitted as a normal job
through the same single-worker inference queue every prediction goes through, so a reload can
never race with an in-flight inference call — reload and inference are mutually exclusive by
construction, using the same mechanism, not a separate lock. A reload that fails (bad directory,
unrecognized model name) reports `problems` in the response and leaves that capability unloaded;
it never crashes the process. **Reload changes are in-memory only** — they revert to `.env`
defaults on container restart. This is intentional: a live override for comparison/testing, not a
config-file-writing feature.

The `/ui/capabilities` and `/ui/playground` pages (Jinja2 + vanilla JS, no separate frontend build)
are a thin client of this admin API plus the regular public endpoints above — the playground page
in particular calls `/predict`, `/table/detect-cells`, etc. directly, the same way any other API
caller would, so there's no separate "test" code path to keep in sync with the real one.

Both `/admin/*` and `/ui/*` go through the same `get_current_principal` dependency as every
inference route, so if `SECURITY_MODE=api_key` is ever implemented (currently a documented no-op
stub in `src/security/auth.py`), the admin surface is covered automatically.

## Concurrency model

A single background worker consumes a bounded `asyncio.Queue` and runs one Paddle inference call
at a time via `run_in_executor` — **shared across every capability** (OCR, table cell detection,
doc orientation), not one queue per model — keeping the event loop responsive to other requests
(uploads, health checks) while inference runs. This preserves a hard guarantee: exactly one Paddle
inference call, across the whole service, is ever in flight at once, avoiding GPU/CPU resource
contention between capabilities. Admin-triggered model reloads (see above) go through this same
queue as ordinary reload "jobs" — so a reload and a prediction can never run concurrently either.
**This requires `uvicorn --workers 1`** — the Dockerfiles pin this. Running multiple uvicorn worker processes would each load a separate copy of every model
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
