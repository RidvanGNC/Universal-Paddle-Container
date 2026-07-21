from src.api.schemas import OcrPageResult, OcrTextResult


def test_predict_rejects_unsupported_content_type(client):
    resp = client.post("/predict", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "InvalidImageError"


def test_predict_returns_503_when_model_not_loaded(client, sample_image_bytes):
    resp = client.post("/predict", files={"file": ("test.png", sample_image_bytes, "image/png")})
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "ModelNotLoadedError"


def test_predict_happy_path_with_mocked_engine(client, sample_image_bytes):
    engine = client.app.state.engines["ocr"]
    fake_pages = [
        OcrPageResult(
            page_index=0,
            results=[OcrTextResult(text="hello", score=0.99, box=[[0, 0], [1, 0], [1, 1], [0, 1]])],
        )
    ]
    engine.run = lambda file_bytes, content_type, params: (fake_pages, 12.3)

    resp = client.post("/predict", files={"file": ("test.png", sample_image_bytes, "image/png")})
    assert resp.status_code == 200

    body = resp.json()
    assert body["pages"][0]["page_index"] == 0
    assert body["pages"][0]["results"][0]["text"] == "hello"
    assert body["device_used"] == "cpu"


def test_predict_accepts_pdf_content_type_with_mocked_engine(client):
    engine = client.app.state.engines["ocr"]
    fake_pages = [
        OcrPageResult(page_index=0, results=[OcrTextResult(text="page one", score=0.9, box=[[0, 0]])]),
        OcrPageResult(page_index=1, results=[OcrTextResult(text="page two", score=0.9, box=[[0, 0]])]),
    ]
    engine.run = lambda file_bytes, content_type, params: (fake_pages, 20.0)

    resp = client.post("/predict", files={"file": ("test.pdf", b"%PDF-fake-bytes", "application/pdf")})
    assert resp.status_code == 200

    body = resp.json()
    assert len(body["pages"]) == 2
    assert body["pages"][0]["results"][0]["text"] == "page one"
    assert body["pages"][1]["results"][0]["text"] == "page two"
