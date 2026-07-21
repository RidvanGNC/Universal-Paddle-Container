from src.api.schemas import OcrTextResult


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
    fake_result = [OcrTextResult(text="hello", score=0.99, box=[[0, 0], [1, 0], [1, 1], [0, 1]])]
    engine.run = lambda image_bytes, params: (fake_result, 12.3)

    resp = client.post("/predict", files={"file": ("test.png", sample_image_bytes, "image/png")})
    assert resp.status_code == 200

    body = resp.json()
    assert body["results"][0]["text"] == "hello"
    assert body["device_used"] == "cpu"
