from src.api.schemas import SealDetectionBox


def test_detect_seal_text_returns_503_when_not_loaded(client, sample_image_bytes):
    resp = client.post(
        "/seal/detect",
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "ModelNotLoadedError"


def test_detect_seal_text_happy_path_with_mocked_engine(client, sample_image_bytes):
    engine = client.app.state.engines["seal_detection"]
    fake_boxes = [SealDetectionBox(score=0.91, box=[[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]])]
    engine.run = lambda image_bytes, **kwargs: (fake_boxes, 5.5)

    resp = client.post(
        "/seal/detect",
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body["boxes"][0]["score"] == 0.91
