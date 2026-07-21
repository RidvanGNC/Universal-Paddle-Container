from src.api.schemas import DetectionBox


def test_detect_layout_returns_503_when_not_loaded(client, sample_image_bytes):
    resp = client.post(
        "/layout/detect",
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "ModelNotLoadedError"


def test_detect_layout_happy_path_with_mocked_engine(client, sample_image_bytes):
    engine = client.app.state.engines["layout_detection"]
    fake_boxes = [DetectionBox(label="table", score=0.88, coordinate=[1.0, 2.0, 3.0, 4.0])]
    engine.run = lambda image_bytes, **kwargs: (fake_boxes, 4.0)

    resp = client.post(
        "/layout/detect",
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body["boxes"][0]["label"] == "table"
