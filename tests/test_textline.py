def test_textline_orientation_returns_503_when_not_loaded(client, sample_image_bytes):
    resp = client.post(
        "/textline/orientation",
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "ModelNotLoadedError"


def test_textline_orientation_happy_path_with_mocked_engine(client, sample_image_bytes):
    engine = client.app.state.engines["textline_orientation"]
    fake_result = [{"angle": "180_degree", "score": 0.85}]
    engine.run = lambda image_bytes, **kwargs: (fake_result, 2.5)

    resp = client.post(
        "/textline/orientation",
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body["angle"] == "180_degree"
