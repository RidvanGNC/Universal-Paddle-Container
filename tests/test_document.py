def test_orientation_returns_503_when_model_not_loaded(client, sample_image_bytes):
    resp = client.post(
        "/document/orientation",
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "ModelNotLoadedError"


def test_orientation_happy_path_with_mocked_engine(client, sample_image_bytes):
    engine = client.app.state.doc_orientation_engine
    fake_result = [{"angle": "90", "score": 0.97}]
    engine.run = lambda image_bytes, **kwargs: (fake_result, 3.1)

    resp = client.post(
        "/document/orientation",
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body["angle"] == "90"
    assert body["score"] == 0.97
