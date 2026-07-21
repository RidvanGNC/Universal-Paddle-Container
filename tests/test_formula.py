def test_recognize_formula_returns_503_when_not_loaded(client, sample_image_bytes):
    resp = client.post(
        "/formula/recognize",
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "ModelNotLoadedError"


def test_recognize_formula_happy_path_with_mocked_engine(client, sample_image_bytes):
    engine = client.app.state.engines["formula_recognition"]
    engine.run = lambda image_bytes, **kwargs: (["x^2 + y^2 = z^2"], 7.0)

    resp = client.post(
        "/formula/recognize",
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body["rec_formula"] == "x^2 + y^2 = z^2"
