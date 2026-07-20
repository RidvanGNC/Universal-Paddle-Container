from src.api.schemas import TableCellBox


def test_detect_cells_rejects_bad_table_type(client, sample_image_bytes):
    resp = client.post(
        "/table/detect-cells",
        data={"table_type": "triangular"},
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 422


def test_detect_cells_returns_503_when_wired_model_not_loaded(client, sample_image_bytes):
    resp = client.post(
        "/table/detect-cells",
        data={"table_type": "wired"},
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "ModelNotLoadedError"


def test_detect_cells_returns_503_when_wireless_model_not_loaded(client, sample_image_bytes):
    resp = client.post(
        "/table/detect-cells",
        data={"table_type": "wireless"},
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "ModelNotLoadedError"


def test_detect_cells_happy_path_with_mocked_wired_engine(client, sample_image_bytes):
    engine = client.app.state.table_cell_wired_engine
    fake_boxes = [TableCellBox(label="cell", score=0.95, coordinate=[1.0, 2.0, 3.0, 4.0])]
    engine.run = lambda image_bytes, **kwargs: (fake_boxes, 5.0)

    resp = client.post(
        "/table/detect-cells",
        data={"table_type": "wired"},
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body["table_type"] == "wired"
    assert body["boxes"][0]["label"] == "cell"
    assert body["boxes"][0]["coordinate"] == [1.0, 2.0, 3.0, 4.0]
