def test_live_is_always_ok(client):
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}


def test_ready_reports_no_capabilities_configured_without_models(client):
    resp = client.get("/health/ready")
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "no_capabilities_configured"
    assert body["capabilities"] == {
        "ocr": False,
        "table_cell_wired": False,
        "table_cell_wireless": False,
        "doc_orientation": False,
        "table_structure": False,
        "layout_detection": False,
        "formula_recognition": False,
        "seal_detection": False,
        "doc_unwarping": False,
        "textline_orientation": False,
    }
    assert body["hardware"]["using_device"] == "cpu"
