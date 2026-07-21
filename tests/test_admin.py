from unittest.mock import Mock

from src.api.worker.load_result import LoadResult


def test_list_capabilities_shape(client):
    resp = client.get("/admin/capabilities")
    assert resp.status_code == 200

    body = resp.json()
    names = {c["name"] for c in body["capabilities"]}
    assert "ocr" in names
    assert "table_cell_wired" in names
    assert all(c["loaded"] is False for c in body["capabilities"])


def test_available_models_against_empty_dir(client):
    resp = client.get("/admin/capabilities/table_cell_wired/available-models")
    assert resp.status_code == 200
    assert resp.json()["directories"] == []


def test_available_models_unknown_capability_404(client):
    resp = client.get("/admin/capabilities/does-not-exist/available-models")
    assert resp.status_code == 404


def test_reload_unknown_capability_404(client):
    resp = client.post("/admin/capabilities/does-not-exist/reload", json={"config": {}})
    assert resp.status_code == 404


def test_reload_calls_engine_reload_with_posted_config(client):
    engine = client.app.state.engines["table_cell_wired"]
    engine.reload = Mock(return_value=LoadResult(loaded=True, problems=[]))

    resp = client.post(
        "/admin/capabilities/table_cell_wired/reload",
        json={"config": {"model_dir_name": "some_dir", "model_name": "some_name"}},
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body == {"name": "table_cell_wired", "loaded": True, "problems": []}
    engine.reload.assert_called_once_with(model_dir_name="some_dir", model_name="some_name")


def test_reload_reports_problems_on_failure(client):
    engine = client.app.state.engines["table_cell_wired"]
    engine.reload = Mock(return_value=LoadResult(loaded=False, problems=["directory 'x' not found"]))

    resp = client.post(
        "/admin/capabilities/table_cell_wired/reload",
        json={"config": {"model_dir_name": "x"}},
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body["loaded"] is False
    assert body["problems"] == ["directory 'x' not found"]
