from unittest.mock import Mock, patch

from src.api.worker.paddlex_engine import PaddleXModelEngine
from src.config import Settings
from src.hardware import HardwareInfo


def _make_engine(tmp_path, model_dir_name=None, model_name=None):
    settings = Settings(strict_model_loading=False)
    hardware = HardwareInfo(
        compiled_with_cuda=False,
        cuda_device_count=0,
        device_name=None,
        using_device="cpu",
        paddle_device_string="cpu",
    )
    return PaddleXModelEngine(
        capability_label="test_capability",
        model_dir_name=model_dir_name,
        model_name=model_name,
        default_model_name="RT-DETR-L_wired_table_cell_det",
        result_mapper=lambda page: [],
        model_files_dir=tmp_path,
        hardware=hardware,
        settings=settings,
    )


def test_initial_state_not_loaded(tmp_path):
    engine = _make_engine(tmp_path)
    assert not engine.is_loaded()
    assert engine.get_config() == {"model_dir_name": None, "model_name": "RT-DETR-L_wired_table_cell_det"}


def test_reload_updates_config_and_loads(tmp_path):
    (tmp_path / "my_model_dir").mkdir()
    engine = _make_engine(tmp_path)

    with patch("paddlex.create_model", return_value=Mock()) as mock_create:
        result = engine.reload(model_dir_name="my_model_dir", model_name="RT-DETR-L_wired_table_cell_det")

    assert result.loaded is True
    assert result.problems == []
    assert engine.is_loaded()
    assert engine.get_config() == {
        "model_dir_name": "my_model_dir",
        "model_name": "RT-DETR-L_wired_table_cell_det",
    }
    mock_create.assert_called_once()


def test_reload_with_missing_directory_reports_problem_and_stays_unloaded(tmp_path):
    engine = _make_engine(tmp_path)

    result = engine.reload(model_dir_name="does_not_exist")

    assert result.loaded is False
    assert any("not found" in p for p in result.problems)
    assert not engine.is_loaded()


def test_reload_with_unknown_model_name_reports_problem(tmp_path):
    (tmp_path / "my_model_dir").mkdir()
    engine = _make_engine(tmp_path)

    result = engine.reload(model_dir_name="my_model_dir", model_name="Totally_Made_Up_Model_Name")

    assert result.loaded is False
    assert any("not recognized" in p for p in result.problems)
