from __future__ import annotations

import sys

import pytest

from ttheart_sender.config import Config, ConfigError, deep_merge, from_dict, load_config


def test_defaults_are_sane():
    config = Config()
    assert config.matching.confidence == 0.85
    assert config.window.position.x == 0
    assert config.runner.stop_key == "f12"
    assert config.window.insets.is_empty


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    overlay = {"a": {"y": 3}, "c": 4}
    merged = deep_merge(base, overlay)
    assert merged == {"a": {"x": 1, "y": 3}, "b": 1, "c": 4}
    assert base == {"a": {"x": 1, "y": 2}, "b": 1}


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError) as exc:
        from_dict(Config, {"matching": {"confidance": 0.9}}, exclude={"root"})
    assert "confidance" in str(exc.value)


def test_wrong_type_is_rejected():
    with pytest.raises(ConfigError):
        from_dict(Config, {"matching": {"confidence": "high"}}, exclude={"root"})


def test_load_config_reads_file_and_local_overlay(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "matching:\n  confidence: 0.7\nwindow:\n  instance: 1\n", encoding="utf-8"
    )
    (tmp_path / "config.local.yaml").write_text("matching:\n  confidence: 0.95\n", encoding="utf-8")

    config = load_config(tmp_path)
    assert config.matching.confidence == 0.95  # local wins
    assert config.window.instance == 1  # base preserved
    assert config.root == tmp_path.resolve()
    assert config.templates_dir == tmp_path.resolve() / "templates"


def test_missing_config_file_falls_back_to_defaults(tmp_path):
    config = load_config(tmp_path)
    assert config.matching.confidence == 0.85


def test_collected_samples_land_beside_the_exe_not_inside_the_bundle(tmp_path, monkeypatch):
    """A one-file build unpacks into a temp dir that is wiped on exit.

    Logs already dodge that; training data has to as well, or a session's
    samples are deleted the moment the app closes and nobody notices until
    they go looking for the folder to send.
    """
    exe_dir, bundle = tmp_path / "app", tmp_path / "_MEI"
    exe_dir.mkdir()
    bundle.mkdir()
    (bundle / "config.yaml").write_text("dataset:\n  enabled: true\n", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "ttheart-sender.exe"))
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)

    config = load_config()

    assert config.root == bundle.resolve(), "flows and templates still come from the bundle"
    assert config.dataset_dir == exe_dir.resolve() / "dataset"
