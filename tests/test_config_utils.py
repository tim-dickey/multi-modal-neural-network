from pathlib import Path

from src.utils.config import (
    load_config,
    _resolve_env_vars,
    resolve_path,
    save_config,
    merge_configs,
    validate_config,
    ConfigNamespace,
)


def test_resolve_env_vars_and_save_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("FOO", "bar")
    cfg = {"a": "${FOO}", "home": "~/mydir", "nested": {"list": ["${FOO}", "plain"]}}
    resolved = _resolve_env_vars(cfg)
    assert resolved["a"] == "bar"
    assert "mydir" in resolved["home"]
    assert resolved["nested"]["list"][0] == "bar"

    # save and load roundtrip
    p = tmp_path / "cfg.yaml"
    save_config(resolved, p)
    loaded = load_config(p)
    assert loaded["a"] == "bar"


def test_merge_and_validate_and_namespace(tmp_path):
    base = {
        "model": {"vision_encoder": {}, "text_encoder": {}, "fusion": {}, "heads": {}},
        "training": {"max_epochs": 1, "inner_lr": 0.001},
        "data": {},
    }
    override = {"training": {"inner_lr": 0.01}, "extra": 1}
    merged = merge_configs(base, override)
    assert merged["training"]["inner_lr"] == 0.01

    assert validate_config(merged) is True

    ns = ConfigNamespace({"a": 1, "b": {"c": 2}})
    d = ns.to_dict()
    assert d["b"]["c"] == 2
    assert "ConfigNamespace" not in repr(ns)


def test_resolve_path_absolute_and_relative(tmp_path, monkeypatch):
    # absolute path returns itself
    abs_p = Path(tmp_path / "abs.txt").resolve()
    assert resolve_path(str(abs_p)) == abs_p

    # relative to project root: use a fake project structure by specifying relative_to
    rp = resolve_path("some_dir/file.txt", relative_to=tmp_path)
    # Should create a resolved path under tmp_path
    assert str(tmp_path) in str(rp)
