import importlib

import pytest


def _reload_config(monkeypatch, **env):
    keys = [
        "ENVIRONMENT",
        "ENABLE_JUDGE",
        "NEBIUS_MODEL",
        "SUMMARIZER_MODEL",
        "EVAL_MODEL",
        "NEBIUS_API_KEY",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    import config

    return importlib.reload(config)


def test_config_defaults(monkeypatch):
    cfg = _reload_config(monkeypatch)

    assert cfg.ENVIRONMENT == "prod"
    assert cfg.ENABLE_JUDGE is False
    assert cfg.IS_PROD is True
    assert cfg.IS_TEST is False
    assert cfg.IS_EVAL is False
    assert cfg.SUMMARIZER_MODEL == cfg.NEBIUS_MODEL
    assert cfg.EVAL_MODEL == "Meta/Llama-3.3-70B-Instruct"


def test_config_mode_and_model_overrides(monkeypatch):
    cfg = _reload_config(
        monkeypatch,
        ENVIRONMENT="eval",
        ENABLE_JUDGE="1",
        NEBIUS_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct",
        SUMMARIZER_MODEL="meta-llama/Meta-Llama-3.3-70B-Instruct",
        EVAL_MODEL="Meta/Llama-3.3-70B-Instruct",
    )

    assert cfg.ENVIRONMENT == "eval"
    assert cfg.ENABLE_JUDGE is True
    assert cfg.IS_PROD is False
    assert cfg.IS_TEST is False
    assert cfg.IS_EVAL is True
    assert cfg.SUMMARIZER_MODEL == "meta-llama/Meta-Llama-3.3-70B-Instruct"
    assert cfg.EVAL_MODEL == "Meta/Llama-3.3-70B-Instruct"


def test_get_nebius_api_key_guard(monkeypatch):
    cfg = _reload_config(monkeypatch)

    with pytest.raises(RuntimeError):
        cfg.get_nebius_api_key()

    cfg = _reload_config(monkeypatch, NEBIUS_API_KEY="test-key")
    assert cfg.get_nebius_api_key() == "test-key"
