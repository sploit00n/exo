import os

import pytest

from exo.worker.runner.bootstrap import configure_fast_synch_environment


def test_fast_synch_uses_mlx_default_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXO_FAST_SYNCH", raising=False)
    monkeypatch.delenv("MLX_METAL_FAST_SYNCH", raising=False)

    assert configure_fast_synch_environment() == "unset (MLX default)"
    assert "MLX_METAL_FAST_SYNCH" not in os.environ


@pytest.mark.parametrize(("override", "expected"), [("true", "1"), ("false", "0")])
def test_fast_synch_applies_explicit_override(
    monkeypatch: pytest.MonkeyPatch, override: str, expected: str
) -> None:
    monkeypatch.setenv("EXO_FAST_SYNCH", override)
    monkeypatch.delenv("MLX_METAL_FAST_SYNCH", raising=False)

    assert configure_fast_synch_environment() == expected


def test_fast_synch_rejects_invalid_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_FAST_SYNCH", "sometimes")
    with pytest.raises(ValueError, match="must be either"):
        configure_fast_synch_environment()
