import pytest

from exo.worker.engines.mlx.constants import (
    DEFAULT_PREFILL_STEP_SIZE,
    get_prefill_step_size,
)


def test_prefill_step_size_defaults_to_4096(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXO_MLX_PREFILL_STEP_SIZE", raising=False)
    assert get_prefill_step_size() == DEFAULT_PREFILL_STEP_SIZE == 4096


def test_prefill_step_size_accepts_positive_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_MLX_PREFILL_STEP_SIZE", "1024")
    assert get_prefill_step_size() == 1024


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_prefill_step_size_rejects_invalid_override(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("EXO_MLX_PREFILL_STEP_SIZE", value)
    with pytest.raises(ValueError, match="must be a positive integer"):
        get_prefill_step_size()
