import os

# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

KV_GROUP_SIZE: int | None = 32
KV_BITS: int | None = None
ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = 3200
KEEP_KV_SIZE: int | None = 1600
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
KV_CACHE_BITS: int | None = None

DEFAULT_TOP_LOGPROBS: int = 5
DEFAULT_PREFILL_STEP_SIZE: int = 4096


def get_prefill_step_size() -> int:
    """Return the MLX prompt chunk size, optionally overridden per runner."""
    raw_value = os.environ.get("EXO_MLX_PREFILL_STEP_SIZE")
    if raw_value is None:
        return DEFAULT_PREFILL_STEP_SIZE

    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(
            f"EXO_MLX_PREFILL_STEP_SIZE must be a positive integer; got {raw_value!r}"
        ) from error

    if value <= 0:
        raise ValueError(
            f"EXO_MLX_PREFILL_STEP_SIZE must be a positive integer; got {raw_value!r}"
        )
    return value


# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
