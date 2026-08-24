#!/usr/bin/env python3
"""Run deterministic GLM-5.2 needle probes through exo's chat API."""

import argparse
import json
import re
import time
from collections.abc import Sequence
from typing import Protocol

import httpx
from transformers import AutoTokenizer

DEFAULT_LENGTHS = (1024, 2049, 4096)
FILLER = (
    "Background archive entry: copper lanterns were catalogued beside quiet "
    "cedar shelves; this line contains no verification code.\n"
)


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


def parse_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("lengths must be positive comma-separated integers")
    return lengths


def build_needle_prompt(
    tokenizer: Tokenizer, target_tokens: int
) -> tuple[str, str, int]:
    code = f"GLM52-{target_tokens}-7319"
    prefix = (
        "Read the archive and remember the one verification code. "
        "Ignore all statements saying that no code is present.\n"
    )
    needle = f"AUTHORITATIVE VERIFICATION CODE: {code}\n"
    question = "Return only the authoritative verification code, with no explanation."

    fixed_tokens = len(
        tokenizer.encode(prefix + needle + question, add_special_tokens=False)
    )
    filler_tokens = max(
        1, len(tokenizer.encode(FILLER, add_special_tokens=False))
    )
    repeats = max(0, (target_tokens - fixed_tokens) // filler_tokens)
    before = repeats // 2
    after = repeats - before
    prompt = prefix + FILLER * before + needle + FILLER * after + question
    actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt, code, actual_tokens


def repeated_four_gram_ratio(text: str) -> float:
    tokens = text.split()
    if len(tokens) < 4:
        return 0.0
    grams = [tuple(tokens[index : index + 4]) for index in range(len(tokens) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


def run_probe(
    client: httpx.Client,
    endpoint: str,
    model: str,
    tokenizer: Tokenizer,
    target_tokens: int,
) -> dict[str, object]:
    prompt, expected, content_tokens = build_needle_prompt(tokenizer, target_tokens)
    started = time.perf_counter()
    response = client.post(
        endpoint,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 42,
            "max_tokens": 64,
            "enable_thinking": False,
            "use_prefix_cache": False,
        },
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    payload = response.json()
    message = payload["choices"][0]["message"]
    answer = message.get("content") or message.get("reasoning_content") or ""
    usage = payload.get("usage") or {}
    printable_ratio = (
        sum(character.isprintable() for character in answer) / len(answer)
        if answer
        else 0.0
    )

    return {
        "target_content_tokens": target_tokens,
        "actual_content_tokens": content_tokens,
        "api_prompt_tokens": usage.get("prompt_tokens"),
        "expected": expected,
        "answer": answer,
        "needle_found": expected in answer,
        "printable_ratio": round(printable_ratio, 4),
        "repeated_4gram_ratio": round(repeated_four_gram_ratio(answer), 4),
        "zero_noise": bool(re.search(r"(?:0[\s.,;:_-]*){8,}", answer)),
        "elapsed_seconds": round(elapsed, 3),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:52415", help="exo API origin"
    )
    parser.add_argument("--model", default="kernelpool/GLM-5.2-8bit")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer repo/path; defaults to --model",
    )
    parser.add_argument(
        "--lengths",
        type=parse_lengths,
        default=DEFAULT_LENGTHS,
        help="Comma-separated target content lengths",
    )
    parser.add_argument("--timeout", type=float, default=3600.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, trust_remote_code=True
    )
    endpoint = f"{args.base_url.rstrip('/')}/bench/chat/completions"

    failed = False
    with httpx.Client(timeout=args.timeout) as client:
        for target_tokens in args.lengths:
            result = run_probe(
                client, endpoint, args.model, tokenizer, target_tokens
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
            failed = failed or not bool(result["needle_found"])
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
