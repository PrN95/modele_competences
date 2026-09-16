"""Teste explicitement les trois endpoints OpenAI-compatibles, sans campagne."""

from __future__ import annotations

from pathlib import Path
import sys

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    BGE_API_KEY, BGE_BASE_URL, GEMMA_API_KEY, GEMMA_BASE_URL,
    QWEN_API_KEY, QWEN_BASE_URL, REMOTE_TIMEOUT_SECONDS,
)


def _test(name: str, base_url: str, api_key: str | None) -> bool:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = requests.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=REMOTE_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json().get("data", [])
        ids = [item.get("id") for item in data if isinstance(item, dict) and item.get("id")]
        if not ids:
            raise ValueError("aucun identifiant dans data")
    except (requests.RequestException, ValueError, TypeError) as exc:
        print(f"{name}: ÉCHEC — {exc}")
        return False
    print(f"{name}: OK — {base_url} — modèles : {', '.join(ids)}")
    return True


def main() -> int:
    ok = [
        _test("BGE-M3", BGE_BASE_URL, BGE_API_KEY),
        _test("Qwen3-Embedding-8B", QWEN_BASE_URL, QWEN_API_KEY),
        _test("Gemma 4", GEMMA_BASE_URL, GEMMA_API_KEY),
    ]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
