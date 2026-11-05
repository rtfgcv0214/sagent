"""Tests for the vLLM provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sagent.providers import VLLM
from sagent.providers.openai.compat import OpenAICompatModel
from sagent.providers.providers import build_provider, default_auth_for_provider
from sagent.types.model import ModelRequest
from sagent.types.runtime import UserMessage


if TYPE_CHECKING:
    import pytest


def test_from_env_reads_url_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-node:8000/v1")
    monkeypatch.setenv("VLLM_API_KEY", "k")
    p = build_provider("VLLM", default_auth_for_provider("VLLM"))
    assert isinstance(p, VLLM)
    assert (p.base_url, p.api_key) == ("http://gpu-node:8000/v1", "k")


def test_from_env_without_key_or_url_uses_the_local_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    p = VLLM.from_env()
    assert (p.base_url, p.api_key) == ("http://127.0.0.1:8000/v1", "EMPTY")


def test_default_model_is_qwen_flash_next_on_its_served_name() -> None:
    model = VLLM.from_key("k").model()
    assert model.capability.model_id == "qwen-flash-next-3.8"
    assert model.capability.wire_model_id == "qwen3.8-flash-next"


def test_body_names_the_served_model_and_sends_no_effort() -> None:
    model = VLLM.from_key("k").model()
    assert isinstance(model, OpenAICompatModel)
    body = model._build_body(
        ModelRequest(messages=[UserMessage(text="x")]),
        stream=False,
    )
    assert body["model"] == "qwen3.8-flash-next"
    assert "reasoning_effort" not in body
