"""vLLM provider: a self-hosted OpenAI-compatible server.

Usage::

    from sagent.providers import VLLM

    provider = VLLM.from_env()      # VLLM_API_KEY, VLLM_BASE_URL
    model = provider.model()        # qwen-flash-next-3.8
    response = await model.buffer(request)

Unlike ``LlamaCpp`` and ``SelfHosted`` it manages no process: it talks to
a server already running elsewhere (a Slurm job on a GPU node, say).
``VLLM_BASE_URL`` names it; ``VLLM_API_KEY`` is the server's ``--api-key``,
or unset for a server started without one.

vLLM's reasoning parser returns reasoning in the ``reasoning`` field and
reads it back from the same field on replayed assistant messages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self, override

import os

from sagent.catalog.openai import compatible
from sagent.catalog.table import ModelCatalog
from sagent.catalog.vllm import models
from sagent.providers.openai.compat import (
    OpenAICompat,
    OpenAICompatModel,
)


if TYPE_CHECKING:
    from treekle import MutablePlainTree

    from sagent.types.model import ModelRequest


class _VLLMModel(OpenAICompatModel):
    """vLLM backend: reasoning in ``reasoning``, no effort knob."""

    _reasoning_field: ClassVar[str] = "reasoning"

    # The catalog rows offer no effort, and the chat template decides
    # thinking itself; an OpenAI-spelled effort would only confuse it.
    @override
    def _transform_body(
        self,
        body: dict[str, MutablePlainTree],
        request: ModelRequest,
    ) -> dict[str, MutablePlainTree]:
        """Drop the OpenAI effort knob the base adds."""
        del request
        body.pop("reasoning_effort", None)
        return body


class VLLM(OpenAICompat):
    """Self-hosted vLLM server provider."""

    ENV_VAR: ClassVar[str] = "VLLM_API_KEY"
    BASE_URL: ClassVar[str] = "http://127.0.0.1:8000/v1"

    catalog = ModelCatalog(models=models(), transport=compatible())

    MODEL_CLASS: ClassVar[type[OpenAICompatModel]] = _VLLMModel

    @classmethod
    @override
    def from_env(cls, *, base_url: str | None = None) -> Self:
        """Build provider from ``VLLM_BASE_URL`` and ``VLLM_API_KEY``.

        A server started without ``--api-key`` accepts any key, so an unset
        key is sent as ``EMPTY`` rather than refused here.

        Args:
          base_url: Override for ``VLLM_BASE_URL``.

        Returns:
          provider: Configured provider instance.

        """
        return cls(
            api_key=os.environ.get(cls.ENV_VAR) or "EMPTY",
            base_url=base_url or os.environ.get("VLLM_BASE_URL") or None,
        )
