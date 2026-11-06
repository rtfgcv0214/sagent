"""vLLM (self-hosted) model catalog, expressed as ``ModelCapability`` rows.

Source: n/a -- self-hosted. A self-hosted model bills nothing, but its row carries the list
price of the same model on its vendor's API, so a run's cost compares with runs on other
providers: Qwen3.8 Flash Next is priced as DashScope's ``qwen-flash-3.8``
(:mod:`sagent.catalog.dashscope`, whose published rates it takes rather than copies).

A row carries only what the MODEL can do; caching, retry, and auth mode
are transport facts declared by ``openai.compatible()``. The wire id is
the server's ``--served-model-name``. The two token windows split the
server's ``--max-model-len`` between request and response.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from sagent.catalog import dashscope
from sagent.catalog.table import ModelTable
from sagent.types.capability import ContextTag, ModelCapability, ModelLimits


if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = ["models"]


def models() -> ModelTable:
    """Return every model this vendor serves, keyed by base id.

    Returns:
      models: Capability per base model id.

    """
    # The reference row: Qwen3.8 Flash Next (FP8) served with
    # ``--max-model-len 262144 --reasoning-parser qwen3 --tool-call-parser
    # qwen3_coder``, priced as Qwen3.8 Flash on DashScope (its list price; the
    # open-weights Next has none of its own).
    flash = next(r for r in dashscope.models().rows if r.model_id == "qwen-flash-3.8")
    default = ModelCapability(
        model_id="qwen-flash-next-3.8",
        wire_model_id="qwen3.8-flash-next",
        context=_limits(max_tokens=229_376, output_tokens=32_768),
        prices=flash.prices,
    )
    rows = (default,)
    # A tier is offered exactly when it is priced.
    rows = tuple(replace(row, service_tier=row.prices.service_tiers) for row in rows)
    return ModelTable(
        rows=rows,
        roles={"default": default.model_id, "utility": default.model_id},
    )


def _limits(
    *,
    max_tokens: int,
    output_tokens: int,
) -> Mapping[ContextTag, ModelLimits]:
    """One context tag: a self-hosted server ships no window variants."""
    return MappingProxyType(
        {
            "": ModelLimits(
                max_request_tokens=max_tokens,
                max_response_tokens=output_tokens,
            ),
        },
    )
