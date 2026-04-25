"""LLM and embedding clients behind narrow Protocols.

The agent graph and ingestion pipeline depend only on the Protocols here, not
on OpenAI directly — this is what lets unit tests inject deterministic fakes.

Two embedder backends ship:
- `OpenAIEmbedder` — real, used when `OPENAI_API_KEY` is present.
- `LocalDeterministicEmbedder` — hash-based projection into a 1536-dim unit
  vector. NOT semantically meaningful; used so the system boots end-to-end
  without an API key (demo / CI). The retrieval recall on this fallback is
  essentially a baseline; the moment a real key is set it switches over.

A `FakeLLM` is provided for tests; the real `OpenAIChat` wraps the v1 SDK with
tenacity-backed retries and structured token accounting.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from openai import APIError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.logging_config import get_logger

log = get_logger("llm_client")

EMBEDDING_DIM = 1536


@dataclasses.dataclass(frozen=True)
class TokenUsage:
    """A single LLM call's token + cost accounting."""

    node: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclasses.dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclasses.dataclass(frozen=True)
class ChatResult:
    content: str
    usage: TokenUsage


@runtime_checkable
class Embedder(Protocol):
    """Embed text strings into a fixed-dimension dense vector."""

    backend_name: str
    dimension: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


@runtime_checkable
class LLMClient(Protocol):
    """Chat completion with structured output expectations."""

    backend_name: str

    async def chat(
        self,
        *,
        node: str,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float = 0.1,
        response_format_json: bool = False,
    ) -> ChatResult:
        ...


# ---------- OpenAI embedder ------------------------------------------------------


class OpenAIEmbedder:
    """Production embedder against OpenAI's `text-embedding-3-small` (1536 dim)."""

    backend_name = "openai"
    dimension = EMBEDDING_DIM

    def __init__(self, *, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIEmbedder")
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    @retry(
        retry=retry_if_exception_type((APIError, RateLimitError)),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(model=self._model, input=list(texts))
        return [list(item.embedding) for item in resp.data]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        # OpenAI handles batching server-side, but cap to 96 per call to be safe.
        out: list[list[float]] = []
        batch_size = 96
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embedded = await self._embed_batch(batch)
            out.extend(embedded)
        log.info(
            "embedder.openai", count=len(out), model=self._model, dim=self.dimension
        )
        return out


# ---------- Local deterministic embedder ----------------------------------------


class LocalDeterministicEmbedder:
    """Deterministic hash-based projection into a unit vector.

    Method:
        - Build an n-gram bag (1-grams + 3-grams) from the text.
        - For each token, hash with sha256 → 4 32-bit ints → 4 dimension indices
          modulo `dimension`. Add `1.0 / sqrt(token_count)` to each.
        - L2-normalize.

    This produces stable, reproducible vectors with weak but non-zero semantic
    signal. Sufficient for end-to-end demo without an API key; *not* a
    substitute for real embeddings on a real corpus.
    """

    backend_name = "local-deterministic"
    dimension = EMBEDDING_DIM

    def __init__(self, *, dimension: int = EMBEDDING_DIM) -> None:
        self.dimension = dimension

    def _tokens(self, text: str) -> Iterable[str]:
        cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
        words = [w for w in cleaned.split() if w]
        for w in words:
            yield w
        joined = " ".join(words)
        for i in range(len(joined) - 2):
            yield joined[i : i + 3]

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dimension, dtype=np.float32)
        token_list = list(self._tokens(text))
        if not token_list:
            return v.tolist()
        weight = 1.0 / math.sqrt(len(token_list))
        for tok in token_list:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            for offset in range(0, 16, 4):
                idx = int.from_bytes(digest[offset : offset + 4], "big") % self.dimension
                v[idx] += weight
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm
        return v.tolist()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out = [self._vec(t) for t in texts]
        log.info(
            "embedder.local",
            count=len(out),
            dim=self.dimension,
            note="deterministic-fallback",
        )
        return out


def build_embedder() -> Embedder:
    settings = get_settings()
    if settings.openai_api_key:
        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )
    log.warning(
        "embedder.no_api_key",
        message="OPENAI_API_KEY not set; using local deterministic embedder",
    )
    return LocalDeterministicEmbedder()


# ---------- Chat clients ---------------------------------------------------------

_PRICING = {
    # Per-1K tokens, USD. Configurable; this is an honest snapshot — keep in sync
    # with `app/observability/pricing.json` when that file lands in Stage 4.
    "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
    "gpt-4o": {"input": 0.005, "output": 0.015},
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = _PRICING.get(model, _PRICING["gpt-4o-mini"])
    return round(
        (prompt_tokens / 1000.0) * rates["input"]
        + (completion_tokens / 1000.0) * rates["output"],
        6,
    )


class OpenAIChat:
    """Production chat client. Used by the LangGraph agent in Stage 3."""

    backend_name = "openai"

    def __init__(self, *, api_key: str) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIChat")
        self._client = AsyncOpenAI(api_key=api_key)

    @retry(
        retry=retry_if_exception_type((APIError, RateLimitError)),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def chat(
        self,
        *,
        node: str,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float = 0.1,
        response_format_json: bool = False,
    ) -> ChatResult:
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if response_format_json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        content = choice.message.content or ""
        usage_obj = resp.usage
        prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
        usage = TokenUsage(
            node=node,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=_estimate_cost(model, prompt_tokens, completion_tokens),
        )
        return ChatResult(content=content, usage=usage)


class FakeLLM:
    """Deterministic chat client for tests.

    Configure with a `responder` that maps the last user message to a string,
    or a fixed `default_response`. Records every call for assertions.
    """

    backend_name = "fake"

    def __init__(
        self,
        *,
        default_response: str = "{}",
        responder: Any = None,
    ) -> None:
        self._default = default_response
        self._responder = responder
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        *,
        node: str,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float = 0.1,
        response_format_json: bool = False,
    ) -> ChatResult:
        self.calls.append(
            {
                "node": node,
                "model": model,
                "messages": [(m.role, m.content) for m in messages],
                "temperature": temperature,
                "json": response_format_json,
            }
        )
        if self._responder is not None:
            content = self._responder(messages)
        else:
            content = self._default
        if response_format_json:
            try:
                json.loads(content)
            except json.JSONDecodeError:
                content = self._default
        usage = TokenUsage(
            node=node,
            model=model,
            prompt_tokens=10,
            completion_tokens=10,
            cost_usd=_estimate_cost(model, 10, 10),
        )
        return ChatResult(content=content, usage=usage)


def _offline_responder(messages: Sequence[ChatMessage]) -> str:
    """Deterministic JSON-shaped responder used when no OPENAI_API_KEY is set.

    The responder reads the *last user message* to figure out which node is
    calling (each node prefixes its USER block with a `# NODE: <name>` tag),
    and produces a minimally-valid JSON payload of the right shape. This keeps
    the agent graph runnable end-to-end in a key-less local / CI environment.

    The output is honest about being offline: the synthesizer composes its
    answer from the evidence snippets verbatim and prefixes a notice.
    """
    last = messages[-1].content if messages else ""

    if "# NODE: plan" in last:
        question = _extract_field(last, "QUESTION:")
        return json.dumps({"sub_questions": [question or "(empty)"]})

    if "# NODE: synthesize" in last:
        # Compose a baseline answer that quotes evidence [1] (and [2] if
        # present), prefixed with a transparency notice.
        snippets = _extract_evidence(last)
        if not snippets:
            return json.dumps(
                {
                    "answer": (
                        "[Offline mode] No evidence was retrieved for this "
                        "question; cannot answer from the corpus."
                    ),
                    "used_indices": [],
                }
            )
        first = snippets[0]
        text_one = first[:240].replace("\n", " ").strip()
        body = (
            "[Offline mode — running without OPENAI_API_KEY; this answer "
            "quotes the top retrieved excerpt verbatim.]\n\n"
            f"{text_one}…  [1]"
        )
        used = [1]
        if len(snippets) > 1:
            used.append(2)
        return json.dumps({"answer": body, "used_indices": used})

    if "# NODE: critique" in last:
        return json.dumps(
            {
                "grounding_score": 0.85,
                "hallucination_risk": 0.15,
                "passed": True,
                "issues": [],
            }
        )

    return json.dumps({"answer": "OPENAI_API_KEY not configured."})


def _extract_field(text: str, label: str) -> str:
    """Pull the line(s) following a `LABEL:` marker, stopping at the next blank."""
    if label not in text:
        return ""
    after = text.split(label, 1)[1]
    lines: list[str] = []
    for line in after.splitlines():
        if not line.strip():
            if lines:
                break
            continue
        lines.append(line.strip())
    return " ".join(lines).strip()


def _extract_evidence(text: str) -> list[str]:
    """Recover the textual bodies of `[N]` evidence snippets from a synth prompt."""
    if "EVIDENCE" not in text:
        return []
    after = text.split("EVIDENCE", 1)[1]
    snippets: list[str] = []
    current: list[str] = []
    seen_marker = False
    for raw in after.splitlines():
        line = raw.rstrip()
        stripped = line.lstrip()
        if stripped.startswith("[") and "]" in stripped[:6]:
            if current and seen_marker:
                snippets.append("\n".join(current).strip())
            current = [stripped.split("]", 1)[1].strip()]
            seen_marker = True
            continue
        if "# NODE:" in stripped or stripped.startswith("QUESTION:"):
            break
        if seen_marker:
            current.append(stripped)
    if current and seen_marker:
        snippets.append("\n".join(current).strip())
    return [s for s in snippets if s]


def build_llm() -> LLMClient:
    settings = get_settings()
    if settings.openai_api_key:
        return OpenAIChat(api_key=settings.openai_api_key)
    log.warning(
        "llm.no_api_key",
        message=(
            "OPENAI_API_KEY not set; using offline FakeLLM responder. "
            "Answers will be evidence-quoting only."
        ),
    )
    return FakeLLM(responder=_offline_responder)
