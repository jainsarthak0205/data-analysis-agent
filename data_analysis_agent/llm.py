"""LLM client abstraction.

Defaults to Claude Opus 4.8 but the ``ClaudeClient`` is model-agnostic: any
Claude model that speaks the Messages API with tool use will work. Use
``SUPPORTED_MODELS`` to enumerate the picks we have shipped guidance for, and
``THINKING_MODELS`` to know which support the
``thinking={"type": "adaptive"}`` extension.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple


SUPPORTED_MODELS: Tuple[str, ...] = (
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-fable-5",
)

THINKING_MODELS = {
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-fable-5",
}


def supports_adaptive_thinking(model: str) -> bool:
    """True if the given model accepts ``thinking={"type": "adaptive"}``."""
    base = model.rsplit("-2", 1)[0]
    return base in THINKING_MODELS or model in THINKING_MODELS


@dataclass
class ToolCall:
    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"


class LLMClient(Protocol):
    def complete(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        ...


class ClaudeClient:
    """Wraps the anthropic SDK. Model is configurable; adaptive thinking is
    auto-disabled on models that don't support it (e.g. Haiku 4.5)."""

    DEFAULT_MODEL = "claude-opus-4-8"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        adaptive_thinking: Optional[bool] = None,
    ):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "ClaudeClient requires the `anthropic` package. "
                "Install with: pip install anthropic"
            ) from e
        self.anthropic = __import__("anthropic")
        self.client = self.anthropic.Anthropic(api_key=api_key)
        self.model = model
        if adaptive_thinking is None:
            self.adaptive_thinking = supports_adaptive_thinking(model)
        else:
            self.adaptive_thinking = bool(adaptive_thinking) and supports_adaptive_thinking(model)

    def complete(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "tools": tools,
        }
        if self.adaptive_thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        msg = self.client.messages.create(**kwargs)

        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        for block in msg.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=dict(block.input))
                )
        return LLMResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=msg.stop_reason or "end_turn",
        )


@dataclass
class FakeStep:
    tool_calls: List[ToolCall] = field(default_factory=list)
    final_text: str = ""


class FakeLLMClient:
    """Scripted LLM for tests and offline demos."""

    def __init__(
        self,
        script: List[FakeStep] | Callable[[List[Dict[str, Any]]], FakeStep],
        final_text: str = "Analysis complete.",
    ):
        self._script = script
        self._idx = 0
        self._final_text = final_text
        self.calls: List[Dict[str, Any]] = []

    def complete(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        self.calls.append({
            "system": system,
            "messages": list(messages),
            "tools": list(tools),
            "max_tokens": max_tokens,
        })
        if callable(self._script):
            step = self._script(list(messages))
        else:
            if self._idx >= len(self._script):
                return LLMResponse(text=self._final_text, stop_reason="end_turn")
            step = self._script[self._idx]
            self._idx += 1

        if step.tool_calls:
            return LLMResponse(tool_calls=step.tool_calls, stop_reason="tool_use")
        return LLMResponse(
            text=step.final_text or self._final_text,
            stop_reason="end_turn",
        )
