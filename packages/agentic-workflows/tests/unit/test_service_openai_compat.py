"""Unit tests for AiAgentServiceOpenAICompat strict JSON + OpenRouter params."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from agents.model_settings import ModelSettings
from zeroshot_agentic_workflows import AgentConfig, AgentRunConfig
from zeroshot_agentic_workflows import service_openai_compat as svc_mod


@dataclass
class _Schema:
    answer: str


class _FakeAgentOutputSchema:
    """Recording fake for AgentOutputSchema(output_type, strict_json_schema=...)."""

    instances: ClassVar[list[_FakeAgentOutputSchema]] = []

    def __init__(self, output_type: Any, strict_json_schema: bool = True) -> None:
        self.output_type = output_type
        self.strict_json_schema = strict_json_schema
        _FakeAgentOutputSchema.instances.append(self)


class _FakeAgent:
    instances: ClassVar[list[_FakeAgent]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _FakeAgent.instances.append(self)


class _FakeResult:
    def __init__(self, output: Any) -> None:
        self.final_output = output


class _FakeRunner:
    @staticmethod
    async def run(_agent: Any, **_kwargs: Any) -> _FakeResult:
        return _FakeResult(output=_Schema(answer="ok"))


@pytest.fixture(autouse=True)
def _patch_sdk() -> Any:
    _FakeAgentOutputSchema.instances.clear()
    _FakeAgent.instances.clear()
    with (
        patch.object(svc_mod, "AgentOutputSchema", _FakeAgentOutputSchema),
        patch.object(svc_mod, "Agent", _FakeAgent),
        patch.object(svc_mod, "Runner", _FakeRunner),
        patch.object(svc_mod, "OpenAIChatCompletionsModel", lambda **kw: ("model", kw)),
    ):
        yield


def _make_service() -> svc_mod.AiAgentServiceOpenAICompat:
    return svc_mod.AiAgentServiceOpenAICompat(
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        default_model="openai/gpt-5",
    )


class TestStrictJsonSchema:
    async def test_defaults_to_strict_true(self) -> None:
        service = _make_service()
        config = AgentConfig[_Schema](
            name="t",
            instructions="hi",
            output_schema=_Schema,
        )

        result = await service.create_and_run(config, AgentRunConfig(input="x"))

        assert result.success is True
        assert len(_FakeAgentOutputSchema.instances) == 1
        assert _FakeAgentOutputSchema.instances[0].strict_json_schema is True
        assert _FakeAgentOutputSchema.instances[0].output_type is _Schema

    async def test_caller_can_opt_out(self) -> None:
        service = _make_service()
        config = AgentConfig[_Schema](
            name="t",
            instructions="hi",
            output_schema=_Schema,
            strict_json_schema=False,
        )

        result = await service.create_and_run(config, AgentRunConfig(input="x"))

        assert result.success is True
        assert len(_FakeAgentOutputSchema.instances) == 1
        assert _FakeAgentOutputSchema.instances[0].strict_json_schema is False

    async def test_no_output_schema_means_no_output_type(self) -> None:
        service = _make_service()
        config = AgentConfig[str](name="t", instructions="hi")

        result = await service.create_and_run(config, AgentRunConfig(input="x"))

        assert result.success is True
        assert _FakeAgentOutputSchema.instances == []
        assert _FakeAgent.instances[0].kwargs["output_type"] is None


class TestOpenRouterParams:
    async def test_extra_body_flows_into_model_settings(self) -> None:
        service = _make_service()
        extra_body = {
            "provider": {
                "require_parameters": True,
                "allow_fallbacks": False,
                "order": ["OpenAI"],
            },
            "plugins": [{"id": "response-healing"}],
        }
        config = AgentConfig[_Schema](
            name="t",
            instructions="hi",
            output_schema=_Schema,
            model_settings={
                "tool_choice": "none",
                "extra_body": extra_body,
            },
        )

        result = await service.create_and_run(config, AgentRunConfig(input="x"))

        assert result.success is True
        ms = _FakeAgent.instances[0].kwargs["model_settings"]
        assert isinstance(ms, ModelSettings)
        assert ms.extra_body == extra_body
        assert ms.tool_choice == "none"

    async def test_no_model_settings_yields_default_model_settings(self) -> None:
        service = _make_service()
        config = AgentConfig[_Schema](
            name="t",
            instructions="hi",
            output_schema=_Schema,
        )

        await service.create_and_run(config, AgentRunConfig(input="x"))

        ms = _FakeAgent.instances[0].kwargs["model_settings"]
        assert isinstance(ms, ModelSettings)
        assert ms.extra_body is None
