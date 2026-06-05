from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Protocol, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class AgentRunConfig:
    input: str
    context: dict[str, Any] | None = None
    session: Any | None = None
    max_turns: int | None = None
    branch: str | None = None


@dataclass(frozen=True, slots=True)
class AgentRunResult[T]:
    output: T
    success: bool
    error: str | None = None
    raw_result: Any | None = None
    working_dir: str | None = None


class ConsensusStrategy(StrEnum):
    MAJORITY = "majority"
    UNANIMOUS = "unanimous"
    JUDGE = "judge"


@dataclass(frozen=True, slots=True)
class ConsensusRunResult(AgentRunResult[T]):
    runs: list[AgentRunResult[T]] = field(default_factory=list)
    agreement: float = 0.0
    total_runs: int = 0
    successful_runs: int = 0


@dataclass(frozen=True, slots=True)
class AgentConfig[T]:
    name: str
    instructions: str
    model: str | None = None
    tools: list[Any] = field(default_factory=list)
    output_schema: Any | None = None
    model_settings: dict[str, Any] | None = None
    input_guardrails: list[Any] | None = None
    strict_json_schema: bool = True


@dataclass(frozen=True, slots=True)
class AgentType[T]:
    config: AgentConfig[T]


class AiAgentService(Protocol):
    def create_agent(self, config: AgentConfig[T]) -> AgentType[T]: ...

    async def run_agent(
        self,
        agent: AgentType[T],
        config: AgentRunConfig,
    ) -> AgentRunResult[T]: ...

    async def create_and_run(
        self,
        agent_config: AgentConfig[T],
        run_config: AgentRunConfig,
    ) -> AgentRunResult[T]: ...


class AiAgentProvider(StrEnum):
    OPENAI = "openai"
    OPENAI_COMPAT = "openai_compat"


@dataclass(frozen=True, slots=True)
class AiAgentConfig:
    local: bool
    provider: AiAgentProvider = AiAgentProvider.OPENAI
    openai_api_token: str | None = None
    openai_compat_base_url: str | None = None
    openai_compat_api_key: str | None = None
    default_model: str | None = None


class AiAgentServiceLocal:
    _instance: ClassVar[AiAgentServiceLocal | None] = None
    _responses_by_agent_name: ClassVar[dict[str, list[Any]]] = {}
    _last_response_by_agent_name: ClassVar[dict[str, Any]] = {}
    _errors_by_agent_name: ClassVar[dict[str, str]] = {}
    _mock_working_dir: ClassVar[str | None] = None

    @classmethod
    def get_instance(cls) -> AiAgentServiceLocal:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def set_response(cls, agent_name: str, output: Any) -> None:
        cls._responses_by_agent_name.setdefault(agent_name, []).append(output)

    @classmethod
    def set_responses(cls, agent_name: str, outputs: list[Any]) -> None:
        cls._responses_by_agent_name.setdefault(agent_name, []).extend(outputs)

    @classmethod
    def set_mock_working_dir(cls, directory: str | None) -> None:
        cls._mock_working_dir = directory

    @classmethod
    def set_error(cls, agent_name: str, error_message: str) -> None:
        cls._errors_by_agent_name[agent_name] = error_message

    @classmethod
    def clear_responses(cls) -> None:
        cls._responses_by_agent_name.clear()
        cls._last_response_by_agent_name.clear()

    @classmethod
    def clear_errors(cls) -> None:
        cls._errors_by_agent_name.clear()

    @classmethod
    def clear_all_overrides(cls) -> None:
        cls._responses_by_agent_name.clear()
        cls._last_response_by_agent_name.clear()
        cls._errors_by_agent_name.clear()
        cls._mock_working_dir = None

    def create_agent(self, config: AgentConfig[T]) -> AgentType[T]:
        return AgentType(config=config)

    async def run_agent(
        self,
        agent: AgentType[T],
        config: AgentRunConfig,
    ) -> AgentRunResult[T]:
        return await self._execute_agent(agent.config, config)

    async def create_and_run(
        self,
        agent_config: AgentConfig[T],
        run_config: AgentRunConfig,
    ) -> AgentRunResult[T]:
        return await self._execute_agent(agent_config, run_config)

    async def _execute_agent(
        self,
        agent_config: AgentConfig[T],
        run_config: AgentRunConfig,
    ) -> AgentRunResult[T]:
        result = await self._get_agent_result(agent_config)

        if run_config.session is not None and result.success:
            await run_config.session.add_items(
                [
                    {"role": "user", "content": run_config.input},
                    {
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    result.output
                                    if isinstance(result.output, str)
                                    else json.dumps(result.output)
                                ),
                            }
                        ],
                    },
                ]
            )

        return result

    async def _get_agent_result(self, agent_config: AgentConfig[T]) -> AgentRunResult[T]:
        error = self._errors_by_agent_name.get(agent_config.name)
        if error is not None:
            return AgentRunResult(
                success=False,
                error=error,
                output=None,  # type: ignore[arg-type]
            )

        responses = self._responses_by_agent_name.get(agent_config.name)
        if responses:
            response = responses.pop(0)
            self._last_response_by_agent_name[agent_config.name] = response
            return AgentRunResult(
                success=True,
                output=response,
                working_dir=self._mock_working_dir,
            )

        if agent_config.name in self._last_response_by_agent_name:
            return AgentRunResult(
                success=True,
                output=self._last_response_by_agent_name[agent_config.name],
                working_dir=self._mock_working_dir,
            )

        return AgentRunResult(
            success=True,
            output=self._generate_default_response(agent_config),
            working_dir=self._mock_working_dir,
        )

    def _generate_default_response(self, config: AgentConfig[T]) -> T:
        if config.output_schema is not None:
            return {}  # type: ignore[return-value]
        return f"Mock response for {config.name}"  # type: ignore[return-value]
