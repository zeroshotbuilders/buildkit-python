"""Port of the TS decorators.spec.ts test suite."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from zeroshot_agentic_workflows import (
    AgentConfig,
    AgentRunConfig,
    AgentRunResult,
    AiAgentServiceLocal,
    ConsensusRunResult,
    ConsensusStrategy,
    InMemoryConversationSessionRepository,
    RepositorySession,
    agent,
    agentic_workflow,
    consensus_agent,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    AiAgentServiceLocal.clear_all_overrides()


# ---------------------------------------------------------------------------
# Helpers: build a test workflow class dynamically with a temp prompts dir
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prompts_dir() -> str:
    d = tempfile.mkdtemp()
    (Path(d) / "test_method.md").write_text("Test prompt content")
    (Path(d) / "test_model_override_method.md").write_text("Model override prompt")
    (Path(d) / "majority_method.md").write_text("Majority prompt")
    (Path(d) / "unanimous_method.md").write_text("Unanimous prompt")
    (Path(d) / "judge_method.md").write_text("Judge prompt")
    (Path(d) / "temp_spread_method.md").write_text("Temp spread prompt")
    return d


def _make_workflow_class(prompts_dir: str) -> type:
    """Build a workflow class with all the decorated methods."""

    async def _judge_fn(
        instance: object, results: list[AgentRunResult[str]]
    ) -> AgentRunResult[str]:
        return results[0]

    @agentic_workflow(prompts_directory=prompts_dir)
    class TestWorkflow:
        def __init__(self, service: AiAgentServiceLocal) -> None:
            self._ai_agent_service = service

        @agent(tools=[])
        async def test_method(
            self, input_text: str, session: RepositorySession | None = None
        ) -> AgentRunResult[str]: ...

        @agent(model="test-model-override")
        async def test_model_override_method(self, input_text: str) -> AgentRunResult[str]: ...

        @consensus_agent(
            runs=3,
            consensus_strategy=ConsensusStrategy.MAJORITY,
        )
        async def majority_method(self, input_text: str) -> ConsensusRunResult[str]: ...

        @consensus_agent(
            runs=3,
            consensus_strategy=ConsensusStrategy.UNANIMOUS,
        )
        async def unanimous_method(self, input_text: str) -> ConsensusRunResult[str]: ...

        @consensus_agent(
            runs=3,
            consensus_strategy=ConsensusStrategy.JUDGE,
            judge=_judge_fn,
        )
        async def judge_method(self, input_text: str) -> ConsensusRunResult[str]: ...

        @consensus_agent(
            runs=3,
            consensus_strategy=ConsensusStrategy.MAJORITY,
            temperature_spread=(0.2, 1.0),
        )
        async def temp_spread_method(self, input_text: str) -> ConsensusRunResult[str]: ...

    return TestWorkflow


# ---------------------------------------------------------------------------
# @agent decorator tests
# ---------------------------------------------------------------------------


class TestAgentDecorator:
    async def test_picks_up_session_and_passes_to_service(self, prompts_dir: str) -> None:
        Cls = _make_workflow_class(prompts_dir)
        service = AiAgentServiceLocal.get_instance()
        AiAgentServiceLocal.set_response("TestWorkflow:test_method", "hello")
        wf = Cls(service)

        repo = InMemoryConversationSessionRepository()
        backing = await repo.create_session("client-1")
        session = RepositorySession(backing.session_id, repo)

        result = await wf.test_method("test input", session)

        assert result.success is True
        assert result.output == "hello"

        # Session should have recorded the chat turns
        items = await repo.get_conversation_items(backing.session_id)
        assert len(items) == 2
        assert items[0].role == "user"
        assert items[1].role == "assistant"

    async def test_excludes_session_from_input_json(self, prompts_dir: str) -> None:
        Cls = _make_workflow_class(prompts_dir)
        service = AiAgentServiceLocal.get_instance()

        # Capture what input the service received
        captured: list[AgentRunConfig] = []
        original = service.create_and_run

        async def spy(config: AgentConfig[str], run_config: AgentRunConfig) -> AgentRunResult[str]:
            captured.append(run_config)
            return await original(config, run_config)

        service.create_and_run = spy  # type: ignore[assignment]

        repo = InMemoryConversationSessionRepository()
        backing = await repo.create_session("client-1")
        session = RepositorySession(backing.session_id, repo)

        await wf_call(Cls, service, "test_method", "some text", session)

        assert len(captured) == 1
        input_json = json.loads(captured[0].input)
        assert "session" not in input_json
        assert input_json["input_text"] == "some text"

    async def test_kwargs_are_mapped_to_input_json(self, prompts_dir: str) -> None:
        """When the decorated method is called with keyword arguments, they
        should still appear in the input JSON sent to the agent service."""
        Cls = _make_workflow_class(prompts_dir)
        service = AiAgentServiceLocal.get_instance()

        captured: list[AgentRunConfig] = []
        original = service.create_and_run

        async def spy(config: AgentConfig[str], run_config: AgentRunConfig) -> AgentRunResult[str]:
            captured.append(run_config)
            return await original(config, run_config)

        service.create_and_run = spy  # type: ignore[assignment]

        wf = Cls(service)
        await wf.test_method(input_text="kwarg value")

        assert len(captured) == 1
        input_json = json.loads(captured[0].input)
        assert input_json["input_text"] == "kwarg value"

    async def test_passes_model_override(self, prompts_dir: str) -> None:
        Cls = _make_workflow_class(prompts_dir)
        service = AiAgentServiceLocal.get_instance()

        captured_configs: list[AgentConfig[str]] = []
        original = service.create_and_run

        async def spy(config: AgentConfig[str], run_config: AgentRunConfig) -> AgentRunResult[str]:
            captured_configs.append(config)
            return await original(config, run_config)

        service.create_and_run = spy  # type: ignore[assignment]

        wf = Cls(service)
        await wf.test_model_override_method("test")

        assert len(captured_configs) == 1
        assert captured_configs[0].model == "test-model-override"

    async def test_strict_json_schema_defaults_true_and_opt_out(self, prompts_dir: str) -> None:
        (Path(prompts_dir) / "strict_default.md").write_text("Strict default")
        (Path(prompts_dir) / "strict_off.md").write_text("Strict off")

        @agentic_workflow(prompts_directory=prompts_dir)
        class Wf:
            def __init__(self, service: AiAgentServiceLocal) -> None:
                self._ai_agent_service = service

            @agent(output_schema=dict)
            async def strict_default(self, input_text: str) -> AgentRunResult[dict]: ...

            @agent(output_schema=dict, strict_json_schema=False)
            async def strict_off(self, input_text: str) -> AgentRunResult[dict]: ...

        service = AiAgentServiceLocal.get_instance()
        captured: list[AgentConfig[dict]] = []
        original = service.create_and_run

        async def spy(
            config: AgentConfig[dict], run_config: AgentRunConfig
        ) -> AgentRunResult[dict]:
            captured.append(config)
            return await original(config, run_config)

        service.create_and_run = spy  # type: ignore[assignment]

        wf = Wf(service)
        await wf.strict_default("x")
        await wf.strict_off("x")

        assert captured[0].strict_json_schema is True
        assert captured[1].strict_json_schema is False


# ---------------------------------------------------------------------------
# @consensus_agent validation tests
# ---------------------------------------------------------------------------


class TestConsensusValidation:
    def test_rejects_even_runs(self) -> None:
        with pytest.raises(ValueError, match="runs must be odd"):

            @consensus_agent(
                runs=2,
                consensus_strategy=ConsensusStrategy.MAJORITY,
            )
            async def bad(self, x: str) -> None: ...

    def test_judge_strategy_requires_judge_fn(self) -> None:
        with pytest.raises(ValueError, match="judge function required"):

            @consensus_agent(
                runs=3,
                consensus_strategy=ConsensusStrategy.JUDGE,
            )
            async def bad(self, x: str) -> None: ...


# ---------------------------------------------------------------------------
# MAJORITY strategy tests
# ---------------------------------------------------------------------------


class TestMajorityStrategy:
    async def test_all_agree(self, prompts_dir: str) -> None:
        Cls = _make_workflow_class(prompts_dir)
        service = AiAgentServiceLocal.get_instance()
        AiAgentServiceLocal.set_responses("TestWorkflow:majority_method", ["same", "same", "same"])
        wf = Cls(service)

        result = await wf.majority_method("test")

        assert result.success is True
        assert result.output == "same"
        assert result.agreement == 1.0
        assert result.total_runs == 3
        assert result.successful_runs == 3

    async def test_two_of_three_agree(self, prompts_dir: str) -> None:
        Cls = _make_workflow_class(prompts_dir)
        service = AiAgentServiceLocal.get_instance()
        AiAgentServiceLocal.set_responses(
            "TestWorkflow:majority_method", ["winner", "winner", "loser"]
        )
        wf = Cls(service)

        result = await wf.majority_method("test")

        assert result.success is True
        assert result.output == "winner"
        assert abs(result.agreement - 2.0 / 3.0) < 0.01
        assert result.successful_runs == 3

    async def test_handles_failed_runs(self, prompts_dir: str) -> None:
        Cls = _make_workflow_class(prompts_dir)
        service = AiAgentServiceLocal.get_instance()
        # Only 2 responses queued; 3rd run gets last response repeated
        AiAgentServiceLocal.set_responses("TestWorkflow:majority_method", ["ok", "ok"])
        wf = Cls(service)

        result = await wf.majority_method("test")

        assert result.success is True
        assert result.output == "ok"

    async def test_all_runs_fail(self, prompts_dir: str) -> None:
        Cls = _make_workflow_class(prompts_dir)
        service = AiAgentServiceLocal.get_instance()
        AiAgentServiceLocal.set_error("TestWorkflow:majority_method", "boom")
        wf = Cls(service)

        result = await wf.majority_method("test")

        assert result.success is False
        assert result.error == "All runs failed"
        assert result.successful_runs == 0


# ---------------------------------------------------------------------------
# UNANIMOUS strategy tests
# ---------------------------------------------------------------------------


class TestUnanimousStrategy:
    async def test_all_agree(self, prompts_dir: str) -> None:
        Cls = _make_workflow_class(prompts_dir)
        service = AiAgentServiceLocal.get_instance()
        AiAgentServiceLocal.set_responses(
            "TestWorkflow:unanimous_method", ["agree", "agree", "agree"]
        )
        wf = Cls(service)

        result = await wf.unanimous_method("test")

        assert result.success is True
        assert result.output == "agree"
        assert result.agreement == 1.0

    async def test_disagree(self, prompts_dir: str) -> None:
        Cls = _make_workflow_class(prompts_dir)
        service = AiAgentServiceLocal.get_instance()
        AiAgentServiceLocal.set_responses("TestWorkflow:unanimous_method", ["a", "a", "b"])
        wf = Cls(service)

        result = await wf.unanimous_method("test")

        assert result.success is False
        assert "consensus not reached" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# JUDGE strategy tests
# ---------------------------------------------------------------------------


class TestJudgeStrategy:
    async def test_judge_selects_first(self, prompts_dir: str) -> None:
        Cls = _make_workflow_class(prompts_dir)
        service = AiAgentServiceLocal.get_instance()
        AiAgentServiceLocal.set_responses("TestWorkflow:judge_method", ["alpha", "beta", "gamma"])
        wf = Cls(service)

        result = await wf.judge_method("test")

        assert result.success is True
        assert result.output == "alpha"

    async def test_all_fail_before_judge(self, prompts_dir: str) -> None:
        Cls = _make_workflow_class(prompts_dir)
        service = AiAgentServiceLocal.get_instance()
        AiAgentServiceLocal.set_error("TestWorkflow:judge_method", "boom")
        wf = Cls(service)

        result = await wf.judge_method("test")

        assert result.success is False
        assert result.error == "All runs failed"


# ---------------------------------------------------------------------------
# Temperature spread test
# ---------------------------------------------------------------------------


class TestTemperatureSpread:
    async def test_temperature_varies_across_runs(self, prompts_dir: str) -> None:
        Cls = _make_workflow_class(prompts_dir)
        service = AiAgentServiceLocal.get_instance()

        captured_settings: list[dict[str, object] | None] = []
        original = service.create_and_run

        async def spy(config: AgentConfig[str], run_config: AgentRunConfig) -> AgentRunResult[str]:
            captured_settings.append(config.model_settings)
            return await original(config, run_config)

        service.create_and_run = spy  # type: ignore[assignment]

        AiAgentServiceLocal.set_responses("TestWorkflow:temp_spread_method", ["a", "a", "a"])
        wf = Cls(service)
        await wf.temp_spread_method("test")

        assert len(captured_settings) == 3
        temps: list[float] = [s["temperature"] for s in captured_settings if s]  # type: ignore[index]
        assert abs(temps[0] - 0.2) < 0.01
        assert abs(temps[1] - 0.6) < 0.01
        assert abs(temps[2] - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def wf_call(cls: type, service: AiAgentServiceLocal, method: str, *args: object) -> object:
    wf = cls(service)
    return await getattr(wf, method)(*args)
