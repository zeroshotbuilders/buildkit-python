from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .agent_service import (
    AgentConfig,
    AgentRunConfig,
    AgentRunResult,
    AiAgentService,
    ConsensusRunResult,
    ConsensusStrategy,
)
from .param_mapper import AgentParameterMapper
from .prompt_utils import generate_tools_reference, parse_prompt_frontmatter

logger = logging.getLogger(__name__)


def agentic_workflow(
    *,
    prompts_directory: str,
    tool_registry: dict[str, str] | None = None,
) -> Any:
    """Class decorator that registers prompt directory and tool registry."""

    def decorator(cls: type) -> type:
        cls._agentic_workflow_options = {  # type: ignore[attr-defined]
            "prompts_directory": prompts_directory,
            "tool_registry": tool_registry,
        }
        return cls

    return decorator


def agent(
    *,
    name: str | None = None,
    tools: list[Any] | Callable[..., list[Any]] | None = None,
    model: str | None = None,
    model_settings: dict[str, Any] | None = None,
    output_schema: Any | None = None,
    max_turns: int | None = None,
    branch_param: str | None = None,
    strict_json_schema: bool = True,
) -> Any:
    """Method decorator that turns a method into an agent invocation.

    The decorated method's body is replaced. Its parameters are mapped
    to the agent input JSON, and the prompt is loaded from a markdown file
    in the class's ``prompts_directory``.
    """

    def decorator(fn: Any) -> Any:
        mapper = AgentParameterMapper.from_function(fn)

        @functools.wraps(fn)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> AgentRunResult[Any]:
            ai_service: AiAgentService = self._ai_agent_service
            options = getattr(self.__class__, "_agentic_workflow_options", {})
            prompts_dir = options.get("prompts_directory", "")

            # Load prompt from markdown file
            prompt_path = Path(prompts_dir) / f"{fn.__name__}.md"
            if prompt_path.exists():
                raw_prompt = prompt_path.read_text()
            else:
                raw_prompt = ""

            parsed = parse_prompt_frontmatter(raw_prompt)
            instructions = parsed.content

            # Resolve tools
            resolved_tools = tools
            if callable(resolved_tools):
                resolved_tools = resolved_tools(self)
            resolved_tools = resolved_tools or []

            # Generate tools reference and prepend
            tools_ref = generate_tools_reference(resolved_tools)
            if tools_ref:
                instructions = tools_ref + "\n" + instructions

            # Map parameters — merge kwargs into positional order
            if kwargs:
                import inspect as _inspect

                sig = _inspect.signature(fn)
                params = [p for p in sig.parameters if p != "self"]
                all_args = tuple(
                    kwargs.get(p, args[i] if i < len(args) else None) for i, p in enumerate(params)
                )
            else:
                all_args = args
            mapped = mapper.map_arguments(all_args)
            session = mapper.find_session(all_args)

            # Determine agent name
            agent_name = name or f"{self.__class__.__name__}:{fn.__name__}"

            # Get branch if specified
            branch = None
            if branch_param:
                branch = mapper.get_param_value(branch_param, all_args)

            # Determine max_turns
            effective_max_turns = max_turns
            if effective_max_turns is None:
                effective_max_turns = 8 if resolved_tools else 1

            config = AgentConfig(
                name=agent_name,
                instructions=instructions,
                model=model,
                tools=resolved_tools,
                output_schema=output_schema,
                model_settings=model_settings,
                strict_json_schema=strict_json_schema,
            )

            run_config = AgentRunConfig(
                input=mapped.input,
                context=mapped.context,
                session=session,
                max_turns=effective_max_turns,
                branch=branch,
            )

            start = time.monotonic()
            result = await ai_service.create_and_run(config, run_config)
            elapsed = time.monotonic() - start
            logger.debug(
                "Agent %s completed in %.2fs (success=%s)",
                agent_name,
                elapsed,
                result.success,
            )

            return result

        return wrapper

    return decorator


def consensus_agent(
    *,
    name: str | None = None,
    tools: list[Any] | Callable[..., list[Any]] | None = None,
    model: str | None = None,
    model_settings: dict[str, Any] | None = None,
    output_schema: Any | None = None,
    max_turns: int | None = None,
    runs: int,
    consensus_strategy: ConsensusStrategy,
    judge: Callable[..., Any] | None = None,
    temperature_spread: tuple[float, float] | None = None,
    strict_json_schema: bool = True,
) -> Any:
    """Method decorator for consensus-based multi-run agent invocation."""

    if runs % 2 == 0:
        raise ValueError("runs must be odd")
    if consensus_strategy == ConsensusStrategy.JUDGE and judge is None:
        raise ValueError("judge function required for JUDGE strategy")

    def decorator(fn: Any) -> Any:
        mapper = AgentParameterMapper.from_function(fn)

        @functools.wraps(fn)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> ConsensusRunResult[Any]:
            ai_service: AiAgentService = self._ai_agent_service
            options = getattr(self.__class__, "_agentic_workflow_options", {})
            prompts_dir = options.get("prompts_directory", "")

            prompt_path = Path(prompts_dir) / f"{fn.__name__}.md"
            raw_prompt = prompt_path.read_text() if prompt_path.exists() else ""
            parsed = parse_prompt_frontmatter(raw_prompt)
            instructions = parsed.content

            resolved_tools = tools
            if callable(resolved_tools):
                resolved_tools = resolved_tools(self)
            resolved_tools = resolved_tools or []

            tools_ref = generate_tools_reference(resolved_tools)
            if tools_ref:
                instructions = tools_ref + "\n" + instructions

            # Map parameters — merge kwargs into positional order
            if kwargs:
                import inspect as _inspect

                sig = _inspect.signature(fn)
                params = [p for p in sig.parameters if p != "self"]
                all_args = tuple(
                    kwargs.get(p, args[i] if i < len(args) else None) for i, p in enumerate(params)
                )
            else:
                all_args = args
            mapped = mapper.map_arguments(all_args)
            session = mapper.find_session(all_args)
            agent_name = name or f"{self.__class__.__name__}:{fn.__name__}"
            effective_max_turns = max_turns or (8 if resolved_tools else 1)

            # Build configs for each run
            async def single_run(run_index: int) -> AgentRunResult[Any]:
                ms = dict(model_settings or {})
                if temperature_spread:
                    lo, hi = temperature_spread
                    t = lo + (hi - lo) * run_index / max(runs - 1, 1)
                    ms["temperature"] = t

                config = AgentConfig(
                    name=agent_name,
                    instructions=instructions,
                    model=model,
                    tools=resolved_tools,
                    output_schema=output_schema,
                    model_settings=ms or None,
                    strict_json_schema=strict_json_schema,
                )
                run_config = AgentRunConfig(
                    input=mapped.input,
                    context=mapped.context,
                    session=session,
                    max_turns=effective_max_turns,
                )
                return await ai_service.create_and_run(config, run_config)

            all_results = await asyncio.gather(*(single_run(i) for i in range(runs)))
            all_results_list = list(all_results)

            return await _resolve_consensus(
                all_results_list,
                consensus_strategy,
                runs,
                judge_fn=judge,
                instance=self,
            )

        return wrapper

    return decorator


async def _resolve_consensus(
    all_results: list[AgentRunResult[Any]],
    strategy: ConsensusStrategy,
    total_runs: int,
    judge_fn: Callable[..., Any] | None = None,
    instance: Any = None,
) -> ConsensusRunResult[Any]:
    successful = [r for r in all_results if r.success]

    if not successful:
        return ConsensusRunResult(
            output=None,  # type: ignore[arg-type]
            success=False,
            error="All runs failed",
            runs=all_results,
            agreement=0.0,
            total_runs=total_runs,
            successful_runs=0,
        )

    if strategy == ConsensusStrategy.MAJORITY:
        serialized = [json.dumps(r.output, sort_keys=True, default=str) for r in successful]
        counts = Counter(serialized)
        winner_key, winner_count = counts.most_common(1)[0]
        winner_result = next(
            r for r, s in zip(successful, serialized, strict=False) if s == winner_key
        )
        return ConsensusRunResult(
            output=winner_result.output,
            success=True,
            raw_result=winner_result.raw_result,
            runs=all_results,
            agreement=winner_count / len(successful),
            total_runs=total_runs,
            successful_runs=len(successful),
        )

    if strategy == ConsensusStrategy.UNANIMOUS:
        serialized = [json.dumps(r.output, sort_keys=True, default=str) for r in successful]
        if len(set(serialized)) != 1:
            return ConsensusRunResult(
                output=None,  # type: ignore[arg-type]
                success=False,
                error="Unanimous consensus not reached",
                runs=all_results,
                agreement=0.0,
                total_runs=total_runs,
                successful_runs=len(successful),
            )
        return ConsensusRunResult(
            output=successful[0].output,
            success=True,
            raw_result=successful[0].raw_result,
            runs=all_results,
            agreement=1.0,
            total_runs=total_runs,
            successful_runs=len(successful),
        )

    if strategy == ConsensusStrategy.JUDGE:
        assert judge_fn is not None
        judge_result = await judge_fn(instance, successful)
        return ConsensusRunResult(
            output=judge_result.output,
            success=judge_result.success,
            runs=all_results,
            agreement=0.0,
            total_runs=total_runs,
            successful_runs=len(successful),
        )

    raise ValueError(f"Unknown consensus strategy: {strategy}")
