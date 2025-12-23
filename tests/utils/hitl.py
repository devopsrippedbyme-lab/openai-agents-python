from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any, Callable, cast

from openai.types.responses import ResponseCustomToolCall, ResponseFunctionToolCall

from agents import Agent, Runner, RunResult
from agents._run_impl import NextStepInterruption, SingleStepResult
from agents.items import ToolApprovalItem, ToolCallOutputItem, TResponseOutputItem
from agents.run_context import RunContextWrapper
from agents.run_state import RunState as RunStateClass

from ..fake_model import FakeModel


async def run_and_resume(
    agent: Agent[Any],
    model: Any,
    raw_call: Any,
    *,
    user_input: str,
) -> RunResult:
    """Run once, then resume from the produced state."""
    model.set_next_output([raw_call])
    first = await Runner.run(agent, user_input)
    return await Runner.run(agent, first.to_state())


async def run_and_resume_after_approval(
    agent: Agent[Any],
    model: Any,
    raw_call: Any,
    final_output: Any,
    *,
    user_input: str,
) -> RunResult:
    """Run, approve the first interruption, and resume."""
    model.set_next_output([raw_call])
    first = await Runner.run(agent, user_input)
    assert first.interruptions, "expected an approval interruption"
    state = first.to_state()
    state.approve(first.interruptions[0], always_approve=True)
    model.set_next_output([final_output])
    return await Runner.run(agent, state)


def collect_tool_outputs(
    items: Iterable[Any],
    *,
    output_type: str,
) -> list[ToolCallOutputItem]:
    """Return ToolCallOutputItems matching a raw_item type."""
    return [
        item
        for item in items
        if isinstance(item, ToolCallOutputItem)
        and isinstance(item.raw_item, dict)
        and item.raw_item.get("type") == output_type
    ]


async def consume_stream(result: Any) -> None:
    """Drain all stream events to completion."""
    async for _ in result.stream_events():
        pass


def assert_single_approval_interruption(
    result: SingleStepResult,
    *,
    tool_name: str | None = None,
) -> ToolApprovalItem:
    """Assert the result contains exactly one approval interruption and return it."""
    assert isinstance(result.next_step, NextStepInterruption)
    assert len(result.next_step.interruptions) == 1
    interruption = result.next_step.interruptions[0]
    assert isinstance(interruption, ToolApprovalItem)
    if tool_name:
        assert interruption.tool_name == tool_name
    return interruption


async def require_approval(
    _ctx: Any | None = None, _params: Any = None, _call_id: str | None = None
) -> bool:
    """Approval helper that always requires a HITL decision."""
    return True


class RecordingEditor:
    """Editor that records operations for testing."""

    def __init__(self) -> None:
        self.operations: list[Any] = []

    def create_file(self, operation: Any) -> Any:
        self.operations.append(operation)
        return {"output": f"Created {operation.path}", "status": "completed"}

    def update_file(self, operation: Any) -> Any:
        self.operations.append(operation)
        return {"output": f"Updated {operation.path}", "status": "completed"}

    def delete_file(self, operation: Any) -> Any:
        self.operations.append(operation)
        return {"output": f"Deleted {operation.path}", "status": "completed"}


def make_shell_call(
    call_id: str,
    *,
    id_value: str | None = None,
    commands: list[str] | None = None,
    status: str = "in_progress",
) -> TResponseOutputItem:
    """Build a shell_call payload with optional overrides."""
    return cast(
        TResponseOutputItem,
        {
            "type": "shell_call",
            "id": id_value or call_id,
            "call_id": call_id,
            "status": status,
            "action": {"type": "exec", "commands": commands or ["echo test"], "timeout_ms": 1000},
        },
    )


def make_apply_patch_call(call_id: str, diff: str = "-a\n+b\n") -> ResponseCustomToolCall:
    """Create a ResponseCustomToolCall for apply_patch."""
    operation_json = json.dumps({"type": "update_file", "path": "test.md", "diff": diff})
    return ResponseCustomToolCall(
        type="custom_tool_call",
        name="apply_patch",
        call_id=call_id,
        input=operation_json,
    )


def make_apply_patch_dict(call_id: str, diff: str = "-a\n+b\n") -> TResponseOutputItem:
    """Create an apply_patch_call dict payload."""
    return cast(
        TResponseOutputItem,
        {
            "type": "apply_patch_call",
            "call_id": call_id,
            "operation": {"type": "update_file", "path": "test.md", "diff": diff},
        },
    )


def make_function_tool_call(
    name: str,
    *,
    call_id: str = "call-1",
    arguments: str = "{}",
) -> ResponseFunctionToolCall:
    """Create a ResponseFunctionToolCall for HITL scenarios."""
    return ResponseFunctionToolCall(
        type="function_call",
        name=name,
        call_id=call_id,
        arguments=arguments,
    )


def queue_function_call_and_text(
    model: FakeModel,
    function_call: TResponseOutputItem,
    *,
    first_turn_extra: Sequence[TResponseOutputItem] | None = None,
    followup: Sequence[TResponseOutputItem] | None = None,
) -> None:
    """Queue a function call turn followed by a follow-up turn on the fake model."""
    raw_type = (
        function_call.get("type")
        if isinstance(function_call, dict)
        else getattr(function_call, "type", None)
    )
    assert raw_type == "function_call", "queue_function_call_and_text expects a function call item"
    model.add_multiple_turn_outputs(
        [
            [function_call, *(first_turn_extra or [])],
            list(followup or []),
        ]
    )


async def run_and_resume_with_mutation(
    agent: Agent[Any],
    model: Any,
    turn_outputs: Sequence[Sequence[Any]],
    *,
    user_input: str,
    mutate_state: Callable[[RunStateClass[Any, Agent[Any]], ToolApprovalItem], None] | None = None,
) -> tuple[RunResult, RunResult]:
    """Run until interruption, optionally mutate state, then resume."""
    model.add_multiple_turn_outputs(turn_outputs)
    first = await Runner.run(agent, input=user_input)
    assert first.interruptions, "expected an approval interruption"
    state = first.to_state()
    if mutate_state and first.interruptions:
        mutate_state(state, first.interruptions[0])
    resumed = await Runner.run(agent, input=state)
    return first, resumed


async def assert_pending_resume(
    tool: Any,
    model: Any,
    raw_call: TResponseOutputItem,
    *,
    user_input: str,
    output_type: str,
) -> RunResult:
    """Run, resume, and assert pending approvals stay pending."""
    agent = make_agent(model=model, tools=[tool])

    resumed = await run_and_resume(agent, model, raw_call, user_input=user_input)

    assert resumed.interruptions, "pending approval should remain after resuming"
    assert any(
        isinstance(item, ToolApprovalItem) and item.tool_name == tool.name
        for item in resumed.interruptions
    )
    assert not collect_tool_outputs(resumed.new_items, output_type=output_type), (
        f"{output_type} should not execute without approval"
    )
    return resumed


def make_mcp_raw_item(
    *,
    call_id: str = "call_mcp_1",
    include_provider_data: bool = True,
    tool_name: str = "test_mcp_tool",
    provider_data: dict[str, Any] | None = None,
    include_name: bool = True,
    use_call_id: bool = True,
) -> dict[str, Any]:
    """Build a hosted MCP tool call payload for approvals."""

    raw_item: dict[str, Any] = {"type": "hosted_tool_call"}
    if include_name:
        raw_item["name"] = tool_name
    if include_provider_data:
        if use_call_id:
            raw_item["call_id"] = call_id
        else:
            raw_item["id"] = call_id
        raw_item["providerData"] = provider_data or {
            "type": "mcp_approval_request",
            "id": "req-1",
            "server_label": "test_server",
        }
    else:
        raw_item["id"] = call_id
    return raw_item


def make_mcp_approval_item(
    agent: Agent[Any],
    *,
    call_id: str = "call_mcp_1",
    include_provider_data: bool = True,
    tool_name: str | None = "test_mcp_tool",
    provider_data: dict[str, Any] | None = None,
    include_name: bool = True,
    use_call_id: bool = True,
) -> ToolApprovalItem:
    """Create a ToolApprovalItem for MCP or hosted tool calls."""

    raw_item = make_mcp_raw_item(
        call_id=call_id,
        include_provider_data=include_provider_data,
        tool_name=tool_name or "unknown_mcp_tool",
        provider_data=provider_data,
        include_name=include_name,
        use_call_id=use_call_id,
    )
    return ToolApprovalItem(agent=agent, raw_item=raw_item, tool_name=tool_name)


def make_context_wrapper() -> RunContextWrapper[dict[str, Any]]:
    """Create an empty RunContextWrapper for HITL tests."""
    return RunContextWrapper(context={})


def make_agent(
    *,
    model: Any | None = None,
    tools: Sequence[Any] | None = None,
    name: str = "TestAgent",
) -> Agent[Any]:
    """Build a test Agent with optional model and tools."""
    return Agent(name=name, model=model, tools=list(tools or []))


def make_model_and_agent(
    *,
    tools: Sequence[Any] | None = None,
    name: str = "TestAgent",
) -> tuple[FakeModel, Agent[Any]]:
    """Build a FakeModel with a paired Agent for HITL tests."""
    model = FakeModel()
    agent = make_agent(model=model, tools=tools, name=name)
    return model, agent
