"""Regression tests for HITL edge cases."""

from __future__ import annotations

from typing import Any, Callable, cast

import pytest
from openai.types.responses.response_input_param import (
    ComputerCallOutput,
    LocalShellCallOutput,
)
from openai.types.responses.response_output_item import LocalShellCall, McpApprovalRequest

from agents import (
    Agent,
    ApplyPatchTool,
    LocalShellTool,
    Runner,
    RunResult,
    RunState,
    ShellTool,
    ToolApprovalItem,
    function_tool,
)
from agents._run_impl import (
    NextStepInterruption,
    ProcessedResponse,
    RunImpl,
    ToolRunMCPApprovalRequest,
    ToolRunShellCall,
)
from agents.exceptions import ModelBehaviorError
from agents.items import (
    MCPApprovalResponseItem,
    MessageOutputItem,
    ModelResponse,
    ToolCallOutputItem,
    TResponseOutputItem,
)
from agents.lifecycle import RunHooks
from agents.run import RunConfig
from agents.run_state import RunState as RunStateClass
from agents.usage import Usage

from .fake_model import FakeModel
from .test_responses import get_text_message
from .utils.hitl import (
    RecordingEditor,
    assert_pending_resume,
    collect_tool_outputs,
    make_agent,
    make_apply_patch_call,
    make_apply_patch_dict,
    make_context_wrapper,
    make_function_tool_call,
    make_mcp_approval_item,
    make_model_and_agent,
    make_shell_call,
    queue_function_call_and_text,
    require_approval,
    run_and_resume_after_approval,
)

ApprovalSetup = tuple[Any, TResponseOutputItem, TResponseOutputItem, Callable[[RunResult], None]]
PendingSetup = tuple[Any, TResponseOutputItem, Callable[[RunResult], None] | None]


async def _roundtrip_interruption_via_run(
    agent: Agent[Any],
    model: FakeModel,
    raw_call: Any,
    user_input: str = "test",
) -> list[ToolApprovalItem]:
    """Run once with a tool call, serialize state, and deserialize it."""
    model.set_next_output([raw_call])
    result = await Runner.run(agent, user_input)
    assert result.interruptions, "expected an interruption"
    state = result.to_state()
    deserialized_state = await RunStateClass.from_json(agent, state.to_json())
    return deserialized_state.get_interruptions()


async def assert_roundtrip_tool_name(
    agent: Agent[Any],
    model: FakeModel,
    raw_call: TResponseOutputItem,
    expected_tool_name: str,
    *,
    user_input: str,
) -> None:
    """Assert that deserialized interruptions keep the tool name intact."""
    interruptions = await _roundtrip_interruption_via_run(agent, model, raw_call, user_input)
    assert interruptions, "Interruptions should be preserved after deserialization"
    assert interruptions[0].tool_name == expected_tool_name, (
        f"{expected_tool_name} tool approval should be preserved, not converted to function"
    )


def make_state_with_interruptions(
    agent: Agent[Any], interruptions: list[ToolApprovalItem]
) -> RunStateClass[Any]:
    """Create a RunState primed with interruptions."""
    context = make_context_wrapper()
    state = RunState(
        context=context,
        original_input="test",
        starting_agent=agent,
        max_turns=10,
    )
    state._current_step = NextStepInterruption(interruptions=interruptions)
    return state


async def assert_tool_output_roundtrip(
    agent: Agent[Any], raw_output: Any, expected_type: str, *, output: Any = "command output"
) -> None:
    """Ensure tool outputs keep their type through serialization and deserialization."""
    context = make_context_wrapper()
    state = RunState(context=context, original_input="test", starting_agent=agent, max_turns=3)
    state._generated_items = [
        ToolCallOutputItem(
            agent=agent,
            raw_item=raw_output,
            output=output,
        )
    ]

    json_data = state.to_json()

    generated_items_json = json_data.get("generatedItems", [])
    assert len(generated_items_json) == 1, f"{expected_type} item should be serialized"
    serialized_type = generated_items_json[0].get("rawItem", {}).get("type")

    assert serialized_type == expected_type, (
        f"Expected {expected_type} in serialized JSON, but got {serialized_type}. "
        "Serialization should not coerce tool outputs."
    )

    deserialized_state = await RunStateClass.from_json(agent, json_data)

    assert len(deserialized_state._generated_items) == 1, (
        f"{expected_type} item should be deserialized."
    )
    deserialized_item = deserialized_state._generated_items[0]
    assert isinstance(deserialized_item, ToolCallOutputItem)

    raw_item = deserialized_item.raw_item
    output_type = raw_item.get("type") if isinstance(raw_item, dict) else raw_item.type

    assert output_type == expected_type, (
        f"Expected {expected_type}, but got {output_type}. "
        "Serialization should preserve the tool output type."
    )


def _shell_approval_setup() -> ApprovalSetup:
    tool = ShellTool(executor=lambda request: "shell_output", needs_approval=require_approval)
    shell_call = make_shell_call("call_shell_1", id_value="shell_1", commands=["echo test"])

    def _assert(result: RunResult) -> None:
        shell_outputs = collect_tool_outputs(result.new_items, output_type="shell_call_output")
        assert shell_outputs, "Shell tool should have been executed after approval"
        assert any("shell_output" in str(item.output) for item in shell_outputs)

    return tool, shell_call, get_text_message("done"), _assert


def _apply_patch_approval_setup() -> ApprovalSetup:
    editor = RecordingEditor()
    tool = ApplyPatchTool(editor=editor, needs_approval=require_approval)
    apply_patch_call = make_apply_patch_call("call_apply_1")

    def _assert(result: RunResult) -> None:
        apply_patch_outputs = collect_tool_outputs(
            result.new_items, output_type="apply_patch_call_output"
        )
        assert apply_patch_outputs, "ApplyPatch tool should have been executed after approval"
        assert editor.operations, "Editor should have been called"

    return tool, apply_patch_call, get_text_message("done"), _assert


@pytest.mark.parametrize(
    "setup_fn, user_input",
    [
        (_shell_approval_setup, "run shell command"),
        (_apply_patch_approval_setup, "update file"),
    ],
    ids=["shell_approved", "apply_patch_approved"],
)
@pytest.mark.asyncio
async def test_resumed_hitl_executes_approved_tools(
    setup_fn: Callable[[], ApprovalSetup],
    user_input: str,
) -> None:
    """Approved tools should run once the interrupted turn resumes."""
    tool, raw_call, final_output, extra_assert = setup_fn()
    model, agent = make_model_and_agent(tools=[tool])

    result = await run_and_resume_after_approval(
        agent,
        model,
        raw_call,
        final_output,
        user_input=user_input,
    )

    extra_assert(result)


@pytest.mark.parametrize(
    "tool_kind", ["shell", "apply_patch"], ids=["shell_auto", "apply_patch_auto"]
)
@pytest.mark.asyncio
async def test_resuming_skips_approvals_for_non_hitl_tools(tool_kind: str) -> None:
    """Auto-approved tools should not trigger new approvals when resuming a turn."""
    shell_runs: list[str] = []
    editor: RecordingEditor | None = None
    auto_tool: ShellTool | ApplyPatchTool

    if tool_kind == "shell":

        def _executor(_req: Any) -> str:
            shell_runs.append("run")
            return "shell_output"

        auto_tool = ShellTool(executor=_executor)
        raw_call = make_shell_call("call_shell_auto", id_value="shell_auto", commands=["echo auto"])
        output_type = "shell_call_output"
    else:
        editor = RecordingEditor()
        auto_tool = ApplyPatchTool(editor=editor)
        raw_call = make_apply_patch_call("call_apply_auto")
        output_type = "apply_patch_call_output"

    async def needs_hitl() -> str:
        return "approved"

    approval_tool = function_tool(needs_hitl, needs_approval=require_approval)
    model, agent = make_model_and_agent(tools=[auto_tool, approval_tool])

    function_call = make_function_tool_call(approval_tool.name, call_id="call-func-auto")

    queue_function_call_and_text(
        model,
        function_call,
        first_turn_extra=[raw_call],
        followup=[get_text_message("done")],
    )

    first = await Runner.run(agent, "resume approvals")
    assert first.interruptions, "function tool should require approval"

    state = first.to_state()
    state.approve(first.interruptions[0], always_approve=True)

    resumed = await Runner.run(agent, state)

    assert not resumed.interruptions, "non-HITL tools should not request approval on resume"

    outputs = collect_tool_outputs(resumed.new_items, output_type=output_type)
    assert len(outputs) == 1, f"{tool_kind} should run exactly once without extra approvals"

    if tool_kind == "shell":
        assert len(shell_runs) == 1, "shell should execute automatically when resuming"
    else:
        assert editor is not None
        assert len(editor.operations) == 1, "apply_patch should execute once when resuming"


def _apply_patch_pending_setup() -> PendingSetup:
    editor = RecordingEditor()
    apply_patch_tool = ApplyPatchTool(editor=editor, needs_approval=True)

    def _assert_editor(_resumed: RunResult) -> None:
        assert editor.operations == [], "editor should not run before approval"

    return apply_patch_tool, make_apply_patch_call("call_apply_pending"), _assert_editor


@pytest.mark.parametrize(
    "setup_fn, output_type",
    [
        (
            lambda: (
                ShellTool(executor=lambda _req: "shell_output", needs_approval=True),
                make_shell_call(
                    "call_shell_pending", id_value="shell_pending", commands=["echo pending"]
                ),
                None,
            ),
            "shell_call_output",
        ),
        (_apply_patch_pending_setup, "apply_patch_call_output"),
    ],
    ids=["shell_pending", "apply_patch_pending"],
)
@pytest.mark.asyncio
async def test_pending_approvals_stay_pending_on_resume(
    setup_fn: Callable[[], PendingSetup],
    output_type: str,
) -> None:
    """Unapproved tool calls should remain pending after resuming a run."""
    tool, raw_call, extra_assert = setup_fn()
    model, _ = make_model_and_agent()

    resumed = await assert_pending_resume(
        tool,
        model,
        raw_call,
        user_input="resume pending approval",
        output_type=output_type,
    )

    if extra_assert:
        extra_assert(resumed)


@pytest.mark.asyncio
async def test_resuming_pending_mcp_approvals_raises_typeerror():
    """ToolApprovalItem must be hashable so pending MCP approvals can be tracked in a set."""
    _, agent = make_model_and_agent(tools=[])

    mcp_approval_item = make_mcp_approval_item(
        agent, call_id="mcp-approval-1", include_provider_data=False
    )

    pending_hosted_mcp_approvals: set[ToolApprovalItem] = set()
    pending_hosted_mcp_approvals.add(mcp_approval_item)
    assert mcp_approval_item in pending_hosted_mcp_approvals


@pytest.mark.asyncio
async def test_route_local_shell_calls_to_remote_shell_tool():
    """Test that local shell calls are routed to the local shell tool.

    When processing model output with LocalShellCall items, they should be handled by
    LocalShellTool (not ShellTool), even when both tools are registered. This ensures
    local shell operations use the correct executor and approval hooks.
    """
    remote_shell_executed = []
    local_shell_executed = []

    def remote_executor(request: Any) -> str:
        remote_shell_executed.append(request)
        return "remote_output"

    def local_executor(request: Any) -> str:
        local_shell_executed.append(request)
        return "local_output"

    shell_tool = ShellTool(executor=remote_executor)
    local_shell_tool = LocalShellTool(executor=local_executor)
    model, agent = make_model_and_agent(tools=[shell_tool, local_shell_tool])

    # Model emits a local_shell_call
    local_shell_call = LocalShellCall(
        id="local_1",
        call_id="call_local_1",
        type="local_shell_call",
        action={"type": "exec", "command": ["echo", "test"], "env": {}},  # type: ignore[arg-type]
        status="in_progress",
    )
    model.set_next_output([local_shell_call])

    await Runner.run(agent, "run local shell")

    # Local shell call should be handled by LocalShellTool, not ShellTool
    # This test will fail because LocalShellCall is routed to shell_tool first
    assert len(local_shell_executed) > 0, "LocalShellTool should have been executed"
    assert len(remote_shell_executed) == 0, (
        "ShellTool should not have been executed for local shell call"
    )


@pytest.mark.asyncio
async def test_preserve_max_turns_when_resuming_from_runresult_state():
    """Test that max_turns is preserved when resuming from RunResult state.

    A run configured with max_turns=20 should keep that limit after resuming from
    result.to_state() without re-passing max_turns.
    """

    async def test_tool() -> str:
        return "tool_result"

    # Create the tool with needs_approval directly
    # The tool name will be "test_tool" based on the function name
    tool = function_tool(test_tool, needs_approval=require_approval)
    model, agent = make_model_and_agent(tools=[tool])

    model.add_multiple_turn_outputs([[make_function_tool_call("test_tool", call_id="call-1")]])

    result1 = await Runner.run(agent, "call test_tool", max_turns=20)
    assert result1.interruptions, "should have an interruption"

    state = result1.to_state()
    state.approve(result1.interruptions[0], always_approve=True)

    # Provide 10 more turns (turns 2-11) to ensure we exceed the default 10 but not 20.
    model.add_multiple_turn_outputs(
        [
            [
                get_text_message(f"turn {i + 2}"),  # Text message first (doesn't finish)
                make_function_tool_call("test_tool", call_id=f"call-{i + 2}"),
            ]
            for i in range(10)
        ]
    )

    result2 = await Runner.run(agent, state)
    assert result2 is not None, "Run should complete successfully with max_turns=20 from state"


@pytest.mark.asyncio
async def test_current_turn_not_preserved_in_to_state():
    """Test that current turn counter is preserved when converting RunResult to RunState."""

    async def test_tool() -> str:
        return "tool_result"

    tool = function_tool(test_tool, needs_approval=require_approval)
    model, agent = make_model_and_agent(tools=[tool])

    # Model emits a tool call requiring approval
    model.set_next_output([make_function_tool_call("test_tool", call_id="call-1")])

    # First turn with interruption
    result1 = await Runner.run(agent, "call test_tool")
    assert result1.interruptions, "should have interruption on turn 1"

    # Convert to state - this should preserve current_turn=1
    state1 = result1.to_state()

    # Regression guard: to_state should keep the turn counter instead of resetting it.
    assert state1._current_turn == 1, (
        f"Expected current_turn=1 after 1 turn, got {state1._current_turn}. "
        "to_state() should preserve the current turn counter."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_factory, raw_call_factory, expected_tool_name, user_input",
    [
        (
            lambda: ShellTool(executor=lambda request: "output", needs_approval=require_approval),
            lambda: make_shell_call("call_shell_1", id_value="shell_1", commands=["echo test"]),
            "shell",
            "run shell",
        ),
        (
            lambda: ApplyPatchTool(editor=RecordingEditor(), needs_approval=require_approval),
            lambda: cast(Any, make_apply_patch_dict("call_apply_1")),
            "apply_patch",
            "update file",
        ),
    ],
    ids=["shell", "apply_patch"],
)
@pytest.mark.asyncio
async def test_deserialize_interruptions_preserve_tool_calls(
    tool_factory: Callable[[], Any],
    raw_call_factory: Callable[[], TResponseOutputItem],
    expected_tool_name: str,
    user_input: str,
) -> None:
    """Ensure deserialized interruptions preserve tool types instead of forcing function calls."""
    model, agent = make_model_and_agent(tools=[tool_factory()])
    await assert_roundtrip_tool_name(
        agent, model, raw_call_factory(), expected_tool_name, user_input=user_input
    )


@pytest.mark.parametrize("include_provider_data", [True, False])
@pytest.mark.asyncio
async def test_deserialize_interruptions_preserve_mcp_tools(
    include_provider_data: bool,
) -> None:
    """Ensure MCP/hosted tool approvals survive serialization."""
    model, agent = make_model_and_agent(tools=[])

    mcp_approval_item = make_mcp_approval_item(
        agent, call_id="mcp-approval-1", include_provider_data=include_provider_data
    )
    state = make_state_with_interruptions(agent, [mcp_approval_item])

    state_json = state.to_json()

    deserialized_state = await RunStateClass.from_json(agent, state_json)
    interruptions = deserialized_state.get_interruptions()
    assert len(interruptions) > 0, "Interruptions should be preserved after deserialization"
    assert interruptions[0].tool_name == "test_mcp_tool", (
        "MCP tool approval should be preserved, not converted to function"
    )


@pytest.mark.asyncio
async def test_hosted_mcp_approval_matches_unknown_tool_key() -> None:
    """Approved hosted MCP interruptions should resume even when the tool name is missing."""
    agent = make_agent()
    context_wrapper = make_context_wrapper()

    approval_item = make_mcp_approval_item(
        agent,
        call_id="mcp-123",
        provider_data={"type": "mcp_approval_request"},
        tool_name=None,
        include_name=False,
        use_call_id=False,
    )
    context_wrapper.approve_tool(approval_item)

    class DummyMcpTool:
        on_approval_request: Any = None

    processed_response = ProcessedResponse(
        new_items=[],
        handoffs=[],
        functions=[],
        computer_actions=[],
        local_shell_calls=[],
        shell_calls=[],
        apply_patch_calls=[],
        tools_used=[],
        mcp_approval_requests=[
            ToolRunMCPApprovalRequest(
                request_item=McpApprovalRequest(
                    id="mcp-123",
                    type="mcp_approval_request",
                    server_label="test_server",
                    arguments="{}",
                    name="hosted_mcp",
                ),
                mcp_tool=cast(Any, DummyMcpTool()),
            )
        ],
        interruptions=[],
    )

    result = await RunImpl.resolve_interrupted_turn(
        agent=agent,
        original_input="test",
        original_pre_step_items=[approval_item],
        new_response=ModelResponse(output=[], usage=Usage(), response_id="resp"),
        processed_response=processed_response,
        hooks=RunHooks(),
        context_wrapper=context_wrapper,
        run_config=RunConfig(),
        run_state=None,
    )

    assert any(
        isinstance(item, MCPApprovalResponseItem) and item.raw_item.get("approve") is True
        for item in result.new_step_items
    ), "Approved hosted MCP call should emit an approval response"


@pytest.mark.asyncio
async def test_shell_call_without_call_id_raises() -> None:
    """Shell calls missing call_id should raise ModelBehaviorError instead of being skipped."""
    agent = make_agent()
    context_wrapper = make_context_wrapper()
    shell_tool = ShellTool(executor=lambda _request: "")
    shell_call = {"type": "shell_call", "action": {"commands": ["echo", "hi"]}}

    processed_response = ProcessedResponse(
        new_items=[],
        handoffs=[],
        functions=[],
        computer_actions=[],
        local_shell_calls=[],
        shell_calls=[ToolRunShellCall(tool_call=shell_call, shell_tool=shell_tool)],
        apply_patch_calls=[],
        tools_used=[],
        mcp_approval_requests=[],
        interruptions=[],
    )

    with pytest.raises(ModelBehaviorError):
        await RunImpl.resolve_interrupted_turn(
            agent=agent,
            original_input="test",
            original_pre_step_items=[],
            new_response=ModelResponse(output=[], usage=Usage(), response_id="resp"),
            processed_response=processed_response,
            hooks=RunHooks(),
            context_wrapper=context_wrapper,
            run_config=RunConfig(),
            run_state=None,
        )


@pytest.mark.asyncio
async def test_preserve_persisted_item_counter_when_resuming_streamed_runs():
    """Preserve the persisted-item counter on streamed resume to avoid losing history."""
    model, agent = make_model_and_agent()

    # Simulate a turn interrupted mid-persistence: 5 items generated, 3 actually saved.
    context_wrapper = make_context_wrapper()
    state = RunState(
        context=context_wrapper,
        original_input="test input",
        starting_agent=agent,
        max_turns=10,
    )

    # Create 5 generated items (simulating multiple outputs before interruption)
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    for i in range(5):
        message_item = MessageOutputItem(
            agent=agent,
            raw_item=ResponseOutputMessage(
                id=f"msg_{i}",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(
                        type="output_text", text=f"Message {i}", annotations=[], logprobs=[]
                    )
                ],
            ),
        )
        state._generated_items.append(message_item)

    # Persisted count reflects what was already written before interruption.
    state._current_turn_persisted_item_count = 3

    # Add a model response so the state is valid for resumption
    state._model_responses = [
        ModelResponse(
            output=[get_text_message("test")],
            usage=Usage(),
            response_id="resp_1",
        )
    ]

    # Set up model to return final output immediately (so the run completes)
    model.set_next_output([get_text_message("done")])

    result = Runner.run_streamed(agent, state)

    assert result._current_turn_persisted_item_count == 3, (
        f"Expected _current_turn_persisted_item_count=3 (the actual persisted count), "
        f"but got {result._current_turn_persisted_item_count}. "
        f"The counter should reflect persisted items, not len(_generated_items)="
        f"{len(state._generated_items)}."
    )

    # Consume events to complete the run
    async for _ in result.stream_events():
        pass


@pytest.mark.asyncio
async def test_preserve_tool_output_types_during_serialization():
    """Keep tool output types intact during RunState serialization/deserialization."""

    model, agent = make_model_and_agent(tools=[])

    computer_output: ComputerCallOutput = {
        "type": "computer_call_output",
        "call_id": "call_computer_1",
        "output": {"type": "computer_screenshot", "image_url": "base64_screenshot_data"},
    }
    await assert_tool_output_roundtrip(
        agent, computer_output, "computer_call_output", output="screenshot_data"
    )

    # TypedDict requires "id", but runtime objects use "call_id"; cast to align with runtime shape.
    shell_output = cast(
        LocalShellCallOutput,
        {
            "type": "local_shell_call_output",
            "id": "shell_1",
            "call_id": "call_shell_1",
            "output": "command output",
        },
    )
    await assert_tool_output_roundtrip(agent, shell_output, "local_shell_call_output")
