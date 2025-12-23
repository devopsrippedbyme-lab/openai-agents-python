from __future__ import annotations

import pytest

from agents import Agent, Runner

from ..fake_model import FakeModel
from ..test_responses import get_function_tool_call, get_text_message
from ..utils.hitl import queue_function_call_and_text
from .helpers import FakeMCPServer


@pytest.mark.asyncio
async def test_mcp_require_approval_pauses_and_resumes():
    """MCP servers should honor require_approval for non-hosted tools."""

    server = FakeMCPServer(require_approval="always")
    server.add_tool("add", {"type": "object", "properties": {}})

    model = FakeModel()
    agent = Agent(name="TestAgent", model=model, mcp_servers=[server])

    queue_function_call_and_text(
        model,
        get_function_tool_call("add", "{}"),
        followup=[get_text_message("done")],
    )

    first = await Runner.run(agent, "call add")

    assert first.interruptions, "MCP tool should request approval"
    assert first.interruptions[0].tool_name == "add"

    state = first.to_state()
    state.approve(first.interruptions[0], always_approve=True)

    resumed = await Runner.run(agent, state)

    assert not resumed.interruptions
    assert server.tool_calls == ["add"]
    assert resumed.final_output == "done"


@pytest.mark.asyncio
async def test_mcp_require_approval_tool_lists():
    """TS-style requireApproval toolNames should map to needs_approval."""

    require_approval: dict[str, object] = {
        "always": {"tool_names": ["add"]},
        "never": {"tool_names": ["noop"]},
    }
    server = FakeMCPServer(require_approval=require_approval)
    server.add_tool("add", {"type": "object", "properties": {}})

    model = FakeModel()
    agent = Agent(name="TestAgent", model=model, mcp_servers=[server])

    queue_function_call_and_text(
        model,
        get_function_tool_call("add", "{}"),
        followup=[get_text_message("done")],
    )

    first = await Runner.run(agent, "call add")
    assert first.interruptions, "add should require approval via require_approval toolNames"

    state = first.to_state()
    state.approve(first.interruptions[0], always_approve=True)

    resumed = await Runner.run(agent, state)
    assert resumed.final_output == "done"
    assert server.tool_calls == ["add"]
