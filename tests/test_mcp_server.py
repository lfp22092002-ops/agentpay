"""
Tests for MCP server — tool registry, JSON-RPC protocol, and error handling.
"""
import json
import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _mock_sdk(monkeypatch):
    """Prevent real SDK imports from failing."""
    monkeypatch.setattr("mcp.server._sdk_available", False)


def _load_tool_json():
    path = os.path.join(os.path.dirname(__file__), "..", "mcp", "agentpay_tool.json")
    with open(path) as f:
        return json.load(f)


class TestToolRegistry:
    """Verify tool definitions match handler registry."""

    def test_tool_json_names_match_handlers(self):
        from mcp.server import TOOL_HANDLERS

        tool_data = _load_tool_json()
        json_names = {t["name"] for t in tool_data["tools"]}
        handler_names = set(TOOL_HANDLERS.keys())
        assert json_names == handler_names, (
            f"Mismatch — JSON-only: {json_names - handler_names}, "
            f"Handler-only: {handler_names - json_names}"
        )

    def test_all_tools_have_descriptions(self):
        tool_data = _load_tool_json()
        for tool in tool_data["tools"]:
            assert tool.get("description"), f"Tool {tool['name']} missing description"

    def test_all_tools_have_input_schema(self):
        tool_data = _load_tool_json()
        for tool in tool_data["tools"]:
            schema = tool.get("inputSchema", tool.get("input_schema"))
            assert schema is not None, f"Tool {tool['name']} missing input schema"
            assert schema.get("type") == "object", f"Tool {tool['name']} schema type must be 'object'"

    def test_handler_count(self):
        from mcp.server import TOOL_HANDLERS

        assert len(TOOL_HANDLERS) == 10, f"Expected 10 tools, got {len(TOOL_HANDLERS)}"

    def test_handlers_are_callable(self):
        from mcp.server import TOOL_HANDLERS

        for name, handler in TOOL_HANDLERS.items():
            assert callable(handler), f"Handler for {name} is not callable"


class TestToolHandlers:
    """Test individual handlers raise RuntimeError without SDK."""

    def test_balance_requires_sdk(self):
        from mcp.server import handle_balance

        with pytest.raises(RuntimeError, match="SDK not installed"):
            handle_balance({})

    def test_spend_requires_sdk(self):
        from mcp.server import handle_spend

        with pytest.raises(RuntimeError, match="SDK not installed"):
            handle_spend({"amount": 1.0, "description": "test"})

    def test_chains_requires_sdk(self):
        from mcp.server import handle_chains

        with pytest.raises(RuntimeError, match="SDK not installed"):
            handle_chains({})
