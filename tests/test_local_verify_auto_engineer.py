"""Tests for local_verify wiring into ClaudeAutoEngineer (Phase 2 — agent-drive
support for reverse-api-engineer's own standalone CLI users; route-reveal
itself never calls into this, since its own agent-drive is always
Browserbase-cloud-driven). Mirrors test_auto_engineer.py's TestClaudeAutoEngineerAnalyze
_make_engineer fixture and test_local_verify.py's LocalVerifyConfig/_config
conventions.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import ResultMessage

from reverse_api.auto_engineer import ClaudeAutoEngineer
from reverse_api.base_engineer import LocalVerifyConfig
from reverse_api.engineer import _LOCAL_VERIFY_MCP_SERVER_NAME, _RUN_ON_USERS_MACHINE_TOOL_NAME


def _make_engineer(tmp_path, **kwargs):
    defaults = {
        "run_id": "test123",
        "prompt": "browse and capture",
        "model": "claude-sonnet-4-6",
        "output_dir": str(tmp_path),
    }
    defaults.update(kwargs)
    with patch("reverse_api.auto_engineer.get_har_dir", return_value=tmp_path / "har"):
        with patch("reverse_api.base_engineer.get_scripts_dir", return_value=tmp_path / "scripts"):
            with patch("reverse_api.base_engineer.MessageStore") as mock_ms:
                mock_ms.return_value = MagicMock()
                eng = ClaudeAutoEngineer(**defaults)
                eng.scripts_dir = tmp_path / "scripts"
                eng.scripts_dir.mkdir(parents=True, exist_ok=True)
                return eng


def _config(**overrides):
    defaults = {
        "callback_url": "https://route-reveal.example/internal/verify-callback/job123",
        "callback_token": "cb-token-xyz",
    }
    defaults.update(overrides)
    return LocalVerifyConfig(**defaults)


def _sdk_result_message(*, is_error: bool = False, result: str | None = "ok") -> ResultMessage:
    return ResultMessage(
        subtype="test",
        duration_ms=0,
        duration_api_ms=0,
        is_error=is_error,
        num_turns=1,
        session_id="test-session",
        result=result,
    )


async def _run_analyze(eng):
    mock_client = AsyncMock()
    mock_client.query = AsyncMock()

    async def mock_receive():
        yield _sdk_result_message()

    mock_client.receive_response = mock_receive

    with patch.object(eng, "_prompt_follow_up", new=AsyncMock(return_value=None)):
        with patch("reverse_api.auto_engineer.ClaudeSDKClient") as mock_sdk:
            mock_sdk.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_sdk.return_value.__aexit__ = AsyncMock(return_value=False)
            await eng.analyze_and_generate()


class TestMcpProviderBranch:
    """agent_provider in {"auto", "chrome-mcp"} — the mcp_servers={mcp_name: mcp_config} branch."""

    @pytest.mark.asyncio
    async def test_registers_local_verify_alongside_browser_mcp_when_configured(self, tmp_path):
        eng = _make_engineer(tmp_path, agent_provider="auto", local_verify=_config())
        with patch("reverse_api.auto_engineer.ClaudeAgentOptions") as mock_options:
            await _run_analyze(eng)
        mcp_servers = mock_options.call_args.kwargs["mcp_servers"]
        assert _LOCAL_VERIFY_MCP_SERVER_NAME in mcp_servers
        assert "playwright" in mcp_servers  # the browser MCP is still there too

    @pytest.mark.asyncio
    async def test_chrome_mcp_also_gets_local_verify(self, tmp_path):
        eng = _make_engineer(tmp_path, agent_provider="chrome-mcp", local_verify=_config())
        with patch("reverse_api.auto_engineer.ClaudeAgentOptions") as mock_options:
            await _run_analyze(eng)
        mcp_servers = mock_options.call_args.kwargs["mcp_servers"]
        assert _LOCAL_VERIFY_MCP_SERVER_NAME in mcp_servers
        assert "chrome-devtools" in mcp_servers

    @pytest.mark.asyncio
    async def test_no_local_verify_server_without_config(self, tmp_path):
        eng = _make_engineer(tmp_path, agent_provider="auto", local_verify=None)
        with patch("reverse_api.auto_engineer.ClaudeAgentOptions") as mock_options:
            await _run_analyze(eng)
        mcp_servers = mock_options.call_args.kwargs["mcp_servers"]
        assert _LOCAL_VERIFY_MCP_SERVER_NAME not in mcp_servers


class TestAgentBrowserBranch:
    """agent_provider == "agent-browser" — the allowlist-based branch."""

    @pytest.mark.asyncio
    async def test_registers_server_and_allows_the_tool_when_configured(self, tmp_path):
        eng = _make_engineer(tmp_path, agent_provider="agent-browser", local_verify=_config())
        with patch("reverse_api.auto_engineer.ensure_agent_browser_runtime") as mock_ensure:
            mock_ensure.return_value = MagicMock(ok=True, notices=())
            with patch("reverse_api.auto_engineer.ClaudeAgentOptions") as mock_options:
                await _run_analyze(eng)
        mcp_servers = mock_options.call_args.kwargs["mcp_servers"]
        allowed_tools = mock_options.call_args.kwargs["allowed_tools"]
        assert _LOCAL_VERIFY_MCP_SERVER_NAME in mcp_servers
        assert f"mcp__{_LOCAL_VERIFY_MCP_SERVER_NAME}__{_RUN_ON_USERS_MACHINE_TOOL_NAME}" in allowed_tools

    @pytest.mark.asyncio
    async def test_existing_allowlist_entries_are_preserved(self, tmp_path):
        eng = _make_engineer(tmp_path, agent_provider="agent-browser", local_verify=_config())
        with patch("reverse_api.auto_engineer.ensure_agent_browser_runtime") as mock_ensure:
            mock_ensure.return_value = MagicMock(ok=True, notices=())
            with patch("reverse_api.auto_engineer.ClaudeAgentOptions") as mock_options:
                await _run_analyze(eng)
        allowed_tools = mock_options.call_args.kwargs["allowed_tools"]
        assert "Bash" in allowed_tools
        assert "AskUserQuestion" in allowed_tools

    @pytest.mark.asyncio
    async def test_no_local_verify_tool_without_config(self, tmp_path):
        eng = _make_engineer(tmp_path, agent_provider="agent-browser", local_verify=None)
        with patch("reverse_api.auto_engineer.ensure_agent_browser_runtime") as mock_ensure:
            mock_ensure.return_value = MagicMock(ok=True, notices=())
            with patch("reverse_api.auto_engineer.ClaudeAgentOptions") as mock_options:
                await _run_analyze(eng)
        mcp_servers = mock_options.call_args.kwargs["mcp_servers"]
        allowed_tools = mock_options.call_args.kwargs["allowed_tools"]
        assert _LOCAL_VERIFY_MCP_SERVER_NAME not in mcp_servers
        assert not any("local_verify" in t for t in allowed_tools)


class TestCodegenInstructionInheritance:
    """ClaudeAutoEngineer doesn't override _get_codegen_instructions — it
    should inherit ClaudeEngineer's local_verify-conditional append."""

    def test_instruction_present_when_configured(self, tmp_path):
        from reverse_api.base_engineer import RUN_ON_USERS_MACHINE_INSTRUCTION

        eng = _make_engineer(tmp_path, local_verify=_config())
        assert RUN_ON_USERS_MACHINE_INSTRUCTION in eng._get_codegen_instructions()

    def test_instruction_absent_without_config(self, tmp_path):
        from reverse_api.base_engineer import RUN_ON_USERS_MACHINE_INSTRUCTION

        eng = _make_engineer(tmp_path, local_verify=None)
        assert RUN_ON_USERS_MACHINE_INSTRUCTION not in eng._get_codegen_instructions()
