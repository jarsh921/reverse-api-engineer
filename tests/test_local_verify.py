"""Tests for the local-verification-agent relay: LocalVerifyConfig, the
run_on_users_machine SDK tool on ClaudeEngineer, and the env-var wiring in
cli.py. Mirrors test_engineer.py's TestClaudeEngineerAnalyzeAndGenerate
_make_engineer fixture pattern and mocks `requests` rather than hitting a
real HTTP server, matching this repo's existing mocking conventions.
"""

from unittest.mock import MagicMock, patch

import pytest

from reverse_api.base_engineer import RUN_ON_USERS_MACHINE_INSTRUCTION, LocalVerifyConfig
from reverse_api.engineer import ClaudeEngineer, _LOCAL_VERIFY_MCP_SERVER_NAME, _RUN_ON_USERS_MACHINE_TOOL_NAME


def _make_engineer(tmp_path, **kwargs):
    har_path = tmp_path / "test.har"
    har_path.touch()
    defaults = {
        "run_id": "test123",
        "har_path": har_path,
        "prompt": "test prompt",
        "output_dir": str(tmp_path),
    }
    defaults.update(kwargs)
    with patch("reverse_api.base_engineer.get_scripts_dir", return_value=tmp_path / "scripts"):
        with patch("reverse_api.base_engineer.MessageStore") as mock_ms:
            mock_ms.return_value = MagicMock()
            eng = ClaudeEngineer(**defaults)
            eng.scripts_dir = tmp_path / "scripts"
            eng.scripts_dir.mkdir(parents=True, exist_ok=True)
            return eng


def _config(**overrides):
    defaults = {
        "callback_url": "https://route-reveal.example/internal/verify-callback/job123",
        "callback_token": "cb-token-xyz",
        "poll_interval_seconds": 0.01,  # fast polling under test
        "wait_timeout_seconds": 0.05,
        "command_timeout_seconds": 60.0,
    }
    defaults.update(overrides)
    return LocalVerifyConfig(**defaults)


class TestGetCodegenInstructions:
    def test_no_local_verify_leaves_instructions_unchanged(self, tmp_path):
        eng = _make_engineer(tmp_path, local_verify=None)
        assert RUN_ON_USERS_MACHINE_INSTRUCTION not in eng._get_codegen_instructions()

    def test_local_verify_appends_instruction(self, tmp_path):
        eng = _make_engineer(tmp_path, local_verify=_config())
        assert RUN_ON_USERS_MACHINE_INSTRUCTION in eng._get_codegen_instructions()

    def test_docs_mode_never_gets_the_instruction_even_with_local_verify(self, tmp_path):
        eng = _make_engineer(tmp_path, local_verify=_config(), output_mode="docs")
        assert RUN_ON_USERS_MACHINE_INSTRUCTION not in eng._get_codegen_instructions()


class TestBuildLocalExecMcpServer:
    def test_server_is_named_and_sdk_typed(self, tmp_path):
        eng = _make_engineer(tmp_path, local_verify=_config())
        server = eng._build_local_exec_mcp_server()
        assert server["name"] == _LOCAL_VERIFY_MCP_SERVER_NAME
        assert server["type"] == "sdk"

    def test_tool_itself_is_named_run_on_users_machine(self, tmp_path):
        eng = _make_engineer(tmp_path, local_verify=_config())
        assert eng._build_local_exec_tool().name == _RUN_ON_USERS_MACHINE_TOOL_NAME


class TestRunOnUsersMachineTool:
    def _handler(self, eng):
        return eng._build_local_exec_tool().handler

    @pytest.mark.asyncio
    async def test_missing_client_file_is_an_error(self, tmp_path):
        eng = _make_engineer(tmp_path, local_verify=_config())
        result = await self._handler(eng)({})
        assert result["is_error"] is True
        assert "write it first" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_successful_run_returns_stdout_and_is_not_an_error(self, tmp_path):
        eng = _make_engineer(tmp_path, local_verify=_config())
        (eng.scripts_dir / eng._get_client_filename()).write_text("print('hi')")

        mock_post = MagicMock()
        mock_post.json.return_value = {"id": "cmd-1", "status": "pending"}
        mock_post.raise_for_status = MagicMock()

        mock_get = MagicMock()
        mock_get.json.return_value = {
            "id": "cmd-1",
            "status": "succeeded",
            "stdout": "hi\n",
            "stderr": "",
            "exit_code": 0,
        }
        mock_get.raise_for_status = MagicMock()

        with patch("reverse_api.engineer.requests.post", return_value=mock_post) as post, patch(
            "reverse_api.engineer.requests.get", return_value=mock_get
        ) as get:
            result = await self._handler(eng)({})

        assert result["is_error"] is False
        assert "hi" in result["content"][0]["text"]
        assert "succeeded" in result["content"][0]["text"]
        post.assert_called_once()
        assert post.call_args.args[0] == "https://route-reveal.example/internal/verify-callback/job123/commands"
        assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer cb-token-xyz"}
        get.assert_called_once_with(
            "https://route-reveal.example/internal/verify-callback/job123/commands/cmd-1",
            headers={"Authorization": "Bearer cb-token-xyz"},
            timeout=15,
        )

    @pytest.mark.asyncio
    async def test_failed_run_is_reported_as_an_error(self, tmp_path):
        eng = _make_engineer(tmp_path, local_verify=_config())
        (eng.scripts_dir / eng._get_client_filename()).write_text("print('hi')")

        mock_post = MagicMock()
        mock_post.json.return_value = {"id": "cmd-1"}
        mock_get = MagicMock()
        mock_get.json.return_value = {
            "id": "cmd-1",
            "status": "failed",
            "stdout": "",
            "stderr": "Traceback...",
            "exit_code": 1,
        }

        with patch("reverse_api.engineer.requests.post", return_value=mock_post), patch(
            "reverse_api.engineer.requests.get", return_value=mock_get
        ):
            result = await self._handler(eng)({})

        assert result["is_error"] is True
        assert "failed" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_dispatch_failure_is_an_error(self, tmp_path):
        import requests as requests_module

        eng = _make_engineer(tmp_path, local_verify=_config())
        (eng.scripts_dir / eng._get_client_filename()).write_text("print('hi')")

        with patch(
            "reverse_api.engineer.requests.post",
            side_effect=requests_module.RequestException("connection refused"),
        ):
            result = await self._handler(eng)({})

        assert result["is_error"] is True
        assert "Could not reach the paired local machine" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_no_result_within_timeout_reports_offline_and_is_an_error(self, tmp_path):
        eng = _make_engineer(tmp_path, local_verify=_config())
        (eng.scripts_dir / eng._get_client_filename()).write_text("print('hi')")

        mock_post = MagicMock()
        mock_post.json.return_value = {"id": "cmd-1"}
        mock_get = MagicMock()
        mock_get.json.return_value = {"id": "cmd-1", "status": "pending"}  # never terminal

        with patch("reverse_api.engineer.requests.post", return_value=mock_post), patch(
            "reverse_api.engineer.requests.get", return_value=mock_get
        ):
            result = await self._handler(eng)({})

        assert result["is_error"] is True
        assert "offline or unreachable" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_can_be_called_more_than_once_no_dedup(self, tmp_path):
        """Unlike report_client_verified, this tool has no once-only guard —
        RUN_ON_USERS_MACHINE_INSTRUCTION explicitly tells the agent it can
        call it repeatedly while iterating."""
        eng = _make_engineer(tmp_path, local_verify=_config())
        (eng.scripts_dir / eng._get_client_filename()).write_text("print('hi')")

        mock_post = MagicMock()
        mock_post.json.return_value = {"id": "cmd-1"}
        mock_get = MagicMock()
        mock_get.json.return_value = {"id": "cmd-1", "status": "succeeded", "stdout": "", "stderr": "", "exit_code": 0}

        handler = self._handler(eng)
        with patch("reverse_api.engineer.requests.post", return_value=mock_post) as post, patch(
            "reverse_api.engineer.requests.get", return_value=mock_get
        ):
            await handler({})
            await handler({})

        assert post.call_count == 2


class TestGatherLocalVerifyFiles:
    """_gather_local_verify_files — the per-language sidecar file logic
    added in Phase 2. Python has no sidecars at all, so its own coverage
    lives in TestRunOnUsersMachineTool above; this focuses on the languages
    that actually have required/optional sidecar files."""

    def test_missing_entrypoint_returns_an_error_string(self, tmp_path):
        eng = _make_engineer(tmp_path, local_verify=_config(), output_language="java")
        result = eng._gather_local_verify_files()
        assert isinstance(result, str)
        assert "write it first" in result

    def test_java_requires_pom_xml(self, tmp_path):
        eng = _make_engineer(tmp_path, local_verify=_config(), output_language="java")
        (eng.scripts_dir / eng._get_client_filename()).write_text("public class ApiClient {}")

        result = eng._gather_local_verify_files()
        assert isinstance(result, str)
        assert "pom.xml" in result

    def test_java_gathers_client_and_pom_when_both_present(self, tmp_path):
        eng = _make_engineer(tmp_path, local_verify=_config(), output_language="java")
        client_filename = eng._get_client_filename()
        (eng.scripts_dir / client_filename).write_text("public class ApiClient {}")
        (eng.scripts_dir / "pom.xml").write_text("<project></project>")

        files = eng._gather_local_verify_files()
        assert files == {client_filename: "public class ApiClient {}", "pom.xml": "<project></project>"}

    def test_csharp_requires_csproj(self, tmp_path):
        eng = _make_engineer(tmp_path, local_verify=_config(), output_language="csharp")
        (eng.scripts_dir / eng._get_client_filename()).write_text("class ApiClient {}")

        result = eng._gather_local_verify_files()
        assert isinstance(result, str)
        assert "ApiClient.csproj" in result

    def test_c_requires_both_cjson_files(self, tmp_path):
        eng = _make_engineer(tmp_path, local_verify=_config(), output_language="c")
        (eng.scripts_dir / eng._get_client_filename()).write_text("int main() { return 0; }")
        (eng.scripts_dir / "cJSON.c").write_text("/* cjson */")
        # cJSON.h still missing

        result = eng._gather_local_verify_files()
        assert isinstance(result, str)
        assert "cJSON.h" in result

    def test_c_gathers_all_three_files_when_present(self, tmp_path):
        eng = _make_engineer(tmp_path, local_verify=_config(), output_language="c")
        client_filename = eng._get_client_filename()
        (eng.scripts_dir / client_filename).write_text("int main() { return 0; }")
        (eng.scripts_dir / "cJSON.c").write_text("/* c */")
        (eng.scripts_dir / "cJSON.h").write_text("/* h */")

        files = eng._gather_local_verify_files()
        assert set(files.keys()) == {client_filename, "cJSON.c", "cJSON.h"}

    def test_javascript_package_json_is_optional_not_an_error(self, tmp_path):
        eng = _make_engineer(tmp_path, local_verify=_config(), output_language="javascript")
        client_filename = eng._get_client_filename()
        (eng.scripts_dir / client_filename).write_text("console.log('hi')")
        # no package.json written — should NOT be treated as missing-required

        files = eng._gather_local_verify_files()
        assert files == {client_filename: "console.log('hi')"}

    def test_javascript_package_json_included_when_present(self, tmp_path):
        eng = _make_engineer(tmp_path, local_verify=_config(), output_language="javascript")
        client_filename = eng._get_client_filename()
        (eng.scripts_dir / client_filename).write_text("console.log('hi')")
        (eng.scripts_dir / "package.json").write_text("{}")

        files = eng._gather_local_verify_files()
        assert files == {client_filename: "console.log('hi')", "package.json": "{}"}

    def test_go_gathers_both_optional_files_when_present(self, tmp_path):
        eng = _make_engineer(tmp_path, local_verify=_config(), output_language="go")
        client_filename = eng._get_client_filename()
        (eng.scripts_dir / client_filename).write_text("package main")
        (eng.scripts_dir / "go.mod").write_text("module x")
        (eng.scripts_dir / "go.sum").write_text("")

        files = eng._gather_local_verify_files()
        assert set(files.keys()) == {client_filename, "go.mod", "go.sum"}

    def test_python_has_no_sidecars(self, tmp_path):
        eng = _make_engineer(tmp_path, local_verify=_config(), output_language="python")
        client_filename = eng._get_client_filename()
        (eng.scripts_dir / client_filename).write_text("print('hi')")

        files = eng._gather_local_verify_files()
        assert files == {client_filename: "print('hi')"}


class TestAnalyzeAndGenerateRegistersLocalVerifyServer:
    @pytest.mark.asyncio
    async def test_mcp_servers_includes_local_verify_when_configured(self, tmp_path):
        from unittest.mock import AsyncMock

        eng = _make_engineer(tmp_path, local_verify=_config())

        mock_client = AsyncMock()
        mock_client.query = AsyncMock()

        async def mock_receive():
            from claude_agent_sdk import ResultMessage

            m = MagicMock(spec=ResultMessage)
            m.is_error = False
            m.result = "ok"
            yield m

        mock_client.receive_response = mock_receive

        with patch("reverse_api.engineer.ClaudeAgentOptions") as mock_options, patch(
            "reverse_api.engineer.ClaudeSDKClient"
        ) as mock_sdk:
            mock_sdk.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_sdk.return_value.__aexit__ = AsyncMock(return_value=False)
            with patch.object(eng, "_prompt_follow_up", new_callable=AsyncMock, return_value=None):
                await eng.analyze_and_generate()

        assert _LOCAL_VERIFY_MCP_SERVER_NAME in mock_options.call_args.kwargs["mcp_servers"]

    @pytest.mark.asyncio
    async def test_mcp_servers_empty_without_local_verify(self, tmp_path):
        from unittest.mock import AsyncMock

        eng = _make_engineer(tmp_path, local_verify=None)

        mock_client = AsyncMock()
        mock_client.query = AsyncMock()

        async def mock_receive():
            from claude_agent_sdk import ResultMessage

            m = MagicMock(spec=ResultMessage)
            m.is_error = False
            m.result = "ok"
            yield m

        mock_client.receive_response = mock_receive

        with patch("reverse_api.engineer.ClaudeAgentOptions") as mock_options, patch(
            "reverse_api.engineer.ClaudeSDKClient"
        ) as mock_sdk:
            mock_sdk.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_sdk.return_value.__aexit__ = AsyncMock(return_value=False)
            with patch.object(eng, "_prompt_follow_up", new_callable=AsyncMock, return_value=None):
                await eng.analyze_and_generate()

        assert mock_options.call_args.kwargs["mcp_servers"] == {}


class TestRunReverseEngineeringPassesLocalVerify:
    def test_claude_branch_forwards_local_verify(self, tmp_path):
        from reverse_api.engineer import run_reverse_engineering

        har_path = tmp_path / "test.har"
        har_path.touch()
        config = _config()

        with patch("reverse_api.engineer.ClaudeEngineer") as mock_cls:
            mock_engineer = MagicMock()
            mock_engineer.analyze_and_generate = MagicMock()
            mock_cls.return_value = mock_engineer
            with patch("reverse_api.engineer.asyncio.run", return_value={"script_path": "x", "usage": {}}):
                run_reverse_engineering(
                    run_id="r1",
                    har_path=har_path,
                    prompt="p",
                    sdk="claude",
                    local_verify=config,
                )

        assert mock_cls.call_args.kwargs["local_verify"] is config


class TestLocalVerifyConfigFromEnv:
    def test_none_when_url_missing(self, monkeypatch):
        from reverse_api.cli import _local_verify_config_from_env

        monkeypatch.delenv("RAE_VERIFY_CALLBACK_URL", raising=False)
        monkeypatch.setenv("RAE_VERIFY_CALLBACK_TOKEN", "tok")
        assert _local_verify_config_from_env() is None

    def test_none_when_token_missing(self, monkeypatch):
        from reverse_api.cli import _local_verify_config_from_env

        monkeypatch.setenv("RAE_VERIFY_CALLBACK_URL", "https://example.com/x")
        monkeypatch.delenv("RAE_VERIFY_CALLBACK_TOKEN", raising=False)
        assert _local_verify_config_from_env() is None

    def test_builds_config_when_both_set(self, monkeypatch):
        from reverse_api.cli import _local_verify_config_from_env

        monkeypatch.setenv("RAE_VERIFY_CALLBACK_URL", "https://example.com/x")
        monkeypatch.setenv("RAE_VERIFY_CALLBACK_TOKEN", "tok")
        config = _local_verify_config_from_env()
        assert config.callback_url == "https://example.com/x"
        assert config.callback_token == "tok"
