"""Reverse engineering module with SDK dispatch."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import requests
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    ResultMessage,
    TextBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

from .base_engineer import RUN_ON_USERS_MACHINE_INSTRUCTION, BaseEngineer, LocalVerifyConfig
from .utils import build_sdk_env, is_context_overflow_error

# Suppress claude_agent_sdk logs
logging.getLogger("claude_agent_sdk").setLevel(logging.WARNING)
logging.getLogger("claude_agent_sdk._internal.transport.subprocess_cli").setLevel(logging.WARNING)

_LOCAL_VERIFY_MCP_SERVER_NAME = "local_verify"
_RUN_ON_USERS_MACHINE_TOOL_NAME = "run_on_users_machine"

# Sidecar files the local agent's own run command depends on, alongside the
# main client file — mirrors BaseEngineer._get_run_command()'s per-language
# knowledge of what has to exist on disk to actually run a client (Maven
# needs pom.xml, dotnet needs the .csproj, C needs its vendored cJSON pair)
# and _get_auto_output_files()'s per-language sidecar list, but as a
# machine-readable table rather than prose for a prompt. "Required": local
# verification can't be attempted at all without these — the local agent's
# own fixed run command hard-depends on them. "Optional": included only if
# reverse-api-engineer actually wrote them (package.json/go.mod only exist
# when the client needed external dependencies) — their absence isn't an
# error, the run command works without them either way.
_LOCAL_VERIFY_REQUIRED_SIDECARS: dict[str, tuple[str, ...]] = {
    "java": ("pom.xml",),
    "csharp": ("ApiClient.csproj",),
    "c": ("cJSON.c", "cJSON.h"),
}
_LOCAL_VERIFY_OPTIONAL_SIDECARS: dict[str, tuple[str, ...]] = {
    "javascript": ("package.json",),
    "typescript": ("package.json",),
    "go": ("go.mod", "go.sum"),
}


class ClaudeEngineer(BaseEngineer):
    """Uses Claude Agent SDK to analyze HAR files and generate Python API scripts."""

    # Set when the session dies with a context-window overflow ("Prompt is
    # too long"). The conversation itself exceeds the model's limit at that
    # point, so no further query on the same client can succeed and the
    # follow-up loop must not offer another turn.
    _context_overflowed = False

    def _print_context_overflow_help(self) -> None:
        """Explain a context-window overflow and how to continue."""
        self.ui.console.print()
        self.ui.console.print("  [yellow]The session ran out of context window (too much captured traffic for one conversation).[/yellow]")
        self.ui.console.print("  [dim]This session can't continue, but progress is saved: generated files are in[/dim]")
        self.ui.console.print(f"  [dim]{self.scripts_dir}[/dim]")
        self.ui.console.print(f"  [dim]Start a new run for this target (run id: {self.run_id}) to keep iterating —[/dim]")
        self.ui.console.print("  [dim]it picks up the existing client and HAR from disk. A shorter browsing[/dim]")
        self.ui.console.print("  [dim]session (fewer pages before closing the browser) also keeps the HAR smaller.[/dim]")

    async def _handle_tool_permission(self, tool_name: str, input_data: dict[str, Any], context: ToolPermissionContext) -> PermissionResultAllow:
        """Handle tool permission requests, with interactive UI for AskUserQuestion."""
        if tool_name == "AskUserQuestion":
            questions = input_data.get("questions", [])
            answers = await self._ask_user_questions(questions)
            return PermissionResultAllow(
                updated_input={"questions": questions, "answers": answers},
            )

        # Auto-approve all other tools
        return PermissionResultAllow(updated_input=input_data)

    def _get_codegen_instructions(self) -> str:
        """Appends RUN_ON_USERS_MACHINE_INSTRUCTION only when this run is
        configured for local verification (self.local_verify is not None)
        and isn't docs mode (there's no client to run in docs mode) — zero
        change to the base instructions otherwise.
        """
        base = super()._get_codegen_instructions()
        if self.local_verify is not None and self.output_mode != "docs":
            return base + RUN_ON_USERS_MACHINE_INSTRUCTION
        return base

    def _gather_local_verify_files(self) -> dict[str, str] | str:
        """Gathers the client file plus every sidecar its run command needs,
        fresh off disk. Returns `{relative_filename: content}` on success, or
        a plain error string (missing entrypoint or a missing *required*
        sidecar) for the caller to surface as a tool error.
        """
        client_filename = self._get_client_filename()
        client_path = self.scripts_dir / client_filename
        if not client_path.exists():
            return f"No {client_filename} exists yet at {client_path} — write it first, then call this tool."

        required = _LOCAL_VERIFY_REQUIRED_SIDECARS.get(self.output_language, ())
        missing = [name for name in required if not (self.scripts_dir / name).exists()]
        if missing:
            return (
                f"Missing required file(s) for {self.output_language}: {', '.join(missing)} "
                f"(expected alongside {client_filename} in {self.scripts_dir}) — write them first, then call this tool."
            )

        files = {client_filename: client_path.read_text(encoding="utf-8", errors="replace")}
        optional = _LOCAL_VERIFY_OPTIONAL_SIDECARS.get(self.output_language, ())
        for name in (*required, *optional):
            path = self.scripts_dir / name
            if path.exists():
                files[name] = path.read_text(encoding="utf-8", errors="replace")
        return files

    def _build_local_exec_tool(self):
        """Builds the `run_on_users_machine` SDK tool for this run's
        local_verify config. Each call gathers the client file plus its
        required/optional sidecars fresh off disk (see
        _gather_local_verify_files), dispatches them to route-reveal's
        job-scoped internal-callback endpoints, and polls for a result.

        Deliberately no dedup/once-only guard, unlike report_client_verified
        — RUN_ON_USERS_MACHINE_INSTRUCTION explicitly tells the agent it can
        call this repeatedly while iterating, so every call must actually
        dispatch and wait, not just the first.
        """
        config = self.local_verify

        @tool(
            _RUN_ON_USERS_MACHINE_TOOL_NAME,
            "Run the current client code on the user's own paired machine (which can reach "
            "the target when this session's Bash can't) and return its stdout/stderr/exit "
            "code. Safe to call again after each fix while iterating.",
            {},
        )
        async def run_on_users_machine(args: dict[str, Any]) -> dict[str, Any]:
            client_filename = self._get_client_filename()
            files = self._gather_local_verify_files()
            if isinstance(files, str):
                return {"content": [{"type": "text", "text": files}], "is_error": True}
            headers = {"Authorization": f"Bearer {config.callback_token}"}

            try:
                create_resp = requests.post(
                    f"{config.callback_url}/commands",
                    json={
                        "output_language": self.output_language,
                        "entrypoint_filename": client_filename,
                        "files": files,
                        "timeout_seconds": config.command_timeout_seconds,
                    },
                    headers=headers,
                    timeout=15,
                )
                create_resp.raise_for_status()
                command_id = create_resp.json()["id"]
            except requests.RequestException as e:
                return {
                    "content": [{"type": "text", "text": f"Could not reach the paired local machine to dispatch this: {e}"}],
                    "is_error": True,
                }

            deadline = time.monotonic() + config.wait_timeout_seconds
            while time.monotonic() < deadline:
                await asyncio.sleep(config.poll_interval_seconds)
                try:
                    poll_resp = requests.get(
                        f"{config.callback_url}/commands/{command_id}", headers=headers, timeout=15
                    )
                    poll_resp.raise_for_status()
                    result = poll_resp.json()
                except requests.RequestException as e:
                    return {
                        "content": [
                            {"type": "text", "text": f"Lost contact with the paired local machine while waiting for a result: {e}"}
                        ],
                        "is_error": True,
                    }

                status = result.get("status")
                if status in ("succeeded", "failed", "timed_out", "cancelled"):
                    text = (
                        f"Ran on the user's paired machine — status: {status}, exit code: {result.get('exit_code')}\n"
                        f"stdout:\n{result.get('stdout') or ''}\n\nstderr:\n{result.get('stderr') or ''}"
                    )
                    return {"content": [{"type": "text", "text": text}], "is_error": status != "succeeded"}

            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"No result from the user's paired machine within {config.wait_timeout_seconds:.0f}s — "
                            "it may be offline or unreachable. This is an external constraint you can't fix from "
                            "here: don't retry this in a loop, and don't fall back to testing via Bash (it can't "
                            "reach this target either). Note it in your summary and continue."
                        ),
                    }
                ],
                "is_error": True,
            }

        return run_on_users_machine

    def _build_local_exec_mcp_server(self):
        return create_sdk_mcp_server(name=_LOCAL_VERIFY_MCP_SERVER_NAME, tools=[self._build_local_exec_tool()])

    _USAGE_ACCUMULATE_KEYS = {
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    }

    def _accumulate_usage(self, usage: dict) -> None:
        """Merge usage data, summing token counts instead of replacing."""
        for key, value in usage.items():
            if key in self._USAGE_ACCUMULATE_KEYS and isinstance(value, (int, float)):
                self.usage_metadata[key] = self.usage_metadata.get(key, 0) + value
            else:
                self.usage_metadata[key] = value

    async def _process_streaming_response(self, client: ClaudeSDKClient) -> dict[str, Any] | None:
        """Process a single streaming response from the SDK client.

        Returns a result dict on success, or None on error.
        """
        async for message in client.receive_response():
            if hasattr(message, "usage") and isinstance(message.usage, dict):
                self._accumulate_usage(message.usage)

            if isinstance(message, AssistantMessage):
                last_tool_name = None
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        last_tool_name = block.name
                        self.ui.tool_start(block.name, block.input)
                        self.message_store.save_tool_start(block.name, block.input)
                    elif isinstance(block, ToolResultBlock):
                        is_error = block.is_error if block.is_error else False

                        output = None
                        if getattr(block, "content", None) is not None:
                            output = block.content
                        elif getattr(block, "result", None) is not None:
                            output = block.result
                        elif getattr(block, "output", None) is not None:
                            output = block.output

                        tool_name = last_tool_name or "Tool"
                        self.ui.tool_result(tool_name, is_error, output)
                        self.message_store.save_tool_result(tool_name, is_error, str(output) if output else None)
                    elif isinstance(block, TextBlock):
                        self.ui.thinking(block.text)
                        self.message_store.save_thinking(block.text)

            elif isinstance(message, ResultMessage):
                if message.is_error:
                    error_text = message.result or "Unknown error"
                    self.ui.error(error_text)
                    self.message_store.save_error(error_text)
                    if is_context_overflow_error(error_text):
                        self._context_overflowed = True
                        self._print_context_overflow_help()
                    return None
                else:
                    script_path = str(self.scripts_dir / self._get_client_filename())
                    local_path = str(self.local_scripts_dir / self._get_client_filename()) if self.local_scripts_dir else None
                    self.ui.success(script_path, local_path)

                    if self.usage_metadata:
                        input_tokens = self.usage_metadata.get("input_tokens", 0)
                        output_tokens = self.usage_metadata.get("output_tokens", 0)
                        cache_creation_tokens = self.usage_metadata.get("cache_creation_input_tokens", 0)
                        cache_read_tokens = self.usage_metadata.get("cache_read_input_tokens", 0)

                        from .pricing import calculate_cost

                        cost = calculate_cost(
                            model_id=self.model,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cache_creation_tokens=cache_creation_tokens,
                            cache_read_tokens=cache_read_tokens,
                        )
                        self.usage_metadata["estimated_cost_usd"] = cost

                        self.ui.console.print("  [dim]Usage:[/dim]")
                        if input_tokens > 0:
                            self.ui.console.print(f"  [dim]  input: {input_tokens:,} tokens[/dim]")
                        if cache_creation_tokens > 0:
                            self.ui.console.print(f"  [dim]  cache creation: {cache_creation_tokens:,} tokens[/dim]")
                        if cache_read_tokens > 0:
                            self.ui.console.print(f"  [dim]  cache read: {cache_read_tokens:,} tokens[/dim]")
                        if output_tokens > 0:
                            self.ui.console.print(f"  [dim]  output: {output_tokens:,} tokens[/dim]")
                        self.ui.console.print(f"  [dim]  total cost: ${cost:.4f}[/dim]")

                    result: dict[str, Any] = {
                        "script_path": script_path,
                        "usage": self.usage_metadata,
                    }
                    self.message_store.save_result(result)
                    return result

        return None

    async def analyze_and_generate(self) -> dict[str, Any] | None:
        """Run the reverse engineering analysis with Claude.

        Supports follow-up messages: after the initial analysis completes,
        the user can send follow-ups in the same session for iterative refinement.
        Press Enter or Ctrl+C to finish and return to the REPL.
        """
        self.ui.header(self.run_id, self.prompt, self.model, self.sdk, mode="engineer")
        self.ui.start_analysis()

        # Fresh SDK session, fresh context window: clear any overflow state
        # left over from a previous run on a reused engineer instance.
        self._context_overflowed = False

        system_prompt, user_message = self._build_prompts()
        self.message_store.save_prompt(user_message)

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            permission_mode="acceptEdits",
            can_use_tool=self._handle_tool_permission,
            cwd=str(self.scripts_dir.parent.parent),
            model=self.model,
            env=build_sdk_env(),
            stderr=self._handle_cli_stderr,
            mcp_servers=(
                {_LOCAL_VERIFY_MCP_SERVER_NAME: self._build_local_exec_mcp_server()}
                if self.local_verify is not None
                else {}
            ),
        )

        last_result: dict[str, Any] | None = None

        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(user_message)

                # Process initial response
                last_result = await self._process_streaming_response(client)
                if last_result is None:
                    return None

                # Conversation loop: prompt for follow-ups
                while True:
                    follow_up = await self._prompt_follow_up()
                    if not follow_up:
                        return last_result

                    self.ui.console.print()
                    self.message_store.save_prompt(follow_up)
                    await client.query(follow_up)

                    result = await self._process_streaming_response(client)
                    if result is not None:
                        last_result = result
                    elif self._context_overflowed:
                        # The conversation exceeds the context window; every
                        # further query would fail the same way, so end the
                        # follow-up loop instead of offering another turn.
                        return last_result

        except KeyboardInterrupt:
            self.ui.console.print("\n  [dim]run aborted[/dim]")
            return last_result

        except Exception as e:
            self.ui.error(str(e))
            self.message_store.save_error(str(e))
            self.ui.console.print("\n[dim]Make sure Claude Code CLI is installed: npm install -g @anthropic-ai/claude-code[/dim]")
            return None


# Keep old class name for backwards compatibility
APIReverseEngineer = ClaudeEngineer


def run_reverse_engineering(
    run_id: str,
    har_path: Path,
    prompt: str,
    model: str | None = None,
    additional_instructions: str | None = None,
    output_dir: str | None = None,
    verbose: bool = True,
    sdk: str = "claude",
    opencode_provider: str | None = None,
    opencode_model: str | None = None,
    copilot_model: str | None = None,
    cursor_model: str | None = None,
    cursor_web_search: bool = True,
    cursor_setting_sources: list[str] | None = None,
    enable_sync: bool = False,
    is_fresh: bool = False,
    output_language: str = "python",
    output_mode: str = "client",
    interactive: bool = True,
    json_event_sink: Any = None,
    local_verify: LocalVerifyConfig | None = None,
) -> dict[str, Any] | None:
    """Run reverse engineering with the specified SDK.

    Args:
        sdk: "claude", "opencode", "copilot", or "cursor" - determines which SDK to use
        opencode_provider: Provider ID for OpenCode (e.g., "anthropic")
        opencode_model: Model ID for OpenCode (e.g., "claude-sonnet-4-6")
        copilot_model: Model ID for Copilot (e.g., "gpt-5")
        cursor_model: Model id for Cursor SDK (e.g., "composer-2.5")
        cursor_web_search: When True, load extra Cursor setting layers so WebFetch/WebSearch and plugins apply.
        cursor_setting_sources: Optional explicit list (overrides cursor_web_search), e.g. ["project","user","all"].
        enable_sync: Enable real-time file syncing during engineering
        is_fresh: Whether to start fresh (ignore previous scripts)
        output_language: Target language - "python", "javascript", "typescript", "go", "java", "csharp", "php", "ruby", or "c"
        output_mode: Output mode - "client" for API client code, "docs" for OpenAPI specification
        local_verify: Only honored for sdk="claude" (see ClaudeEngineer) — the other SDK
            backends don't yet register the run_on_users_machine tool, so this is silently
            ignored for them rather than raising, matching this function's own existing
            per-SDK feature-support pattern (e.g. cursor_model/opencode_provider).
    """
    if sdk == "opencode":
        from .opencode_engineer import OpenCodeEngineer

        engineer = OpenCodeEngineer(
            run_id=run_id,
            har_path=har_path,
            prompt=prompt,
            model=model,
            additional_instructions=additional_instructions,
            output_dir=output_dir,
            verbose=verbose,
            opencode_provider=opencode_provider,
            opencode_model=opencode_model,
            enable_sync=enable_sync,
            sdk=sdk,
            is_fresh=is_fresh,
            output_language=output_language,
            output_mode=output_mode,
            interactive=interactive,
        )
    elif sdk == "cursor":
        from .cursor_engineer import CursorEngineer

        engineer = CursorEngineer(
            run_id=run_id,
            har_path=har_path,
            prompt=prompt,
            model=model,
            additional_instructions=additional_instructions,
            output_dir=output_dir,
            verbose=verbose,
            enable_sync=enable_sync,
            sdk=sdk,
            is_fresh=is_fresh,
            output_language=output_language,
            output_mode=output_mode,
            cursor_model=cursor_model,
            cursor_web_search=cursor_web_search,
            cursor_setting_sources=cursor_setting_sources,
            interactive=interactive,
        )
    elif sdk == "copilot":
        from .copilot_engineer import CopilotEngineer

        engineer = CopilotEngineer(
            run_id=run_id,
            har_path=har_path,
            prompt=prompt,
            model=model,
            additional_instructions=additional_instructions,
            output_dir=output_dir,
            verbose=verbose,
            enable_sync=enable_sync,
            sdk=sdk,
            is_fresh=is_fresh,
            output_language=output_language,
            output_mode=output_mode,
            copilot_model=copilot_model,
            interactive=interactive,
        )
    else:
        engineer = ClaudeEngineer(
            run_id=run_id,
            har_path=har_path,
            prompt=prompt,
            model=model,
            additional_instructions=additional_instructions,
            output_dir=output_dir,
            verbose=verbose,
            enable_sync=enable_sync,
            sdk=sdk,
            is_fresh=is_fresh,
            output_language=output_language,
            output_mode=output_mode,
            interactive=interactive,
            local_verify=local_verify,
        )

    if json_event_sink is not None:
        from .json_stream import attach_json_stream_to_engineer

        attach_json_stream_to_engineer(engineer, json_event_sink)

    # Start sync before analysis
    engineer.start_sync()

    try:
        result = asyncio.run(engineer.analyze_and_generate())
    except KeyboardInterrupt:
        # Absorb interrupt so REPL continues instead of exiting
        result = None
    finally:
        # Always stop sync when done
        engineer.stop_sync()

    return result
