import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from app import copilot_chat, harness as H


class CopilotTests(unittest.TestCase):
    def test_command_is_isolated_and_limited_to_lab_tools(self):
        cmd = copilot_chat.command("copilot", "oi", Path("/tmp/mcp.json"), new_session_id="abc")
        for flag in ("--no-custom-instructions", "--disable-builtin-mcps", "--no-ask-user", "--disallow-temp-dir"):
            self.assertIn(flag, cmd)
        tools = cmd[cmd.index("--available-tools") + 1:cmd.index("--allow-tool")]
        self.assertEqual(tools, [f"dl-lab-{name}" for name in H.TOOL_NAMES])
        self.assertNotIn("--allow-all-tools", cmd)
        self.assertEqual(cmd[cmd.index("--session-id") + 1], "abc")

    def test_resume_and_model(self):
        cmd = copilot_chat.command("copilot", "oi", Path("/tmp/mcp.json"), model="gpt-5", session_id="s1", new_session_id="ignored")
        self.assertIn("--resume=s1", cmd)
        self.assertNotIn("--session-id", cmd)
        self.assertEqual(cmd[cmd.index("--model") + 1], "gpt-5")

    def test_mcp_config_carries_agent_token(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = json.loads(copilot_chat.write_mcp_config(Path(folder) / "mcp.json", "tok").read_text(encoding="utf-8"))
        server = cfg["mcpServers"]["dl-lab"]
        self.assertEqual(server["env"]["DL_LAB_AGENT_TOKEN"], "tok")
        self.assertEqual(server["tools"], ["*"])

    def test_text_is_not_duplicated_after_deltas(self):
        state = {}
        delta = copilot_chat.normalize({"type": "assistant.message_delta", "data": {"messageId": "m", "deltaContent": "Olá"}}, state)
        final = copilot_chat.normalize({"type": "assistant.message", "data": {"messageId": "m", "content": "Olá"}}, state)
        self.assertEqual(delta, [{"type": "text_delta", "text": "Olá"}])
        self.assertEqual(final, [])
        self.assertEqual(copilot_chat.normalize({"type": "assistant.message", "data": {"messageId": "n", "content": "Fim"}}, state), [{"type": "text_delta", "text": "Fim"}])

    def test_lab_tools_are_shown_and_internal_tools_hidden(self):
        state = {}
        use = copilot_chat.normalize({"type": "tool.execution_start", "data": {"toolCallId": "t", "toolName": "dl-lab-list_projects", "arguments": {}}}, state)
        self.assertEqual(use[0]["name"], "list_projects")
        done = copilot_chat.normalize({"type": "tool.execution_complete", "data": {"toolCallId": "t", "success": False, "error": {"message": "bloqueado"}}}, state)
        self.assertEqual(done, [{"type": "tool_result", "tool_use_id": "t", "content": "bloqueado", "is_error": True}])
        self.assertEqual(copilot_chat.normalize({"type": "tool.execution_start", "data": {"toolCallId": "x", "toolName": "report_intent"}}, state), [])
        self.assertEqual(copilot_chat.normalize({"type": "tool.execution_complete", "data": {"toolCallId": "x", "success": True}}, state), [])

    def test_result_returns_session_for_resume(self):
        result = copilot_chat.normalize({"type": "result", "sessionId": "s9", "exitCode": 0, "usage": {"sessionDurationMs": 1200, "premiumRequests": 1}}, {})
        self.assertEqual(result[0]["session_id"], "s9")
        self.assertFalse(result[0]["is_error"])
        self.assertEqual(result[0]["duration_ms"], 1200)

    def test_missing_cli_reports_status(self):
        with mock.patch.object(copilot_chat, "executable", return_value=None):
            self.assertFalse(copilot_chat.status()["available"])
            async def collect():
                return [frame async for frame in copilot_chat.stream_chat("oi", None, None, None)]
            frames = asyncio.run(collect())
        self.assertIn("copilot login", frames[0])
        self.assertIn('"done"', frames[-1])

    def test_auth_failure_message(self):
        self.assertIn("copilot login", copilot_chat.failure("Error: No authentication information found."))


if __name__ == "__main__":
    unittest.main()
