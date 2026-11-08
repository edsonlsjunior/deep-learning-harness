import unittest
from pathlib import Path
from app import codex_chat


class CodexTests(unittest.TestCase):
    def test_bridge_is_read_only_and_isolated(self):
        cmd = codex_chat.command("codex", Path("/tmp/lab"), None)
        self.assertIn("--ignore-user-config", cmd)
        self.assertIn("read-only", cmd)
        self.assertIn("forced_login_method=\"chatgpt\"", cmd)
        self.assertIn("shell_tool", cmd)
        self.assertFalse(any(name.startswith("delete_") or name in ("create_model", "train_model", "import_file") for name in codex_chat.READ_TOOLS))

    def test_text_normalization(self):
        self.assertEqual(codex_chat.normalize({"type": "item.completed", "item": {"type": "agent_message", "text": "Olá"}}), [{"type": "text_delta", "text": "Olá"}])

    def test_tool_normalization(self):
        result = codex_chat.normalize({"type": "item.completed", "item": {"id": "t", "type": "mcp_tool_call", "status": "completed", "result": {"data": []}}})
        self.assertEqual(result[0]["type"], "tool_result")
        self.assertFalse(result[0]["is_error"])
