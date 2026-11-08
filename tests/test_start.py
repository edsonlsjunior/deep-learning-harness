import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import start


class LauncherTests(unittest.TestCase):
    def test_windows_npm_uses_node_without_shell(self):
        with tempfile.TemporaryDirectory(prefix="node with spaces ") as folder:
            base = Path(folder)
            cli = base / "node_modules/npm/bin/npm-cli.js"
            cli.parent.mkdir(parents=True)
            cli.write_text("// fixture")
            self.assertEqual(start.npm_command("node.exe", str(base / "npm.cmd"), True), ["node.exe", str(cli)])

    def test_platform_interpreter_paths(self):
        root = Path("project with spaces")
        self.assertEqual(start.venv_python(root, True), root / "backend/.venv/Scripts/python.exe")
        self.assertEqual(start.venv_python(root, False), root / "backend/.venv/bin/python")

    def test_dependency_change_invalidates_marker(self):
        with tempfile.TemporaryDirectory() as d:
            source, marker = Path(d) / "requirements.txt", Path(d) / "marker"
            source.write_text("one")
            self.assertTrue(start.needs_install(marker, source))
            marker.write_text(start.fingerprint(source))
            self.assertFalse(start.needs_install(marker, source))
            source.write_text("two")
            self.assertTrue(start.needs_install(marker, source))

    def test_mcp_config_uses_selected_port_and_absolute_paths(self):
        with tempfile.TemporaryDirectory(prefix="lab with spaces ") as d:
            root = Path(d)
            with patch.object(start, "ROOT", root), patch.object(start, "venv_python", return_value=root / "backend/.venv/bin/python"):
                data = json.loads(start.mcp_config(8123).read_text())["mcpServers"]["dl-lab"]
            self.assertEqual(data["env"]["DL_LAB_API"], "http://127.0.0.1:8123")
            self.assertTrue(Path(data["command"]).is_absolute())
            self.assertEqual(data["args"], [str(root / "backend/mcp_server/server.py")])

    def test_busy_port_does_not_install_or_kill_anything(self):
        with patch.object(start, "port_free", return_value=False), patch.object(start, "ensure_setup") as setup:
            with self.assertRaisesRegex(RuntimeError, "já está em uso"):
                start.main(["--port", "8123", "--no-browser"])
            setup.assert_not_called()

    def test_check_is_read_only(self):
        with patch.object(start, "tool", side_effect=lambda n: n), patch.object(start, "ensure_setup") as setup:
            start.main(["--check"])
            setup.assert_not_called()

    def test_completed_process_is_not_signalled(self):
        from unittest.mock import Mock
        proc = Mock()
        proc.poll.return_value = 0
        start.stop(proc, windows=True)
        proc.send_signal.assert_not_called()
