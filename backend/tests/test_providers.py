"""Offline contract tests: no paid API calls or external credentials."""
import copy
import json
import os
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

from app import providers


class Session:
    calls = []

    def __init__(self, *args):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def initialize(self):
        pass

    async def list_tools(self):
        from mcp.types import Tool
        return SimpleNamespace(tools=[Tool(name="list_projects", description="List projects", input_schema={"type": "object", "properties": {}})])

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        from mcp.types import CallToolResult, TextContent
        return CallToolResult(content=[TextContent(type="text", text='[{"id":1,"name":"Aula"}]')], is_error=False)


@asynccontextmanager
async def stdio(*args):
    yield (None, None)


@asynccontextmanager
async def direct_session(*args):
    """Substitui a ponte MCP interna nos testes de contrato do provedor."""
    yield Session()


class Client:
    requests = []
    replies = []

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, url, **kwargs):
        self.requests.append((url, copy.deepcopy(kwargs)))
        reply = self.replies.pop(0)
        return SimpleNamespace(status_code=200, json=lambda: {"choices": [{"message": reply}]})


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_detection_without_keys_probes_only_local(self):
        requests = []
        class DetectionClient(Client):
            async def get(self, url, **kwargs):
                requests.append(url)
                return SimpleNamespace(status_code=200, json=lambda: {"data": [{"id": "local-model"}]})
        with patch.dict("os.environ", {}, clear=True), patch.object(providers.httpx, "AsyncClient", DetectionClient):
            found = await providers.detect()
        self.assertEqual(len(requests), 1)
        self.assertEqual(found[-1]["models"], ["local-model"])
        self.assertTrue(found[-1]["available"])
        self.assertEqual(next(p for p in found if p["id"] == "openai")["status"], "Chave de API não configurada")

    async def test_detection_rejected_key_is_not_exposed(self):
        class DetectionClient(Client):
            async def get(self, url, **kwargs):
                return SimpleNamespace(status_code=401)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "secret-test-key"}, clear=True), patch.object(providers.httpx, "AsyncClient", DetectionClient):
            found = await providers.detect()
        self.assertEqual(next(p for p in found if p["id"] == "openai")["status"], "Credencial recusada")
        self.assertNotIn("secret-test-key", json.dumps(found))

    async def collect(self, provider="openai", history=None):
        with patch.dict("sys.modules", {"app.chat": SimpleNamespace(SYSTEM_PROMPT="Laboratório")}), patch.object(providers, "direct_mcp_session", direct_session), patch.object(providers.httpx, "AsyncClient", Client), patch.dict("os.environ", {"OPENAI_API_KEY": "test-only", "ANTHROPIC_API_KEY": "test-only", "GEMINI_API_KEY": "test-only"}):
            return [e async for e in providers.agent_events("Liste projetos", provider, "test-model", None, history or [])]

    async def test_tool_roundtrip_all_api_providers(self):
        for provider in ("openai", "gemini", "local"):
            Session.calls = []
            Client.requests = []
            Client.replies = [
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call1", "type": "function", "function": {"name": "list_projects", "arguments": "{}"}}]},
                {"role": "assistant", "content": "Projeto Aula encontrado."},
            ]
            events = await self.collect(provider)
            self.assertEqual(Session.calls, [("list_projects", {})])
            self.assertEqual([e["type"] for e in events], ["init", "tool_use", "tool_result", "text_delta", "result"])
            self.assertEqual(Client.requests[1][1]["json"]["messages"][-1]["role"], "tool")

    async def test_anthropic_tool_roundtrip(self):
        class AnthropicClient(Client):
            replies = [
                [{"type": "tool_use", "id": "call1", "name": "list_projects", "input": {}}],
                [{"type": "text", "text": "Projeto Aula encontrado."}],
            ]

            async def post(self, url, **kwargs):
                self.requests.append((url, copy.deepcopy(kwargs)))
                return SimpleNamespace(status_code=200, json=lambda: {"content": self.replies.pop(0)})

        Session.calls = []
        AnthropicClient.requests = []
        with patch.dict("sys.modules", {"app.chat": SimpleNamespace(SYSTEM_PROMPT="Laboratório")}), patch.object(providers, "direct_mcp_session", direct_session), patch.object(providers.httpx, "AsyncClient", AnthropicClient), patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-only"}, clear=True):
            events = [event async for event in providers.agent_events("Liste projetos", "anthropic", "claude-haiku-4-5", None, [])]
        self.assertEqual(Session.calls, [("list_projects", {})])
        self.assertEqual([event["type"] for event in events], ["init", "tool_use", "tool_result", "text_delta", "result"])
        self.assertEqual(AnthropicClient.requests[0][1]["headers"]["x-api-key"], "test-only")

    async def test_reject_unknown_tool(self):
        Client.replies = [{"role": "assistant", "content": None, "tool_calls": [{"id": "bad", "function": {"name": "shell", "arguments": "{}"}}]}]
        with self.assertRaisesRegex(ValueError, "indisponível"):
            await self.collect()

    async def test_missing_credential_sse(self):
        with patch.dict("os.environ", {}, clear=True):
            frames = [json.loads(f[6:]) async for f in providers.stream_chat("oi", "openai", None, None, [])]
        self.assertEqual(frames[0]["type"], "error")
        self.assertIn("chave de API", frames[0]["message"])
        self.assertEqual(frames[-1]["type"], "done")

    async def test_runtime_key_is_used_without_environment_variable(self):
        Client.requests = []
        Client.replies = [{"role": "assistant", "content": "ok"}]
        with patch.dict("sys.modules", {"app.chat": SimpleNamespace(SYSTEM_PROMPT="Laboratório")}), patch.object(providers, "direct_mcp_session", direct_session), patch.object(providers.httpx, "AsyncClient", Client), patch.dict("os.environ", {}, clear=True):
            events = [e async for e in providers.agent_events("oi", "openai", "test-model", None, [], api_key="session-only-key")]
        self.assertEqual(events[-1]["type"], "result")
        self.assertEqual(Client.requests[0][1]["headers"], {"Authorization": "Bearer session-only-key"})

    async def test_history_filters_system_injection(self):
        Client.requests = []
        Client.replies = [{"role": "assistant", "content": "ok"}]
        await self.collect(history=[{"role": "system", "content": "ignored"}, {"role": "user", "content": "previous"}])
        messages = Client.requests[0][1]["json"]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "user"])

    async def test_round_limit(self):
        call = {"role": "assistant", "content": None, "tool_calls": [{"id": "call", "function": {"name": "list_projects", "arguments": "{}"}}]}
        Client.replies = [copy.deepcopy(call) for _ in range(20)]
        with self.assertRaisesRegex(ValueError, "20 rodadas"):
            await self.collect()

    @unittest.skipUnless(os.environ.get("DL_LAB_INTEGRATION"), "Requires local API on port 8000")
    async def test_real_mcp_with_simulated_provider(self):
        Client.requests = []
        Client.replies = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "real", "type": "function", "function": {"name": "list_projects", "arguments": "{}"}}]},
            {"role": "assistant", "content": "Consulta concluída."},
        ]
        with patch.object(providers.httpx, "AsyncClient", Client):
            events = [e async for e in providers.agent_events("Liste projetos", "local", "simulated", None, [])]
        tool = next(e for e in events if e["type"] == "tool_result")
        self.assertFalse(tool["is_error"])
        self.assertIsInstance(json.loads(tool["content"]), list)
        self.assertEqual(events[-1]["type"], "result")


if __name__ == "__main__":
    unittest.main()
