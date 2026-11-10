"""Provider-neutral agent loop using the laboratory's existing MCP tools.

Credentials stay on the backend. OpenAI-compatible endpoints include Gemini's
compatibility API and local servers such as Ollama.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
from . import codex_chat, copilot_chat
from . import harness as H

PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4.1-mini"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", "claude-haiku-4-5"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-2.5-flash"),
    "local": ("http://127.0.0.1:11434/v1", "", ""),
}


async def mcp_status() -> dict:
    """Diagnóstico seguro do subprocesso MCP, sem chamar modelo nem usar credencial.

    Além da inicialização, executa uma leitura real pelo gateway do Harness. Isso
    confirma que o subprocesso consegue voltar ao servidor no contêiner.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    server = Path(__file__).resolve().parent.parent / "mcp_server" / "server.py"
    run = H.create("consult")
    params = StdioServerParameters(command=sys.executable, args=[str(server)], env={**os.environ, "PYTHONIOENCODING": "utf-8", "DL_LAB_AGENT_TOKEN": run.token})
    try:
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                async with asyncio.timeout(20):
                    await session.initialize()
                    tools = (await session.list_tools()).tools
                    probe = await session.call_tool("list_projects", {})
                return {"ok": not bool(probe.is_error), "tools": len(tools), "gateway": "ok" if not probe.is_error else "erro"}
    except TimeoutError:
        return {"ok": False, "detail": "O subprocesso MCP não respondeu em 20 segundos."}
    except Exception:
        return {"ok": False, "detail": "O subprocesso MCP não iniciou corretamente."}
    finally:
        H.close(run)


def public_mode() -> bool:
    """Hospedagem pública nunca usa logins/variáveis do operador do servidor."""
    return os.environ.get("DL_LAB_PUBLIC_MODE", "").lower() in {"1", "true", "yes"}


def catalog(session_providers: set[str] | frozenset[str] = frozenset()):
    cli = [] if public_mode() else [{"id": "codex", "name": "Codex — assinatura ChatGPT", "configured": False, "default_model": ""}, {"id": "claude-code", "name": "Claude Code", "configured": bool(__import__("shutil").which("claude")), "default_model": ""}, {"id": "copilot", "name": "GitHub Copilot CLI", "configured": bool(copilot_chat.executable()), "default_model": ""}]
    provider_ids = ("openai",) if public_mode() else PROVIDERS
    return cli + [
        {"id": key, "name": {"openai": "OpenAI API", "anthropic": "Claude API (Anthropic)", "gemini": "Gemini API", "local": "Local (Ollama)"}[key],
         "configured": key in session_providers or (not public_mode() and (not env or bool(os.environ.get(env)))), "default_model": model}
        for key in provider_ids for _, env, model in [PROVIDERS[key]]
    ]


async def detect(session_credentials: dict[str, str] | None = None):
    """Check configured catalogs without generating text or returning credentials."""
    session_credentials = session_credentials or {}
    async def probe(item):
        item = {**item, "models": [], "available": False}
        if item["id"] == "codex":
            return await codex_chat.status()
        if item["id"] == "copilot":
            return copilot_chat.status()
        if item["id"] == "claude-code":
            item["status"] = "CLI instalado; login não verificado" if item["configured"] else "CLI não instalado"
            return item
        base, env, _ = PROVIDERS[item["id"]]
        key = session_credentials.get(item["id"]) or ("" if public_mode() else (os.environ.get(env, "") if env else "local"))
        if env and not key:
            item["status"] = "Chave de API não configurada"
            return item
        if item["id"] == "local":
            base = os.environ.get("LOCAL_LLM_BASE_URL", base)
        headers = ({"x-api-key": key, "anthropic-version": "2023-06-01"}
                   if item["id"] == "anthropic" else {"Authorization": f"Bearer {key}"} if env else {})
        try:
            async with httpx.AsyncClient(timeout=5, follow_redirects=False) as client:
                response = await client.get(base.rstrip("/") + "/models", headers=headers)
            if response.status_code != 200:
                item["status"] = "Credencial recusada" if response.status_code in (401, 403) else f"Serviço respondeu HTTP {response.status_code}"
                return item
            models = response.json().get("data", [])
            item["models"] = sorted({m["id"] for m in models if isinstance(m, dict) and isinstance(m.get("id"), str)})
            item["available"] = bool(item["models"])
            item["status"] = "Conectado; confirme suporte a ferramentas do modelo" if item["available"] else "Conectado, sem modelos disponíveis"
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            item["status"] = "Serviço indisponível ou resposta inválida"
        return item
    return await asyncio.gather(*(probe(p) for p in catalog(set(session_credentials))))


async def anthropic_agent_events(message, model, project, history, agent_token, policy, api_key, owner_session):
    """Loop nativo da Messages API da Anthropic, incluindo ferramentas MCP."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from .chat import SYSTEM_PROMPT

    started = time.monotonic()
    prompt = SYSTEM_PROMPT + (f"\nProjeto atual: {project['id']} ({project['name']})." if project else "")
    if agent_token:
        prompt += H.instructions(policy)
    messages = []
    for item in history[-30:]:
        if item.get("role") in ("user", "assistant") and isinstance(item.get("content"), str):
            messages.append({"role": item["role"], "content": item["content"][:20000]})
    messages.append({"role": "user", "content": message})
    server = Path(__file__).resolve().parent.parent / "mcp_server" / "server.py"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if agent_token:
        env["DL_LAB_AGENT_TOKEN"] = agent_token
    if owner_session:
        env["DL_LAB_OWNER_SESSION"] = owner_session
    params = StdioServerParameters(command=sys.executable, args=[str(server)], env=env)
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            try:
                async with asyncio.timeout(20):
                    await session.initialize()
                    available = (await session.list_tools()).tools
            except TimeoutError as exc:
                raise ValueError("O Harness interno não respondeu em 20 segundos. Tente novamente em instantes.") from exc
            tools = [{"name": tool.name, "description": tool.description or "", "input_schema": tool.input_schema} for tool in available]
            names = {tool.name for tool in available}
            yield {"type": "init", "session_id": "", "model": model, "tools": sorted(names), "mcp_servers": [{"name": "dl-lab", "status": "connected"}]}
            headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            async with httpx.AsyncClient(timeout=120) as client:
                for turn in range(1, 21):
                    response = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json={"model": model, "max_tokens": 4096, "system": prompt, "messages": messages, "tools": tools})
                    if response.status_code >= 400:
                        try:
                            reason = response.json().get("error", {}).get("message", "")
                        except (ValueError, AttributeError):
                            reason = ""
                        suffix = f": {reason[:300]}" if reason else ""
                        raise ValueError(f"A API Claude recusou a solicitação (HTTP {response.status_code}){suffix}")
                    try:
                        blocks = response.json()["content"]
                    except (ValueError, KeyError, TypeError) as exc:
                        raise ValueError("A API Claude devolveu uma resposta inválida.") from exc
                    if not isinstance(blocks, list):
                        raise ValueError("A API Claude devolveu uma resposta inválida.")
                    messages.append({"role": "assistant", "content": blocks})
                    calls = [block for block in blocks if isinstance(block, dict) and block.get("type") == "tool_use"]
                    for block in blocks:
                        if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                            yield {"type": "text_delta", "text": block["text"]}
                    if not calls:
                        yield {"type": "result", "session_id": "", "num_turns": turn, "duration_ms": int((time.monotonic() - started) * 1000), "total_cost_usd": None, "is_error": False}
                        return
                    results = []
                    for call in calls:
                        name, args, call_id = call.get("name"), call.get("input", {}), call.get("id")
                        if name not in names or not isinstance(args, dict) or not call_id:
                            raise ValueError("O modelo solicitou uma ferramenta inválida.")
                        yield {"type": "tool_use", "id": call_id, "name": name, "input": args}
                        result = await session.call_tool(name, args)
                        content = "\n".join(getattr(item, "text", "") for item in result.content)
                        yield {"type": "tool_result", "tool_use_id": call_id, "content": content, "is_error": bool(result.is_error)}
                        results.append({"type": "tool_result", "tool_use_id": call_id, "content": content, "is_error": bool(result.is_error)})
                    messages.append({"role": "user", "content": results})
    raise ValueError("Limite de 20 rodadas atingido. Consulte o projeto antes de repetir ações.")


async def agent_events(message, provider, model, project, history, agent_token=None, policy="consult", api_key=None, owner_session=None):
    started = time.monotonic()
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from .chat import SYSTEM_PROMPT

    if provider not in PROVIDERS:
        raise ValueError("Provedor desconhecido")
    base, env, default = PROVIDERS[provider]
    key = api_key or ("" if public_mode() else (os.environ.get(env, "") if env else "local"))
    if not key:
        raise ValueError("Conecte sua chave de API nesta sessão para usar este provedor.")
    model = model or default
    if not model:
        raise ValueError("Informe o nome do modelo instalado no servidor local.")
    if provider == "anthropic":
        async for event in anthropic_agent_events(message, model, project, history, agent_token, policy, key, owner_session):
            yield event
        return
    if provider == "local":
        base = os.environ.get("LOCAL_LLM_BASE_URL", base)
    prompt = SYSTEM_PROMPT + (f"\nProjeto atual: {project['id']} ({project['name']})." if project else "")
    if agent_token:
        prompt += H.instructions(policy)
    messages = [{"role": "system", "content": prompt}]
    for item in history[-30:]:
        if item.get("role") in ("user", "assistant") and isinstance(item.get("content"), str):
            messages.append({"role": item["role"], "content": item["content"][:20000]})
    messages.append({"role": "user", "content": message})
    server = Path(__file__).resolve().parent.parent / "mcp_server" / "server.py"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if agent_token:
        env["DL_LAB_AGENT_TOKEN"] = agent_token
    if owner_session:
        env["DL_LAB_OWNER_SESSION"] = owner_session
    params = StdioServerParameters(command=sys.executable, args=[str(server)], env=env)
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            try:
                async with asyncio.timeout(20):
                    await session.initialize()
                    available = (await session.list_tools()).tools
            except TimeoutError as exc:
                raise ValueError("O Harness interno não respondeu em 20 segundos. Tente novamente em instantes.") from exc
            tools = [{"type": "function", "function": {"name": t.name, "description": t.description or "", "parameters": t.input_schema}} for t in available]
            names = {t.name for t in available}
            yield {"type": "init", "session_id": "", "model": model, "tools": sorted(names), "mcp_servers": [{"name": "dl-lab", "status": "connected"}]}
            async with httpx.AsyncClient(timeout=120) as client:
                for turn in range(1, 21):
                    response = await client.post(base.rstrip("/") + "/chat/completions", headers={"Authorization": f"Bearer {key}"}, json={"model": model, "messages": messages, "tools": tools})
                    if response.status_code >= 400:
                        raise ValueError(f"O provedor recusou a solicitação (HTTP {response.status_code}). Confira credencial, modelo e suporte a ferramentas.")
                    reply = response.json()["choices"][0]["message"]
                    calls = reply.get("tool_calls") or []
                    messages.append({k: v for k, v in reply.items() if k in ("role", "content", "tool_calls")})
                    if reply.get("content"):
                        yield {"type": "text_delta", "text": reply["content"]}
                    if not calls:
                        yield {"type": "result", "session_id": "", "num_turns": turn, "duration_ms": int((time.monotonic() - started) * 1000), "total_cost_usd": None, "is_error": False}
                        return
                    for call in calls:
                        name = call["function"]["name"]
                        args = json.loads(call["function"]["arguments"])
                        if name not in names:
                            raise ValueError("O modelo solicitou uma ferramenta indisponível.")
                        yield {"type": "tool_use", "id": call["id"], "name": name, "input": args}
                        result = await session.call_tool(name, args)
                        content = "\n".join(getattr(c, "text", "") for c in result.content)
                        yield {"type": "tool_result", "tool_use_id": call["id"], "content": content, "is_error": bool(result.is_error)}
                        messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
                raise ValueError("Limite de 20 rodadas atingido. Consulte o projeto antes de repetir ações.")


async def stream_chat(message, provider, model, project, history, agent_token=None, policy="consult", api_key=None, owner_session=None):
    try:
        async for event in agent_events(message, provider, model, project, history, agent_token, policy, api_key, owner_session):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # Do not expose upstream request objects or credentials in UI errors.
        detail = str(exc) if isinstance(exc, ValueError) else "Falha ao comunicar com o provedor ou servidor MCP. Consulte a configuração do backend."
        yield f"data: {json.dumps({'type': 'error', 'message': detail}, ensure_ascii=False)}\n\n"
    yield 'data: {"type":"done"}\n\n'
