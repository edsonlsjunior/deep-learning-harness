"""Read-only local Codex bridge using the user's existing ChatGPT login."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from . import harness as H

READ_TOOLS = ["list_architectures", "list_dataset_sources", "list_projects", "get_project", "describe_dataset", "get_run", "compare_runs", "predict"]


def executable():
    found = os.environ.get("DL_LAB_CODEX_BIN") or shutil.which("codex")
    if found:
        return found
    bundled = Path("/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex")
    return str(bundled) if bundled.is_file() else None


async def status():
    item = {"id": "codex", "name": "Codex — assinatura ChatGPT", "configured": False, "default_model": "", "models": [], "available": False}
    exe = executable()
    if not exe:
        return {**item, "status": "Instale Codex CLI e execute codex login"}
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(exe, "login", "status", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await asyncio.wait_for(proc.communicate(), 8)
        logged = proc.returncode == 0 and "chatgpt" in (out + err).decode(errors="replace").lower()
        return {**item, "configured": logged, "available": logged, "status": "Conectado com ChatGPT" if logged else "Execute codex login e entre com ChatGPT"}
    except (OSError, asyncio.TimeoutError):
        return {**item, "status": "Não foi possível verificar o login local"}
    finally:
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()


def command(exe, workspace, model, agent_token=None):
    server = Path(__file__).resolve().parent.parent / "mcp_server" / "server.py"
    cfg = {
        "forced_login_method": "chatgpt", "web_search": "disabled",
        "model_instructions_file": str(workspace / "instructions.md"),
        "mcp_servers.dl-lab.command": sys.executable,
        "mcp_servers.dl-lab.args": [str(server)],
        "mcp_servers.dl-lab.env": {"DL_LAB_API": os.environ.get("DL_LAB_API", "http://127.0.0.1:8000"), "PYTHONIOENCODING": "utf-8"},
        "mcp_servers.dl-lab.required": True,
        "mcp_servers.dl-lab.startup_timeout_sec": 120,
        "mcp_servers.dl-lab.tool_timeout_sec": 660,
        "mcp_servers.dl-lab.enabled_tools": H.TOOL_NAMES if agent_token else READ_TOOLS,
    }
    if agent_token:
        cfg["mcp_servers.dl-lab.env"]["DL_LAB_AGENT_TOKEN"] = agent_token
    args = [exe, "exec", "--json", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check", "--sandbox", "read-only", "--cd", str(workspace)]
    for feature in ("shell_tool", "view_image", "apps", "plugins", "remote_plugin", "hooks", "multi_agent", "computer_use", "browser_use", "image_generation", "memories", "skill_search", "workspace_dependencies"):
        args += ["--disable", feature]
    for key, value in cfg.items():
        encoded = "{" + ", ".join(f"{k} = {json.dumps(v)}" for k, v in value.items()) + "}" if isinstance(value, dict) else json.dumps(value)
        args += ["-c", f"{key}={encoded}"]
    if model:
        args += ["--model", model]
    return args + ["-"]


def normalize(event):
    kind, item = event.get("type"), event.get("item", {})
    if kind == "item.completed" and item.get("type") == "agent_message":
        return [{"type": "text_delta", "text": item.get("text", "")}]
    if item.get("type") == "mcp_tool_call":
        if kind == "item.started":
            return [{"type": "tool_use", "id": item["id"], "name": item.get("tool", ""), "input": item.get("arguments", {})}]
        if kind == "item.completed":
            return [{"type": "tool_result", "tool_use_id": item["id"], "content": json.dumps(item.get("result") or item.get("error"), ensure_ascii=False), "is_error": item.get("status") == "failed"}]
    if kind in ("error", "turn.failed"):
        return [{"type": "error", "message": "Codex não concluiu a resposta. Verifique o login e os limites do plano."}]
    return []


async def stream_chat(message, model, project, history, agent_token=None, policy="consult"):
    started = time.monotonic()
    def sse(obj):
        return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"
    info = await status()
    if not info["available"]:
        yield sse({"type": "error", "message": info["status"]})
        yield sse({"type": "done"})
        return
    with tempfile.TemporaryDirectory(prefix="dl-lab-codex-") as folder:
        workspace = Path(folder)
        instructions = "Você é o assistente do Laboratório de Deep Learning. Responda em português. Use apenas as ferramentas MCP dl-lab de consulta. Não afirme que criou, treinou ou excluiu algo; esta integração permite somente consultas. Oriente o usuário a usar os botões da interface para alterações. Não invente resultados."
        if agent_token:
            from .chat import SYSTEM_PROMPT
            instructions = SYSTEM_PROMPT + H.instructions(policy)
        if project:
            instructions += f"\nProjeto atual: {project['id']} ({project['name']})."
        (workspace / "instructions.md").write_text(instructions, encoding="utf-8")
        previous = [{"role": m["role"], "content": m["content"][:20000]} for m in history[-30:] if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
        prompt = json.dumps({"historico": previous, "mensagem_atual": message}, ensure_ascii=False)
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(*command(executable(), workspace, model, agent_token), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, limit=4 * 1024 * 1024)
            proc.stdin.write(prompt.encode())
            await proc.stdin.drain()
            proc.stdin.close()
            yield sse({"type": "init", "session_id": "", "model": model or "padrão do Codex", "tools": H.TOOL_NAMES if agent_token else READ_TOOLS, "mcp_servers": [{"name": "dl-lab", "status": "starting"}]})
            async with asyncio.timeout(H.MAX_SECONDS):
                async for line in proc.stdout:
                    try:
                        event = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    for normalized in normalize(event):
                        yield sse(normalized)
                code = await proc.wait()
            if code:
                yield sse({"type": "error", "message": "Codex encerrou sem concluir. Verifique o login, a versão do CLI e os limites do plano."})
            else:
                yield sse({"type": "result", "session_id": "", "num_turns": 1, "duration_ms": int((time.monotonic() - started) * 1000), "total_cost_usd": None, "is_error": False})
        except (OSError, TimeoutError):
            yield sse({"type": "error", "message": "Não foi possível concluir a execução local do Codex."})
        finally:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
    yield sse({"type": "done"})
