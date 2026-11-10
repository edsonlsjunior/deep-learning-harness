"""Local GitHub Copilot CLI bridge using the user's existing `copilot login`."""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from . import harness as H

SERVER = "dl-lab"
SESSION_RE = re.compile(r"^[0-9a-fA-F-]{8,64}$")


def executable():
    return os.environ.get("DL_LAB_COPILOT_BIN") or shutil.which("copilot")


def status():
    exe = executable()
    return {"id": "copilot", "name": "GitHub Copilot CLI", "configured": bool(exe), "default_model": "", "models": [], "available": bool(exe),
            "status": "CLI instalado; login verificado na primeira mensagem" if exe else "Instale o GitHub Copilot CLI e execute copilot login"}


def write_mcp_config(path, agent_token):
    from .chat import write_mcp_config as base
    base(path, agent_token)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["mcpServers"][SERVER]["tools"] = ["*"]
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


def command(exe, prompt, config_path, model=None, session_id=None, new_session_id=None):
    args = [exe, "-p", prompt, "--output-format", "json", "--stream", "on", "--silent",
            "--additional-mcp-config", f"@{config_path}", "--disable-builtin-mcps",
            "--no-custom-instructions", "--no-ask-user", "--disallow-temp-dir", "--no-auto-update",
            "--available-tools", *(f"{SERVER}-{name}" for name in H.TOOL_NAMES), "--allow-tool", SERVER]
    if model:
        args += ["--model", model]
    if session_id:
        args.append(f"--resume={session_id}")
    elif new_session_id:
        args += ["--session-id", new_session_id]
    return args


def tool_name(name):
    name = name or ""
    for prefix in (f"{SERVER}-", f"{SERVER}/", f"{SERVER}__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name if name in H.TOOL_NAMES else None


def normalize(event, state):
    """Convert one Copilot CLI JSONL event into chat UI events."""
    kind, data = event.get("type"), event.get("data") or {}
    if kind == "assistant.message_delta" and data.get("deltaContent"):
        state.setdefault("streamed", set()).add(data.get("messageId"))
        return [{"type": "text_delta", "text": data["deltaContent"]}]
    if kind == "assistant.message":
        if data.get("content") and data.get("messageId") not in state.get("streamed", set()):
            return [{"type": "text_delta", "text": data["content"]}]
        return []
    if kind == "tool.execution_start":
        name = tool_name(data.get("toolName"))
        if not name:
            return []
        state.setdefault("tools", set()).add(data.get("toolCallId"))
        return [{"type": "tool_use", "id": data.get("toolCallId"), "name": name, "input": data.get("arguments") or {}}]
    if kind == "tool.execution_complete":
        if data.get("toolCallId") not in state.get("tools", set()):
            return []
        ok = bool(data.get("success"))
        content = (data.get("result") or {}).get("content") if ok else (data.get("error") or {}).get("message")
        return [{"type": "tool_result", "tool_use_id": data.get("toolCallId"),
                 "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False), "is_error": not ok}]
    if kind == "session.error":
        state["failed"] = True
        return [{"type": "error", "message": "Copilot não concluiu a resposta. Verifique o login (copilot login) e os limites da assinatura."}]
    if kind == "result":
        usage = event.get("usage") or {}
        state["session_id"] = event.get("sessionId") or state.get("session_id", "")
        failed = bool(event.get("exitCode")) or state.get("failed", False)
        state["finished"] = True
        return [{"type": "result", "session_id": state["session_id"], "num_turns": max(1, len(state.get("tools", ())) + 1),
                 "duration_ms": usage.get("sessionDurationMs") or int((time.monotonic() - state.get("started", time.monotonic())) * 1000),
                 "total_cost_usd": None, "premium_requests": usage.get("premiumRequests"), "is_error": failed}]
    return []


def failure(stderr):
    text = stderr.lower()
    if "authentication" in text or "login" in text or "unauthorized" in text:
        return "GitHub Copilot sem login. Execute copilot login no ambiente do backend e tente novamente."
    if "session" in text and ("not found" in text or "resume" in text):
        return "A conversa anterior do Copilot não foi encontrada. Clique em Nova conversa e tente novamente."
    return "Copilot encerrou sem concluir. Verifique o login, a versão do CLI e os limites da assinatura."


async def stream_chat(message, session_id, model, project, agent_token=None, policy="consult"):
    from .chat import SYSTEM_PROMPT
    state = {"started": time.monotonic()}
    def sse(obj):
        return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"
    exe = executable()
    if not exe:
        yield sse({"type": "error", "message": status()["status"]})
        yield sse({"type": "done"})
        return
    if session_id and not SESSION_RE.match(session_id):
        session_id = None
    new_id = None if session_id else str(uuid.uuid4())
    instructions = (SYSTEM_PROMPT if not session_id else "") + H.instructions(policy)
    if project:
        instructions += f"\nProjeto atual: {project['id']} ({project['name']})."
    prompt = f"<instrucoes_do_laboratorio>\n{instructions}\n</instrucoes_do_laboratorio>\n\n{message}"
    with tempfile.TemporaryDirectory(prefix="dl-lab-copilot-") as folder:
        workspace = Path(folder)
        config = write_mcp_config(workspace / "mcp.json", agent_token)
        proc = stderr_task = None
        try:
            proc = await asyncio.create_subprocess_exec(*command(exe, prompt, config, model, session_id, new_id), cwd=folder,
                                                        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.PIPE, limit=4 * 1024 * 1024)
            stderr_task = asyncio.create_task(proc.stderr.read())
            state["session_id"] = session_id or new_id
            yield sse({"type": "init", "session_id": state["session_id"], "model": model or "padrão do Copilot",
                       "tools": H.TOOL_NAMES, "mcp_servers": [{"name": SERVER, "status": "starting"}]})
            async with asyncio.timeout(H.MAX_SECONDS):
                async for line in proc.stdout:
                    try:
                        event = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    for normalized in normalize(event, state):
                        yield sse(normalized)
                code = await proc.wait()
                stderr = (await stderr_task).decode(errors="replace")
            if code and not state.get("failed"):
                yield sse({"type": "error", "message": failure(stderr)})
            elif not state.get("finished") and not state.get("failed"):
                yield sse({"type": "result", "session_id": state["session_id"], "num_turns": 1,
                           "duration_ms": int((time.monotonic() - state["started"]) * 1000), "total_cost_usd": None, "is_error": False})
        except (OSError, TimeoutError):
            yield sse({"type": "error", "message": "Não foi possível concluir a execução local do Copilot."})
        finally:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
            if stderr_task is not None and not stderr_task.done():
                stderr_task.cancel()
    yield sse({"type": "done"})
