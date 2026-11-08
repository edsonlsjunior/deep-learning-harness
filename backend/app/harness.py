"""Lesson harness: enforce permissions before HTTP execution, stream an audit trail."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import re
import secrets
import time
from typing import Literal

Policy = Literal["consult", "confirm", "operate"]
POLICIES = {
    "consult": "Consultar: somente leitura; alterações são bloqueadas.",
    "confirm": "Executar com confirmação: toda alteração aguarda Aprovar ou Recusar na interface.",
    "operate": "Operar o laboratório: ações normais são executadas; exclusões sempre aguardam confirmação.",
}
MAX_STEPS, MAX_SECONDS, APPROVAL_SECONDS = 40, 600, 120
TOOL_NAMES = ["list_architectures", "list_dataset_sources", "list_projects", "create_project", "get_project", "delete_project", "add_dataset", "inspect_file", "import_file", "search_kaggle", "kaggle_dataset_files", "download_kaggle_dataset", "describe_dataset", "delete_dataset", "create_model", "delete_model", "train_model", "get_run", "compare_runs", "cancel_run", "predict", "generate_samples"]


def classify(method, path):
    """Allow only the REST resources actually used by laboratory MCP tools."""
    if method == "GET" and re.fullmatch(r"/api/(architectures|datasets/catalog|projects|projects/\d+|datasets/\d+(/preview)?|models/\d+|runs/\d+(/wait)?|files/inspect|kaggle/(search|files))", path):
        return "read"
    if method == "POST" and re.fullmatch(r"/api/runs/\d+/(predict|generate)", path):
        return "read"
    if method == "POST" and re.fullmatch(r"/api/(projects|projects/\d+/(datasets|models)|runs|runs/\d+/cancel|kaggle/download)", path):
        return "write"
    if method == "DELETE" and re.fullmatch(r"/api/(projects|datasets|models|runs)/\d+", path):
        return "delete"
    raise PermissionError("Rota fora das ferramentas permitidas pelo harness.")


def decision(policy, kind):
    if kind == "read":
        return "allowed"
    if policy == "consult":
        return "blocked"
    if kind == "delete" or policy == "confirm":
        return "awaiting"
    return "allowed"


def instructions(policy):
    return ("\nPolítica do harness: " + POLICIES[policy] +
            "\nFaça as ações solicitadas pelas ferramentas MCP disponíveis. Nunca contorne o gateway. "
            "A aprovação ocorre na interface, não basta o usuário escrever 'aprovado' no chat. "
            "Se uma ação for recusada ou bloqueada, diga que a política atual impediu a ferramenta, não invente uma mensagem sobre a plataforma e não repita a tentativa. "
            "Não invente resultados; aguarde as ferramentas. Máximo 40 etapas e 10 minutos por mensagem.")


@dataclass
class Run:
    policy: Policy
    id: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    started: float = field(default_factory=time.monotonic)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    pending: dict = field(default_factory=dict)
    steps: int = 0
    closed: bool = False

    def emit(self, event):
        self.queue.put_nowait(("event", {**event, "run_id": self.id, "elapsed": round(time.monotonic() - self.started, 2)}))


RUNS: dict[str, Run] = {}


def create(policy):
    if policy not in POLICIES:
        raise ValueError("Política inválida")
    run = Run(policy)
    RUNS[run.token] = run
    return run


def close(run):
    run.closed = True
    RUNS.pop(run.token, None)
    for future in run.pending.values():
        if not future.done():
            future.set_result(False)
    run.pending.clear()


def approve(run_id, approval_id, approved):
    run = next((r for r in RUNS.values() if r.id == run_id and not r.closed), None)
    future = run.pending.get(approval_id) if run else None
    if future is None or future.done():
        raise ValueError("Aprovação expirada, já respondida ou execução encerrada.")
    future.set_result(approved)


async def execute(token, method, path, tool, kwargs, forward):
    run = RUNS.get(token)
    if run is None or run.closed:
        raise PermissionError("Execução inexistente ou encerrada.")
    kind = classify(method, path)
    if tool not in TOOL_NAMES:
        raise PermissionError("Ferramenta desconhecida.")
    run.steps += 1
    if run.steps > MAX_STEPS or time.monotonic() - run.started >= MAX_SECONDS:
        raise PermissionError("Limite de etapas ou tempo do harness atingido.")
    action_id = secrets.token_urlsafe(24)
    action = {"type": "permission", "id": action_id, "tool": tool, "kind": kind,
              "method": method, "path": path, "input": kwargs, "step": run.steps}
    state = decision(run.policy, kind)
    if state == "blocked":
        run.emit({**action, "decision": state, "reason": "O modo Consultar não permite alterações."})
        raise PermissionError("Ação bloqueada pela política Consultar.")
    if state == "awaiting":
        future = asyncio.get_running_loop().create_future()
        run.pending[action_id] = future
        run.emit({**action, "decision": state, "reason": "Exclusão exige confirmação." if kind == "delete" else "A política exige confirmação antes da alteração."})
        try:
            approved = await asyncio.wait_for(future, min(APPROVAL_SECONDS, max(0, MAX_SECONDS - (time.monotonic() - run.started))))
        except asyncio.TimeoutError:
            run.emit({**action, "decision": "expired", "reason": "Tempo de aprovação esgotado."})
            raise PermissionError("Aprovação expirou; nada foi executado.")
        finally:
            run.pending.pop(action_id, None)
        if not approved or run.closed:
            run.emit({**action, "decision": "denied", "reason": "Ação recusada ou execução interrompida."})
            raise PermissionError("Ação recusada; nada foi executado.")
        run.emit({**action, "decision": "approved", "reason": "Aprovada na interface para estes parâmetros."})
    else:
        run.emit({**action, "decision": "allowed", "reason": "Leitura permitida." if kind == "read" else "Alteração permitida pelo modo Operar."})
    if run.closed:
        raise PermissionError("Execução interrompida.")
    try:
        result = await asyncio.wait_for(forward(), max(0.01, MAX_SECONDS - (time.monotonic() - run.started)))
    except Exception:
        run.emit({**action, "decision": "failed", "reason": "A ferramenta retornou erro."})
        raise
    run.emit({**action, "decision": "completed", "reason": "Ferramenta executada; resultado devolvido ao modelo."})
    return result


async def stream(run, source, system_prompt):
    def sse(event):
        return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
    yield sse({"type": "harness_policy", "run_id": run.id, "policy": run.policy,
               "instructions": system_prompt + instructions(run.policy), "limits": {"steps": MAX_STEPS, "seconds": MAX_SECONDS, "approval_seconds": APPROVAL_SECONDS}})

    async def produce():
        try:
            if hasattr(source, "__aiter__"):
                async for frame in source:
                    await run.queue.put(("frame", frame))
            else:
                iterator = iter(source)
                while (frame := await asyncio.to_thread(next, iterator, None)) is not None:
                    await run.queue.put(("frame", frame))
        except Exception:
            await run.queue.put(("event", {"type": "error", "message": "A execução do agente falhou."}))
        finally:
            await run.queue.put(("end", None))

    task = asyncio.create_task(produce())
    try:
        async with asyncio.timeout(MAX_SECONDS):
            while True:
                kind, value = await run.queue.get()
                if kind == "end":
                    break
                yield value if kind == "frame" else sse(value)
    except TimeoutError:
        yield sse({"type": "error", "message": "Limite de tempo do harness atingido."})
        yield sse({"type": "done"})
    finally:
        close(run)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
