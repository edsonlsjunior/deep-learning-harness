"""API REST do Laboratório de Deep Learning.

Consumida por dois clientes:
  1. o frontend React (humano clicando)
  2. o servidor MCP (Claude agindo via ferramentas)
Ambos enxergam exatamente o mesmo estado — esse é o ponto do harness.
"""
from __future__ import annotations

import uuid
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import architectures as A
from . import chat, db, files, kaggle, training, providers, codex_chat, copilot_chat, harness as H
from . import datasets as D

app = FastAPI(title="DL Lab API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(ValueError)
async def value_error(_: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(httpx.HTTPError)
async def upstream_error(_: Request, exc: httpx.HTTPError):
    return JSONResponse(status_code=502, content={"detail": f"Falha de rede ao acessar serviço externo: {exc}"})


def _404(obj: Any, what: str):
    if not obj:
        raise HTTPException(404, f"{what} não encontrado")
    return obj


# ------------------------------------------------------------------ schemas
class ProjectIn(BaseModel):
    name: str
    description: str = ""


class DatasetIn(BaseModel):
    source: str = Field(description="nome de uma base embutida ou 'file' (params.path)")
    name: str | None = None
    params: dict[str, Any] = {}


class ModelIn(BaseModel):
    name: str
    architecture: str
    config: dict[str, Any] = {}


class RunIn(BaseModel):
    model_id: int
    dataset_id: int
    hyperparams: dict[str, Any] = {}


class PredictIn(BaseModel):
    inputs: list[Any]


class KaggleDownloadIn(BaseModel):
    ref: str
    file_name: str | None = None


class ChatIn(BaseModel):
    message: str
    session_id: str | None = None
    project_id: int | None = None
    model: str | None = None
    provider: str = "claude-code"
    history: list[dict[str, str]] = Field(default_factory=list, max_length=30)
    policy: H.Policy = "operate"


class AgentCall(BaseModel):
    method: str
    path: str
    tool: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class ApprovalIn(BaseModel):
    approved: bool


# ------------------------------------------------------------------ catálogo
@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/architectures")
def architectures():
    return [{"name": k, **{f: v[f] for f in ("aula", "description", "tasks", "default_config", "config_doc")},
             "default_hyperparams": {**training.COMMON_HP, **training.DEFAULT_HP[k]}}
            for k, v in A.ARCHITECTURES.items()]


@app.get("/api/datasets/catalog")
def dataset_catalog():
    return D.catalog()


# ------------------------------------------------------------------ projetos
def _run_summary(r: dict) -> dict:
    r = dict(r)
    r.pop("extras", None)
    return r


@app.get("/api/projects")
def list_projects():
    return db.query("""SELECT p.*,
        (SELECT COUNT(*) FROM datasets d WHERE d.project_id = p.id) AS n_datasets,
        (SELECT COUNT(*) FROM models m WHERE m.project_id = p.id) AS n_models,
        (SELECT COUNT(*) FROM runs r WHERE r.project_id = p.id) AS n_runs
        FROM projects p ORDER BY p.id DESC""")


@app.post("/api/projects")
def create_project(body: ProjectIn):
    pid = db.insert("projects", {"name": body.name, "description": body.description, "created_at": db.now()})
    return db.one("SELECT * FROM projects WHERE id = ?", (pid,))


@app.get("/api/projects/{pid}")
def get_project(pid: int):
    p = _404(db.one("SELECT * FROM projects WHERE id = ?", (pid,)), "Projeto")
    p["datasets"] = db.query("SELECT * FROM datasets WHERE project_id = ? ORDER BY id", (pid,))
    p["models"] = db.query("SELECT * FROM models WHERE project_id = ? ORDER BY id", (pid,))
    p["runs"] = [_run_summary(r) for r in db.query("SELECT * FROM runs WHERE project_id = ? ORDER BY id DESC", (pid,))]
    return p


@app.delete("/api/projects/{pid}")
def delete_project(pid: int):
    _404(db.one("SELECT id FROM projects WHERE id = ?", (pid,)), "Projeto")
    for d in db.query("SELECT id FROM datasets WHERE project_id = ?", (pid,)):
        D.remove(d["id"])
    db.delete("projects", pid)
    return {"deleted": pid}


# ------------------------------------------------------------------ bases de dados
@app.post("/api/projects/{pid}/datasets")
def add_dataset(pid: int, body: DatasetIn):
    _404(db.one("SELECT id FROM projects WHERE id = ?", (pid,)), "Projeto")
    return D.create(pid, body.source, body.name, body.params)


async def _save_upload(file: UploadFile, prefix: str) -> Path:
    """Grava o upload em data/uploads/ respeitando formato e tamanho máximos."""
    name = Path(file.filename or "dados.csv").name
    files.check_ext(name)
    up = db.DATA_DIR / "uploads"
    up.mkdir(exist_ok=True)
    dest = up / f"{prefix}_{uuid.uuid4().hex[:8]}_{name}"
    limit, total = files.MAX_UPLOAD_MB * 1024 * 1024, 0
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            total += len(chunk)
            if total > limit:
                f.close()
                dest.unlink(missing_ok=True)
                raise ValueError(f"Arquivo maior que {files.MAX_UPLOAD_MB} MB")
            f.write(chunk)
    return dest


@app.post("/api/projects/{pid}/datasets/upload")
async def upload_dataset(pid: int, file: UploadFile = File(...), target_column: str = Form(""),
                         task: str = Form(""), name: str = Form("")):
    _404(db.one("SELECT id FROM projects WHERE id = ?", (pid,)), "Projeto")
    dest = await _save_upload(file, f"p{pid}")
    params = {"path": str(dest), "target_column": target_column or None, "task": task or None,
              "origin": file.filename}
    return D.create(pid, "file", name or file.filename, params)


@app.post("/api/uploads")
async def stage_upload(file: UploadFile = File(...)):
    """Upload pelo chat: o arquivo fica guardado e o Claude decide como importá-lo (alvo, tarefa)."""
    dest = await _save_upload(file, "chat")
    try:
        info = files.inspect(dest)
    except ValueError as e:
        dest.unlink(missing_ok=True)
        raise ValueError(f"Não consegui ler o arquivo: {e}") from e
    info["filename"] = file.filename
    return info


@app.get("/api/files/formats")
def file_formats():
    return {"extensions": files.SUPPORTED, "max_upload_mb": files.MAX_UPLOAD_MB}


@app.get("/api/files/inspect")
def inspect_file(path: str, sheet: str | None = None):
    return files.inspect(path, sheet)


# ------------------------------------------------------------------ Kaggle
@app.get("/api/kaggle/status")
def kaggle_status():
    return kaggle.status()


@app.get("/api/kaggle/search")
def kaggle_search(q: str, page: int = 1, file_type: str | None = None, sort_by: str = "hottest"):
    return kaggle.search(q, page, file_type, sort_by)


@app.get("/api/kaggle/files")
def kaggle_files(ref: str):
    return kaggle.list_files(ref)


@app.post("/api/kaggle/download")
def kaggle_download(body: KaggleDownloadIn):
    return kaggle.download(body.ref, body.file_name)


@app.get("/api/datasets/{did}")
def get_dataset(did: int):
    return _404(D.get(did), "Base")


@app.get("/api/datasets/{did}/preview")
def preview_dataset(did: int, n: int = 5):
    _404(D.get(did), "Base")
    return D.preview(did, n)


@app.delete("/api/datasets/{did}")
def delete_dataset(did: int):
    _404(D.get(did), "Base")
    D.remove(did)
    return {"deleted": did}


# ------------------------------------------------------------------ modelos
@app.post("/api/projects/{pid}/models")
def create_model(pid: int, body: ModelIn):
    _404(db.one("SELECT id FROM projects WHERE id = ?", (pid,)), "Projeto")
    cfg = A.validate_config(body.architecture, body.config)
    mid = db.insert("models", {"project_id": pid, "name": body.name, "architecture": body.architecture,
                               "config": cfg, "created_at": db.now()})
    return db.one("SELECT * FROM models WHERE id = ?", (mid,))


@app.get("/api/models/{mid}")
def get_model(mid: int):
    return _404(db.one("SELECT * FROM models WHERE id = ?", (mid,)), "Modelo")


@app.delete("/api/models/{mid}")
def delete_model(mid: int):
    _404(db.one("SELECT id FROM models WHERE id = ?", (mid,)), "Modelo")
    db.delete("models", mid)
    return {"deleted": mid}


# ------------------------------------------------------------------ execuções (treino)
@app.post("/api/runs")
def start_run(body: RunIn):
    return training.start(body.model_id, body.dataset_id, body.hyperparams)


@app.get("/api/runs/{rid}")
def get_run(rid: int):
    return _404(training.get(rid), "Execução")


@app.get("/api/runs/{rid}/wait")
def wait_run(rid: int, timeout: float = 60):
    _404(training.get(rid), "Execução")
    return training.wait(rid, min(timeout, 600))


@app.post("/api/runs/{rid}/cancel")
def cancel_run(rid: int):
    training.cancel(rid)
    return {"cancelling": rid}


@app.delete("/api/runs/{rid}")
def delete_run(rid: int):
    _404(training.get(rid), "Execução")
    db.delete("runs", rid)
    (db.ARTIFACTS_DIR / f"run_{rid}.pt").unlink(missing_ok=True)
    return {"deleted": rid}


@app.post("/api/runs/{rid}/predict")
def predict(rid: int, body: PredictIn):
    return training.predict(rid, body.inputs)


@app.post("/api/runs/{rid}/generate")
def generate(rid: int, n: int = 16):
    return training.generate(rid, min(n, 64))


# ------------------------------------------------------------------ chat (harness)
@app.get("/api/harness")
def harness():
    return chat.describe_harness()


@app.post("/api/chat")
def chat_endpoint(body: ChatIn):
    project = db.one("SELECT * FROM projects WHERE id = ?", (body.project_id,)) if body.project_id else None
    run = H.create(body.policy)
    if body.provider == "codex":
        source = codex_chat.stream_chat(body.message, body.model, project, body.history, agent_token=run.token, policy=body.policy)
    elif body.provider == "copilot":
        source = copilot_chat.stream_chat(body.message, body.session_id, body.model, project, agent_token=run.token, policy=body.policy)
    elif body.provider != "claude-code":
        source = providers.stream_chat(body.message, body.provider, body.model, project, body.history, agent_token=run.token, policy=body.policy)
    else:
        source = chat.stream_chat(body.message, body.session_id, project, body.model, agent_token=run.token, policy=body.policy)
    return StreamingResponse(H.stream(run, source, chat.SYSTEM_PROMPT),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/providers")
async def list_providers():
    return [await codex_chat.status() if p["id"] == "codex" else p for p in providers.catalog()]


@app.post("/api/harness/execute")
async def agent_execute(body: AgentCall, request: Request):
    token = request.headers.get("Authorization", "").removeprefix("Bearer ")
    if set(body.kwargs) - {"params", "json"}:
        raise HTTPException(400, "Parâmetros HTTP não permitidos.")
    async def forward():
        async with httpx.AsyncClient(base_url=chat.API_URL, timeout=660) as client:
            response = await client.request(body.method, body.path, **body.kwargs)
        if response.status_code >= 400:
            raise HTTPException(response.status_code, response.json().get("detail", "Erro na ferramenta"))
        return response.json()
    try:
        return await H.execute(token, body.method, body.path, body.tool, body.kwargs, forward)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc


@app.post("/api/harness/runs/{run_id}/approvals/{approval_id}")
async def agent_approve(run_id: str, approval_id: str, body: ApprovalIn):
    try:
        H.approve(run_id, approval_id, body.approved)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"accepted": True}


@app.post("/api/providers/detect")
async def detect_providers():
    return await providers.detect()


# Launcher serves the built React UI from the same origin as the API.
if os.environ.get("DL_LAB_SERVE_FRONTEND") == "1":
    frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if not (frontend_dist / "index.html").is_file():
        raise RuntimeError("Interface não compilada. Inicie usando start.py.")
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
