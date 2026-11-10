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
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, SecretStr

from . import architectures as A
from . import chat, db, files, kaggle, training, providers, codex_chat, copilot_chat, harness as H
from . import datasets as D
from .session_credentials import COOKIE_NAME, SUPPORTED_PROVIDERS, TTL_SECONDS, store as session_credentials

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


def _request_uses_https(request: Request) -> bool:
    forwarded = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    return request.url.scheme == "https" or forwarded == "https"


def _secure_cookie(request: Request) -> bool:
    """Em produção HTTPS é obrigatório; HTTP local continua funcional para a aula."""
    configured = os.environ.get("DL_LAB_COOKIE_SECURE", "").lower()
    return configured in {"1", "true", "yes"} if configured else _request_uses_https(request)


def _session_id(request: Request, response: Response) -> str:
    session_id = request.cookies.get(COOKIE_NAME)
    if not session_id:
        session_id = session_credentials.new_session_id()
        response.set_cookie(
            key=COOKIE_NAME,
            value=session_id,
            max_age=TTL_SECONDS,
            httponly=True,
            secure=_secure_cookie(request),
            samesite="strict",
            path="/",
        )
    response.headers["Cache-Control"] = "no-store"
    return session_id


def _session_provider_keys(request: Request) -> dict[str, str]:
    session_id = request.cookies.get(COOKIE_NAME)
    return {provider: key for provider in SUPPORTED_PROVIDERS
            if (key := session_credentials.get(session_id, provider))}


def _owner_session(request: Request, response: Response | None = None) -> str | None:
    """Dono anônimo do espaço público; chamadas MCP só entram pelo gateway ativo."""
    if not providers.public_mode():
        return None
    if owner := request.cookies.get(COOKIE_NAME):
        return owner
    internal_owner = request.headers.get("X-DL-Lab-Owner")
    gateway_token = request.headers.get("X-DL-Lab-Gateway", "")
    if internal_owner and H.owner_for(gateway_token) == internal_owner:
        return internal_owner
    if response is not None:
        return _session_id(request, response)
    raise HTTPException(401, "Sessão anônima ausente ou expirada. Recarregue a página.")


def _owned_project(request: Request, pid: int, response: Response | None = None) -> dict:
    owner = _owner_session(request, response)
    if owner:
        return _404(db.one("SELECT * FROM projects WHERE id = ? AND owner_session = ?", (pid, owner)), "Projeto")
    return _404(db.one("SELECT * FROM projects WHERE id = ?", (pid,)), "Projeto")


def _owned_dataset(request: Request, did: int) -> dict:
    owner = _owner_session(request)
    sql = "SELECT d.* FROM datasets d JOIN projects p ON p.id = d.project_id WHERE d.id = ?"
    args: tuple = (did,)
    if owner:
        sql += " AND p.owner_session = ?"
        args += (owner,)
    return _404(db.one(sql, args), "Base")


def _owned_model(request: Request, mid: int) -> dict:
    owner = _owner_session(request)
    sql = "SELECT m.* FROM models m JOIN projects p ON p.id = m.project_id WHERE m.id = ?"
    args: tuple = (mid,)
    if owner:
        sql += " AND p.owner_session = ?"
        args += (owner,)
    return _404(db.one(sql, args), "Modelo")


def _owned_run(request: Request, rid: int) -> dict:
    owner = _owner_session(request)
    sql = "SELECT r.* FROM runs r JOIN projects p ON p.id = r.project_id WHERE r.id = ?"
    args: tuple = (rid,)
    if owner:
        sql += " AND p.owner_session = ?"
        args += (owner,)
    return _404(db.one(sql, args), "Execução")


def _public_storage_blocked():
    if providers.public_mode():
        raise HTTPException(403, "Uploads e Kaggle ficam desativados na versão pública.")


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


class SessionCredentialIn(BaseModel):
    # SecretStr evita revelar o valor caso uma validação seja registrada por engano.
    api_key: SecretStr


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
def list_projects(request: Request):
    owner = _owner_session(request)
    sql = """SELECT p.*,
        (SELECT COUNT(*) FROM datasets d WHERE d.project_id = p.id) AS n_datasets,
        (SELECT COUNT(*) FROM models m WHERE m.project_id = p.id) AS n_models,
        (SELECT COUNT(*) FROM runs r WHERE r.project_id = p.id) AS n_runs
        FROM projects p"""
    args: tuple = ()
    if owner:
        sql += " WHERE p.owner_session = ?"
        args = (owner,)
    projects = db.query(sql + " ORDER BY p.id DESC", args)
    for project in projects:
        project.pop("owner_session", None)
    return projects


@app.post("/api/projects")
def create_project(body: ProjectIn, request: Request, response: Response):
    owner = _owner_session(request, response)
    values = {"name": body.name, "description": body.description, "created_at": db.now()}
    if owner:
        values["owner_session"] = owner
    pid = db.insert("projects", values)
    project = db.one("SELECT * FROM projects WHERE id = ?", (pid,))
    project.pop("owner_session", None)
    return project


@app.get("/api/projects/{pid}")
def get_project(pid: int, request: Request):
    p = _owned_project(request, pid)
    p.pop("owner_session", None)
    p["datasets"] = db.query("SELECT * FROM datasets WHERE project_id = ? ORDER BY id", (pid,))
    p["models"] = db.query("SELECT * FROM models WHERE project_id = ? ORDER BY id", (pid,))
    p["runs"] = [_run_summary(r) for r in db.query("SELECT * FROM runs WHERE project_id = ? ORDER BY id DESC", (pid,))]
    return p


@app.delete("/api/projects/{pid}")
def delete_project(pid: int, request: Request):
    _owned_project(request, pid)
    for d in db.query("SELECT id FROM datasets WHERE project_id = ?", (pid,)):
        D.remove(d["id"])
    db.delete("projects", pid)
    return {"deleted": pid}


# ------------------------------------------------------------------ bases de dados
@app.post("/api/projects/{pid}/datasets")
def add_dataset(pid: int, body: DatasetIn, request: Request):
    _owned_project(request, pid)
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
async def upload_dataset(pid: int, request: Request, file: UploadFile = File(...), target_column: str = Form(""),
                         task: str = Form(""), name: str = Form("")):
    _public_storage_blocked()
    _owned_project(request, pid)
    dest = await _save_upload(file, f"p{pid}")
    params = {"path": str(dest), "target_column": target_column or None, "task": task or None,
              "origin": file.filename}
    return D.create(pid, "file", name or file.filename, params)


@app.post("/api/uploads")
async def stage_upload(file: UploadFile = File(...)):
    """Upload pelo chat: o arquivo fica guardado e o Claude decide como importá-lo (alvo, tarefa)."""
    _public_storage_blocked()
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
    _public_storage_blocked()
    return files.inspect(path, sheet)


# ------------------------------------------------------------------ Kaggle
@app.get("/api/kaggle/status")
def kaggle_status():
    _public_storage_blocked()
    return kaggle.status()


@app.get("/api/kaggle/search")
def kaggle_search(q: str, page: int = 1, file_type: str | None = None, sort_by: str = "hottest"):
    _public_storage_blocked()
    return kaggle.search(q, page, file_type, sort_by)


@app.get("/api/kaggle/files")
def kaggle_files(ref: str):
    _public_storage_blocked()
    return kaggle.list_files(ref)


@app.post("/api/kaggle/download")
def kaggle_download(body: KaggleDownloadIn):
    _public_storage_blocked()
    return kaggle.download(body.ref, body.file_name)


@app.get("/api/datasets/{did}")
def get_dataset(did: int, request: Request):
    _owned_dataset(request, did)
    return _404(D.get(did), "Base")


@app.get("/api/datasets/{did}/preview")
def preview_dataset(did: int, request: Request, n: int = 5):
    _owned_dataset(request, did)
    return D.preview(did, n)


@app.delete("/api/datasets/{did}")
def delete_dataset(did: int, request: Request):
    _owned_dataset(request, did)
    D.remove(did)
    return {"deleted": did}


# ------------------------------------------------------------------ modelos
@app.post("/api/projects/{pid}/models")
def create_model(pid: int, body: ModelIn, request: Request):
    _owned_project(request, pid)
    cfg = A.validate_config(body.architecture, body.config)
    mid = db.insert("models", {"project_id": pid, "name": body.name, "architecture": body.architecture,
                               "config": cfg, "created_at": db.now()})
    return db.one("SELECT * FROM models WHERE id = ?", (mid,))


@app.get("/api/models/{mid}")
def get_model(mid: int, request: Request):
    return _owned_model(request, mid)


@app.delete("/api/models/{mid}")
def delete_model(mid: int, request: Request):
    _owned_model(request, mid)
    db.delete("models", mid)
    return {"deleted": mid}


# ------------------------------------------------------------------ execuções (treino)
@app.post("/api/runs")
def start_run(body: RunIn, request: Request):
    model = _owned_model(request, body.model_id)
    dataset = _owned_dataset(request, body.dataset_id)
    if model["project_id"] != dataset["project_id"]:
        raise HTTPException(400, "Modelo e base precisam pertencer ao mesmo projeto.")
    return training.start(body.model_id, body.dataset_id, body.hyperparams)


@app.get("/api/runs/{rid}")
def get_run(rid: int, request: Request):
    _owned_run(request, rid)
    return _404(training.get(rid), "Execução")


@app.get("/api/runs/{rid}/wait")
def wait_run(rid: int, request: Request, timeout: float = 60):
    _owned_run(request, rid)
    return training.wait(rid, min(timeout, 600))


@app.post("/api/runs/{rid}/cancel")
def cancel_run(rid: int, request: Request):
    _owned_run(request, rid)
    training.cancel(rid)
    return {"cancelling": rid}


@app.delete("/api/runs/{rid}")
def delete_run(rid: int, request: Request):
    _owned_run(request, rid)
    db.delete("runs", rid)
    (db.ARTIFACTS_DIR / f"run_{rid}.pt").unlink(missing_ok=True)
    return {"deleted": rid}


@app.post("/api/runs/{rid}/predict")
def predict(rid: int, body: PredictIn, request: Request):
    _owned_run(request, rid)
    return training.predict(rid, body.inputs)


@app.post("/api/runs/{rid}/generate")
def generate(rid: int, request: Request, n: int = 16):
    _owned_run(request, rid)
    return training.generate(rid, min(n, 64))


# ------------------------------------------------------------------ chat (harness)
@app.get("/api/harness")
def harness():
    return chat.describe_harness()


@app.get("/api/harness/status")
async def harness_status():
    """Diagnóstico sem modelos ou credenciais para a tela e suporte técnico."""
    return await providers.mcp_status()


@app.post("/api/chat")
def chat_endpoint(body: ChatIn, request: Request):
    owner = _owner_session(request)
    project = _owned_project(request, body.project_id) if body.project_id else None
    run = H.create(body.policy, owner)
    if body.provider == "codex":
        source = codex_chat.stream_chat(body.message, body.model, project, body.history, agent_token=run.token, policy=body.policy)
    elif body.provider == "copilot":
        source = copilot_chat.stream_chat(body.message, body.session_id, body.model, project, agent_token=run.token, policy=body.policy)
    elif body.provider != "claude-code":
        # A chave não veio no corpo do chat: só é obtida da memória da sessão.
        api_key = _session_provider_keys(request).get(body.provider)
        source = providers.stream_chat(body.message, body.provider, body.model, project, body.history, agent_token=run.token, policy=body.policy, api_key=api_key, owner_session=owner)
    else:
        source = chat.stream_chat(body.message, body.session_id, project, body.model, agent_token=run.token, policy=body.policy)
    return StreamingResponse(H.stream(run, source, chat.SYSTEM_PROMPT),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/providers")
async def list_providers(request: Request, response: Response):
    session_id = _session_id(request, response)
    configured = {p for p in SUPPORTED_PROVIDERS if session_credentials.configured(session_id, p)}
    return [await codex_chat.status() if p["id"] == "codex" else p for p in providers.catalog(configured)]


@app.get("/api/session/credentials")
def credential_status(request: Request, response: Response):
    """Devolve somente estado e prazo; nunca o valor de uma chave."""
    session_id = _session_id(request, response)
    return {"providers": {provider: {
        "configured": session_credentials.configured(session_id, provider),
        "expires_in_seconds": session_credentials.expires_in(session_id, provider),
    } for provider in SUPPORTED_PROVIDERS}}


@app.post("/api/session/credentials/{provider}")
def save_credential(provider: str, body: SessionCredentialIn, request: Request, response: Response):
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(404, "Provedor não suporta chave por sessão.")
    if providers.public_mode() and not _request_uses_https(request):
        raise HTTPException(400, "A página pública exige HTTPS para receber uma chave de API.")
    session_id = _session_id(request, response)
    # O segredo é entregue diretamente ao cofre em memória e não é incluído na resposta.
    session_credentials.set(session_id, provider, body.api_key.get_secret_value())
    return {"provider": provider, "configured": True, "expires_in_seconds": TTL_SECONDS}


@app.delete("/api/session/credentials/{provider}")
def remove_credential(provider: str, request: Request, response: Response):
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(404, "Provedor não suporta chave por sessão.")
    session_id = _session_id(request, response)
    session_credentials.remove(session_id, provider)
    return {"provider": provider, "configured": False}


@app.post("/api/harness/execute")
async def agent_execute(body: AgentCall, request: Request):
    token = request.headers.get("Authorization", "").removeprefix("Bearer ")
    if set(body.kwargs) - {"params", "json"}:
        raise HTTPException(400, "Parâmetros HTTP não permitidos.")
    async def forward():
        # Esta é uma chamada de retorno para a própria API; ela não deve usar
        # proxies de ambiente presentes na hospedagem.
        async with httpx.AsyncClient(base_url=chat.API_URL, timeout=660, trust_env=False) as client:
            headers = {}
            if owner := H.owner_for(token):
                headers = {"X-DL-Lab-Owner": owner, "X-DL-Lab-Gateway": token}
            response = await client.request(body.method, body.path, headers=headers, **body.kwargs)
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
async def detect_providers(request: Request, response: Response):
    _session_id(request, response)
    return await providers.detect(_session_provider_keys(request))


# Launcher serves the built React UI from the same origin as the API.
if os.environ.get("DL_LAB_SERVE_FRONTEND") == "1":
    frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if not (frontend_dist / "index.html").is_file():
        raise RuntimeError("Interface não compilada. Inicie usando start.py.")
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
