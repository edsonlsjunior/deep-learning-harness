"""Busca e download de bases públicas do Kaggle (API REST v1, via httpx).

Bases públicas funcionam sem login. Se houver credenciais, elas são usadas (evita limites e libera
bases que exigem aceite de termos): variáveis KAGGLE_USERNAME + KAGGLE_KEY, KAGGLE_API_TOKEN,
ou os arquivos ~/.kaggle/kaggle.json / ~/.kaggle/access_token.

Os arquivos baixados ficam em data/downloads/kaggle/<dono>/<base>/ e são importados com a mesma
rotina dos uploads (`datasets.from_file`).
"""
from __future__ import annotations

import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any

import httpx

from . import db, files

API = "https://www.kaggle.com/api/v1"
ROOT = db.DATA_DIR / "downloads" / "kaggle"
MAX_DOWNLOAD_MB = 200
REF_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")  # sem "..": o ref vira caminho local


def _auth() -> dict[str, Any]:
    token = os.environ.get("KAGGLE_API_TOKEN")
    tok_file = Path.home() / ".kaggle" / "access_token"
    if not token and tok_file.exists():
        token = tok_file.read_text(encoding="utf-8").strip()
    if token:
        return {"headers": {"Authorization": f"Bearer {token}"}}
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not (user and key) and cred.exists():
        d = json.loads(cred.read_text(encoding="utf-8"))
        user, key = d.get("username"), d.get("key")
    return {"auth": (user, key)} if user and key else {}


def status() -> dict:
    return {"authenticated": bool(_auth()), "max_download_mb": MAX_DOWNLOAD_MB,
            "note": "bases públicas funcionam sem login; credenciais são opcionais"}


def _get(path: str, **kw) -> httpx.Response:
    with httpx.Client(timeout=60, follow_redirects=True, **_auth()) as c:
        r = c.get(f"{API}{path}", **kw)
    if r.status_code == 404:
        raise ValueError("Base não encontrada no Kaggle (confira o ref 'dono/base').")
    if r.status_code in (401, 403):
        raise ValueError("O Kaggle recusou o acesso: a base pode exigir login/aceite de termos. "
                         "Configure ~/.kaggle/kaggle.json ou KAGGLE_API_TOKEN.")
    if r.status_code >= 400:
        raise ValueError(f"Erro {r.status_code} do Kaggle: {r.text[:300]}")
    return r


def _check_ref(ref: str) -> tuple[str, str]:
    ref = ref.strip().removeprefix("https://www.kaggle.com/datasets/").strip("/")
    if not REF_RE.match(ref):
        raise ValueError(f"Ref inválido '{ref}'. Use o formato 'dono/base', ex.: 'heptapod/titanic'.")
    owner, slug = ref.split("/")
    return owner, slug


def search(query: str, page: int = 1, file_type: str | None = None, sort_by: str = "hottest") -> list[dict]:
    params: dict[str, Any] = {"search": query, "page": page, "sortBy": sort_by}
    if file_type in ("csv", "json", "sqlite", "bigQuery", "parquet"):
        params["filetype"] = file_type
    items = _get("/datasets/list", params=params).json()
    return [{"ref": d.get("ref"), "title": d.get("title"), "subtitle": d.get("subtitle"),
             "size_mb": round((d.get("totalBytes") or 0) / 1e6, 2), "downloads": d.get("downloadCount"),
             "votes": d.get("voteCount"), "usability": round(d.get("usabilityRating") or 0, 2),
             "license": d.get("licenseName"), "updated": (d.get("lastUpdated") or "")[:10],
             "url": d.get("url") or f"https://www.kaggle.com/datasets/{d.get('ref')}"} for d in items]


def list_files(ref: str) -> list[dict]:
    owner, slug = _check_ref(ref)
    data = _get(f"/datasets/list/{owner}/{slug}").json()
    out = []
    for f in data.get("datasetFiles", []):
        name = f.get("name") or f.get("nameNullable")
        ext = Path(name).suffix.lower()
        out.append({"name": name, "size_mb": round((f.get("totalBytes") or 0) / 1e6, 3),
                    "importable": ext in files.TABLE_EXT | files.ARRAY_EXT})
    return out


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for m in zf.infolist():
        target = (dest / m.filename).resolve()
        if not target.is_relative_to(dest):  # zip slip
            raise ValueError(f"Caminho suspeito dentro do zip: {m.filename}")
    zf.extractall(dest)


def download(ref: str, file_name: str | None = None) -> dict:
    """Baixa a base inteira (zip) ou um único arquivo, extrai e lista os arquivos importáveis."""
    owner, slug = _check_ref(ref)
    listing = list_files(f"{owner}/{slug}")
    if file_name:
        match = [f for f in listing if f["name"] == file_name]
        if not match:
            raise ValueError(f"Arquivo '{file_name}' não existe na base. Arquivos: {[f['name'] for f in listing]}")
        size = match[0]["size_mb"]
    else:
        size = sum(f["size_mb"] for f in listing)
    if size > MAX_DOWNLOAD_MB:
        raise ValueError(f"A base tem {size:.0f} MB (limite {MAX_DOWNLOAD_MB} MB). "
                         "Escolha um arquivo específico com `file_name`.")
    dest = ROOT / owner / slug
    dest.mkdir(parents=True, exist_ok=True)
    path = f"/datasets/download/{owner}/{slug}" + (f"/{file_name}" if file_name else "")
    content = _get(path).content
    if content[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            _safe_extract(zf, dest)
    else:
        target = (dest / file_name).resolve() if file_name else dest / "data"
        if not target.is_relative_to(dest.resolve()):
            raise ValueError("Nome de arquivo inválido")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    found = [p for p in dest.rglob("*") if p.is_file()]
    return {"ref": f"{owner}/{slug}", "folder": str(dest),
            "files": [{"path": str(p), "name": str(p.relative_to(dest)), "size_mb": round(p.stat().st_size / 1e6, 3),
                       "importable": p.suffix.lower() in files.TABLE_EXT | files.ARRAY_EXT} for p in found]}
