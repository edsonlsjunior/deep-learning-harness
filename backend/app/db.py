"""Persistência mínima em SQLite (stdlib), com colunas JSON.

Tabelas:
  projects  -> agrupam bases de dados, modelos e execuções de treino
  datasets  -> metadados; os arrays ficam em data/datasets/<id>.npz
  models    -> arquitetura + configuração (hiperparâmetros estruturais)
  runs      -> execuções de treino (histórico por época, métricas, artefatos)
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASETS_DIR = DATA_DIR / "datasets"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
for d in (DATA_DIR, DATASETS_DIR, ARTIFACTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "lab.sqlite3"
_lock = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    params TEXT NOT NULL DEFAULT '{}',
    meta TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    architecture TEXT NOT NULL,
    config TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    hyperparams TEXT NOT NULL DEFAULT '{}',
    history TEXT NOT NULL DEFAULT '[]',
    metrics TEXT NOT NULL DEFAULT '{}',
    extras TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT
);
"""

JSON_COLS = {"params", "meta", "config", "hyperparams", "history", "metrics", "extras"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


_conn = _connect()
_conn.executescript(SCHEMA)


def _row(r: sqlite3.Row | None) -> dict[str, Any] | None:
    if r is None:
        return None
    out = dict(r)
    for k in JSON_COLS & out.keys():
        out[k] = json.loads(out[k]) if out[k] else None
    return out


def query(sql: str, args: tuple = ()) -> list[dict[str, Any]]:
    with _lock:
        return [_row(r) for r in _conn.execute(sql, args).fetchall()]  # type: ignore[misc]


def one(sql: str, args: tuple = ()) -> dict[str, Any] | None:
    with _lock:
        return _row(_conn.execute(sql, args).fetchone())


def insert(table: str, values: dict[str, Any]) -> int:
    values = {k: json.dumps(v) if k in JSON_COLS else v for k, v in values.items()}
    cols = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    with _lock:
        cur = _conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(values.values()))
        _conn.commit()
        return int(cur.lastrowid)  # type: ignore[arg-type]


def update(table: str, row_id: int, values: dict[str, Any]) -> None:
    values = {k: json.dumps(v) if k in JSON_COLS else v for k, v in values.items()}
    sets = ", ".join(f"{k} = ?" for k in values)
    with _lock:
        _conn.execute(f"UPDATE {table} SET {sets} WHERE id = ?", (*values.values(), row_id))
        _conn.commit()


def delete(table: str, row_id: int) -> None:
    with _lock:
        _conn.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
        _conn.commit()
