"""Leitura de arquivos de dados enviados pelo usuário (upload no chat/UI ou baixados do Kaggle).

Formatos aceitos (os que fazem sentido para as tarefas do laboratório):

  .csv .tsv .txt      tabelas delimitadas (separador , ; ou TAB detectado; decimal com vírgula aceito)
  .json .jsonl        lista de registros [{"col": v, ...}] ou JSON Lines (um registro por linha)
  .xlsx .xls          planilha Excel (primeira aba, ou `sheet`)
  .parquet            tabela colunar (requer pyarrow)
  .npz                arrays NumPy prontos: X (N,F) | (N,T,F) | (N,C,H,W) e, opcional, y

Tabelas viram (header, linhas de strings) para reaproveitar a mesma lógica de conversão
(numéricas -> atributos, categóricas -> one-hot) em `datasets._from_table`.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

TABLE_EXT = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".xlsx", ".xls", ".parquet"}
ARRAY_EXT = {".npz"}
SUPPORTED = sorted(TABLE_EXT | ARRAY_EXT)
MAX_UPLOAD_MB = 200


def check_ext(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in TABLE_EXT | ARRAY_EXT:
        raise ValueError(f"Formato '{ext or '?'}' não suportado. Use: {', '.join(SUPPORTED)}")
    return ext


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    s = str(v)
    return "" if s in ("nan", "NaN", "None", "NaT", "<NA>") else s


def _df_to_rows(df) -> tuple[list[str], list[list[str]]]:
    header = [str(c) for c in df.columns]
    rows = [[_cell(v) for v in r] for r in df.itertuples(index=False, name=None)]
    return header, rows


def _dedupe(header: list[str]) -> list[str]:
    """Colunas repetidas (comum no Kaggle) viram 'col', 'col.1', 'col.2'… em vez de se sobrescreverem."""
    seen: dict[str, int] = {}
    out = []
    for i, h in enumerate(header):
        h = h.strip() or f"col_{i}"
        if h in seen:
            seen[h] += 1
            h = f"{h}.{seen[h]}"
        else:
            seen[h] = 0
        out.append(h)
    return out


def read_table(path: str | Path, sheet: str | None = None) -> tuple[list[str], list[list[str]]]:
    header, rows = _read_table(path, sheet)
    return _dedupe(header), rows


def _read_table(path: str | Path, sheet: str | None = None) -> tuple[list[str], list[list[str]]]:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Arquivo não encontrado: {path}")
    ext = check_ext(p.name)
    if ext in (".csv", ".tsv", ".txt"):
        with p.open(newline="", encoding="utf-8-sig", errors="replace") as f:
            sample = f.read(8192)
            f.seek(0)
            try:
                dialect: Any = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error:
                dialect = "excel-tab" if ext == ".tsv" else "excel"
            rows = list(csv.reader(f, dialect))
        if not rows:
            raise ValueError("Arquivo vazio")
        return rows[0], [r for r in rows[1:] if any(c.strip() for c in r)]

    import pandas as pd

    if ext == ".jsonl":
        df = pd.read_json(p, lines=True)
    elif ext == ".json":
        try:
            df = pd.read_json(p)
        except ValueError:
            df = pd.read_json(p, lines=True)
    elif ext in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(p, sheet_name=sheet or 0)
        except ImportError as e:
            raise ValueError("Leitura de Excel requer 'openpyxl' (pip install openpyxl).") from e
    else:  # parquet
        try:
            df = pd.read_parquet(p)
        except ImportError as e:
            raise ValueError("Leitura de Parquet requer 'pyarrow' (pip install pyarrow).") from e
    if df.empty:
        raise ValueError("A tabela está vazia (ou o JSON não é uma lista de registros).")
    return _df_to_rows(df)


def _is_numeric(vals: list[str]) -> bool:
    try:
        for v in vals:
            if v != "":
                float(v.replace(",", "."))
        return True
    except ValueError:
        return False


def inspect(path: str | Path, sheet: str | None = None, n_preview: int = 5) -> dict[str, Any]:
    """Resumo do arquivo para decidir a coluna alvo e a tarefa antes de importar."""
    p = Path(path)
    ext = check_ext(p.name)
    info: dict[str, Any] = {"path": str(p), "filename": p.name, "format": ext.lstrip("."),
                            "size_kb": round(p.stat().st_size / 1024, 1)}
    if ext == ".npz":
        with np.load(p, allow_pickle=False) as z:
            info["arrays"] = {k: {"shape": list(z[k].shape), "dtype": str(z[k].dtype)} for k in z.files}
        info["hint"] = "importe direto: X vira a entrada e y (se existir) o alvo; a tarefa é inferida pelo formato de X"
        return info
    header, rows = read_table(p, sheet)
    info["n_rows"], info["n_columns"] = len(rows), len(header)
    cols = []
    for i, h in enumerate(header):
        vals = [r[i] if i < len(r) else "" for r in rows]
        uniq = set(vals)
        cols.append({"name": h, "kind": "numeric" if _is_numeric(vals) else "categorical",
                     "n_unique": len(uniq), "n_missing": sum(v == "" for v in vals),
                     "examples": [v for v in list(dict.fromkeys(vals))[:4]]})
    info["columns"] = cols
    info["preview"] = [dict(zip(header, r)) for r in rows[:n_preview]]
    return info
