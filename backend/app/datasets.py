"""Bases de dados: geradores embutidos (offline) + importação de CSV.

Cada base é materializada como um arquivo .npz (X, y) e um dicionário `meta`
que descreve a tarefa. A tarefa determina quais arquiteturas fazem sentido:

  tabular_classification  X:(N,F)        y:int   -> perceptron, adaline, mlp
  tabular_regression      X:(N,F)        y:float -> adaline, mlp
  image_classification    X:(N,C,H,W)    y:int   -> cnn, mlp, autoencoder, gan
  sequence_regression     X:(N,T,F)      y:float -> rnn, lstm, gru, transformer
  sequence_classification X:(N,T,F)      y:int   -> rnn, lstm, gru, transformer
  unsupervised            X:(N,F)        (sem y) -> autoencoder, gan
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import db

TASKS = [
    "tabular_classification",
    "tabular_regression",
    "image_classification",
    "sequence_regression",
    "sequence_classification",
    "unsupervised",
]

# Série clássica "Airline Passengers" (Box & Jenkins, 1949-1960, milhares/mês).
AIRLINE = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170, 170,
    158, 133, 114, 140, 145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166, 171, 180, 193, 181,
    183, 218, 230, 242, 209, 191, 172, 194, 196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229, 242, 233, 267, 269, 270, 315, 364, 347,
    312, 274, 237, 278, 284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306, 315, 301, 356, 348,
    355, 422, 465, 467, 404, 347, 305, 336, 340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405, 417, 391, 419, 461, 472, 535, 622, 606,
    508, 461, 390, 432,
]


def _rng(params: dict) -> np.random.Generator:
    return np.random.default_rng(int(params.get("seed", 42)))


# ---------------------------------------------------------------- geradores
def _logic(kind: str) -> Callable[[dict], tuple]:
    table = {"and": [0, 0, 0, 1], "or": [0, 1, 1, 1], "xor": [0, 1, 1, 0]}[kind]

    def gen(params: dict):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array(table, dtype=np.int64)
        reps = int(params.get("repeat", 1))
        noise = float(params.get("noise", 0.0))
        X, y = np.tile(X, (reps, 1)), np.tile(y, reps)
        if noise > 0:
            X = X + _rng(params).normal(0, noise, X.shape).astype(np.float32)
        return X, y, {"task": "tabular_classification", "feature_names": ["x1", "x2"],
                      "class_names": ["0", "1"]}

    return gen


def _blobs(params: dict):
    rng = _rng(params)
    n = int(params.get("n_samples", 200))
    centers = int(params.get("centers", 2))
    spread = float(params.get("std", 0.8))
    angles = np.linspace(0, 2 * np.pi, centers, endpoint=False)
    c = np.stack([np.cos(angles), np.sin(angles)], 1) * 3
    y = rng.integers(0, centers, n)
    X = c[y] + rng.normal(0, spread, (n, 2))
    return X.astype(np.float32), y.astype(np.int64), {
        "task": "tabular_classification", "feature_names": ["x1", "x2"],
        "class_names": [str(i) for i in range(centers)]}


def _moons(params: dict):
    from sklearn.datasets import make_moons
    X, y = make_moons(int(params.get("n_samples", 400)), noise=float(params.get("noise", 0.2)),
                      random_state=int(params.get("seed", 42)))
    return X.astype(np.float32), y.astype(np.int64), {
        "task": "tabular_classification", "feature_names": ["x1", "x2"], "class_names": ["0", "1"]}


def _circles(params: dict):
    from sklearn.datasets import make_circles
    X, y = make_circles(int(params.get("n_samples", 400)), noise=float(params.get("noise", 0.1)),
                        factor=float(params.get("factor", 0.5)), random_state=int(params.get("seed", 42)))
    return X.astype(np.float32), y.astype(np.int64), {
        "task": "tabular_classification", "feature_names": ["x1", "x2"], "class_names": ["0", "1"]}


def _sklearn_table(loader: str, task: str):
    def gen(params: dict):
        import sklearn.datasets as skd
        d = getattr(skd, loader)()
        meta: dict[str, Any] = {"task": task, "feature_names": [str(f) for f in d.feature_names]}
        if task == "tabular_classification":
            meta["class_names"] = [str(c) for c in d.target_names]
            y = d.target.astype(np.int64)
        else:
            y = d.target.astype(np.float32)
        return d.data.astype(np.float32), y, meta

    return gen


def _sine_regression(params: dict):
    rng = _rng(params)
    n = int(params.get("n_samples", 400))
    x = rng.uniform(-np.pi, np.pi, (n, 1))
    y = np.sin(2 * x[:, 0]) + 0.3 * x[:, 0] + rng.normal(0, float(params.get("noise", 0.1)), n)
    return x.astype(np.float32), y.astype(np.float32), {"task": "tabular_regression", "feature_names": ["x"]}


def _digits(params: dict):
    from sklearn.datasets import load_digits
    d = load_digits()
    X = (d.images / 16.0).astype(np.float32)[:, None, :, :]
    return X, d.target.astype(np.int64), {
        "task": "image_classification", "class_names": [str(i) for i in range(10)]}


def _torchvision(name: str):
    def gen(params: dict):
        import torchvision
        root = db.DATA_DIR / "downloads"
        cls = getattr(torchvision.datasets, name)
        ds = cls(root=str(root), train=True, download=True)
        n = int(params.get("n_samples", 6000))
        idx = _rng(params).permutation(len(ds.data))[:n]
        X = (ds.data.numpy()[idx] / 255.0).astype(np.float32)[:, None, :, :]
        y = ds.targets.numpy()[idx].astype(np.int64)
        return X, y, {"task": "image_classification", "class_names": [str(c) for c in ds.classes]}

    return gen


def _windows(series: np.ndarray, window: int, horizon: int = 1):
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i:i + window])
        y.append(series[i + window + horizon - 1])
    return np.array(X, dtype=np.float32)[:, :, None], np.array(y, dtype=np.float32)


def _sine_series(params: dict):
    rng = _rng(params)
    n = int(params.get("length", 1000))
    t = np.arange(n)
    s = np.sin(2 * np.pi * t / 50) + 0.5 * np.sin(2 * np.pi * t / 13) + rng.normal(0, float(params.get("noise", 0.1)), n)
    X, y = _windows(s, int(params.get("window", 30)))
    return X, y, {"task": "sequence_regression", "feature_names": ["valor"],
                  "note": "janelas deslizantes: prever o próximo ponto da série"}


def _airline(params: dict):
    s = np.array(AIRLINE, dtype=np.float32)
    mu, sd = float(s.mean()), float(s.std())
    X, y = _windows((s - mu) / sd, int(params.get("window", 12)))
    return X, y, {"task": "sequence_regression", "feature_names": ["passageiros"],
                  "scale": {"mean": mu, "std": sd},
                  "note": "série normalizada (z-score); janelas de 12 meses"}


TEMPS_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"


def _daily_min_temp(params: dict):
    """Temperatura mínima diária em Melbourne (1981-1990, 3650 dias, Australian Bureau of Meteorology).

    Exemplo da Aula 3: "previsão do tempo a partir do histórico climático".
    Entrada: últimos `window` dias (z-score). Alvo: temperatura do dia t+`horizon`, em °C,
    para que RMSE/MAE fiquem diretamente interpretáveis em graus."""
    import urllib.request

    path = db.DATA_DIR / "downloads" / "daily-min-temperatures.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(TEMPS_URL, timeout=30) as r:
            path.write_bytes(r.read())
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    dates = [r[0] for r in rows]
    temps = np.array([float(r[1]) for r in rows], dtype=np.float32)
    window, horizon = int(params.get("window", 30)), int(params.get("horizon", 1))
    mu, sd = float(temps.mean()), float(temps.std())
    X, _ = _windows((temps - mu) / sd, window, horizon)
    _, y = _windows(temps, window, horizon)
    last = temps[window - 1: window - 1 + len(y)]  # último dia observado de cada janela
    naive_rmse = float(np.sqrt(np.mean((y - last) ** 2)))
    return X, y, {"task": "sequence_regression", "feature_names": ["temp_min_C"], "unit": "°C",
                  "baseline_persistence_rmse": round(naive_rmse, 3),
                  "baseline_note": "RMSE do modelo ingênuo 'amanhã = último dia observado'; a rede precisa ficar abaixo disso",
                  "scale": {"mean": mu, "std": sd}, "period": f"{dates[0]} a {dates[-1]}", "source_url": TEMPS_URL,
                  "note": f"entrada: {window} dias anteriores (z-score); alvo: temperatura mínima em °C "
                          f"{horizon} dia(s) à frente"}


def _seq_parity(params: dict):
    """Classificação de sequências: a sequência de bits tem número par de 1s?

    Exige memória de longo prazo — bom para comparar RNN x LSTM x GRU x Transformer."""
    rng = _rng(params)
    n, T = int(params.get("n_samples", 2000)), int(params.get("length", 12))
    X = rng.integers(0, 2, (n, T, 1)).astype(np.float32)
    y = (X.sum(axis=(1, 2)) % 2).astype(np.int64)
    return X, y, {"task": "sequence_classification", "class_names": ["par", "ímpar"]}


BUILTIN: dict[str, dict[str, Any]] = {
    "and_gate": {"gen": _logic("and"), "aula": 1, "desc": "Porta lógica AND (4 pontos). Linearmente separável — Perceptron/Adaline."},
    "or_gate": {"gen": _logic("or"), "aula": 1, "desc": "Porta lógica OR (4 pontos). Linearmente separável."},
    "xor_gate": {"gen": _logic("xor"), "aula": 1, "desc": "Porta lógica XOR. NÃO linearmente separável — motiva o MLP. Params: repeat, noise."},
    "blobs": {"gen": _blobs, "aula": 1, "desc": "Nuvens gaussianas 2D. Params: n_samples, centers, std."},
    "moons": {"gen": _moons, "aula": 1, "desc": "Duas luas entrelaçadas (não linear). Params: n_samples, noise."},
    "circles": {"gen": _circles, "aula": 1, "desc": "Círculos concêntricos (não linear). Params: n_samples, noise, factor."},
    "iris": {"gen": _sklearn_table("load_iris", "tabular_classification"), "aula": 1, "desc": "Iris: 150 flores, 4 atributos, 3 classes."},
    "wine": {"gen": _sklearn_table("load_wine", "tabular_classification"), "aula": 1, "desc": "Vinhos: 178 amostras, 13 atributos, 3 classes."},
    "breast_cancer": {"gen": _sklearn_table("load_breast_cancer", "tabular_classification"), "aula": 1, "desc": "Câncer de mama (Wisconsin): 569 amostras, 30 atributos, binário."},
    "diabetes": {"gen": _sklearn_table("load_diabetes", "tabular_regression"), "aula": 1, "desc": "Diabetes: regressão, 442 amostras, 10 atributos."},
    "sine_regression": {"gen": _sine_regression, "aula": 1, "desc": "Regressão 1D: y = sin(2x) + 0.3x + ruído. Params: n_samples, noise."},
    "digits": {"gen": _digits, "aula": 2, "desc": "Dígitos manuscritos 8x8 (1797 imagens, offline). CNN, autoencoder, GAN."},
    "mnist": {"gen": _torchvision("MNIST"), "aula": 2, "desc": "MNIST 28x28 (download na 1ª vez). Params: n_samples (padrão 6000)."},
    "fashion_mnist": {"gen": _torchvision("FashionMNIST"), "aula": 2, "desc": "Fashion-MNIST 28x28 (download na 1ª vez). Params: n_samples."},
    "sine_series": {"gen": _sine_series, "aula": 3, "desc": "Série temporal (soma de senoides + ruído) em janelas. Params: length, window, noise."},
    "airline_passengers": {"gen": _airline, "aula": 3, "desc": "Série real de passageiros aéreos (1949-1960), janelas de 12 meses."},
    "daily_min_temperatures": {"gen": _daily_min_temp, "aula": 3, "desc": "Histórico climático real: temperatura mínima diária em Melbourne (1981-1990, 3650 dias). Prever a temperatura de amanhã a partir dos últimos dias (download ~70 KB na 1ª vez). Params: window (padrão 30), horizon (padrão 1)."},
    "sequence_parity": {"gen": _seq_parity, "aula": 3, "desc": "Paridade de sequências de bits (memória longa). Params: n_samples, length."},
}


def catalog() -> list[dict[str, Any]]:
    return [{"source": k, "aula": v["aula"], "description": v["desc"]} for k, v in BUILTIN.items()]


# ---------------------------------------------------------------- CSV
def _from_csv(path: str, target_column: str | None, task: str | None):
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Arquivo não encontrado: {path}")
    with p.open(newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        rows = list(csv.reader(f, dialect))
    header, rows = rows[0], [r for r in rows[1:] if any(c.strip() for c in r)]
    cols = {h: [r[i] if i < len(r) else "" for r in rows] for i, h in enumerate(header)}

    def numeric(vals: list[str]) -> bool:
        try:
            [float(v.replace(",", ".")) for v in vals if v != ""]
            return True
        except ValueError:
            return False

    target = cols.pop(target_column) if target_column else None
    if target_column and target is None:
        raise ValueError(f"Coluna alvo '{target_column}' não existe. Colunas: {header}")

    feats, names = [], []
    for name, vals in cols.items():
        if numeric(vals):
            arr = np.array([float(v.replace(",", ".")) if v != "" else np.nan for v in vals], dtype=np.float32)
            arr[np.isnan(arr)] = np.nanmean(arr) if not np.all(np.isnan(arr)) else 0
            feats.append(arr[:, None])
            names.append(name)
        else:  # one-hot encoding de categóricas
            cats = sorted(set(vals))
            if len(cats) > 50:
                continue
            for c in cats:
                feats.append(np.array([v == c for v in vals], dtype=np.float32)[:, None])
                names.append(f"{name}={c}")
    X = np.concatenate(feats, axis=1)
    meta: dict[str, Any] = {"feature_names": names, "csv_path": str(p)}
    if target is None:
        meta["task"] = "unsupervised"
        return X, np.zeros(0, dtype=np.float32), meta
    if task is None:
        task = "tabular_regression" if numeric(target) and len(set(target)) > 20 else "tabular_classification"
    meta["task"] = task
    if task == "tabular_classification":
        classes = sorted(set(target))
        meta["class_names"] = classes
        y = np.array([classes.index(v) for v in target], dtype=np.int64)
    else:
        y = np.array([float(v.replace(",", ".")) for v in target], dtype=np.float32)
    meta["target_column"] = target_column
    return X, y, meta


# ---------------------------------------------------------------- API interna
def _describe(X: np.ndarray, y: np.ndarray, meta: dict) -> dict:
    meta = dict(meta)
    meta["n_samples"] = int(X.shape[0])
    meta["input_shape"] = list(X.shape[1:])
    if y.size and meta["task"].endswith("classification"):
        counts = np.bincount(y)
        meta["n_classes"] = int(len(counts))
        meta["class_counts"] = counts.tolist()
    elif y.size:
        meta["target_stats"] = {"min": float(y.min()), "max": float(y.max()), "mean": float(y.mean()), "std": float(y.std())}
    return meta


def create(project_id: int, source: str, name: str | None = None, params: dict | None = None) -> dict:
    params = params or {}
    if source == "csv":
        X, y, meta = _from_csv(params["path"], params.get("target_column"), params.get("task"))
        meta["description"] = f"CSV importado de {params['path']}"
    elif source in BUILTIN:
        X, y, meta = BUILTIN[source]["gen"](params)
        meta["description"] = BUILTIN[source]["desc"]
    else:
        raise ValueError(f"Fonte desconhecida '{source}'. Disponíveis: {list(BUILTIN)} ou 'csv'.")
    meta = _describe(X, y, meta)
    ds_id = db.insert("datasets", {"project_id": project_id, "name": name or source, "source": source,
                                    "params": params, "meta": meta, "created_at": db.now()})
    np.savez_compressed(db.DATASETS_DIR / f"{ds_id}.npz", X=X, y=y)
    return get(ds_id)  # type: ignore[return-value]


def load(ds_id: int) -> tuple[np.ndarray, np.ndarray, dict]:
    row = db.one("SELECT * FROM datasets WHERE id = ?", (ds_id,))
    if not row:
        raise ValueError(f"Base {ds_id} não existe")
    with np.load(db.DATASETS_DIR / f"{ds_id}.npz") as z:
        return z["X"], z["y"], row["meta"]


def get(ds_id: int) -> dict | None:
    return db.one("SELECT * FROM datasets WHERE id = ?", (ds_id,))


def preview(ds_id: int, n: int = 5) -> dict:
    X, y, meta = load(ds_id)
    out: dict[str, Any] = {"dataset_id": ds_id, "meta": meta}
    if X.ndim == 2:
        out["rows"] = [{"x": X[i].round(4).tolist(), "y": y[i].item() if y.size else None} for i in range(min(n, len(X)))]
    elif meta["task"] == "image_classification":
        out["images"] = [{"pixels": X[i, 0].round(3).tolist(), "y": int(y[i])} for i in range(min(n, len(X)))]
    else:
        out["sequences"] = [{"x": X[i, :, 0].round(4).tolist(), "y": y[i].item()} for i in range(min(n, len(X)))]
    return out


def remove(ds_id: int) -> None:
    db.delete("datasets", ds_id)
    (db.DATASETS_DIR / f"{ds_id}.npz").unlink(missing_ok=True)
