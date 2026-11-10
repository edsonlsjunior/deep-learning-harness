"""Execução de treinos em threads de fundo + inferência com modelos treinados.

Cada execução (run) grava o histórico por época no SQLite à medida que treina, então
o frontend (e o Claude, via MCP) conseguem acompanhar o progresso em tempo real.
"""
from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from . import architectures as A
from . import datasets as D
from . import db

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

DEFAULT_HP: dict[str, dict[str, Any]] = {
    "perceptron": {"epochs": 30, "lr": 0.1, "batch_size": 1, "optimizer": "rule"},
    "adaline": {"epochs": 100, "lr": 0.01, "batch_size": 0, "optimizer": "delta_rule"},
    "mlp": {"epochs": 100, "lr": 0.01, "batch_size": 32, "optimizer": "adam"},
    "cnn": {"epochs": 5, "lr": 0.001, "batch_size": 64, "optimizer": "adam"},
    "rnn": {"epochs": 30, "lr": 0.003, "batch_size": 32, "optimizer": "adam"},
    "lstm": {"epochs": 30, "lr": 0.003, "batch_size": 32, "optimizer": "adam"},
    "gru": {"epochs": 30, "lr": 0.003, "batch_size": 32, "optimizer": "adam"},
    "transformer": {"epochs": 30, "lr": 0.001, "batch_size": 32, "optimizer": "adam"},
    "autoencoder": {"epochs": 30, "lr": 0.001, "batch_size": 64, "optimizer": "adam"},
    "gan": {"epochs": 300, "lr": 0.0002, "batch_size": 64, "optimizer": "adam"},
}
COMMON_HP = {"test_size": 0.2, "normalize": True, "seed": 42, "weight_decay": 0.0, "momentum": 0.9}

_cancel: dict[int, threading.Event] = {}


# ---------------------------------------------------------------- utilidades
class Prep:
    """Pré-processamento ajustado SÓ no treino (evita vazamento de dados)."""

    def __init__(self, d: dict | None = None):
        d = d or {}
        self.x_mean, self.x_std = d.get("x_mean"), d.get("x_std")
        self.y_mean, self.y_std = d.get("y_mean"), d.get("y_std")

    def fit(self, X, y, normalize_x: bool, regression: bool):
        if normalize_x:
            self.x_mean = X.mean(0)
            self.x_std = X.std(0) + 1e-8
        if regression:
            self.y_mean, self.y_std = float(y.mean()), float(y.std() + 1e-8)
        return self

    def tx(self, X):
        return (X - self.x_mean) / self.x_std if self.x_mean is not None else X

    def ty(self, y):
        return (y - self.y_mean) / self.y_std if self.y_mean is not None else y

    def inv_y(self, y):
        return y * self.y_std + self.y_mean if self.y_mean is not None else y

    def dump(self):
        return {"x_mean": self.x_mean, "x_std": self.x_std, "y_mean": self.y_mean, "y_std": self.y_std}


def _split(n: int, test_size: float, seed: int):
    idx = np.random.default_rng(seed).permutation(n)
    if n < 20 or test_size <= 0:
        return idx, idx, True  # bases minúsculas (ex.: XOR): avalia no próprio treino
    k = max(1, int(round(n * test_size)))
    return idx[k:], idx[:k], False


def _optimizer(name: str, params, hp: dict):
    lr, wd = float(hp["lr"]), float(hp.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=float(hp.get("momentum", 0.0)), weight_decay=wd)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    return torch.optim.Adam(params, lr=lr, weight_decay=wd)


def _reg_metrics(y, p) -> dict:
    mse = float(np.mean((y - p) ** 2))
    ss = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    return {"mse": mse, "rmse": mse ** 0.5, "mae": float(np.mean(np.abs(y - p))),
            "r2": 1 - float(np.sum((y - p) ** 2)) / ss}


def _cls_metrics(y, p, k) -> dict:
    cm = np.zeros((k, k), dtype=int)
    for t, q in zip(y, p):
        cm[int(t), int(q)] += 1
    return {"accuracy": float(np.mean(y == p)), "confusion_matrix": cm.tolist()}


def _batches(n: int, bs: int, rng):
    idx = rng.permutation(n)
    bs = n if bs <= 0 else bs
    for s in range(0, n, bs):
        yield idx[s:s + bs]


def _boundary(predict_fn: Callable, X: np.ndarray, y: np.ndarray, task: str, prep: Prep) -> dict:
    """Fronteira de decisão (2 atributos) ou curva ajustada (1 atributo), em escala original."""
    sel = np.random.default_rng(0).permutation(len(X))[:400]
    if X.shape[1] == 2 and task == "tabular_classification":
        lo, hi = X.min(0), X.max(0)
        pad = (hi - lo) * 0.15 + 1e-3
        gx = np.linspace(lo[0] - pad[0], hi[0] + pad[0], 60)
        gy = np.linspace(lo[1] - pad[1], hi[1] + pad[1], 60)
        G = np.array([[a, b] for b in gy for a in gx], dtype=np.float32)
        Z = predict_fn(prep.tx(G)).reshape(60, 60)
        return {"kind": "boundary", "x_range": [float(gx[0]), float(gx[-1])], "y_range": [float(gy[0]), float(gy[-1])],
                "grid": Z.astype(int).tolist(), "points": X[sel].round(4).tolist(), "labels": y[sel].astype(int).tolist()}
    if X.shape[1] == 1 and task == "tabular_regression":
        gx = np.linspace(X.min(), X.max(), 120, dtype=np.float32)[:, None]
        return {"kind": "curve", "x": gx[:, 0].round(4).tolist(), "y_pred": prep.inv_y(predict_fn(prep.tx(gx))).round(4).tolist(),
                "points": X[sel, 0].round(4).tolist(), "targets": y[sel].round(4).tolist()}
    return {}


# ---------------------------------------------------------------- API pública
def start(model_id: int, dataset_id: int, hyperparams: dict | None = None) -> dict:
    model = db.one("SELECT * FROM models WHERE id = ?", (model_id,))
    ds = D.get(dataset_id)
    if not model:
        raise ValueError(f"Modelo {model_id} não existe")
    if not ds:
        raise ValueError(f"Base {dataset_id} não existe")
    if model["project_id"] != ds["project_id"]:
        raise ValueError("Modelo e base de dados pertencem a projetos diferentes")
    A.check_compat(model["architecture"], ds["meta"]["task"])
    hp = {**COMMON_HP, **DEFAULT_HP[model["architecture"]], **(hyperparams or {})}
    run_id = db.insert("runs", {"project_id": model["project_id"], "model_id": model_id, "dataset_id": dataset_id,
                                "status": "queued", "hyperparams": hp, "history": [], "metrics": {}, "extras": {},
                                "created_at": db.now()})
    _cancel[run_id] = threading.Event()
    threading.Thread(target=_run, args=(run_id,), daemon=True, name=f"run-{run_id}").start()
    return get(run_id)  # type: ignore[return-value]


def get(run_id: int) -> dict | None:
    return db.one("SELECT * FROM runs WHERE id = ?", (run_id,))


def cancel(run_id: int) -> None:
    if run_id in _cancel:
        _cancel[run_id].set()


def wait(run_id: int, timeout: float) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = get(run_id)
        if r and r["status"] in ("completed", "failed", "cancelled"):
            return r
        time.sleep(0.5)
    return get(run_id)  # type: ignore[return-value]


def _run(run_id: int) -> None:
    run = get(run_id)
    assert run
    model = db.one("SELECT * FROM models WHERE id = ?", (run["model_id"],))
    db.update("runs", run_id, {"status": "running"})
    history: list[dict] = []
    last_flush = [0.0]

    def log(entry: dict):
        history.append({k: (round(v, 6) if isinstance(v, float) else v) for k, v in entry.items()})
        if time.time() - last_flush[0] > 0.3:  # grava no máx. ~3x/s para acompanhar ao vivo
            db.update("runs", run_id, {"history": history})
            last_flush[0] = time.time()
        if _cancel[run_id].is_set():
            raise InterruptedError

    try:
        X, y, meta = D.load(run["dataset_id"])
        torch.manual_seed(int(run["hyperparams"]["seed"]))
        trainer = {"perceptron": _train_classic, "adaline": _train_classic, "autoencoder": _train_ae,
                   "gan": _train_gan}.get(model["architecture"], _train_supervised)
        t0 = time.time()
        metrics, extras, artifact = trainer(model, run["hyperparams"], X, y, meta, log)
        metrics["train_seconds"] = round(time.time() - t0, 2)
        torch.save(artifact, db.ARTIFACTS_DIR / f"run_{run_id}.pt")
        db.update("runs", run_id, {"status": "completed", "metrics": metrics, "extras": extras,
                                   "history": history, "finished_at": db.now()})
    except InterruptedError:
        db.update("runs", run_id, {"status": "cancelled", "history": history, "finished_at": db.now()})
    except Exception as e:  # noqa: BLE001
        db.update("runs", run_id, {"status": "failed", "history": history,
                                   "error": f"{e}\n{traceback.format_exc(limit=3)}",
                                   "finished_at": db.now()})


# ---------------------------------------------------------------- treinadores
def _train_classic(model, hp, X, y, meta, log):
    """Perceptron / Adaline em NumPy, com as regras de atualização das aulas."""
    arch, task = model["architecture"], meta["task"]
    regression = task == "tabular_regression"
    tr, va, same = _split(len(X), hp["test_size"], hp["seed"])
    prep = Prep().fit(X[tr], y[tr], hp["normalize"], regression)
    Xt, Xv = prep.tx(X[tr]), prep.tx(X[va])
    yt, yv = prep.ty(y[tr]), prep.ty(y[va])
    k = meta.get("n_classes", 2)
    net = A.Perceptron(X.shape[1], k, hp["seed"]) if arch == "perceptron" else \
        A.Adaline(X.shape[1], k, hp["seed"], regression=regression)
    rng = np.random.default_rng(hp["seed"])
    for ep in range(1, int(hp["epochs"]) + 1):
        if arch == "perceptron":
            loss = net.fit_epoch(Xt, yt, hp["lr"], rng)
            entry = {"epoch": ep, "train_loss": loss, "val_loss": float(np.mean(net.predict(Xv) != yv))}
        else:
            loss = net.fit_epoch(Xt, yt, hp["lr"], rng, int(hp["batch_size"]) or None)
            entry = {"epoch": ep, "train_loss": loss, "val_loss": net.loss(Xv, yv)}
        if not regression:
            entry["train_acc"] = float(np.mean(net.predict(Xt) == yt))
            entry["val_acc"] = float(np.mean(net.predict(Xv) == yv))
        log(entry)
    if regression:
        metrics = _reg_metrics(y[va], prep.inv_y(net.predict(Xv)))
    else:
        metrics = _cls_metrics(yv, net.predict(Xv), k)
    metrics["n_params"] = A.count_params(net)
    metrics["evaluated_on"] = "treino (base pequena)" if same else "validação"
    extras = {"weights": net.W.round(4).tolist(), "bias": net.b.round(4).tolist(),
              "loss_name": "taxa de erro" if arch == "perceptron" else "MSE/2 (Regra Delta)"}
    extras["plot"] = _boundary(net.predict, X, y, task, prep)
    return metrics, extras, {"arch": arch, "config": model["config"], "meta": meta, "prep": prep.dump(),
                             "state": net.state()}


def _train_supervised(model, hp, X, y, meta, log):
    """MLP, CNN, RNN/LSTM/GRU e Transformer — classificação ou regressão."""
    arch, task = model["architecture"], meta["task"]
    regression = task.endswith("regression")
    tabular = task.startswith("tabular")
    tr, va, same = _split(len(X), hp["test_size"], hp["seed"])
    prep = Prep().fit(X[tr], y[tr], hp["normalize"] and tabular, regression)
    net = A.build_torch(arch, model["config"], meta)
    opt = _optimizer(hp["optimizer"], net.parameters(), hp)
    crit: nn.Module = nn.MSELoss() if regression else nn.CrossEntropyLoss()
    Xt, Xv = torch.tensor(prep.tx(X[tr]), dtype=torch.float32), torch.tensor(prep.tx(X[va]), dtype=torch.float32)
    ydt = torch.float32 if regression else torch.long
    yt, yv = torch.tensor(prep.ty(y[tr]), dtype=ydt), torch.tensor(prep.ty(y[va]), dtype=ydt)
    rng = np.random.default_rng(hp["seed"])

    def fwd(xb):
        out = net(xb)
        return out.squeeze(-1) if regression else out

    def evaluate(Xe, ye):
        net.eval()
        with torch.no_grad():
            out = torch.cat([fwd(Xe[i:i + 512]) for i in range(0, len(Xe), 512)])
            loss = float(crit(out, ye))
        return loss, out

    for ep in range(1, int(hp["epochs"]) + 1):
        net.train()
        tot = 0.0
        for b in _batches(len(Xt), int(hp["batch_size"]), rng):
            opt.zero_grad()
            loss = crit(fwd(Xt[b]), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)  # evita gradiente explosivo (RNNs)
            opt.step()
            tot += float(loss) * len(b)
        vl, vout = evaluate(Xv, yv)
        entry = {"epoch": ep, "train_loss": tot / len(Xt), "val_loss": vl}
        if not regression:
            _, tout = evaluate(Xt, yt)
            entry["train_acc"] = float((tout.argmax(1) == yt).float().mean())
            entry["val_acc"] = float((vout.argmax(1) == yv).float().mean())
        log(entry)

    _, vout = evaluate(Xv, yv)
    if regression:
        metrics = _reg_metrics(y[va], prep.inv_y(vout.numpy()))
    else:
        metrics = _cls_metrics(y[va], vout.argmax(1).numpy(), meta["n_classes"])
    metrics["n_params"] = A.count_params(net)
    metrics["evaluated_on"] = "treino (base pequena)" if same else "validação"

    def predict_np(Xn):
        net.eval()
        with torch.no_grad():
            o = fwd(torch.tensor(Xn, dtype=torch.float32))
        return o.numpy() if regression else o.argmax(1).numpy()

    extras: dict[str, Any] = {}
    if tabular:
        extras["plot"] = _boundary(predict_np, X, y, task, prep)
    if task == "sequence_regression":
        n = min(len(X), 400)
        extras["plot"] = {"kind": "series", "y_true": y[:n].round(4).tolist(),
                          "y_pred": prep.inv_y(predict_np(X[:n])).round(4).tolist()}
    return metrics, extras, {"arch": arch, "config": model["config"], "meta": meta, "prep": prep.dump(),
                             "state": net.state_dict()}


def _train_ae(model, hp, X, y, meta, log):
    cfg, task = model["config"], meta["task"]
    images = task == "image_classification"
    tr, va, same = _split(len(X), hp["test_size"], hp["seed"])
    prep = Prep().fit(X[tr].reshape(len(tr), -1), y, hp["normalize"] and not images, False)
    flat = prep.tx(X.reshape(len(X), -1)).astype(np.float32)
    net = A.build_torch("autoencoder", cfg, meta)
    opt = _optimizer(hp["optimizer"], net.parameters(), hp)
    Xt, Xv = torch.tensor(flat[tr]), torch.tensor(flat[va])
    rng = np.random.default_rng(hp["seed"])
    variant = cfg["variant"]

    def corrupt(x):
        if variant != "denoising":
            return x
        xn = x + cfg["noise_factor"] * torch.randn_like(x)
        return xn.clamp(0, 1) if images else xn

    def loss_fn(x, train=True):
        recon, mu, logvar = net(corrupt(x) if train else x)
        rec = ((recon - x) ** 2).mean()
        total = rec
        if variant == "sparse":
            total = total + cfg["sparsity_weight"] * mu.abs().mean()
        if variant == "variational":
            kl = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
            total = total + cfg["beta"] * kl / x.shape[1]
        return total, rec

    for ep in range(1, int(hp["epochs"]) + 1):
        net.train()
        tot = 0.0
        for b in _batches(len(Xt), int(hp["batch_size"]), rng):
            opt.zero_grad()
            loss, _ = loss_fn(Xt[b])
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        net.eval()
        with torch.no_grad():
            _, vrec = loss_fn(Xv, train=False)
        log({"epoch": ep, "train_loss": tot / len(Xt), "val_loss": float(vrec)})

    net.eval()
    with torch.no_grad():
        recon, mu, _ = net(Xv)
        err = ((recon - Xv) ** 2).mean(1).numpy()
        noisy = corrupt(Xv[:8])
        rec_noisy, _, _ = net(noisy)
    metrics = {"reconstruction_mse": float(err.mean()), "n_params": A.count_params(net),
               "latent_dim": cfg["latent_dim"], "variant": variant,
               "anomaly_threshold_p95": float(np.percentile(err, 95)),
               "evaluated_on": "treino (base pequena)" if same else "validação"}
    z = mu.numpy()
    if z.shape[1] > 2:  # projeta o latente em 2D via PCA para visualização
        zc = z - z.mean(0)
        z = zc @ np.linalg.svd(zc, full_matrices=False)[2][:2].T
    sel = np.arange(min(500, len(z)))
    extras: dict[str, Any] = {"latent": {"points": z[sel].round(4).tolist(),
                                         "labels": y[va][sel].astype(int).tolist() if task.endswith("classification") else []},
                              "error_hist": np.histogram(err, bins=30)[0].tolist()}
    if images:
        shape = meta["input_shape"][1:]
        extras["images"] = {"shape": shape, "original": Xv[:8].reshape(-1, *shape).numpy().round(3).tolist(),
                            "input": noisy.reshape(-1, *shape).numpy().round(3).tolist() if variant == "denoising" else None,
                            "reconstruction": rec_noisy.reshape(-1, *shape).numpy().round(3).tolist()}
    return metrics, extras, {"arch": "autoencoder", "config": cfg, "meta": meta, "prep": prep.dump(),
                             "state": net.state_dict()}


def _train_gan(model, hp, X, y, meta, log):
    cfg, task = model["config"], meta["task"]
    images = task == "image_classification"
    flat = X.reshape(len(X), -1).astype(np.float32)
    prep = Prep()
    if images:
        data = flat * 2 - 1  # [0,1] -> [-1,1] para casar com a saída tanh do gerador
    else:
        prep.fit(flat, y, True, False)
        data = prep.tx(flat).astype(np.float32)
    net = A.build_torch("gan", cfg, meta)
    optG = torch.optim.Adam(net.G.parameters(), lr=hp["lr"], betas=(0.5, 0.999))
    optD = torch.optim.Adam(net.D.parameters(), lr=hp["lr"], betas=(0.5, 0.999))
    objective = cfg.get("objective", "bce")  # mantém compatibilidade com GANs criadas antes desta melhoria
    critic_steps = int(cfg.get("critic_steps", 1)) if objective == "wgan_gp" else 1
    gp_weight = float(cfg.get("gp_weight", 10.0))
    bce = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(data)
    rng = np.random.default_rng(hp["seed"])
    torch.manual_seed(int(hp["seed"]))
    for ep in range(1, int(hp["epochs"]) + 1):
        dl = gl = dx = 0.0
        nb = 0
        for b in _batches(len(Xt), int(hp["batch_size"]), rng):
            real = Xt[b]
            n = len(b)
            # 1) Crítico/perito. WGAN-GP mede a distância entre distribuições e limita
            # o gradiente do crítico; para as luas isso reduz bastante o mode collapse.
            for _ in range(critic_steps):
                fake = net.sample(n).detach()
                optD.zero_grad()
                d_real, d_fake = net.D(real), net.D(fake)
                if objective == "wgan_gp":
                    alpha = torch.rand(n, 1)
                    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
                    mixed_score = net.D(mixed)
                    grad = torch.autograd.grad(mixed_score, mixed, torch.ones_like(mixed_score),
                                               create_graph=True, only_inputs=True)[0]
                    penalty = ((grad.flatten(1).norm(2, dim=1) - 1) ** 2).mean()
                    lossD = d_fake.mean() - d_real.mean() + gp_weight * penalty
                else:
                    lossD = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
                lossD.backward()
                optD.step()
            # 2) Gerador/falsificador: procura aumentar a avaliação do crítico.
            optG.zero_grad()
            d_gen = net.D(net.sample(n))
            lossG = -d_gen.mean() if objective == "wgan_gp" else bce(d_gen, torch.ones_like(d_gen))
            lossG.backward()
            optG.step()
            dl, gl = dl + float(lossD.detach()), gl + float(lossG.detach())
            score = d_real.mean() if objective == "wgan_gp" else torch.sigmoid(d_real).mean()
            dx += float(score.detach())
            nb += 1
        log({"epoch": ep, "d_loss": dl / nb, "g_loss": gl / nb, "d_real_score": dx / nb})

    net.eval()
    with torch.no_grad():
        samples = net.sample(64).numpy()
    extras: dict[str, Any] = {}
    if images:
        shape = meta["input_shape"][1:]
        extras["images"] = {"shape": shape, "generated": ((samples[:16] + 1) / 2).clip(0, 1).reshape(-1, *shape).round(3).tolist()}
    elif flat.shape[1] == 2:
        gen = samples[:, :] * prep.x_std + prep.x_mean
        with torch.no_grad():
            gen = np.concatenate([gen, (net.sample(436).numpy() * prep.x_std + prep.x_mean)])
        sel = rng.permutation(len(flat))[:500]
        extras["plot"] = {"kind": "gan2d", "real": flat[sel].round(4).tolist(), "generated": gen.round(4).tolist()}
    metrics = {"n_params": A.count_params(net), "n_params_generator": A.count_params(net.G),
               "n_params_discriminator": A.count_params(net.D), "objective": objective,
               "critic_steps": critic_steps}
    return metrics, extras, {"arch": "gan", "config": cfg, "meta": meta, "prep": prep.dump(), "state": net.state_dict()}


# ---------------------------------------------------------------- inferência
def _load_artifact(run_id: int):
    run = get(run_id)
    if not run:
        raise ValueError(f"Execução {run_id} não existe")
    if run["status"] != "completed":
        raise ValueError(f"Execução {run_id} está '{run['status']}', não 'completed'")
    art = torch.load(db.ARTIFACTS_DIR / f"run_{run_id}.pt", weights_only=False)
    return run, art


def predict(run_id: int, inputs: list) -> dict:
    run, art = _load_artifact(run_id)
    arch, meta, prep = art["arch"], art["meta"], Prep(art["prep"])
    X = np.asarray(inputs, dtype=np.float32).reshape(-1, *meta["input_shape"])
    names = meta.get("class_names")
    if arch in ("perceptron", "adaline"):
        net = A.Perceptron(1, 2) if arch == "perceptron" else A.Adaline(1, 2)
        net.load(art["state"])
        p = net.predict(prep.tx(X))
        if meta["task"].endswith("regression"):
            return {"predictions": prep.inv_y(p).round(4).tolist()}
        return {"predictions": [int(v) for v in p], "labels": [names[int(v)] for v in p] if names else None}
    net = A.build_torch(arch, art["config"], meta)
    net.load_state_dict(art["state"])
    net.eval()
    if arch == "gan":
        raise ValueError("GANs não fazem predição; use generate_samples")
    with torch.no_grad():
        if arch == "autoencoder":
            flat = torch.tensor(prep.tx(X.reshape(len(X), -1)), dtype=torch.float32)
            recon, mu, _ = net(flat)
            err = ((recon - flat) ** 2).mean(1)
            # Devolve também a reconstrução no formato original. Assim a interface pode
            # comparar uma amostra desenhada/fornecida pelo aluno com a saída do decoder.
            restored = recon.numpy().reshape(len(X), *meta["input_shape"])
            return {"reconstruction_error": err.numpy().round(6).tolist(),
                    "latent": mu.numpy().round(4).tolist(),
                    "reconstruction": restored.round(4).tolist()}
        Xn = prep.tx(X) if meta["task"].startswith("tabular") else X
        out = net(torch.tensor(Xn, dtype=torch.float32))
    if meta["task"].endswith("regression"):
        return {"predictions": prep.inv_y(out.squeeze(-1).numpy()).round(4).tolist()}
    probs = torch.softmax(out, 1).numpy()
    cls = probs.argmax(1)
    return {"predictions": cls.tolist(), "labels": [names[c] for c in cls] if names else None,
            "probabilities": probs.round(4).tolist()}


def generate(run_id: int, n: int = 16) -> dict:
    run, art = _load_artifact(run_id)
    meta = art["meta"]
    net = A.build_torch(art["arch"], art["config"], meta)
    net.load_state_dict(art["state"])
    net.eval()
    with torch.no_grad():
        if art["arch"] == "gan":
            s = net.sample(n).numpy()
        elif art["arch"] == "autoencoder" and art["config"]["variant"] == "variational":
            s = net.decoder(torch.randn(n, art["config"]["latent_dim"])).numpy()
        else:
            raise ValueError("Geração disponível apenas para GAN ou autoencoder variacional")
    if meta["task"] == "image_classification":
        s = (s + 1) / 2 if art["arch"] == "gan" else s
        return {"shape": meta["input_shape"][1:], "samples": s.clip(0, 1).reshape(n, *meta["input_shape"][1:]).round(3).tolist()}
    prep = Prep(art["prep"])
    if prep.x_mean is not None:
        s = s * prep.x_std + prep.x_mean
    return {"samples": s.round(4).tolist()}
