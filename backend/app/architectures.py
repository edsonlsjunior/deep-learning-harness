"""Zoológico de arquiteturas — cada entrada corresponde a um modelo das aulas.

Aula 1: perceptron, adaline, mlp
Aula 2: cnn
Aula 3: rnn, lstm, gru, autoencoder (basic/denoising/sparse/variational), gan, transformer

Perceptron e Adaline são implementados "na mão" em NumPy, exatamente com as regras
de atualização vistas em aula. Os demais usam PyTorch.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

ARCHITECTURES: dict[str, dict[str, Any]] = {
    "perceptron": {
        "aula": 1,
        "description": "Perceptron de Rosenblatt (1957): soma ponderada + função degrau, regra w += η(y - ŷ)x. "
                       "Implementado em NumPy. Multiclasse via um-contra-todos.",
        "tasks": ["tabular_classification"],
        "default_config": {},
        "config_doc": {},
    },
    "adaline": {
        "aula": 1,
        "description": "ADALINE (Widrow & Hoff, 1960): mesma estrutura do perceptron, mas aprende pela Regra Delta "
                       "(gradiente descendente no erro quadrático da saída LINEAR). NumPy.",
        "tasks": ["tabular_classification", "tabular_regression"],
        "default_config": {},
        "config_doc": {},
    },
    "mlp": {
        "aula": 1,
        "description": "Perceptron multicamadas (feedforward) treinado por retropropagação. Resolve problemas não "
                       "linearmente separáveis (ex.: XOR) graças às ativações não lineares.",
        "tasks": ["tabular_classification", "tabular_regression", "image_classification"],
        "default_config": {"hidden_layers": [16, 16], "activation": "relu", "dropout": 0.0},
        "config_doc": {
            "hidden_layers": "lista com nº de neurônios por camada oculta, ex.: [32, 16]",
            "activation": "relu | tanh | sigmoid | leaky_relu | gelu | linear",
            "dropout": "probabilidade de dropout entre camadas (0 a 0.8)",
        },
    },
    "cnn": {
        "aula": 2,
        "description": "Rede convolucional: blocos Conv2d (kernel, stride, padding) + ativação + pooling, depois "
                       "Flatten e camadas densas.",
        "tasks": ["image_classification"],
        "default_config": {
            "conv_layers": [
                {"filters": 16, "kernel_size": 3, "stride": 1, "padding": "same", "pool": 2},
                {"filters": 32, "kernel_size": 3, "stride": 1, "padding": "same", "pool": 2},
            ],
            "dense_layers": [64],
            "activation": "relu",
            "pooling": "max",
            "dropout": 0.25,
        },
        "config_doc": {
            "conv_layers": "lista de blocos {filters, kernel_size, stride, padding: same|valid, pool: 0|2|...}",
            "dense_layers": "neurônios das camadas densas após o Flatten",
            "activation": "relu | tanh | sigmoid | leaky_relu | gelu",
            "pooling": "max | avg",
            "dropout": "dropout antes das camadas densas",
        },
    },
    "rnn": {
        "aula": 3,
        "description": "Rede recorrente simples (Elman): o estado oculto h_t depende de x_t e h_{t-1}. Sofre com "
                       "gradientes que desaparecem/explodem em sequências longas.",
        "tasks": ["sequence_regression", "sequence_classification"],
        "default_config": {"hidden_size": 32, "num_layers": 1, "dropout": 0.0, "bidirectional": False},
        "config_doc": {
            "hidden_size": "nº de unidades do estado oculto",
            "num_layers": "camadas recorrentes empilhadas",
            "dropout": "dropout entre camadas recorrentes (se num_layers > 1)",
            "bidirectional": "processa a sequência nos dois sentidos",
        },
    },
    "lstm": {
        "aula": 3,
        "description": "LSTM: célula de memória + portas de esquecimento, entrada e saída. Preserva dependências "
                       "de longo prazo.",
        "tasks": ["sequence_regression", "sequence_classification"],
        "default_config": {"hidden_size": 32, "num_layers": 1, "dropout": 0.0, "bidirectional": False},
        "config_doc": "igual ao rnn",
    },
    "gru": {
        "aula": 3,
        "description": "GRU: variação simplificada da LSTM com portas de atualização e reset, sem célula separada.",
        "tasks": ["sequence_regression", "sequence_classification"],
        "default_config": {"hidden_size": 32, "num_layers": 1, "dropout": 0.0, "bidirectional": False},
        "config_doc": "igual ao rnn",
    },
    "transformer": {
        "aula": 3,
        "description": "Encoder Transformer (Attention Is All You Need): embedding + posição, self-attention "
                       "multi-head, Add&Norm, feed-forward. Pooling médio + cabeça de saída.",
        "tasks": ["sequence_regression", "sequence_classification"],
        "default_config": {"d_model": 32, "nhead": 4, "num_layers": 2, "dim_feedforward": 64, "dropout": 0.1},
        "config_doc": {
            "d_model": "dimensão dos embeddings (múltiplo de nhead)",
            "nhead": "nº de cabeças de atenção",
            "num_layers": "nº de blocos encoder empilhados",
            "dim_feedforward": "neurônios da camada feed-forward de cada bloco",
            "dropout": "dropout interno",
        },
    },
    "autoencoder": {
        "aula": 3,
        "description": "Autoencoder MLP: encoder comprime para o espaço latente, decoder reconstrói. Variantes: "
                       "basic, denoising (entrada com ruído), sparse (penalidade L1 no latente), variational (VAE).",
        "tasks": ["image_classification", "tabular_classification", "tabular_regression", "unsupervised"],
        "default_config": {"hidden_layers": [128, 64], "latent_dim": 2, "variant": "basic",
                           "activation": "relu", "noise_factor": 0.3, "sparsity_weight": 0.001, "beta": 1.0},
        "config_doc": {
            "hidden_layers": "camadas do encoder (o decoder é o espelho)",
            "latent_dim": "dimensão do espaço latente (2 permite visualizar o latente)",
            "variant": "basic | denoising | sparse | variational",
            "noise_factor": "desvio do ruído gaussiano (denoising)",
            "sparsity_weight": "peso da penalidade L1 no latente (sparse)",
            "beta": "peso da divergência KL (variational)",
        },
    },
    "gan": {
        "aula": 3,
        "description": "GAN (Goodfellow, 2014): gerador (falsificador) x discriminador (perito) em jogo adversarial. "
                       "Versão MLP (vanilla GAN).",
        "tasks": ["image_classification", "tabular_classification", "unsupervised"],
        "default_config": {"latent_dim": 16, "generator_layers": [64, 128], "discriminator_layers": [128, 64]},
        "config_doc": {
            "latent_dim": "dimensão do ruído z de entrada do gerador",
            "generator_layers": "camadas ocultas do gerador",
            "discriminator_layers": "camadas ocultas do discriminador",
        },
    },
}


def validate_config(arch: str, config: dict | None) -> dict:
    if arch not in ARCHITECTURES:
        raise ValueError(f"Arquitetura '{arch}' desconhecida. Opções: {list(ARCHITECTURES)}")
    merged = {**ARCHITECTURES[arch]["default_config"], **(config or {})}
    if arch == "transformer" and merged["d_model"] % merged["nhead"]:
        raise ValueError("d_model precisa ser múltiplo de nhead")
    if arch == "autoencoder" and merged["variant"] not in ("basic", "denoising", "sparse", "variational"):
        raise ValueError("variant deve ser basic | denoising | sparse | variational")
    return merged


def check_compat(arch: str, task: str) -> None:
    tasks = ARCHITECTURES[arch]["tasks"]
    if task not in tasks:
        raise ValueError(f"A arquitetura '{arch}' não se aplica à tarefa '{task}'. Tarefas suportadas: {tasks}")


# ======================================================================
# Modelos clássicos em NumPy (Aula 1)
# ======================================================================
class Perceptron:
    """Um neurônio por classe (um-contra-todos); binário usa 1 neurônio."""

    def __init__(self, n_features: int, n_classes: int, seed: int = 0):
        self.k = 1 if n_classes == 2 else n_classes
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, (n_features, self.k))
        self.b = np.zeros(self.k)

    def net(self, X):
        return X @ self.W + self.b

    def predict(self, X):
        z = self.net(X)
        return (z[:, 0] >= 0).astype(int) if self.k == 1 else z.argmax(1)

    def targets(self, y):
        return y[:, None].astype(float) if self.k == 1 else np.eye(self.k)[y]

    def fit_epoch(self, X, y, lr, rng):
        """Regra do perceptron, amostra a amostra: w ← w + η (y − ŷ) x."""
        T = self.targets(y)
        for i in rng.permutation(len(X)):
            yhat = (self.net(X[i:i + 1]) >= 0).astype(float)[0]
            err = T[i] - yhat
            self.W += lr * np.outer(X[i], err)
            self.b += lr * err
        return float(np.mean(self.predict(X) != y))  # "perda" = taxa de erro

    def state(self):
        return {"W": self.W, "b": self.b, "k": self.k}

    def load(self, s):
        self.W, self.b, self.k = s["W"], s["b"], s["k"]


class Adaline(Perceptron):
    """Saída linear; Regra Delta = gradiente descendente no erro quadrático médio."""

    def __init__(self, n_features: int, n_classes: int, seed: int = 0, regression: bool = False):
        super().__init__(n_features, 2 if regression else n_classes, seed)
        self.regression = regression

    def predict(self, X):
        z = self.net(X)
        if self.regression:
            return z[:, 0]
        return (z[:, 0] >= 0).astype(int) if self.k == 1 else z.argmax(1)

    def targets(self, y):
        if self.regression:
            return y[:, None].astype(float)
        return (2 * super().targets(y) - 1)  # alvos em {-1, +1}

    def loss(self, X, y):
        return float(np.mean((self.targets(y) - self.net(X)) ** 2) / 2)

    def fit_epoch(self, X, y, lr, rng, batch_size=None):
        T = self.targets(y)
        idx = rng.permutation(len(X))
        bs = batch_size or len(X)
        for s in range(0, len(X), bs):
            b = idx[s:s + bs]
            err = T[b] - self.net(X[b])            # erro contínuo (antes da ativação)
            self.W += lr * X[b].T @ err / len(b)   # Δw = η · Σ (y − z) x
            self.b += lr * err.mean(0)
        return self.loss(X, y)

    def state(self):
        return {**super().state(), "regression": self.regression}

    def load(self, s):
        super().load(s)
        self.regression = s["regression"]


# ======================================================================
# Redes em PyTorch (Aulas 1–3)
# ======================================================================
def act(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(), "leaky_relu": nn.LeakyReLU(0.2),
            "gelu": nn.GELU(), "linear": nn.Identity()}[name]


def mlp_stack(sizes: list[int], activation: str, dropout: float = 0.0, last_act: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(act(activation))
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, cfg: dict):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), mlp_stack([in_dim, *cfg["hidden_layers"], out_dim],
                                                         cfg["activation"], cfg["dropout"]))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, in_shape: list[int], out_dim: int, cfg: dict):
        super().__init__()
        c = in_shape[0]
        blocks: list[nn.Module] = []
        for L in cfg["conv_layers"]:
            k = int(L.get("kernel_size", 3))
            pad = k // 2 if L.get("padding", "same") == "same" else 0
            blocks += [nn.Conv2d(c, int(L["filters"]), k, stride=int(L.get("stride", 1)), padding=pad),
                       act(cfg["activation"])]
            if int(L.get("pool", 0)) > 1:
                blocks.append((nn.MaxPool2d if cfg.get("pooling", "max") == "max" else nn.AvgPool2d)(int(L["pool"])))
            c = int(L["filters"])
        self.features = nn.Sequential(*blocks)
        with torch.no_grad():
            flat = self.features(torch.zeros(1, *in_shape)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(cfg.get("dropout", 0.0)),
                                  mlp_stack([flat, *cfg["dense_layers"], out_dim], cfg["activation"]))

    def forward(self, x):
        return self.head(self.features(x))


class SeqRNN(nn.Module):
    def __init__(self, kind: str, in_feat: int, out_dim: int, cfg: dict):
        super().__init__()
        cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = cls(in_feat, cfg["hidden_size"], num_layers=cfg["num_layers"], batch_first=True,
                       dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0,
                       bidirectional=bool(cfg["bidirectional"]))
        self.head = nn.Linear(cfg["hidden_size"] * (2 if cfg["bidirectional"] else 1), out_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])  # usa o estado do último passo de tempo


class SeqTransformer(nn.Module):
    def __init__(self, seq_len: int, in_feat: int, out_dim: int, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.embed = nn.Linear(in_feat, d)
        pe = torch.zeros(seq_len, d)  # codificação posicional senoidal
        pos = torch.arange(seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, : d // 2]
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(d, cfg["nhead"], cfg["dim_feedforward"], cfg["dropout"], batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg["num_layers"])
        self.head = nn.Linear(d, out_dim)

    def forward(self, x):
        h = self.encoder(self.embed(x) + self.pe)
        return self.head(h.mean(1))


class AutoEncoder(nn.Module):
    def __init__(self, in_dim: int, cfg: dict, sigmoid_out: bool):
        super().__init__()
        self.variational = cfg["variant"] == "variational"
        hs, z = cfg["hidden_layers"], cfg["latent_dim"]
        self.encoder = mlp_stack([in_dim, *hs], cfg["activation"], last_act=True) if hs else nn.Identity()
        last = hs[-1] if hs else in_dim
        self.to_z = nn.Linear(last, z * (2 if self.variational else 1))
        self.decoder = nn.Sequential(mlp_stack([z, *reversed(hs), in_dim], cfg["activation"]),
                                     nn.Sigmoid() if sigmoid_out else nn.Identity())
        self.z = z

    def encode(self, x):
        h = self.to_z(self.encoder(x.flatten(1)))
        if self.variational:
            return h[:, : self.z], h[:, self.z:]
        return h, None

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if (self.variational and self.training) else mu
        return self.decoder(z), mu, logvar


class GAN(nn.Module):
    def __init__(self, data_dim: int, cfg: dict, tanh_out: bool):
        super().__init__()
        self.latent_dim = cfg["latent_dim"]
        self.G = nn.Sequential(mlp_stack([self.latent_dim, *cfg["generator_layers"], data_dim], "leaky_relu"),
                               nn.Tanh() if tanh_out else nn.Identity())
        self.D = mlp_stack([data_dim, *cfg["discriminator_layers"], 1], "leaky_relu")

    def sample(self, n: int):
        return self.G(torch.randn(n, self.latent_dim))


def build_torch(arch: str, cfg: dict, meta: dict) -> nn.Module:
    shape = meta["input_shape"]
    in_dim = int(np.prod(shape))
    task = meta["task"]
    out = meta.get("n_classes", 2) if task.endswith("classification") else 1
    if arch == "mlp":
        return MLP(in_dim, out, cfg)
    if arch == "cnn":
        return CNN(shape, out, cfg)
    if arch in ("rnn", "lstm", "gru"):
        return SeqRNN(arch, shape[-1], out, cfg)
    if arch == "transformer":
        return SeqTransformer(shape[0], shape[-1], out, cfg)
    if arch == "autoencoder":
        return AutoEncoder(in_dim, cfg, sigmoid_out=task == "image_classification")
    if arch == "gan":
        return GAN(in_dim, cfg, tanh_out=task == "image_classification")
    raise ValueError(arch)


def count_params(model: Any) -> int:
    if isinstance(model, nn.Module):
        return int(sum(p.numel() for p in model.parameters()))
    return int(model.W.size + model.b.size)
