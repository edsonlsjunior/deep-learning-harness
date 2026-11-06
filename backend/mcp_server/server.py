"""Servidor MCP (stdio) do Laboratório de Deep Learning.

O Claude Code inicia este script como subprocesso e conversa com ele por stdin/stdout
usando JSON-RPC (protocolo MCP). Cada função decorada com @mcp.tool() vira uma
ferramenta que o modelo pode chamar. A docstring e os tipos dos parâmetros viram a
descrição/JSON Schema que o modelo lê para decidir COMO e QUANDO usar a ferramenta.

Este servidor é um cliente "fino" da API REST (backend/app/main.py): ele não guarda
estado próprio — então o que o Claude faz aparece imediatamente no frontend.

Teste isolado:   npx @modelcontextprotocol/inspector <python> backend/mcp_server/server.py
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

API = os.environ.get("DL_LAB_API", "http://127.0.0.1:8000").rstrip("/")

mcp = MCPServer(
    name="dl-lab",
    instructions=(
        "Ferramentas para operar o Laboratório de Deep Learning: projetos, bases de dados, modelos "
        "(perceptron, adaline, mlp, cnn, rnn, lstm, gru, transformer, autoencoder, gan), treinos e predições. "
        "Bases podem vir de arquivos enviados pelo usuário (inspect_file/import_file) ou do Kaggle (search_kaggle)."
    ),
)

READ = ToolAnnotations(readOnlyHint=True)
DESTRUCTIVE = ToolAnnotations(destructiveHint=True)


async def call(method: str, path: str, **kw) -> Any:
    async with httpx.AsyncClient(base_url=API, timeout=660) as c:
        r = await c.request(method, path, **kw)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except Exception:  # noqa: BLE001
            detail = r.text
        raise ValueError(f"Erro {r.status_code}: {detail}")
    return r.json()


def dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)


def run_summary(run: dict, include_history: bool = False) -> dict:
    """Resumo compacto de uma execução — evita despejar grades/imagens no contexto do modelo."""
    hist = run.get("history") or []
    out = {k: run.get(k) for k in ("id", "model_id", "dataset_id", "status", "hyperparams", "metrics", "error")}
    out["epochs_done"] = len(hist)
    if hist:
        out["first_epoch"], out["last_epoch"] = hist[0], hist[-1]
        if include_history:
            step = max(1, len(hist) // 20)
            out["history_sampled"] = hist[::step]
    extras = run.get("extras") or {}
    if "weights" in extras:
        out["weights"], out["bias"] = extras["weights"], extras["bias"]
    if extras:
        out["visualizations_in_ui"] = [k for k in extras if k not in ("weights", "bias", "loss_name")]
    if not out["error"]:
        out.pop("error")
    return out


# ------------------------------------------------------------------ catálogo
@mcp.tool(annotations=READ, structured_output=False)
async def list_architectures() -> str:
    """Lista as arquiteturas disponíveis (por aula), as tarefas compatíveis, a configuração padrão,
    a documentação de cada campo de configuração e os hiperparâmetros de treino padrão."""
    return dump(await call("GET", "/api/architectures"))


@mcp.tool(annotations=READ, structured_output=False)
async def list_dataset_sources() -> str:
    """Lista as bases de dados embutidas que podem ser adicionadas a um projeto (com descrição e
    parâmetros aceitos). Também é possível importar arquivos próprios com `import_file`
    ou buscar bases no Kaggle com `search_kaggle`."""
    return dump(await call("GET", "/api/datasets/catalog"))


# ------------------------------------------------------------------ projetos
@mcp.tool(annotations=READ, structured_output=False)
async def list_projects() -> str:
    """Lista todos os projetos com contagem de bases, modelos e execuções."""
    return dump(await call("GET", "/api/projects"))


@mcp.tool(structured_output=False)
async def create_project(name: str, description: str = "") -> str:
    """Cria um novo projeto (um espaço que agrupa bases de dados, modelos e treinos)."""
    return dump(await call("POST", "/api/projects", json={"name": name, "description": description}))


@mcp.tool(annotations=READ, structured_output=False)
async def get_project(project_id: int) -> str:
    """Mostra um projeto: bases (id, tarefa, formato), modelos (id, arquitetura, config) e execuções
    (status e métricas principais)."""
    p = await call("GET", f"/api/projects/{project_id}")
    return dump({
        "id": p["id"], "name": p["name"], "description": p["description"],
        "datasets": [{"id": d["id"], "name": d["name"], "source": d["source"], "task": d["meta"]["task"],
                      "n_samples": d["meta"]["n_samples"], "input_shape": d["meta"]["input_shape"],
                      "n_classes": d["meta"].get("n_classes")} for d in p["datasets"]],
        "models": [{"id": m["id"], "name": m["name"], "architecture": m["architecture"], "config": m["config"]}
                   for m in p["models"]],
        "runs": [{"id": r["id"], "model_id": r["model_id"], "dataset_id": r["dataset_id"], "status": r["status"],
                  "metrics": {k: v for k, v in (r["metrics"] or {}).items() if k != "confusion_matrix"}}
                 for r in p["runs"]],
    })


@mcp.tool(annotations=DESTRUCTIVE, structured_output=False)
async def delete_project(project_id: int) -> str:
    """Apaga um projeto e TUDO que está nele (bases, modelos, execuções). Só use se o usuário pedir."""
    return dump(await call("DELETE", f"/api/projects/{project_id}"))


# ------------------------------------------------------------------ bases de dados
@mcp.tool(structured_output=False)
async def add_dataset(project_id: int, source: str, name: str | None = None,
                      params: dict[str, Any] | None = None) -> str:
    """Adiciona uma base embutida ao projeto. `source` vem de `list_dataset_sources`
    (ex.: xor_gate, moons, iris, digits, mnist, daily_min_temperatures, sine_series, airline_passengers, sequence_parity).
    `params` opcionais dependem da base, ex.: {"n_samples": 500, "noise": 0.2} ou {"window": 20}."""
    body = {"source": source, "name": name, "params": params or {}}
    d = await call("POST", f"/api/projects/{project_id}/datasets", json=body)
    return dump({"id": d["id"], "name": d["name"], "meta": d["meta"]})


@mcp.tool(annotations=READ, structured_output=False)
async def inspect_file(path: str, sheet: str | None = None) -> str:
    """Inspeciona um arquivo de dados ANTES de importar: nº de linhas, colunas (numérica/categórica,
    valores únicos, faltantes, exemplos) e prévia. Use para escolher a coluna alvo e a tarefa.
    Funciona com arquivos enviados pelo usuário no chat e com os baixados do Kaggle."""
    return dump(await call("GET", "/api/files/inspect", params={"path": path, **({"sheet": sheet} if sheet else {})}))


@mcp.tool(structured_output=False)
async def import_file(project_id: int, path: str, target_column: str | None = None,
                      task: str | None = None, name: str | None = None, sheet: str | None = None,
                      max_rows: int | None = None) -> str:
    """Importa um arquivo local como base de dados do projeto. Formatos: .csv .tsv .txt .json .jsonl
    .xlsx .xls .parquet (tabelas) e .npz (arrays X e y prontos, inclusive imagens (N,C,H,W) e séries (N,T,F)).
    Tabelas: colunas numéricas viram atributos, categóricas (até 50 valores) recebem one-hot e colunas de
    texto livre/ID são descartadas. `task`: tabular_classification | tabular_regression (inferida se omitida).
    Sem `target_column` a base é não supervisionada (autoencoder/GAN). `max_rows` faz amostragem aleatória
    (recomendado <= 20000 para treinar rápido em CPU). `sheet`: aba de planilha Excel."""
    params = {"path": path, "target_column": target_column, "task": task, "sheet": sheet, "max_rows": max_rows}
    body = {"source": "file", "name": name, "params": {k: v for k, v in params.items() if v is not None}}
    d = await call("POST", f"/api/projects/{project_id}/datasets", json=body)
    return dump({"id": d["id"], "name": d["name"], "meta": d["meta"]})


# ------------------------------------------------------------------ Kaggle
@mcp.tool(annotations=READ, structured_output=False)
async def search_kaggle(query: str, page: int = 1, file_type: str | None = None,
                        sort_by: str = "hottest") -> str:
    """Busca bases públicas no Kaggle. Retorna ref (dono/base), título, tamanho, downloads, votos e
    usabilidade. `file_type`: csv | json | parquet. `sort_by`: hottest | votes | updated | active.
    Prefira bases pequenas (poucos MB) e com boa usabilidade para treinar em aula."""
    params = {"q": query, "page": page, "sort_by": sort_by, **({"file_type": file_type} if file_type else {})}
    return dump(await call("GET", "/api/kaggle/search", params=params))


@mcp.tool(annotations=READ, structured_output=False)
async def kaggle_dataset_files(ref: str) -> str:
    """Lista os arquivos de uma base do Kaggle (`ref` = 'dono/base') com tamanho e se são importáveis."""
    return dump(await call("GET", "/api/kaggle/files", params={"ref": ref}))


@mcp.tool(structured_output=False)
async def download_kaggle_dataset(ref: str, file_name: str | None = None) -> str:
    """Baixa uma base do Kaggle (inteira ou só `file_name`) para o servidor do laboratório e devolve os
    caminhos locais. Depois use `inspect_file` e `import_file` com o caminho do arquivo desejado."""
    return dump(await call("POST", "/api/kaggle/download", json={"ref": ref, "file_name": file_name}))


@mcp.tool(annotations=READ, structured_output=False)
async def describe_dataset(dataset_id: int, n_examples: int = 3) -> str:
    """Descreve uma base: tarefa, nº de amostras, formato da entrada, classes/balanceamento ou
    estatísticas do alvo, e alguns exemplos."""
    p = await call("GET", f"/api/datasets/{dataset_id}/preview", params={"n": n_examples})
    if "images" in p:  # imagens são grandes: só informa o formato
        p["images"] = [{"y": im["y"], "shape": [len(im["pixels"]), len(im["pixels"][0])]} for im in p["images"]]
    return dump(p)


@mcp.tool(annotations=DESTRUCTIVE, structured_output=False)
async def delete_dataset(dataset_id: int) -> str:
    """Remove uma base de dados (e as execuções que a usaram)."""
    return dump(await call("DELETE", f"/api/datasets/{dataset_id}"))


# ------------------------------------------------------------------ modelos
@mcp.tool(structured_output=False)
async def create_model(project_id: int, name: str, architecture: str,
                       config: dict[str, Any] | None = None) -> str:
    """Cria um modelo no projeto. `architecture`: perceptron | adaline | mlp | cnn | rnn | lstm | gru |
    transformer | autoencoder | gan. `config` sobrescreve a configuração padrão (veja `list_architectures`),
    ex.: mlp {"hidden_layers": [8], "activation": "tanh"}; cnn {"conv_layers": [{"filters": 8,
    "kernel_size": 3, "pool": 2}], "dense_layers": [32]}; autoencoder {"latent_dim": 2, "variant": "denoising"}."""
    body = {"name": name, "architecture": architecture, "config": config or {}}
    return dump(await call("POST", f"/api/projects/{project_id}/models", json=body))


@mcp.tool(annotations=DESTRUCTIVE, structured_output=False)
async def delete_model(model_id: int) -> str:
    """Remove um modelo (e suas execuções)."""
    return dump(await call("DELETE", f"/api/models/{model_id}"))


# ------------------------------------------------------------------ treino e avaliação
@mcp.tool(structured_output=False)
async def train_model(model_id: int, dataset_id: int, epochs: int | None = None, lr: float | None = None,
                      batch_size: int | None = None, optimizer: str | None = None,
                      test_size: float | None = None, normalize: bool | None = None,
                      extra_hyperparams: dict[str, Any] | None = None, wait_seconds: int = 120) -> str:
    """Treina um modelo numa base (ambos do mesmo projeto). Hiperparâmetros omitidos usam o padrão da
    arquitetura. `optimizer`: adam | sgd | rmsprop (perceptron/adaline usam suas próprias regras).
    `batch_size` 0 = lote inteiro. `extra_hyperparams`: weight_decay, momentum, seed.
    Aguarda até `wait_seconds` pelo fim do treino; se não terminar, retorna o id da execução para
    consultar depois com `get_run`. O progresso aparece ao vivo na interface."""
    hp = {k: v for k, v in {"epochs": epochs, "lr": lr, "batch_size": batch_size, "optimizer": optimizer,
                            "test_size": test_size, "normalize": normalize}.items() if v is not None}
    hp.update(extra_hyperparams or {})
    run = await call("POST", "/api/runs", json={"model_id": model_id, "dataset_id": dataset_id, "hyperparams": hp})
    if wait_seconds > 0:
        run = await call("GET", f"/api/runs/{run['id']}/wait", params={"timeout": wait_seconds})
    return dump(run_summary(run))


@mcp.tool(annotations=READ, structured_output=False)
async def get_run(run_id: int, wait_seconds: int = 0, include_history: bool = False) -> str:
    """Consulta uma execução de treino: status, hiperparâmetros, métricas finais (accuracy/matriz de
    confusão ou mse/mae/r2, perda de reconstrução, etc.), primeira e última época. Use `wait_seconds`
    para aguardar a conclusão e `include_history` para ver a curva de aprendizado amostrada."""
    path = f"/api/runs/{run_id}/wait" if wait_seconds > 0 else f"/api/runs/{run_id}"
    run = await call("GET", path, params={"timeout": wait_seconds} if wait_seconds > 0 else None)
    return dump(run_summary(run, include_history))


@mcp.tool(annotations=READ, structured_output=False)
async def compare_runs(project_id: int) -> str:
    """Tabela comparativa de todas as execuções do projeto (modelo, arquitetura, base, hiperparâmetros
    principais, métricas finais e nº de parâmetros)."""
    p = await call("GET", f"/api/projects/{project_id}")
    models = {m["id"]: m for m in p["models"]}
    dsets = {d["id"]: d for d in p["datasets"]}
    rows = []
    for r in p["runs"]:
        m, d = models.get(r["model_id"], {}), dsets.get(r["dataset_id"], {})
        met = {k: v for k, v in (r["metrics"] or {}).items() if k != "confusion_matrix"}
        last = (r["history"] or [{}])[-1]
        rows.append({"run": r["id"], "status": r["status"], "model": m.get("name"), "arch": m.get("architecture"),
                     "dataset": d.get("name"), "epochs": r["hyperparams"].get("epochs"), "lr": r["hyperparams"].get("lr"),
                     "metrics": met, "last_epoch": last})
    return dump(rows)


@mcp.tool(structured_output=False)
async def cancel_run(run_id: int) -> str:
    """Interrompe um treino em andamento."""
    return dump(await call("POST", f"/api/runs/{run_id}/cancel"))


@mcp.tool(annotations=READ, structured_output=False)
async def predict(run_id: int, inputs: list[Any]) -> str:
    """Faz predições com o modelo treinado numa execução concluída. `inputs` é uma lista de amostras no
    formato de entrada da base (ex.: [[0,1],[1,1]] para XOR; listas de valores para janelas de séries).
    Para autoencoders retorna o erro de reconstrução (escore de anomalia) e o vetor latente."""
    return dump(await call("POST", f"/api/runs/{run_id}/predict", json={"inputs": inputs}))


@mcp.tool(annotations=READ, structured_output=False)
async def generate_samples(run_id: int, n: int = 8) -> str:
    """Gera novas amostras com uma GAN ou autoencoder variacional treinado. Para imagens, retorna apenas
    um resumo (as imagens aparecem na interface); para dados tabulares, retorna os pontos."""
    out = await call("POST", f"/api/runs/{run_id}/generate", params={"n": n})
    if "shape" in out:
        import statistics
        flat = [v for s in out["samples"] for row in s for v in row]
        return dump({"n": len(out["samples"]), "shape": out["shape"], "mean_pixel": round(statistics.fmean(flat), 4),
                     "note": "imagens geradas visíveis na aba da execução na interface"})
    return dump(out)


# ------------------------------------------------------------------ recurso
@mcp.resource("lab://guia", name="guia", description="Mapa aulas -> arquiteturas -> bases sugeridas", mime_type="text/markdown")
def guia() -> str:
    return """# Guia do laboratório
| Aula | Arquitetura | Bases sugeridas | Experimento clássico |
|---|---|---|---|
| 1 | perceptron, adaline | and_gate, or_gate, blobs, iris | Perceptron resolve AND/OR mas falha no XOR |
| 1 | mlp | xor_gate, moons, circles, breast_cancer, diabetes | MLP com 1 camada oculta resolve XOR |
| 2 | cnn | digits, mnist, fashion_mnist | CNN x MLP em imagens (parâmetros x acurácia) |
| 3 | rnn, lstm, gru | daily_min_temperatures, sine_series, airline_passengers, sequence_parity | previsão do tempo com histórico climático; RNN simples x LSTM em memória longa |
| 3 | transformer | sine_series, sequence_parity | self-attention x recorrência |
| 3 | autoencoder | digits, mnist | latente 2D, denoising, detecção de anomalias |
| 3 | gan | moons, digits | gerador x discriminador |
"""


if __name__ == "__main__":
    mcp.run("stdio")
