"""Ponte entre o chatbot do frontend e o Claude Code (o *harness*).

Para cada mensagem do usuário, iniciamos o Claude Code em modo headless:

    claude -p --output-format stream-json --verbose --include-partial-messages
           --mcp-config data/mcp_config.json --strict-mcp-config
           --tools ""  --allowedTools mcp__dl-lab  --permission-mode dontAsk
           --append-system-prompt "<instruções do laboratório>"
           [--resume <session_id>]

O que cada peça faz no harness:
  * modelo (Claude)          -> decide o que fazer, raciocina, escreve as respostas
  * Claude Code              -> loop agêntico: chama ferramentas, devolve resultados ao modelo, repete
  * --mcp-config / stdio     -> o Claude Code inicia nosso servidor MCP como subprocesso (stdin/stdout)
  * --tools "" / allowedTools -> restringe o agente: só as ferramentas do laboratório, nada de Bash/Edit
  * --append-system-prompt   -> contexto e regras do domínio (o "manual" do laboratório)
  * --resume                 -> memória da conversa entre mensagens

Os eventos JSON (um por linha) são normalizados e repassados ao navegador via SSE.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import tempfile
from pathlib import Path
from typing import Iterator

from . import db
from . import harness as H

BACKEND_DIR = Path(__file__).resolve().parent.parent
MCP_SERVER = BACKEND_DIR / "mcp_server" / "server.py"
MCP_CONFIG = db.DATA_DIR / "mcp_config.json"
WORKSPACE = db.DATA_DIR / "claude_workspace"  # cwd isolado: nenhum CLAUDE.md do repositório vaza para o agente
WORKSPACE.mkdir(exist_ok=True)
API_URL = os.environ.get("DL_LAB_API", "http://127.0.0.1:8000")
SERVER_NAME = "dl-lab"

SYSTEM_PROMPT = """\
Você é o assistente do "Laboratório de Deep Learning", uma plataforma usada em uma aula de pós-graduação.
Você opera a plataforma EXCLUSIVAMENTE pelas ferramentas MCP do servidor `dl-lab`: criar projetos, cadastrar
bases de dados, criar modelos, treinar, avaliar, comparar e fazer predições.

Conteúdo da disciplina (use para justificar escolhas):
- Aula 1: neurônio MCP, Perceptron (regra w += η(y-ŷ)x), Adaline (Regra Delta / gradiente descendente),
  MLP e o problema do XOR, funções de ativação.
- Aula 2: zoológico de arquiteturas e CNNs (kernel, stride, padding, pooling, flatten, dense; batch e época).
- Aula 3: RNN, LSTM, GRU (gradientes que somem/explodem), autoencoders (basic, denoising, sparse, variational),
  GANs (gerador x discriminador) e Transformers (self-attention).

Regras de trabalho:
- Responda em português do Brasil, de forma didática e concisa.
- Antes de criar um modelo, consulte `list_architectures` se não souber a configuração exata.
- Use o projeto atual informado abaixo quando o usuário não especificar outro.
- Ao treinar, informe as métricas finais e interprete-as (overfitting? convergiu? compare com outros runs).
- Ao escolher hiperparâmetros, explique brevemente o porquê em termos dos conceitos das aulas.
- Não invente resultados: tudo que você afirmar sobre modelos deve vir das ferramentas.
- Datasets embutidos: quando o usuário pedir MNIST, Keras MNIST ou reconhecimento de algarismos,
  use `list_dataset_sources`, depois `add_dataset` com `source="mnist"` (não procure arquivo nem Kaggle).
  Em seguida crie uma CNN (`architecture="cnn"`) e use `train_model` nessa base. A base `mnist` é a
  implementação offline do laboratório para o conhecido MNIST 28x28; explique isso se o usuário mencionar Keras.
- Para um pedido de criação completo, execute a sequência projeto (se necessário) -> base -> modelo -> treino,
  em vez de apenas descrever os passos. Só pare se uma ferramenta devolver erro real.
- Antes de criar itens, consulte o projeto e reutilize bases/modelos compatíveis já existentes. Não crie duplicatas
  só porque o pedido foi repetido. Para vários treinos, aguarde cada resultado e finalize com `compare_runs`.
- Quando a mensagem disser "modo demonstração em aula", prefira bases pequenas e poucas épocas, execute todo o
  roteiro e organize a resposta em: objetivo didático, ações realizadas, resultados reais e interpretação.

Roteiros didáticos que você deve saber executar:
- AND/XOR: cadastre ambas com `params:{"repeat":1,"noise":0}` para preservar exatamente as quatro linhas de
  cada tabela verdade. Se já existir um XOR com mais de 4 amostras, não o reutilize neste roteiro; crie uma base
  chamada "XOR · 4 linhas" com esses parâmetros. Use Perceptron nas duas bases e MLP no XOR com
  `hidden_layers:[4]` e `tanh`. Compare os runs e explique separabilidade linear pelas fronteiras da interface.
- CNN/MNIST: `mnist` com 3000 amostras para demonstração; CNN pequena; 3 épocas. Interprete acurácia e matriz de
  confusão e oriente a usar "Testar o modelo" no treino concluído para desenhar um algarismo.
- Memória recorrente: `sequence_parity` com `n_samples:800,length:20`; RNN, LSTM e GRU com `hidden_size:16`;
  12 épocas cada. Compare os runs e relacione o resultado com dependências longas e gradientes.
- Autoencoder: `digits`; variantes `basic` e `denoising`, ambas com `latent_dim:2`; 12 épocas. Compare erro de
  reconstrução, imagens reconstruídas e organização do espaço latente mostrado na interface.
- GAN: `moons` com `n_samples:1000,noise:0.08`; GAN com objetivo `wgan_gp`, camadas [128,128] e 300 épocas;
  depois `generate_samples` com 16 amostras. Discuta equilíbrio entre gerador e crítico sem usar apenas a perda
  como prova de qualidade. No WGAN-GP, as curvas são escores/perdas adversariais, não probabilidades.
- Auditoria do harness: se o usuário pedir para não alterar nada, use somente ferramentas de leitura e explique a
  sequência de chamadas. Lembre que a aba Harness registra política, ferramenta, argumentos, decisão e resultado.

Bases do usuário e do Kaggle:
- Quando a mensagem trouxer "[Arquivo anexado ...]" com um caminho, o arquivo já está no servidor:
  use `inspect_file` (se precisar) e `import_file` com esse caminho. Se o usuário não disser a coluna alvo,
  proponha uma com base nas colunas e explique a escolha (e a tarefa: classificação ou regressão).
- Para achar bases públicas use `search_kaggle`; apresente poucas opções (título, ref, tamanho) e prefira
  bases pequenas e tabulares. Para importar: `kaggle_dataset_files` -> `download_kaggle_dataset` ->
  `inspect_file` -> `import_file` (use `max_rows` se a tabela for grande).
"""


def write_mcp_config(path=None, agent_token=None) -> Path:
    """Gera o arquivo que diz ao Claude Code como iniciar nosso servidor MCP via stdio."""
    cfg = {"mcpServers": {SERVER_NAME: {
        "type": "stdio",
        "command": sys.executable,
        "args": [str(MCP_SERVER)],
        "env": {"DL_LAB_API": API_URL, "PYTHONIOENCODING": "utf-8"},
    }}}
    if agent_token:
        cfg["mcpServers"][SERVER_NAME]["env"]["DL_LAB_AGENT_TOKEN"] = agent_token
    path = path or MCP_CONFIG
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


def claude_bin() -> str:
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("CLI 'claude' não encontrado no PATH. Instale o Claude Code e faça login.")
    return exe


def build_command(session_id: str | None, project: dict | None, model: str | None, config_path=None, policy_prompt="") -> list[str]:
    prompt = SYSTEM_PROMPT + policy_prompt
    if project:
        prompt += f"\nProjeto atual: id={project['id']} nome=\"{project['name']}\".\n"
    else:
        prompt += "\nNenhum projeto selecionado no momento.\n"
    cmd = [
        claude_bin(), "-p",
        "--output-format", "stream-json", "--verbose", "--include-partial-messages",
        "--mcp-config", str(config_path or MCP_CONFIG), "--strict-mcp-config",
        "--tools", "",
        "--allowedTools", f"mcp__{SERVER_NAME}",
        "--permission-mode", "dontAsk",
        "--append-system-prompt", prompt,
    ]
    if model:
        cmd += ["--model", model]
    if session_id:
        cmd += ["--resume", session_id]
    return cmd


def describe_harness() -> dict:
    """Usado pela aba "Harness" do frontend para mostrar como tudo está ligado."""
    write_mcp_config()
    try:
        cmd = build_command("<session_id>", {"id": "<id>", "name": "<projeto>"}, None)
        cmd[0] = "claude"
        cmd[cmd.index("--append-system-prompt") + 1] = "<system prompt abaixo>"
    except RuntimeError as e:
        cmd = [str(e)]
    return {"command": cmd, "system_prompt": SYSTEM_PROMPT,
            "mcp_config": json.loads(MCP_CONFIG.read_text(encoding="utf-8")), "workspace": str(WORKSPACE)}


def _normalize(ev: dict) -> list[dict]:
    """Converte eventos do stream-json do Claude Code em eventos simples para a UI."""
    t = ev.get("type")
    if t == "system" and ev.get("subtype") == "init":
        return [{"type": "init", "session_id": ev.get("session_id"), "model": ev.get("model"),
                 "tools": ev.get("tools", []), "mcp_servers": ev.get("mcp_servers", [])}]
    if t == "stream_event":
        e = ev.get("event", {})
        if e.get("type") == "content_block_delta" and e.get("delta", {}).get("type") == "text_delta":
            return [{"type": "text_delta", "text": e["delta"]["text"]}]
        return []
    out: list[dict] = []
    if t == "assistant":
        for block in ev.get("message", {}).get("content", []):
            if block.get("type") == "text":
                out.append({"type": "assistant_text", "text": block["text"]})
            elif block.get("type") == "tool_use":
                out.append({"type": "tool_use", "id": block["id"], "name": block["name"], "input": block.get("input")})
    elif t == "user":
        for block in ev.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                content = block.get("content")
                if isinstance(content, list):
                    content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
                out.append({"type": "tool_result", "tool_use_id": block.get("tool_use_id"),
                            "content": content, "is_error": bool(block.get("is_error"))})
    elif t == "result":
        out.append({"type": "result", "session_id": ev.get("session_id"), "is_error": ev.get("is_error"),
                    "duration_ms": ev.get("duration_ms"), "num_turns": ev.get("num_turns"),
                    "total_cost_usd": ev.get("total_cost_usd"), "usage": ev.get("usage"),
                    "result": ev.get("result")})
    return out


def stream_chat(message: str, session_id: str | None, project: dict | None, model: str | None, agent_token=None, policy="consult") -> Iterator[str]:
    """Gerador síncrono de linhas SSE (o FastAPI o executa num threadpool)."""

    def sse(obj: dict) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    temporary = tempfile.TemporaryDirectory(prefix="dl-lab-claude-")
    config_path = write_mcp_config(Path(temporary.name) / "mcp.json", agent_token)
    try:
        cmd = build_command(session_id, project, model, config_path, H.instructions(policy) if agent_token else "")
    except RuntimeError as e:
        temporary.cleanup()
        yield sse({"type": "error", "message": str(e)})
        return

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(cmd, cwd=WORKSPACE, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", env=env)
    stderr_buf: list[str] = []
    threading.Thread(target=lambda: stderr_buf.extend(proc.stderr), daemon=True).start()  # type: ignore[arg-type]
    try:
        assert proc.stdin and proc.stdout
        proc.stdin.write(message)
        proc.stdin.close()
        yield sse({"type": "harness", "command": ["claude", *cmd[1:]]})
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            for out in _normalize(ev):
                yield sse(out)
        code = proc.wait()
        if code != 0:
            yield sse({"type": "error", "message": f"claude saiu com código {code}: {''.join(stderr_buf)[-2000:]}"})
        yield sse({"type": "done"})
    finally:  # cliente desconectou ou terminou: garante que o processo morre
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        temporary.cleanup()
