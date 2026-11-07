# Laboratório de Deep Learning — um exemplo de *harness*

Material de apoio para a disciplina **Redes Neurais e Deep Learning** (pós-graduação).
Uma plataforma web onde se criam **projetos**, cadastram **bases de dados**, criam e treinam os **modelos vistos nas aulas** —
e tudo isso pode ser feito também **conversando com um chatbot**, que é o **Claude Code** operando a plataforma via **MCP (stdio)**.

```
┌──────────────┐  POST /api/chat (SSE)  ┌──────────────┐  claude -p --mcp-config …  ┌──────────────┐
│  React (UI)  │ ─────────────────────▶ │   FastAPI    │ ─────────────────────────▶ │ Claude Code  │
│  :5173       │ ◀───── eventos ─────── │   :8000      │ ◀──── stream-json ──────── │  (headless)  │
└──────┬───────┘                        └──────▲───────┘                            └──────┬───────┘
       │ REST (cliques)                        │ REST                                      │ stdin/stdout
       └───────────────────────────────────────┤                                           ▼ JSON-RPC (MCP)
                                               └──────────────────────────────────── mcp_server/server.py
```

## O que é o "harness" aqui?

O modelo de linguagem, sozinho, só recebe texto e devolve texto. O **harness** é toda a estrutura ao redor que o transforma em um **agente**:

| Peça do harness | Onde está neste projeto |
|---|---|
| **Loop agêntico** (modelo → ferramenta → resultado → modelo …) | Claude Code em modo headless (`claude -p`) |
| **Ferramentas** que o agente pode usar | `backend/mcp_server/server.py` (22 ferramentas MCP) |
| **Transporte** das ferramentas | MCP via **stdio**: o Claude Code inicia o servidor como subprocesso |
| **Contexto / instruções** | `--append-system-prompt` em `backend/app/chat.py` (conteúdo das aulas + regras) |
| **Permissões / sandbox** | `--tools ""` (sem Bash/Edit/Read), `--allowedTools mcp__dl-lab`, `--permission-mode dontAsk`, `cwd` isolado |
| **Memória da conversa** | `--resume <session_id>` |
| **Observabilidade** | `--output-format stream-json` → SSE → aba **Harness** do chat (comando, system prompt, eventos) |

A aba **Harness** no painel de chat mostra tudo isso ao vivo: o comando exato executado, o system prompt, o `mcp_config.json`,
a lista de ferramentas que o modelo enxerga e o log de eventos de cada turno.

Ponto didático importante: **o servidor MCP é um cliente fino da mesma API REST que a interface usa.** Humano clicando e agente
chamando ferramentas operam exatamente o mesmo estado — por isso o que o Claude faz aparece imediatamente na tela.

## Conteúdo coberto (das aulas)

| Aula | Arquiteturas | Bases embutidas |
|---|---|---|
| 1 | **Perceptron** (regra `w += η(y−ŷ)x`, em NumPy), **Adaline** (Regra Delta, em NumPy), **MLP** | `and_gate`, `or_gate`, `xor_gate`, `blobs`, `moons`, `circles`, `iris`, `wine`, `breast_cancer`, `diabetes`, `sine_regression` |
| 2 | **CNN** (kernel, stride, padding same/valid, pooling max/avg, flatten, dense) | `digits` (8×8, offline), `mnist`, `fashion_mnist` (download na 1ª vez) |
| 3 | **RNN, LSTM, GRU**, **Transformer** (encoder, self-attention), **Autoencoder** (basic, denoising, sparse, variational), **GAN** | `daily_min_temperatures` (histórico climático real), `sine_series`, `airline_passengers`, `sequence_parity`, + imagens/tabulares acima |

### Bases próprias e do Kaggle

- **Upload pelo chat**: botão **📎 Anexar** (ou arraste o arquivo para o painel). Formatos: `.csv`, `.tsv`, `.txt`
  (separador `,` `;` TAB detectado, decimal com vírgula aceito), `.json`/`.jsonl` (lista de registros), `.xlsx`/`.xls`,
  `.parquet` e `.npz` (arrays `X` e `y` prontos — inclusive imagens `(N,C,H,W)` e séries `(N,T,F)`); até 200 MB.
  O arquivo é enviado ao backend e a mensagem leva o caminho e o resumo das colunas; o Claude escolhe a coluna alvo e
  a tarefa (ferramentas `inspect_file` e `import_file`). O mesmo upload existe no botão "Adicionar base" do projeto.
- **Kaggle**: botão **🔍 Kaggle** no chat busca bases públicas e manda o Claude importar a escolhida; ou peça direto
  (*"procure no Kaggle uma base de doenças cardíacas e importe"*). Ferramentas: `search_kaggle`, `kaggle_dataset_files`,
  `download_kaggle_dataset`. Bases públicas funcionam **sem login**; credenciais são opcionais
  (`~/.kaggle/kaggle.json`, `~/.kaggle/access_token`, `KAGGLE_USERNAME`+`KAGGLE_KEY` ou `KAGGLE_API_TOKEN`).
  Downloads ficam em `backend/data/downloads/kaggle/` (limite 200 MB por base; tabelas grandes podem ser amostradas com `max_rows`).

Visualizações por treino: curvas de perda/acurácia por época (ao vivo), matriz de confusão, **fronteira de decisão** (bases 2D),
curva ajustada (regressão 1D), série real × prevista, **espaço latente** e reconstruções (autoencoders), amostras geradas e
duelo D × G (GANs), pesos aprendidos (Perceptron/Adaline).

## Como rodar

Pré-requisitos: Python 3.11+, Node 18+, e o **Claude Code** instalado e logado (`claude` no PATH).

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

Ou manualmente:

```powershell
# backend
cd backend
python -m venv --system-site-packages .venv      # reaproveita o torch já instalado, se houver
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000

# frontend (outro terminal)
cd frontend
npm install
npm run dev        # http://localhost:5173
```

### Usando o mesmo MCP direto do terminal

O arquivo `.mcp.json` na raiz registra o servidor `dl-lab` para o Claude Code. Com o backend rodando, abra `claude` na raiz
do repositório e peça, por exemplo, *"liste meus projetos do laboratório"*. É o mesmo servidor, agora com outro cliente —
bom para mostrar que o MCP desacopla as ferramentas de quem as usa.

Para inspecionar o servidor isoladamente: `npx @modelcontextprotocol/inspector backend/.venv/Scripts/python.exe backend/mcp_server/server.py`.

## Roteiro sugerido para a aula

Há uma apresentação de apoio (19 slides, “O que é um harness”) em **http://localhost:5173/apresentacao** — também acessível pelo
botão **Apresentação** na barra superior. Navegue com ← →, `F` = tela cheia, `N` = notas do apresentador.
O arquivo é `frontend/apresentacao.html` (HTML autocontido, sem React).

1. **Aula 1** — *"Crie um projeto 'Aula 1', adicione and_gate e xor_gate e mostre que o perceptron resolve AND mas não XOR."*
   Depois: *"Agora treine um MLP com 4 neurônios tanh no XOR."* Compare as fronteiras de decisão.
2. **Aula 2** — *"Adicione digits e compare um MLP com uma CNN pequena."* Discuta parâmetros × acurácia.
3. **Aula 3** — *"Use o histórico climático (daily_min_temperatures) para prever a temperatura de amanhã com LSTM e GRU;
   compare com o modelo ingênuo."* (a base traz o RMSE de "amanhã = hoje" como referência);
   *"Compare RNN, LSTM e GRU em sequence_parity com length 20."* (gradiente que desaparece);
   *"Autoencoder com latente 2D em digits, depois denoising"*; *"GAN em moons"*.
4. **Harness** — abra a aba Harness e mostre o comando, o system prompt, as ferramentas e o log de eventos.
   Experimente remover uma ferramenta do servidor ou mudar o system prompt e veja o comportamento do agente mudar.

## Estrutura

```
backend/
  app/
    main.py           API REST (FastAPI) — usada pela UI e pelo MCP
    chat.py           ponte com o Claude Code (o harness): subprocesso + stream-json → SSE
    architectures.py  zoológico de modelos (NumPy + PyTorch)
    training.py       treino em threads, métricas, visualizações, predição/geração
    datasets.py       bases embutidas + importação de arquivos
    files.py          leitura de CSV/TSV/JSON/Excel/Parquet/NPZ
    kaggle.py         busca e download de bases do Kaggle
    db.py             SQLite (data/lab.sqlite3)
  mcp_server/
    server.py         servidor MCP stdio (ferramentas para o agente)
  data/               banco, bases (.npz), artefatos (.pt) — gerado em runtime
frontend/
  src/                React + TypeScript (Vite); tema derivado do material interativo da disciplina
```

## Notas

- Cada mensagem no chat dispara uma execução do Claude Code e consome sua cota/assinatura; o rodapé de cada resposta mostra
  turnos, duração e custo estimado.
- O agente roda com `cwd` em `backend/data/claude_workspace` e **sem** ferramentas de arquivo/shell: ele só consegue agir
  pelo MCP do laboratório.
- Treinos rodam em CPU; os padrões de épocas são pequenos para caber no tempo de aula.

## Contribuidores

- edsonlsjunior
- bela-bs (bela-bs@hotmail.com)
