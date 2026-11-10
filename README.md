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
| **Loop agêntico** (modelo → ferramenta → resultado → modelo …) | Codex, Claude Code ou GitHub Copilot CLI em modo headless |
| **Ferramentas** que o agente pode usar | `backend/mcp_server/server.py` (22 ferramentas MCP) |
| **Transporte** das ferramentas | MCP via **stdio**: o Claude Code inicia o servidor como subprocesso |
| **Contexto / instruções** | `--append-system-prompt` em `backend/app/chat.py` (conteúdo das aulas + regras) |
| **Permissões / sandbox** | Política Consultar / Confirmar / Operar, gateway MCP e sandbox read-only do processo |
| **Memória da conversa** | `--resume <session_id>` |
| **Observabilidade** | `--output-format stream-json` → SSE → aba **Harness** do chat (comando, system prompt, eventos) |

A aba **Harness** no painel de chat mostra tudo isso ao vivo: o comando exato executado, o system prompt, o `mcp_config.json`,
a lista de ferramentas que o modelo enxerga e o log de eventos de cada turno.

Ponto didático importante: **o servidor MCP é um cliente fino da mesma API REST que a interface usa.** Humano clicando e agente
chamando ferramentas operam exatamente o mesmo estado — por isso o que o Claude faz aparece imediatamente na tela.

### Modos de permissão

No seletor **Permissão do agente**, use **Consultar** para permitir apenas leituras, **Executar com confirmação** para pausar cada alteração aguardando o botão Aprovar, ou **Operar o laboratório** para permitir alterações de projetos, bases, modelos e treinos. Exclusões sempre pedem confirmação. A aba **Harness** registra a decisão, a ferramenta, os parâmetros, o passo e o motivo. O limite didático é de 40 passos ou 10 minutos por turno; o agente não pode usar shell, arquivos, navegador ou instalar plugins.

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

Pré-requisitos em Windows e macOS: **Python 3.11+ e Node.js 18+ (preferencialmente LTS)**.
Claude Code é opcional; a interface e os treinos funcionam sem um provedor de chat.
Mantenha o repositório numa pasta local, fora de iCloud/OneDrive/Dropbox, para evitar
que arquivos e dependências sejam removidos do disco durante a execução.

- **macOS:** dê dois cliques em `start.command`, ou execute `python3 start.py` no terminal.
- **Windows:** dê dois cliques em `start.cmd`, ou execute `py -3 start.py` no terminal.
- **PowerShell:** `./start.ps1` chama o mesmo inicializador.

O inicializador cria `backend/.venv` com o caminho correto para o sistema,
instala as dependências locais (usa uv se já estiver disponível; senão pip),
executa `npm ci`, compila o React e inicia a API servindo também a interface.
Abre **http://127.0.0.1:8000** depois de confirmar que o backend está pronto.
Na primeira execução, o download de PyTorch e demais pacotes pode demorar.
As credenciais da conta do chat permanecem no computador do backend.

Mantenha o terminal aberto; **Ctrl+C** encerra o servidor. Portas ocupadas geram
uma mensagem, sem encerrar processos de outros aplicativos.

```sh
python3 start.py --check           # diagnóstico sem instalar (Windows: py -3)
python3 start.py --setup-only      # preparar dependências e interface
python3 start.py --port 8002       # usar outra porta para API, interface e MCP
python3 start.py --no-browser      # não abrir navegador automaticamente
python3 start.py --skip-setup      # ambiente já preparado, sem recompilar
python3 start.py --reinstall       # reinstalar dependências e recompilar
```

### Escolher o assistente

Clique em **Detectar disponíveis**. Para **Codex — assinatura ChatGPT**, execute
`codex login` uma vez e entre com ChatGPT. A integração verifica o login local e
faz consultas ao laboratório, sem solicitar chave de API. Nesta versão, criação
e treinamento com Codex são feitos pelos botões da interface. Uso sujeito aos
limites da conta. O CLI deve estar no PATH; `DL_LAB_CODEX_BIN` aceita seu caminho
absoluto. No macOS, também é detectado o CLI incluído no aplicativo ChatGPT.

Para **GitHub Copilot CLI**, instale o `copilot` (por exemplo,
`winget install GitHub.Copilot` ou `npm install -g @github/copilot`) e execute
`copilot login` uma vez no mesmo usuário que roda o backend. Também funcionam
`COPILOT_GITHUB_TOKEN`/`GH_TOKEN` no ambiente do backend. `DL_LAB_COPILOT_BIN`
aceita o caminho absoluto do CLI. O Copilot roda em modo headless
(`copilot -p --output-format json`) num diretório temporário, sem instruções
personalizadas, sem MCPs embutidos e só com as ferramentas `dl-lab`. A política
Consultar / Confirmar / Operar vale como no Claude Code, e a conversa continua
com `--resume`. O login é verificado na primeira mensagem. Cada mensagem consome
*premium requests* da sua assinatura Copilot.

Claude Code exige seu próprio CLI instalado e logado. OpenAI API/Gemini API usam
`OPENAI_API_KEY`/`GEMINI_API_KEY` no ambiente do backend. Para servidor local,
configure `LOCAL_LLM_BASE_URL` (padrão `http://127.0.0.1:11434/v1`) e informe um
modelo com suporte a ferramentas. A detecção consulta os catálogos sem gerar
texto; catálogo acessível não garante suporte a ferramentas em todos os modelos.

### Chave própria em uma página pública (OpenAI)

No provedor **OpenAI API**, cada pessoa pode colar a própria chave no campo da
interface. A chave é enviada ao backend e fica apenas na memória do processo,
associada a um cookie anônimo `HttpOnly`, por até 30 minutos (ou até clicar em
**Remover**). Ela não é escrita no banco, em arquivo, em `localStorage`, nos
logs ou no repositório; reiniciar o servidor também a elimina. A resposta da API
só informa se existe uma chave conectada — nunca devolve seu valor.

Para publicar, defina `DL_LAB_PUBLIC_MODE=1` e `DL_LAB_COOKIE_SECURE=1`. Nesse
modo, a aplicação não oferece CLIs nem usa `OPENAI_API_KEY` do operador do
servidor: somente a chave temporária de quem está usando a página. Sirva sempre
por HTTPS e não inclua nenhuma chave em variáveis públicas do frontend.

Uma assinatura ChatGPT/Codex não cria automaticamente crédito de API: a pessoa
deve gerar uma chave de projeto na Plataforma OpenAI e controlar limites e
gastos na própria conta.
Não versione credenciais. Cada instalação usa o login do seu próprio usuário.

### Desenvolvimento

O Vite continua disponível em `frontend` com `npm run dev`, na porta 5173.
Nesse modo, inicie separadamente o backend na porta 8000, com
`backend/.venv/bin/python` no macOS ou `backend/.venv/Scripts/python.exe` no Windows,
executando `-m uvicorn app.main:app --app-dir backend --port 8000` a partir da raiz.

### Usando o mesmo MCP direto do terminal

`start.py` gera `.mcp.local.json` com os caminhos absolutos corretos para o
computador e a porta selecionada. Esse arquivo é ignorado pelo Git e deve ser
regenerado após mover a pasta. Para usar com Claude Code, execute
`claude --mcp-config .mcp.local.json` com o laboratório rodando. O exemplo
`.mcp.example.json` substitui a configuração antiga que continha caminho Windows fixo.

O chat gera suas configurações MCP em tempo de execução usando o próprio Python
do backend, tanto no Windows quanto no macOS.

### Validação

`python -m unittest discover -s tests -v` testa o inicializador, incluindo a
seleção de caminhos Windows/macOS. Em `backend`, execute com o Python de `.venv`
`-m unittest discover -s tests -v` para os testes de provedores. A execução real
foi validada no macOS; o teste do sistema Windows requer uma máquina Windows.

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
