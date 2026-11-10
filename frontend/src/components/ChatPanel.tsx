import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { DATA_ACCEPT, api, streamChat } from "../api";
import type { ChatEvent, FileInfo, KaggleDataset, Project } from "../types";
import PolicyPanel, { Approvals, MODES, DESCRIPTIONS } from "./PolicyPanel";

type ExecutionStatus = "running" | "completed" | "error" | "cancelled";
type Part = { kind: "text"; text: string } | { kind: "tool"; id: string; name: string; input: unknown; result?: string; isError?: boolean; status: ExecutionStatus };
interface Msg {
  role: "user" | "assistant";
  parts: Part[];
  footer?: string;
  status?: ExecutionStatus;
  attachment?: { filename: string; summary: string };
}

/** Resumo do arquivo anexado que vai junto da mensagem: o Claude já sabe caminho e colunas. */
function describeAttachment(f: FileInfo) {
  const cols = f.columns
    ?.slice(0, 40)
    .map((c) => `${c.name} (${c.kind === "numeric" ? "num" : "cat"}, ${c.n_unique} únicos${c.n_missing ? `, ${c.n_missing} faltantes` : ""})`)
    .join("; ");
  return [
    `[Arquivo anexado: ${f.filename}`,
    `caminho no servidor: ${f.path}`,
    `formato ${f.format}, ${f.size_kb} KB, ${attachmentLabel(f)}`,
    cols ? `colunas: ${cols}${(f.columns?.length ?? 0) > 40 ? "; …" : ""}]` : "]",
  ].join("\n");
}

function attachmentLabel(f: FileInfo) {
  return f.arrays
    ? Object.entries(f.arrays)
        .map(([k, a]) => `${k} ${a.shape.join("×")}`)
        .join(" · ")
    : `${f.n_rows} linhas × ${f.n_columns} colunas`;
}

const DEMOS = [
  {title: "1 · AND, XOR e MLP", prompt: "Modo demonstração em aula: crie o projeto 'Demo 1 · Linearidade'. Adicione and_gate e xor_gate com repeat 1 e noise 0, mantendo exatamente as 4 linhas de cada tabela verdade. Treine um Perceptron nas duas bases e depois um MLP com uma camada oculta de 4 neurônios e ativação tanh no XOR. Compare os treinos e explique visualmente por que o Perceptron falha no XOR."},
  {title: "2 · CNN e MNIST", prompt: "Modo demonstração em aula: crie o projeto 'Demo 2 · CNN e MNIST'. Adicione MNIST com 3000 amostras, crie uma CNN pequena, treine por 3 épocas e interprete acurácia e matriz de confusão. Ao final, explique como testar o modelo desenhando um algarismo na tela."},
  {title: "3 · RNN, LSTM e GRU", prompt: "Modo demonstração em aula: crie o projeto 'Demo 3 · Memória recorrente'. Adicione sequence_parity com 800 amostras e sequências de tamanho 20. Crie RNN, LSTM e GRU com hidden_size 16, treine cada uma por 12 épocas, compare os resultados e relacione as diferenças com memória e desaparecimento do gradiente."},
  {title: "3B · Próxima palavra", prompt: "Modo demonstração em aula: crie o projeto 'Demo 3B · Próxima palavra'. Adicione a base next_word com window 3. Crie uma RNN e uma LSTM com hidden_size 24, treine ambas por 20 épocas, compare a acurácia e explique que cada palavra foi convertida em vetor one-hot. Ao final, mostre como testar escrevendo 'o modelo prevê' para obter uma sugestão de próxima palavra."},
  {title: "4 · Autoencoder", prompt: "Modo demonstração em aula: crie o projeto 'Demo 4 · Autoencoder'. Adicione digits. Treine um autoencoder básico com espaço latente 2D e depois uma versão denoising, ambos por 12 épocas. Compare o erro de reconstrução, as reconstruções e o espaço latente."},
  {title: "5 · GAN", prompt: "Modo demonstração em aula: crie o projeto 'Demo 5 · GAN'. Adicione moons com 1000 amostras e noise 0.08. Crie uma GAN com objetivo wgan_gp, generator_layers e discriminator_layers [128,128], critic_steps 3. Treine por 300 épocas, gere 16 amostras e compare visualmente os pontos reais e gerados. Explique que o crítico do WGAN-GP produz escores, não probabilidades, e que a qualidade não deve ser julgada só pela perda."},
  {title: "6 · Auditoria do Harness", prompt: "Demonstração do Harness: sem alterar nada, liste os projetos, as bases disponíveis e as arquiteturas. Explique quais ferramentas MCP você chamou, em que ordem, o que veio do modelo e o que foi executado pelo harness. Peça que eu abra a aba Harness para conferir o registro."},
];

export default function ChatPanel({ project, onToolResult }: { project: Project | null; onToolResult: () => void }) {
  const [view, setView] = useState<"chat" | "harness">("chat");
  const [policy, setPolicy] = useState(() => {
    // Operar é o caminho didático principal; a política pode ser alterada na aba Harness.
    return "operate";
  });
  useEffect(() => { localStorage.setItem("dl.policy", policy); }, [policy]);
  // A conversa é deliberadamente efêmera: atualizar a página começa um chat limpo.
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [model, setModel] = useState(() => localStorage.getItem("dl.model") ?? "");
  const [provider, setProvider] = useState(() => localStorage.getItem("dl.provider") ?? "");
  const [providers, setProviders] = useState<{id: string; name: string; configured: boolean; default_model: string}[]>([]);
  const [detecting, setDetecting] = useState(false);
  const [detected, setDetected] = useState<Awaited<ReturnType<typeof api.detectProviders>>>([]);
  const [detectError, setDetectError] = useState("");
  async function detectProviders() {
    setDetecting(true); setDetectError("");
    try {
      const found = await api.detectProviders();
      setDetected(found); setProviders(found);
    } catch { setDetectError("Não foi possível detectar. Confira se a API está online."); }
    finally { setDetecting(false); }
  }
  useEffect(() => {
    // Remove conversas gravadas por versões anteriores. Projetos e treinos ficam no backend.
    localStorage.removeItem("dl.msgs");
    localStorage.removeItem("dl.session");
    api.providers().then(setProviders).catch(() => {});
  }, []);
  useEffect(() => { localStorage.setItem("dl.provider", provider); }, [provider]);
  const [events, setEvents] = useState<ChatEvent[]>([]);
  const [init, setInit] = useState<Extract<ChatEvent, { type: "init" }> | null>(null);
  const [attachment, setAttachment] = useState<FileInfo | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadErr, setUploadErr] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [kaggleOpen, setKaggleOpen] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const activeProvider = providers.find(p => p.id === provider);
  const providerReady = !!activeProvider?.configured && (provider !== "local" || !!model.trim());
  // o stream vive por vários renders: guardamos sempre o callback mais recente
  const onToolResultRef = useRef(onToolResult);
  onToolResultRef.current = onToolResult;

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight });
  }, [msgs]);
  useEffect(() => {
    localStorage.setItem("dl.model", model);
  }, [model]);

  /** Aplica um evento do stream ao último balão do assistente. */
  function apply(ev: ChatEvent) {
    if (ev.type !== "text_delta") setEvents((e) => [...e.slice(-300), ev]);
    setMsgs((prev) => {
      if (!prev.length) return prev;
      const out = [...prev];
      const last = { ...out[out.length - 1], parts: [...out[out.length - 1].parts] };
      out[out.length - 1] = last;
      const tail = last.parts[last.parts.length - 1];
      switch (ev.type) {
        case "text_delta":
          if (tail?.kind === "text") last.parts[last.parts.length - 1] = { ...tail, text: tail.text + ev.text };
          else last.parts.push({ kind: "text", text: ev.text });
          break;
        case "tool_use":
          last.parts.push({ kind: "tool", id: ev.id, name: ev.name, input: ev.input, status: "running" });
          break;
        case "tool_result":
          last.parts = last.parts.map((p) => (p.kind === "tool" && p.id === ev.tool_use_id ? { ...p, result: ev.content, isError: ev.is_error, status: ev.is_error ? "error" : "completed" } : p));
          break;
        case "result":
          last.status = ev.is_error ? "error" : "completed";
          last.parts = last.parts.map(p => p.kind === "tool" && p.status === "running" ? {...p, status: ev.is_error ? "error" : "completed"} : p);
          last.footer = `${ev.is_error ? "Falhou" : "Concluído"} · ${ev.num_turns} turnos · ${(ev.duration_ms / 1000).toFixed(1)}s${ev.total_cost_usd ? ` · US$ ${ev.total_cost_usd.toFixed(4)}` : ""}`;
          break;
        case "error":
          last.status = "error";
          last.parts = last.parts.map(p => p.kind === "tool" && p.status === "running" ? {...p, status: "error"} : p);
          last.parts.push({ kind: "text", text: `**Erro:** ${ev.message}` });
          last.footer = "Falhou";
          break;
        case "done":
          if (last.status === "running") last.status = "completed";
          last.parts = last.parts.map(p => p.kind === "tool" && p.status === "running" ? {...p, status: "completed"} : p);
          last.footer ||= "Concluído";
          break;
      }
      return out;
    });
    if (ev.type === "init") {
      setInit(ev);
      setSessionId(ev.session_id);
    }
    if (ev.type === "result" && ev.session_id) setSessionId(ev.session_id);
    if (ev.type === "tool_result") onToolResultRef.current(); // o Claude mudou algo -> recarrega a UI
  }

  function finishLast(status: ExecutionStatus) {
    setMsgs(prev => {
      if (!prev.length) return prev;
      const out = [...prev];
      const last = {...out[out.length - 1], parts: [...out[out.length - 1].parts]};
      if (last.role !== "assistant") return prev;
      last.status = status;
      last.parts = last.parts.map(p => p.kind === "tool" && p.status === "running" ? {...p, status} : p);
      last.footer ||= status === "cancelled" ? "Interrompido" : status === "error" ? "Falhou" : "Concluído";
      out[out.length - 1] = last;
      return out;
    });
  }

  async function attach(file: File) {
    setUploadErr("");
    setUploading(true);
    try {
      setAttachment(await api.stageUpload(file));
    } catch (e: any) {
      setUploadErr(e.message);
    } finally {
      setUploading(false);
    }
  }

  async function send(text?: string) {
    const att = text === undefined ? attachment : null; // sugestões/Kaggle não levam o anexo
    let typed = (text ?? input).trim();
    if (!typed && att) typed = "Importe este arquivo como base de dados no projeto atual e descreva-o.";
    if (!typed || busy || uploading || !providerReady) return;
    const message = att ? `${typed}\n\n${describeAttachment(att)}` : typed;
    setInput("");
    if (att) setAttachment(null);
    setKaggleOpen(false);
    setBusy(true);
    const userMsg: Msg = { role: "user", parts: [{ kind: "text", text: typed }] };
    if (att) userMsg.attachment = { filename: att.filename, summary: attachmentLabel(att) };
    setMsgs((m) => [...m, userMsg, { role: "assistant", parts: [], status: "running" }]);
    abort.current = new AbortController();
    try {
      const history = msgs.slice(-30).map(m => ({role: m.role, content: m.parts.filter(p => p.kind === "text").map(p => p.kind === "text" ? p.text : "").join("\n")}));
      await streamChat({ message, session_id: provider === "claude-code" ? sessionId : null, project_id: project?.id ?? null, model: model || null, provider, history, policy }, apply, abort.current.signal);
      finishLast("completed");
    } catch (e: any) {
      if (e.name === "AbortError") finishLast("cancelled");
      else { apply({ type: "error", message: e.message }); finishLast("error"); }
    } finally {
      setBusy(false);
      setEvents(es => es.map(e => e.type === "permission" && e.decision === "awaiting" ? {...e, decision: "cancelled", reason: "Conversa encerrada sem executar a ação pendente."} : e));
      onToolResultRef.current();
    }
  }

  function reset() {
    abort.current?.abort();
    setMsgs([]);
    setEvents([]);
    setSessionId(null);
  }

  return (
    <aside
      className={`chat ${dragOver ? "drag" : ""}`}
      onDragOver={(e) => {
        if (view !== "chat" || !e.dataTransfer.types.includes("Files")) return;
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragOver(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const f = e.dataTransfer.files?.[0];
        if (f && view === "chat") attach(f);
      }}
    >
      {dragOver && <div className="drop-hint">Solte o arquivo para anexar ao chat</div>}
      <div className="chat-head">
        <div className="spread">
          <div>
            <div className="eyebrow">ASSISTENTE COM FERRAMENTAS MCP</div>
            <h3>Agente</h3>
          </div>
          <button className="btn dark-ghost" onClick={reset} title="Começa uma nova conversa">
            Nova conversa
          </button>
        </div>
        <label className="provider-main">Provedor
          <select id="llm-provider" aria-label="Provedor de IA" value={provider} disabled={busy} onChange={e => {
            const next = e.target.value;
            reset(); setProvider(next); setModel(providers.find(p => p.id === next)?.default_model ?? "");
          }}>
            <option value="">Escolha o provedor…</option>
            {providers.map(p => <option key={p.id} value={p.id}>{p.name}{p.configured ? "" : " (configurar)"}</option>)}
          </select>
        </label>
        <div className="chat-tabs">
          <button className={view === "chat" ? "active" : ""} onClick={() => setView("chat")}>
            Chat
          </button>
          <button className={view === "harness" ? "active" : ""} onClick={() => setView("harness")}>
            Harness
          </button>
        </div>
      </div>

      {view === "chat" ? (
        <>
          <div className="chat-scroll" ref={scroller}>
            {!msgs.length && (
              <div>
                <p className="small" style={{ color: "#91a5bb", lineHeight: 1.6 }}>
                  Peça em linguagem natural. O modelo escolhido decide quais ferramentas MCP chamar
                  {project ? (
                    <>
                      {" "}— ele sabe que o projeto atual é <b style={{ color: "#fff" }}>{project.name}</b>
                    </>
                  ) : null}
                  .
                </p>
                <details className="demo-suggestions">
                  <summary>Exemplos prontos para aula</summary>
                  <div className="suggest">
                    {DEMOS.map((demo) => <button key={demo.title} onClick={() => setInput(demo.prompt)}>{demo.title}</button>)}
                  </div>
                  <p className="small">Clique em um exemplo para colocá-lo no campo de mensagem. Você pode editar antes de enviar.</p>
                </details>
              </div>
            )}
            {msgs.map((m, i) => (
              <div key={i} className={`msg ${m.role}`}>
                {m.attachment && (
                  <div className="att-chip in-msg">
                    <span>📎</span>
                    <b>{m.attachment.filename}</b>
                    <span className="att-meta">{m.attachment.summary}</span>
                  </div>
                )}
                {m.parts.map((p, j) =>
                  p.kind === "text" ? (
                    m.role === "user" ? (
                      <div key={j}>{p.text}</div>
                    ) : (
                      <div key={j} className="md">
                        <ReactMarkdown>{p.text}</ReactMarkdown>
                      </div>
                    )
                  ) : (
                    <ToolCard key={j} part={p} />
                  ),
                )}
                {m.role === "assistant" && m.status === "running" && i === msgs.length - 1 && (
                  <div className="row small" style={{ color: "#91a5bb" }}>
                    <span className="spinner" /> trabalhando…
                  </div>
                )}
                {m.footer && <div className="result-line">{m.footer}</div>}
              </div>
            ))}
          </div>
          <div className="chat-input">
            {(attachment || uploading || uploadErr) && (
              <div className={`att-chip ${uploadErr ? "err" : ""}`}>
                {uploading ? (
                  <>
                    <span className="spinner" /> enviando arquivo…
                  </>
                ) : uploadErr ? (
                  <>
                    <span>✕</span> {uploadErr}
                  </>
                ) : (
                  attachment && (
                    <>
                      <span>📎</span>
                      <b>{attachment.filename}</b>
                      <span className="att-meta">{attachmentLabel(attachment)}</span>
                    </>
                  )
                )}
                {!uploading && (
                  <button
                    className="att-x"
                    title="Remover"
                    onClick={() => {
                      setAttachment(null);
                      setUploadErr("");
                    }}
                  >
                    ×
                  </button>
                )}
              </div>
            )}
            <input
              ref={fileInput}
              type="file"
              accept={DATA_ACCEPT}
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) attach(f);
                e.target.value = "";
              }}
            />
            <textarea
              placeholder={
                attachment
                  ? "Diga o que fazer com o arquivo (ex.: alvo = Survived) ou só envie"
                  : project
                    ? `Peça algo sobre "${project.name}"…`
                    : "Ex.: crie um projeto e treine um perceptron no AND"
              }
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
            />
            <div className="spread" style={{ marginTop: 8 }}>
              <details className="chat-options">
                <summary>Mais opções</summary>
                <div className="row option-actions">
                  {provider !== "claude-code" ? <input aria-label="Modelo de IA" placeholder={provider === "codex" ? "Modelo padrão do Codex" : "Nome do modelo"} value={model} disabled={busy} onChange={e => { setModel(e.target.value); setSessionId(null); }} /> : <select value={model} disabled={busy} onChange={(e) => { setModel(e.target.value); setSessionId(null); }} title="Modelo usado pelo Claude Code">
                    <option value="">modelo padrão</option><option value="opus">opus</option><option value="sonnet">sonnet</option><option value="haiku">haiku</option>
                  </select>}
                  <button className="btn dark-ghost sm" disabled={busy || detecting} onClick={detectProviders}>{detecting ? "Detectando…" : "Verificar conexão"}</button>
                  <button className="btn dark-ghost sm" onClick={() => fileInput.current?.click()} disabled={busy || uploading}>📎 Arquivo</button>
                  <button className={`btn dark-ghost sm ${kaggleOpen ? "on" : ""}`} onClick={() => setKaggleOpen((o) => !o)}>🔍 Kaggle</button>
                </div>
                {detectError && <p className="small" role="status">{detectError}</p>}
                {!!detected.length && <p className="small">{detected.find(p => p.id === provider)?.status ?? "Conexão verificada."}</p>}
              </details>
              {busy ? (
                <button className="btn dark-ghost" onClick={() => abort.current?.abort()}>
                  ■ Parar
                </button>
              ) : (
                <button className="btn primary" onClick={() => send()} disabled={(!input.trim() && !attachment) || uploading || !providerReady}>
                  Enviar ↵
                </button>
              )}
            </div>
            {kaggleOpen && <KagglePanel onPick={(msg) => send(msg)} close={() => setKaggleOpen(false)} disabled={busy} />}
            {!providerReady && <p className="small provider-warning" role="status">{!provider ? "Escolha um provedor no topo." : `${activeProvider?.name ?? "O provedor"} precisa ser configurado ou verificado.`}</p>}
          </div>
        </>
      ) : (
        <>
          <div className="harness-controls">
            <label className="policy-control">Permissão do agente
              <select aria-label="Permissão do agente" value={policy} disabled={busy} onChange={e => {setPolicy(e.target.value); setSessionId(null);}}>
                {Object.entries(MODES).map(([id, name]) => <option key={id} value={id}>{name}</option>)}
              </select>
            </label>
            <p className="small">{DESCRIPTIONS[policy]}</p>
          </div>
          <Approvals events={events} busy={busy} />
          <PolicyPanel events={events} policy={policy} />
        </>
      )}
    </aside>
  );
}

function KagglePanel({ onPick, close, disabled }: { onPick: (message: string) => void; close: () => void; disabled: boolean }) {
  const [q, setQ] = useState("");
  const [fileType, setFileType] = useState("csv");
  const [results, setResults] = useState<KaggleDataset[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  async function search() {
    if (!q.trim()) return;
    setErr("");
    setLoading(true);
    try {
      setResults(await api.kaggleSearch(q.trim(), fileType));
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="kaggle">
      <div className="spread">
        <div className="eyebrow" style={{ color: "var(--cyan)" }}>
          Buscar no Kaggle
        </div>
        <button className="att-x" onClick={close} title="Fechar">
          ×
        </button>
      </div>
      <div className="row" style={{ gap: 6, marginTop: 6 }}>
        <input
          autoFocus
          value={q}
          placeholder="ex.: titanic, heart disease, weather…"
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && search()}
        />
        <select value={fileType} onChange={(e) => setFileType(e.target.value)} title="Tipo de arquivo">
          <option value="csv">csv</option>
          <option value="json">json</option>
          <option value="parquet">parquet</option>
          <option value="">todos</option>
        </select>
        <button className="btn primary sm" onClick={search} disabled={loading || !q.trim()}>
          {loading ? <span className="spinner" /> : "Buscar"}
        </button>
      </div>
      {err && <div className="kaggle-err">{err}</div>}
      {results && (
        <div className="kaggle-list">
          {!results.length && <div className="small">Nenhuma base encontrada.</div>}
          {results.map((r) => (
            <div key={r.ref} className="kaggle-item">
              <div className="ktitle">
                <a href={r.url} target="_blank" rel="noreferrer">
                  {r.title}
                </a>
              </div>
              {r.subtitle && <div className="ksub">{r.subtitle}</div>}
              <div className="kmeta">
                <code>{r.ref}</code> · {r.size_mb < 1 ? `${Math.max(1, Math.round(r.size_mb * 1000))} KB` : `${r.size_mb} MB`} · ⬇ {r.downloads} · ▲{" "}
                {r.votes} · usab. {r.usability}
              </div>
              <div className="row" style={{ gap: 6, marginTop: 6 }}>
                <button
                  className="btn primary sm"
                  disabled={disabled}
                  onClick={() =>
                    onPick(
                      `Importe a base do Kaggle "${r.title}" (ref ${r.ref}) no projeto atual: veja os arquivos, baixe o mais adequado, inspecione as colunas, escolha a coluna alvo e a tarefa explicando o porquê, e importe.`,
                    )
                  }
                >
                  Importar pelo chat
                </button>
                <button
                  className="btn dark-ghost sm"
                  disabled={disabled}
                  onClick={() => onPick(`Descreva a base do Kaggle "${r.title}" (ref ${r.ref}): quais arquivos ela tem e para quais modelos das aulas ela serve?`)}
                >
                  Perguntar
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function shortName(name: string) {
  const m = name.match(/^mcp__(.+?)__(.+)$/);
  return m ? { server: m[1], tool: m[2] } : { server: "", tool: name };
}

function pretty(s?: string) {
  if (!s) return "";
  try {
    return JSON.stringify(JSON.parse(s), null, 1);
  } catch {
    return s;
  }
}

function ToolCard({ part }: { part: Extract<Part, { kind: "tool" }> }) {
  const { server, tool } = shortName(part.name);
  const label = {running: "em execução", completed: "concluída", error: "falhou", cancelled: "interrompida"}[part.status];
  return (
    <details className={`tool ${part.isError ? "err" : ""}`}>
      <summary>
        {part.status === "running" ? <span className="spinner" /> : <span>{part.status === "completed" ? "✓" : part.status === "error" ? "✕" : "■"}</span>}
        <span className="tname">{tool}</span>
        <span className="tserver">{server}</span>
        <span className={`tool-status ${part.status}`}>{label}</span>
      </summary>
      <pre>{`→ entrada\n${JSON.stringify(part.input, null, 1)}`}</pre>
      {part.result !== undefined && <pre>{`← resultado\n${pretty(part.result)}`}</pre>}
    </details>
  );
}

function HarnessView({ init, events, sessionId }: { init: any; events: ChatEvent[]; sessionId: string | null }) {
  const [info, setInfo] = useState<any>(null);
  useEffect(() => {
    api.harness().then(setInfo).catch(() => {});
  }, []);
  return (
    <div className="chat-scroll harness">
      <div className="eyebrow" style={{ color: "var(--cyan)" }}>
        O que é o harness?
      </div>
      <p style={{ marginTop: 6 }}>
        O modelo sozinho só gera texto. O <b>harness</b> é tudo ao redor que o transforma em agente: o loop que executa ferramentas, as permissões, o
        contexto (system prompt), a memória da sessão e as integrações (MCP).
      </p>
      <div className="flow">
        <div>
          <b>1</b>
          <span>
            <strong>Frontend</strong> envia sua mensagem para <code>POST /api/chat</code>.
          </span>
        </div>
        <div>
          <b>2</b>
          <span>
            <strong>Backend</strong> inicia <code>claude -p</code> (Claude Code headless) com system prompt, permissões e <code>--mcp-config</code>.
          </span>
        </div>
        <div>
          <b>3</b>
          <span>
            <strong>Claude Code</strong> inicia o <code>server.py</code> como subprocesso e fala MCP (JSON-RPC) por <b>stdin/stdout</b>.
          </span>
        </div>
        <div>
          <b>4</b>
          <span>
            <strong>Loop agêntico</strong>: o modelo pede uma ferramenta → o MCP chama a API REST → o resultado volta ao modelo → repete.
          </span>
        </div>
        <div>
          <b>5</b>
          <span>
            Eventos <code>stream-json</code> são repassados via SSE; a UI recarrega quando uma ferramenta muda o estado.
          </span>
        </div>
      </div>

      <h4>Sessão atual</h4>
      <p>
        session_id: <code>{sessionId || "—"}</code>
        <br />
        modelo: <code>{init?.model ?? "—"}</code>
        <br />
        servidores MCP: {init?.mcp_servers?.map((s: any) => `${s.name} (${s.status})`).join(", ") ?? "—"}
      </p>
      {init?.tools && (
        <>
          <h4>Ferramentas expostas ao modelo ({init.tools.length})</h4>
          <pre>{init.tools.join("\n")}</pre>
        </>
      )}

      {info && (
        <>
          <h4>Comando executado</h4>
          <pre>{info.command.map((c: string) => (c === "" ? '""' : c.includes(" ") ? `"${c}"` : c)).join(" ")}</pre>
          <h4>System prompt (anexado)</h4>
          <pre>{info.system_prompt}</pre>
          <h4>mcp_config.json</h4>
          <pre>{JSON.stringify(info.mcp_config, null, 2)}</pre>
        </>
      )}

      <h4>Log de eventos ({events.length})</h4>
      <div className="evlog">
        {events.map((e, i) => (
          <div key={i}>
            <span>{e.type}</span>
            {e.type === "tool_use" ? `${e.name} ${JSON.stringify(e.input)}` : e.type === "tool_result" ? (e.content ?? "").slice(0, 160) : e.type === "result" ? `${e.num_turns} turnos, ${e.duration_ms}ms` : ""}
          </div>
        ))}
      </div>
    </div>
  );
}
