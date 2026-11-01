import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { api, streamChat } from "../api";
import type { ChatEvent, Project } from "../types";

type Part = { kind: "text"; text: string } | { kind: "tool"; id: string; name: string; input: unknown; result?: string; isError?: boolean };
interface Msg {
  role: "user" | "assistant";
  parts: Part[];
  footer?: string;
}

const SUGGESTIONS = [
  "Crie um projeto 'Aula 1', adicione as bases and_gate e xor_gate e mostre que o perceptron resolve AND mas não XOR.",
  "Neste projeto, adicione a base xor_gate e treine um MLP com uma camada oculta de 4 neurônios tanh. Explique por que funciona.",
  "Adicione a base digits e compare um MLP com uma CNN pequena. Qual tem menos parâmetros e mais acurácia?",
  "Compare RNN, LSTM e GRU na base sequence_parity com sequências de tamanho 20.",
  "Treine um autoencoder com latente 2D na base digits e depois uma versão denoising. O que muda?",
  "Treine uma GAN na base moons e comente as curvas do discriminador e do gerador.",
];

export default function ChatPanel({ project, onToolResult }: { project: Project | null; onToolResult: () => void }) {
  const [view, setView] = useState<"chat" | "harness">("chat");
  const [msgs, setMsgs] = useState<Msg[]>(() => {
    try {
      return JSON.parse(localStorage.getItem("dl.msgs") ?? "[]");
    } catch {
      return [];
    }
  });
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(() => localStorage.getItem("dl.session"));
  const [model, setModel] = useState(() => localStorage.getItem("dl.model") ?? "");
  const [events, setEvents] = useState<ChatEvent[]>([]);
  const [init, setInit] = useState<Extract<ChatEvent, { type: "init" }> | null>(null);
  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  // o stream vive por vários renders: guardamos sempre o callback mais recente
  const onToolResultRef = useRef(onToolResult);
  onToolResultRef.current = onToolResult;

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight });
  }, [msgs]);
  useEffect(() => {
    localStorage.setItem("dl.session", sessionId ?? "");
  }, [sessionId]);
  useEffect(() => {
    if (busy) return; // grava só ao fim de cada resposta
    try {
      localStorage.setItem("dl.msgs", JSON.stringify(msgs.slice(-60)));
    } catch {
      /* quota cheia: ignora */
    }
  }, [msgs, busy]);
  useEffect(() => {
    localStorage.setItem("dl.model", model);
  }, [model]);

  /** Aplica um evento do stream ao último balão do assistente. */
  function apply(ev: ChatEvent) {
    if (ev.type !== "text_delta") setEvents((e) => [...e.slice(-300), ev]);
    setMsgs((prev) => {
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
          last.parts.push({ kind: "tool", id: ev.id, name: ev.name, input: ev.input });
          break;
        case "tool_result":
          last.parts = last.parts.map((p) => (p.kind === "tool" && p.id === ev.tool_use_id ? { ...p, result: ev.content, isError: ev.is_error } : p));
          break;
        case "result":
          last.footer = `${ev.num_turns} turnos · ${(ev.duration_ms / 1000).toFixed(1)}s${ev.total_cost_usd ? ` · US$ ${ev.total_cost_usd.toFixed(4)}` : ""}`;
          break;
        case "error":
          last.parts.push({ kind: "text", text: `**Erro:** ${ev.message}` });
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

  async function send(text?: string) {
    const message = (text ?? input).trim();
    if (!message || busy) return;
    setInput("");
    setBusy(true);
    setMsgs((m) => [...m, { role: "user", parts: [{ kind: "text", text: message }] }, { role: "assistant", parts: [] }]);
    abort.current = new AbortController();
    try {
      await streamChat({ message, session_id: sessionId, project_id: project?.id ?? null, model: model || null }, apply, abort.current.signal);
    } catch (e: any) {
      if (e.name !== "AbortError") apply({ type: "error", message: e.message });
    } finally {
      setBusy(false);
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
    <aside className="chat">
      <div className="chat-head">
        <div className="spread">
          <div>
            <div className="eyebrow">Claude Code · MCP stdio</div>
            <h3>Assistente do laboratório</h3>
          </div>
          <button className="btn dark-ghost" onClick={reset} title="Começa uma nova sessão do Claude Code">
            Nova conversa
          </button>
        </div>
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
                  Peça em linguagem natural. O Claude decide quais ferramentas MCP chamar
                  {project ? (
                    <>
                      {" "}— ele sabe que o projeto atual é <b style={{ color: "#fff" }}>{project.name}</b>
                    </>
                  ) : null}
                  .
                </p>
                <div className="suggest">
                  {SUGGESTIONS.map((s) => (
                    <button key={s} onClick={() => send(s)}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {msgs.map((m, i) => (
              <div key={i} className={`msg ${m.role}`}>
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
                {m.role === "assistant" && busy && i === msgs.length - 1 && (
                  <div className="row small" style={{ color: "#91a5bb" }}>
                    <span className="spinner" /> trabalhando…
                  </div>
                )}
                {m.footer && <div className="result-line">{m.footer}</div>}
              </div>
            ))}
          </div>
          <div className="chat-input">
            <textarea
              placeholder={project ? `Peça algo sobre "${project.name}"…` : "Ex.: crie um projeto e treine um perceptron no AND"}
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
              <select value={model} onChange={(e) => setModel(e.target.value)} title="Modelo usado pelo Claude Code">
                <option value="">modelo padrão</option>
                <option value="opus">opus</option>
                <option value="sonnet">sonnet</option>
                <option value="haiku">haiku</option>
              </select>
              {busy ? (
                <button className="btn dark-ghost" onClick={() => abort.current?.abort()}>
                  ■ Parar
                </button>
              ) : (
                <button className="btn primary" onClick={() => send()} disabled={!input.trim()}>
                  Enviar ↵
                </button>
              )}
            </div>
          </div>
        </>
      ) : (
        <HarnessView init={init} events={events} sessionId={sessionId} />
      )}
    </aside>
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
  return (
    <details className={`tool ${part.isError ? "err" : ""}`}>
      <summary>
        {part.result === undefined ? <span className="spinner" /> : <span>{part.isError ? "✕" : "✓"}</span>}
        <span className="tname">{tool}</span>
        <span className="tserver">{server}</span>
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
