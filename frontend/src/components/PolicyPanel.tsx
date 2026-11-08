import { useState } from "react";
import { api } from "../api";
import type { ChatEvent, PermissionEvent } from "../types";

export const MODES: Record<string, string> = {
  consult: "Consultar",
  confirm: "Executar com confirmação",
  operate: "Operar o laboratório",
};
export const DESCRIPTIONS: Record<string, string> = {
  consult: "Somente consultas. Pedidos de alteração são bloqueados pelo código.",
  confirm: "Cada alteração aguarda sua aprovação antes da execução.",
  operate: "Cria projetos, adiciona bases e treina quando solicitado. Exclusões exigem aprovação.",
};

const LABELS: Record<string, string> = {allowed: "Permitida", blocked: "Bloqueada", awaiting: "Aguardando aprovação", approved: "Aprovada", denied: "Recusada", expired: "Expirada", completed: "Concluída", failed: "Erro", cancelled: "Interrompida"};

export function Approvals({events, busy}: {events: ChatEvent[]; busy: boolean}) {
  const [sent, setSent] = useState<string[]>([]);
  const [error, setError] = useState("");
  const latest = new Map<string, PermissionEvent>();
  events.forEach(e => {if (e.type === "permission") latest.set(e.id, e);});
  const pending = [...latest.values()].filter(e => e.decision === "awaiting");
  async function decide(e: PermissionEvent, approved: boolean) {
    setSent(s => [...s, e.id]); setError("");
    try { await api.approveAction(e.run_id, e.id, approved); }
    catch (err: any) { setError(err.message); setSent(s => s.filter(id => id !== e.id)); }
  }
  if (!busy || !pending.length) return null;
  return <div className="approval-list" aria-live="polite">
    {pending.map(e => <section className="approval-card" key={e.id}>
      <strong>{e.kind === "delete" ? "Confirmar exclusão" : "Ação proposta"}: {e.tool}</strong>
      <p>{e.reason}</p><code>{e.method} {e.path}</code>
      <pre>{JSON.stringify(e.input, null, 2)}</pre>
      <div className="row">
        <button className="btn primary" disabled={sent.includes(e.id)} onClick={() => decide(e, true)}>Aprovar esta ação</button>
        <button className="btn dark-ghost" disabled={sent.includes(e.id)} onClick={() => decide(e, false)}>Recusar</button>
      </div>
    </section>)}
    {error && <p role="alert">{error}</p>}
  </div>;
}

export default function PolicyPanel({events, policy}: {events: ChatEvent[]; policy: string}) {
  const current = [...events].reverse().find(e => e.type === "harness_policy");
  const actions = events.filter((e): e is PermissionEvent => e.type === "permission");
  return <div className="chat-scroll harness">
    <h3>Política: {MODES[current?.type === "harness_policy" ? current.policy : policy]}</h3>
    <p>{DESCRIPTIONS[policy]}</p>
    <p>Pedido → ferramenta → verificação de permissão → execução → resultado → modelo.</p>
    <p>Limites por mensagem: 40 chamadas, 10 minutos e 2 minutos para aprovar. Parar interrompe a conversa; um treino já iniciado é cancelado na aba Treinos.</p>
    <details><summary>Instruções recebidas pelo modelo</summary><pre>{current?.type === "harness_policy" ? current.instructions : "As instruções aparecem ao enviar a primeira mensagem."}</pre></details>
    <h4>Linha do tempo das permissões</h4>
    {!actions.length && <p>Envie um pedido para observar as decisões do harness.</p>}
    <ol className="permission-timeline">{actions.map((e, i) => <li key={`${e.run_id}-${e.id}-${i}`}>
      <strong>{LABELS[e.decision] ?? e.decision} · {e.tool}</strong>
      <p>Etapa {e.step} · {e.elapsed ?? 0}s · {e.reason}</p>
      <details><summary>Parâmetros da ação</summary><pre>{JSON.stringify(e.input, null, 2)}</pre></details>
    </li>)}</ol>
  </div>;
}
