import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { Architecture, CatalogEntry, Project, ProjectSummary } from "./types";
import ProjectView from "./components/ProjectView";
import ChatPanel from "./components/ChatPanel";
import "./responsive.css";

export default function App() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [selected, setSelected] = useState<number | null>(() => Number(localStorage.getItem("dl.project")) || null);
  const [project, setProject] = useState<Project | null>(null);
  const [archs, setArchs] = useState<Architecture[]>([]);
  const [catalog, setCatalog] = useState<CatalogEntry[]>([]);
  const [online, setOnline] = useState<boolean | null>(null);
  const [chatOpen, setChatOpen] = useState(true);
  const [newName, setNewName] = useState("");
  const [deletingAll, setDeletingAll] = useState(false);

  const refreshProjects = useCallback(async () => {
    try {
      const ps = await api.projects();
      setProjects(ps);
      setOnline(true);
      return ps;
    } catch {
      setOnline(false);
      return [];
    }
  }, []);

  const refreshProject = useCallback(async () => {
    if (!selected) return setProject(null);
    try {
      setProject(await api.project(selected));
    } catch {
      setProject(null);
      setSelected(null);
    }
  }, [selected]);

  /** Chamado quando o Claude executa uma ferramenta: o estado mudou do lado do servidor. */
  const maxSeen = useRef<number | null>(null);
  const refreshAll = useCallback(async () => {
    const ps = await refreshProjects();
    const newest = ps[0]?.id ?? 0;
    // se o Claude criou um projeto novo (ou nenhum está aberto), abre-o
    if (ps.length && (!selected || (maxSeen.current !== null && newest > maxSeen.current))) setSelected(newest);
    maxSeen.current = Math.max(maxSeen.current ?? 0, newest);
    await refreshProject();
  }, [refreshProjects, refreshProject, selected]);

  useEffect(() => {
    if (projects.length && maxSeen.current === null) maxSeen.current = projects[0].id;
  }, [projects]);

  useEffect(() => {
    refreshProjects();
    api.architectures().then(setArchs).catch(() => {});
    api.catalog().then(setCatalog).catch(() => {});
  }, [refreshProjects]);

  useEffect(() => {
    localStorage.setItem("dl.project", String(selected ?? ""));
    refreshProject();
  }, [selected, refreshProject]);

  // Polling enquanto houver treino em andamento (o progresso por época aparece ao vivo)
  useEffect(() => {
    const busy = project?.runs.some((r) => r.status === "running" || r.status === "queued");
    if (!busy) return;
    const t = setInterval(refreshProject, 1000);
    return () => clearInterval(t);
  }, [project, refreshProject]);

  async function create() {
    if (!newName.trim()) return;
    const p = await api.createProject(newName.trim());
    setNewName("");
    await refreshProjects();
    setSelected(p.id);
  }

  async function deleteAllProjects() {
    if (!projects.length || deletingAll) return;
    const message = `Apagar todos os ${projects.length} projetos? Isso também remove as bases, modelos e treinos de cada projeto. Esta ação não pode ser desfeita.`;
    if (!window.confirm(message)) return;
    setDeletingAll(true);
    try {
      // Em sequência para evitar concorrência na atualização da base local.
      for (const item of projects) await api.deleteProject(item.id);
      localStorage.removeItem("dl.project");
      setSelected(null);
      setProject(null);
      await refreshProjects();
    } catch (error) {
      window.alert(`Não foi possível apagar todos os projetos: ${error instanceof Error ? error.message : "erro desconhecido"}`);
      await refreshProjects();
    } finally {
      setDeletingAll(false);
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <b>DL</b>
          <div>
            Laboratório de Deep Learning
            <small>harness · múltiplos provedores + mcp stdio</small>
          </div>
        </div>
        <div className="top-actions">
          <span className="hbadge">
            <span className={`dot ${online ? "ok" : online === false ? "bad" : ""}`} /> API {online ? "online" : online === false ? "offline" : "…"}
          </span>
          <span className="hbadge">
            <b>{archs.length}</b> arquiteturas · <b>{catalog.length}</b> bases
          </span>
          <a className="btn dark-ghost" href="/apresentacao.html" target="_blank" rel="noreferrer" style={{ textDecoration: "none" }} title="Apresentação 1: o que é um harness">
            Apresentação 1
          </a>
          <a className="btn dark-ghost" href="/apresentacao2.html" target="_blank" rel="noreferrer" style={{ textDecoration: "none" }} title="Apresentação 2: vibe coding e uso corporativo">
            Apresentação 2
          </a>
          <button className="btn dark-ghost" onClick={() => setChatOpen((v) => !v)}>
            {chatOpen ? "Ocultar chat" : "Abrir chat"}
          </button>
        </div>
      </header>

      <div className={`body ${chatOpen ? "" : "chat-hidden"}`}>
        <aside className="sidebar">
          <div className="eyebrow">Projetos</div>
          <div className="row" style={{ margin: "10px 0 14px" }}>
            <input placeholder="Novo projeto…" value={newName} onChange={(e) => setNewName(e.target.value)} onKeyDown={(e) => e.key === "Enter" && create()} style={{ flex: 1, minWidth: 0 }} />
            <button className="btn primary" onClick={create} title="Criar projeto">
              +
            </button>
          </div>
          {projects.length > 0 && (
            <button className="btn danger" onClick={deleteAllProjects} disabled={deletingAll} style={{ width: "100%", marginBottom: 10 }}>
              {deletingAll ? "Apagando projetos…" : "Apagar todos os projetos"}
            </button>
          )}
          {projects.map((p) => (
            <button key={p.id} className={`proj ${p.id === selected ? "active" : ""}`} onClick={() => setSelected(p.id)}>
              <strong>{p.name}</strong>
              <span>
                #{p.id} · {p.n_datasets} bases · {p.n_models} modelos · {p.n_runs} treinos
              </span>
            </button>
          ))}
          {!projects.length && <p className="muted small">Nenhum projeto ainda. Crie aqui ou peça ao assistente no chat.</p>}
        </aside>

        <main className="main">
          {project ? (
            <ProjectView
              project={project}
              archs={archs}
              catalog={catalog}
              refresh={refreshProject}
              onDeleted={async () => {
                setSelected(null);
                await refreshProjects();
              }}
            />
          ) : (
            <Welcome />
          )}
        </main>

        {chatOpen && <ChatPanel project={project} onToolResult={refreshAll} />}
      </div>
    </div>
  );
}

function Welcome() {
  return (
    <div className="empty">
      <div className="eyebrow">Redes Neurais e Deep Learning · Pós-graduação</div>
      <h2>
        Um laboratório operado por <em className="accent">um agente</em>
      </h2>
      <p>
        Crie projetos, cadastre bases de dados e treine os modelos das aulas — Perceptron, Adaline, MLP, CNN, RNN/LSTM/GRU, Autoencoders, GANs e
        Transformers — clicando aqui ou conversando com o assistente no painel ao lado.
      </p>
      <p>
        O assistente usa as mesmas ferramentas do laboratório via <b>MCP stdio</b>. Escolha seu provedor no painel e abra a aba <b>Harness</b> do chat
        para ver as engrenagens.
      </p>
    </div>
  );
}
