import { useEffect, useMemo, useState } from "react";
import { DATA_ACCEPT, api } from "../api";
import type { Architecture, CatalogEntry, Dataset, Project, Run } from "../types";
import { ARCH_LABEL, TASK_LABEL, archTheme, taskTheme } from "../theme";
import { GrayImage, LineChart, PALETTE, Sparkline, fmt } from "./Charts";
import RunDetail from "./RunDetail";

type Tab = "datasets" | "models" | "runs";

interface Props {
  project: Project;
  archs: Architecture[];
  catalog: CatalogEntry[];
  refresh: () => Promise<void>;
  onDeleted: () => void;
}

export default function ProjectView({ project, archs, catalog, refresh, onDeleted }: Props) {
  const [tab, setTab] = useState<Tab>("datasets");
  const [modal, setModal] = useState<null | "dataset" | "model" | { train: number | null }>(null);
  const [runId, setRunId] = useState<number | null>(null);
  const [dsPreview, setDsPreview] = useState<number | null>(null);

  // quando um treino novo aparece (ex.: disparado pelo Claude), vai para a aba de treinos
  const latestRun = project.runs[0]?.id;
  const [seenRun, setSeenRun] = useState(latestRun);
  useEffect(() => {
    if (latestRun && latestRun !== seenRun) {
      setSeenRun(latestRun);
      setTab("runs");
      setRunId(latestRun);
    }
  }, [latestRun, seenRun]);
  useEffect(() => {
    setRunId(null);
    setDsPreview(null);
  }, [project.id]);

  const modelName = (id: number) => project.models.find((m) => m.id === id)?.name ?? `#${id}`;
  const dsName = (id: number) => project.datasets.find((d) => d.id === id)?.name ?? `#${id}`;

  return (
    <>
      <div className="page-head">
        <div>
          <div className="eyebrow">Projeto #{project.id}</div>
          <h1>{project.name}</h1>
          {project.description && <p>{project.description}</p>}
        </div>
        <div className="row">
          <button
            className="btn danger"
            onClick={async () => {
              if (confirm(`Apagar o projeto "${project.name}" e tudo dentro dele?`)) {
                await api.deleteProject(project.id);
                onDeleted();
              }
            }}
          >
            Apagar projeto
          </button>
        </div>
      </div>

      <div className="tabs">
        {(
          [
            ["datasets", "Bases de dados", project.datasets.length],
            ["models", "Modelos", project.models.length],
            ["runs", "Treinos", project.runs.length],
          ] as const
        ).map(([k, label, n]) => (
          <button key={k} className={`tab ${tab === k ? "active" : ""}`} onClick={() => setTab(k)}>
            {label}
            <span className="count">{n}</span>
          </button>
        ))}
      </div>

      {tab === "datasets" && (
        <>
          <div className="spread" style={{ marginBottom: 14 }}>
            <p className="muted small" style={{ margin: 0 }}>
              Bases embutidas (offline), arquivo próprio ou Kaggle (pelo chat). Clique numa base para ver exemplos.
            </p>
            <button className="btn primary" onClick={() => setModal("dataset")}>
              + Adicionar base
            </button>
          </div>
          <div className="grid">
            {project.datasets.map((d) => (
              <div key={d.id} className={`card click ${taskTheme(d.meta.task)} ${dsPreview === d.id ? "selected" : ""}`} onClick={() => setDsPreview(d.id)}>
                <div className="spread">
                  <span className="eyebrow">
                    #{d.id} · {d.source}
                  </span>
                  <button
                    className="btn danger"
                    onClick={async (e) => {
                      e.stopPropagation();
                      await api.deleteDataset(d.id);
                      refresh();
                    }}
                  >
                    ✕
                  </button>
                </div>
                <h4>{d.name}</h4>
                <div className="muted small">{d.meta.description}</div>
                <div className="chips">
                  <span className="chip">{TASK_LABEL[d.meta.task]}</span>
                  <span className="chip">{d.meta.n_samples} amostras</span>
                  <span className="chip">entrada {d.meta.input_shape.join("×")}</span>
                  {d.meta.n_classes && <span className="chip">{d.meta.n_classes} classes</span>}
                </div>
              </div>
            ))}
          </div>
          {dsPreview && <DatasetPreview id={dsPreview} />}
        </>
      )}

      {tab === "models" && (
        <>
          <div className="spread" style={{ marginBottom: 14 }}>
            <p className="muted small" style={{ margin: 0 }}>
              Um modelo é arquitetura + configuração. O mesmo modelo pode ser treinado em várias bases.
            </p>
            <button className="btn primary" onClick={() => setModal("model")}>
              + Criar modelo
            </button>
          </div>
          <div className="grid">
            {project.models.map((m) => {
              const a = archs.find((x) => x.name === m.architecture);
              return (
                <div key={m.id} className={`card ${archTheme(m.architecture)}`}>
                  <div className="spread">
                    <span className="eyebrow">
                      Aula {a?.aula} · {ARCH_LABEL[m.architecture]}
                    </span>
                    <button
                      className="btn danger"
                      onClick={async () => {
                        await api.deleteModel(m.id);
                        refresh();
                      }}
                    >
                      ✕
                    </button>
                  </div>
                  <h4>
                    {m.name} <span className="muted small">#{m.id}</span>
                  </h4>
                  {Object.keys(m.config).length > 0 && <pre className="json">{JSON.stringify(m.config, null, 1)}</pre>}
                  <div style={{ marginTop: 10 }}>
                    <button className="btn ghost" onClick={() => setModal({ train: m.id })}>
                      ▶ Treinar
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}

      {tab === "runs" && (
        <>
          <div className="spread" style={{ marginBottom: 14 }}>
            <p className="muted small" style={{ margin: 0 }}>
              Cada treino registra o histórico por época, as métricas finais e visualizações.
            </p>
            <button className="btn primary" onClick={() => setModal({ train: null })} disabled={!project.models.length || !project.datasets.length}>
              ▶ Novo treino
            </button>
          </div>
          <RunsTable runs={project.runs} modelName={modelName} dsName={dsName} selected={runId} onSelect={setRunId} archOf={(id) => project.models.find((m) => m.id === id)?.architecture ?? ""} />
          {project.runs.length > 1 && <CompareChart runs={project.runs} modelName={modelName} />}
          {runId && (
            <RunDetail
              key={runId}
              runId={runId}
              project={project}
              liveRun={project.runs.find((r) => r.id === runId)}
              onDeleted={() => {
                setRunId(null);
                refresh();
              }}
            />
          )}
        </>
      )}

      {modal === "dataset" && <DatasetModal project={project} catalog={catalog} close={() => setModal(null)} done={refresh} />}
      {modal === "model" && <ModelModal project={project} archs={archs} close={() => setModal(null)} done={refresh} />}
      {modal && typeof modal === "object" && (
        <TrainModal
          project={project}
          archs={archs}
          modelId={modal.train}
          close={() => setModal(null)}
          done={async (r) => {
            await refresh();
            setTab("runs");
            setRunId(r.id);
          }}
        />
      )}
    </>
  );
}

/* ------------------------------------------------------------ tabela de treinos */
function mainMetric(r: Run): string {
  const m = r.metrics || {};
  if (m.accuracy !== undefined) return `acc ${(m.accuracy * 100).toFixed(1)}%`;
  if (m.r2 !== undefined) return `R² ${fmt(m.r2)} · RMSE ${fmt(m.rmse)}`;
  if (m.reconstruction_mse !== undefined) return `recon MSE ${fmt(m.reconstruction_mse)}`;
  const last = r.history.at(-1);
  if (last?.d_loss !== undefined) return `D ${fmt(last.d_loss)} · G ${fmt(last.g_loss)}`;
  return "—";
}

function RunsTable({ runs, modelName, dsName, selected, onSelect, archOf }: { runs: Run[]; modelName: (id: number) => string; dsName: (id: number) => string; selected: number | null; onSelect: (id: number) => void; archOf: (id: number) => string }) {
  if (!runs.length) return <p className="muted">Nenhum treino ainda.</p>;
  return (
    <table className="runs">
      <thead>
        <tr>
          <th>#</th>
          <th>Modelo</th>
          <th>Base</th>
          <th>Status</th>
          <th>Época</th>
          <th>Perda</th>
          <th>Métrica</th>
        </tr>
      </thead>
      <tbody>
        {runs.map((r) => {
          const losses = r.history.map((h) => h.train_loss ?? h.d_loss);
          return (
            <tr key={r.id} className={`click ${selected === r.id ? "selected" : ""}`} onClick={() => onSelect(r.id)}>
              <td className="mono">{r.id}</td>
              <td>
                <span className={archTheme(archOf(r.model_id))}>
                  <span className="chip">{ARCH_LABEL[archOf(r.model_id)] ?? "?"}</span>
                </span>{" "}
                {modelName(r.model_id)}
              </td>
              <td>{dsName(r.dataset_id)}</td>
              <td>
                <span className={`status ${r.status}`}>{r.status}</span>
              </td>
              <td className="mono">
                {r.history.length}/{String(r.hyperparams.epochs)}
              </td>
              <td>
                <Sparkline values={losses} />
              </td>
              <td className="mono small">{mainMetric(r)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function CompareChart({ runs, modelName }: { runs: Run[]; modelName: (id: number) => string }) {
  const withVal = runs.filter((r) => r.history.some((h) => h.val_loss !== undefined)).slice(0, 6);
  if (withVal.length < 2) return null;
  return (
    <div className="card" style={{ marginTop: 14 }}>
      <LineChart
        title="Comparação · perda de validação"
        series={withVal.map((r, i) => ({ name: `#${r.id} ${modelName(r.model_id)}`.slice(0, 16), values: r.history.map((h) => h.val_loss), color: PALETTE[i % PALETTE.length] }))}
      />
    </div>
  );
}

/* ------------------------------------------------------------ prévia de base */
function DatasetPreview({ id }: { id: number }) {
  const [p, setP] = useState<any>(null);
  useEffect(() => {
    setP(null);
    api.previewDataset(id, 10).then(setP);
  }, [id]);
  if (!p) return null;
  const m = p.meta;
  return (
    <div className="card section">
      <div className="eyebrow">Prévia · base #{id}</div>
      <div className="stats" style={{ margin: "12px 0" }}>
        <div className="stat">
          <span>amostras</span>
          <b>{m.n_samples}</b>
        </div>
        <div className="stat">
          <span>entrada</span>
          <b>{m.input_shape.join("×")}</b>
        </div>
        {m.n_classes && (
          <div className="stat">
            <span>classes</span>
            <b>{m.n_classes}</b>
          </div>
        )}
        {m.baseline_persistence_rmse !== undefined && (
          <div className="stat">
            <span>rmse ingênuo (amanhã = hoje)</span>
            <b>
              {fmt(m.baseline_persistence_rmse)} {m.unit ?? ""}
            </b>
          </div>
        )}
        {m.period && (
          <div className="stat">
            <span>período</span>
            <b style={{ fontSize: "0.95rem" }}>{m.period}</b>
          </div>
        )}
        {m.target_stats && (
          <div className="stat">
            <span>alvo (média ± dp)</span>
            <b>
              {fmt(m.target_stats.mean)} ± {fmt(m.target_stats.std)}
            </b>
          </div>
        )}
      </div>
      {m.class_counts && (
        <p className="small muted">
          Balanceamento: {m.class_counts.map((c: number, i: number) => `${m.class_names?.[i] ?? i}: ${c}`).join(" · ")}
        </p>
      )}
      {p.images && (
        <div className="img-row">
          {p.images.map((im: any, i: number) => (
            <div key={i} style={{ textAlign: "center" }}>
              <GrayImage pixels={im.pixels} scale={2.4} />
              <div className="small muted">{m.class_names?.[im.y] ?? im.y}</div>
            </div>
          ))}
        </div>
      )}
      {p.sequences && (
        <LineChart xLabel="passo" title="Janelas (entrada) — cada linha é uma amostra" series={p.sequences.slice(0, 5).map((s: any, i: number) => ({ name: `y=${fmt(s.y)}`, values: s.x, color: PALETTE[i] }))} />
      )}
      {p.rows && (
        <table className="runs">
          <thead>
            <tr>
              {(m.feature_names ?? []).slice(0, 8).map((f: string) => (
                <th key={f}>{f}</th>
              ))}
              <th>alvo</th>
            </tr>
          </thead>
          <tbody>
            {p.rows.map((r: any, i: number) => (
              <tr key={i}>
                {r.x.slice(0, 8).map((v: number, j: number) => (
                  <td key={j} className="mono">
                    {fmt(v)}
                  </td>
                ))}
                <td className="mono">{m.class_names ? m.class_names[r.y] : r.y === null ? "—" : fmt(r.y)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

/* ------------------------------------------------------------ modais */
function Modal({ title, eyebrow, close, children }: { title: string; eyebrow: string; close: () => void; children: React.ReactNode }) {
  return (
    <div className="modal-back" onClick={close}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="eyebrow">{eyebrow}</div>
        <h3>{title}</h3>
        {children}
      </div>
    </div>
  );
}

function parseJson(s: string): object {
  if (!s.trim()) return {};
  return JSON.parse(s);
}

function DatasetModal({ project, catalog, close, done }: { project: Project; catalog: CatalogEntry[]; close: () => void; done: () => void }) {
  const [mode, setMode] = useState<"builtin" | "csv">("builtin");
  const [source, setSource] = useState(catalog[0]?.source ?? "");
  const [name, setName] = useState("");
  const [params, setParams] = useState("{}");
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState("");
  const [task, setTask] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const entry = catalog.find((c) => c.source === source);

  async function submit() {
    setErr("");
    setBusy(true);
    try {
      if (mode === "builtin") await api.addDataset(project.id, source, name || null, parseJson(params));
      else {
        if (!file) throw new Error("Escolha um arquivo");
        const f = new FormData();
        f.append("file", file);
        f.append("target_column", target);
        f.append("task", task);
        f.append("name", name);
        await api.uploadFile(project.id, f);
      }
      done();
      close();
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal title="Adicionar base de dados" eyebrow={`Projeto ${project.name}`} close={close}>
      <div className="form">
        <div className="row">
          <button className={`btn ${mode === "builtin" ? "secondary" : "ghost"}`} onClick={() => setMode("builtin")}>
            Base embutida
          </button>
          <button className={`btn ${mode === "csv" ? "secondary" : "ghost"}`} onClick={() => setMode("csv")}>
            Upload de arquivo
          </button>
        </div>
        {mode === "builtin" ? (
          <>
            <label className="field">
              Fonte
              <select value={source} onChange={(e) => setSource(e.target.value)}>
                {[1, 2, 3].map((a) => (
                  <optgroup key={a} label={`Aula ${a}`}>
                    {catalog
                      .filter((c) => c.aula === a)
                      .map((c) => (
                        <option key={c.source} value={c.source}>
                          {c.source}
                        </option>
                      ))}
                  </optgroup>
                ))}
              </select>
            </label>
            {entry && <div className="hint">{entry.description}</div>}
            <label className="field">
              Parâmetros (JSON)
              <textarea className="code" style={{ minHeight: 60 }} value={params} onChange={(e) => setParams(e.target.value)} />
            </label>
          </>
        ) : (
          <>
            <label className="field">
              Arquivo (CSV, TSV, JSON, Excel, Parquet ou NPZ)
              <input type="file" accept={DATA_ACCEPT} onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            </label>
            <div className="form-row">
              <label className="field">
                Coluna alvo
                <input value={target} onChange={(e) => setTarget(e.target.value)} placeholder="vazio = não supervisionado" />
              </label>
              <label className="field">
                Tarefa
                <select value={task} onChange={(e) => setTask(e.target.value)}>
                  <option value="">inferir</option>
                  <option value="tabular_classification">classificação</option>
                  <option value="tabular_regression">regressão</option>
                </select>
              </label>
            </div>
          </>
        )}
        <label className="field">
          Nome (opcional)
          <input value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        {err && <div className="err">{err}</div>}
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <button className="btn ghost" onClick={close}>
            Cancelar
          </button>
          <button className="btn primary" onClick={submit} disabled={busy}>
            {busy ? "Gerando…" : "Adicionar"}
          </button>
        </div>
      </div>
    </Modal>
  );
}

function ModelModal({ project, archs, close, done }: { project: Project; archs: Architecture[]; close: () => void; done: () => void }) {
  const [arch, setArch] = useState(archs[0]?.name ?? "mlp");
  const a = archs.find((x) => x.name === arch);
  const [name, setName] = useState("");
  const [config, setConfig] = useState("{}");
  const [err, setErr] = useState("");
  useEffect(() => setConfig(JSON.stringify(a?.default_config ?? {}, null, 2)), [a]);

  async function submit() {
    setErr("");
    try {
      await api.createModel(project.id, name || `${ARCH_LABEL[arch]} ${project.models.length + 1}`, arch, parseJson(config));
      done();
      close();
    } catch (e: any) {
      setErr(e.message);
    }
  }
  return (
    <Modal title="Criar modelo" eyebrow={`Projeto ${project.name}`} close={close}>
      <div className="form">
        <div className="form-row">
          <label className="field">
            Arquitetura
            <select value={arch} onChange={(e) => setArch(e.target.value)}>
              {archs.map((x) => (
                <option key={x.name} value={x.name}>
                  Aula {x.aula} · {ARCH_LABEL[x.name]}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Nome
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder={ARCH_LABEL[arch]} />
          </label>
        </div>
        {a && (
          <div className={`hint ${archTheme(arch)}`}>
            {a.description}
            <div className="chips">
              {a.tasks.map((t) => (
                <span key={t} className="chip">
                  {TASK_LABEL[t]}
                </span>
              ))}
            </div>
          </div>
        )}
        {a && Object.keys(a.default_config).length > 0 && (
          <label className="field">
            Configuração (JSON)
            <textarea className="code" value={config} onChange={(e) => setConfig(e.target.value)} />
          </label>
        )}
        {a && typeof a.config_doc === "object" && Object.keys(a.config_doc).length > 0 && (
          <div className="small muted">
            {Object.entries(a.config_doc).map(([k, v]) => (
              <div key={k}>
                <code>{k}</code>: {v}
              </div>
            ))}
          </div>
        )}
        {err && <div className="err">{err}</div>}
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <button className="btn ghost" onClick={close}>
            Cancelar
          </button>
          <button className="btn primary" onClick={submit}>
            Criar
          </button>
        </div>
      </div>
    </Modal>
  );
}

function TrainModal({ project, archs, modelId, close, done }: { project: Project; archs: Architecture[]; modelId: number | null; close: () => void; done: (r: Run) => void }) {
  const [mid, setMid] = useState<number>(modelId ?? project.models[0]?.id);
  const model = project.models.find((m) => m.id === mid);
  const a = archs.find((x) => x.name === model?.architecture);
  const compatible = useMemo(() => project.datasets.filter((d: Dataset) => a?.tasks.includes(d.meta.task)), [project.datasets, a]);
  const [did, setDid] = useState<number | undefined>(compatible[0]?.id);
  const [hp, setHp] = useState("{}");
  const [err, setErr] = useState("");
  useEffect(() => {
    setHp(JSON.stringify(a?.default_hyperparams ?? {}, null, 2));
    setDid(compatible[0]?.id);
  }, [a, compatible]);

  async function submit() {
    setErr("");
    try {
      if (!did) throw new Error("Nenhuma base compatível com essa arquitetura");
      const r = await api.startRun(mid, did, parseJson(hp));
      done(r);
      close();
    } catch (e: any) {
      setErr(e.message);
    }
  }
  return (
    <Modal title="Treinar modelo" eyebrow={`Projeto ${project.name}`} close={close}>
      <div className="form">
        <div className="form-row">
          <label className="field">
            Modelo
            <select value={mid} onChange={(e) => setMid(Number(e.target.value))}>
              {project.models.map((m) => (
                <option key={m.id} value={m.id}>
                  #{m.id} {m.name} ({ARCH_LABEL[m.architecture]})
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Base de dados
            <select value={did} onChange={(e) => setDid(Number(e.target.value))}>
              {compatible.map((d) => (
                <option key={d.id} value={d.id}>
                  #{d.id} {d.name}
                </option>
              ))}
            </select>
          </label>
        </div>
        {!compatible.length && <div className="err">Nenhuma base do projeto é compatível com {model?.architecture} ({a?.tasks.map((t) => TASK_LABEL[t]).join(", ")}).</div>}
        <label className="field">
          Hiperparâmetros (JSON)
          <textarea className="code" value={hp} onChange={(e) => setHp(e.target.value)} />
        </label>
        <div className="small muted">
          <code>optimizer</code>: adam | sgd | rmsprop · <code>batch_size</code> 0 = lote inteiro · <code>test_size</code>: fração de validação
        </div>
        {err && <div className="err">{err}</div>}
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <button className="btn ghost" onClick={close}>
            Cancelar
          </button>
          <button className="btn primary" onClick={submit} disabled={!compatible.length}>
            ▶ Treinar
          </button>
        </div>
      </div>
    </Modal>
  );
}
