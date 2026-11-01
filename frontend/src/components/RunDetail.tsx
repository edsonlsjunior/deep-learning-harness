import { useEffect, useState } from "react";
import { api } from "../api";
import type { Project, Run } from "../types";
import { ARCH_LABEL, archTheme } from "../theme";
import { BoundaryPlot, ConfusionMatrix, CurvePlot, GrayImage, ImageRow, LineChart, PALETTE, ScatterPlot, fmt } from "./Charts";

export default function RunDetail({ runId, project, liveRun, onDeleted }: { runId: number; project: Project; liveRun?: Run; onDeleted: () => void }) {
  const [run, setRun] = useState<Run | null>(null);
  const [generated, setGenerated] = useState<any>(null);
  const status = liveRun?.status;

  // detalhes completos (com extras) quando o treino termina; enquanto roda usa o resumo ao vivo
  useEffect(() => {
    api.run(runId).then(setRun);
  }, [runId, status]);

  const r = run && liveRun && liveRun.status !== run.status ? { ...run, ...liveRun } : run && liveRun ? { ...run, history: liveRun.history } : run;
  if (!r) return null;
  const model = project.models.find((m) => m.id === r.model_id);
  const ds = project.datasets.find((d) => d.id === r.dataset_id);
  const arch = model?.architecture ?? "";
  const h = r.history;
  const m = r.metrics || {};
  const x = r.extras || {};
  const names = ds?.meta.class_names;

  const metricCards: [string, string][] = [];
  if (m.accuracy !== undefined) metricCards.push(["acurácia", `${(m.accuracy * 100).toFixed(1)}%`]);
  for (const k of ["r2", "rmse", "mae", "reconstruction_mse", "anomaly_threshold_p95"]) if (m[k] !== undefined) metricCards.push([k, fmt(m[k])]);
  if (m.n_params !== undefined) metricCards.push(["parâmetros", m.n_params.toLocaleString("pt-BR")]);
  if (m.train_seconds !== undefined) metricCards.push(["tempo", `${m.train_seconds}s`]);
  if (h.length) metricCards.push(["épocas", String(h.length)]);

  return (
    <div className={`card section ${archTheme(arch)}`}>
      <div className="spread">
        <div>
          <div className="eyebrow">
            Treino #{r.id} · {ARCH_LABEL[arch]} · <span className={`status ${r.status}`}>{r.status}</span>
          </div>
          <h4 style={{ fontSize: "1.3rem" }}>
            {model?.name} → {ds?.name}
          </h4>
          {m.evaluated_on && <div className="muted small">métricas avaliadas em: {m.evaluated_on}</div>}
        </div>
        <div className="row">
          {(r.status === "running" || r.status === "queued") && (
            <button className="btn ghost" onClick={() => api.cancelRun(r.id)}>
              Interromper
            </button>
          )}
          {r.status === "completed" && (arch === "gan" || model?.config.variant === "variational") && (
            <button className="btn ghost" onClick={async () => setGenerated(await api.generate(r.id, 16))}>
              ✦ Gerar amostras
            </button>
          )}
          <button
            className="btn danger"
            onClick={async () => {
              await api.deleteRun(r.id);
              onDeleted();
            }}
          >
            Apagar
          </button>
        </div>
      </div>

      {r.error && <pre className="err">{r.error}</pre>}

      <div className="stats" style={{ marginTop: 14 }}>
        {metricCards.map(([k, v]) => (
          <div className="stat" key={k}>
            <span>{k}</span>
            <b>{v}</b>
          </div>
        ))}
      </div>

      <div className="viz">
        {h.length > 0 && h[0].train_loss !== undefined && (
          <div>
            <LineChart
              title={`perda${x.loss_name ? ` · ${x.loss_name}` : ""}`}
              series={[
                { name: "treino", values: h.map((e) => e.train_loss), color: PALETTE[0] },
                { name: "validação", values: h.map((e) => e.val_loss), color: PALETTE[1], dashed: true },
              ]}
            />
          </div>
        )}
        {h.length > 0 && h[0].train_acc !== undefined && (
          <div>
            <LineChart
              title="acurácia"
              series={[
                { name: "treino", values: h.map((e) => e.train_acc), color: PALETTE[2] },
                { name: "validação", values: h.map((e) => e.val_acc), color: PALETTE[3], dashed: true },
              ]}
            />
          </div>
        )}
        {h.length > 0 && h[0].d_loss !== undefined && (
          <div>
            <LineChart
              title="GAN · discriminador x gerador"
              series={[
                { name: "D (perito)", values: h.map((e) => e.d_loss), color: PALETTE[3] },
                { name: "G (falsificador)", values: h.map((e) => e.g_loss), color: PALETTE[1] },
              ]}
            />
          </div>
        )}
      </div>

      <div className="viz">
        {x.plot?.kind === "boundary" && (
          <div>
            <div className="eyebrow" style={{ marginBottom: 8 }}>
              Fronteira de decisão
            </div>
            <BoundaryPlot plot={x.plot} />
          </div>
        )}
        {x.plot?.kind === "curve" && (
          <div>
            <div className="eyebrow">Função aprendida</div>
            <CurvePlot plot={x.plot} />
          </div>
        )}
        {x.plot?.kind === "series" && (
          <div style={{ gridColumn: "1 / -1" }}>
            <LineChart
              title="Série · real x previsto (1 passo à frente)"
              height={220}
              series={[
                { name: "real", values: x.plot.y_true, color: PALETTE[0] },
                { name: "previsto", values: x.plot.y_pred, color: PALETTE[1] },
              ]}
            />
          </div>
        )}
        {x.plot?.kind === "gan2d" && (
          <div>
            <div className="eyebrow" style={{ marginBottom: 8 }}>
              <span style={{ color: PALETTE[0] }}>● real</span> · <span style={{ color: PALETTE[1] }}>● gerado</span>
            </div>
            <ScatterPlot
              groups={[
                { points: x.plot.real, color: PALETTE[0] + "99" },
                { points: x.plot.generated, color: PALETTE[1] + "aa" },
              ]}
            />
          </div>
        )}
        {x.latent && x.latent.points.length > 0 && (
          <div>
            <div className="eyebrow" style={{ marginBottom: 8 }}>
              Espaço latente {m.latent_dim > 2 ? "(PCA 2D)" : "(2D)"}
            </div>
            <ScatterPlot groups={[{ points: x.latent.points, color: (i: number) => PALETTE[(x.latent.labels[i] ?? 0) % PALETTE.length], r: 2.5 }]} />
          </div>
        )}
        {m.confusion_matrix && (
          <div>
            <div className="eyebrow" style={{ marginBottom: 8 }}>
              Matriz de confusão
            </div>
            <ConfusionMatrix cm={m.confusion_matrix} names={names} />
          </div>
        )}
        {x.weights && (
          <div>
            <div className="eyebrow" style={{ marginBottom: 8 }}>
              Pesos aprendidos (w) e bias (b)
            </div>
            <pre className="json">{`w = ${JSON.stringify(x.weights)}\nb = ${JSON.stringify(x.bias)}`}</pre>
            <p className="small muted">Para entradas normalizadas: ŷ = degrau(w·x + b).</p>
          </div>
        )}
      </div>

      {x.images && (
        <div className="section">
          {x.images.original && <ImageRow label="original" images={x.images.original} />}
          {x.images.input && <ImageRow label="com ruído" images={x.images.input} />}
          {x.images.reconstruction && <ImageRow label="reconstrução" images={x.images.reconstruction} />}
          {x.images.generated && <ImageRow label="gerado (G)" images={x.images.generated} />}
        </div>
      )}
      {generated?.samples && generated.shape && (
        <div className="section">
          <div className="eyebrow">Novas amostras</div>
          <div className="img-row" style={{ marginTop: 8 }}>
            {generated.samples.map((s: number[][], i: number) => (
              <GrayImage key={i} pixels={s} scale={2.4} />
            ))}
          </div>
        </div>
      )}

      <details className="section">
        <summary className="eyebrow" style={{ cursor: "pointer" }}>
          Hiperparâmetros e configuração
        </summary>
        <pre className="json" style={{ marginTop: 8 }}>
          {JSON.stringify({ hyperparams: r.hyperparams, config: model?.config }, null, 2)}
        </pre>
      </details>
    </div>
  );
}
