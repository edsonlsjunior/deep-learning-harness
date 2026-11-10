import { useEffect, useRef, useState } from "react";
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
  const wgan = arch === "gan" && model?.config.objective === "wgan_gp";
  const h = r.history;
  const m = r.metrics || {};
  const x = r.extras || {};
  const names = ds?.meta.class_names;

  const metricCards: [string, string][] = [];
  if (m.accuracy !== undefined) metricCards.push(["acurácia", `${(m.accuracy * 100).toFixed(1)}%`]);
  for (const k of ["r2", "rmse", "mae", "reconstruction_mse", "anomaly_threshold_p95"]) if (m[k] !== undefined) metricCards.push([k, fmt(m[k])]);
  const baseline = (ds?.meta as any)?.baseline_persistence_rmse;
  if (baseline !== undefined && m.rmse !== undefined) {
    const unit = (ds?.meta as any)?.unit ?? "";
    metricCards.push(["rmse ingênuo (amanhã = hoje)", `${fmt(baseline)} ${unit}`]);
    metricCards.push(["ganho sobre o ingênuo", `${(((baseline - m.rmse) / baseline) * 100).toFixed(1)}%`]);
  }
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
              title={wgan ? "WGAN-GP · crítico x gerador" : "GAN · discriminador x gerador"}
              series={[
                { name: wgan ? "crítico" : "D (perito)", values: h.map((e) => e.d_loss), color: PALETTE[3] },
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
              xLabel="amostra"
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

      {r.status === "completed" && ds?.meta.task === "image_classification" && arch !== "autoencoder" && arch !== "gan" && (
        <DigitPredictor runId={r.id} classNames={names} />
      )}
      {r.status === "completed" && ["rnn", "lstm", "gru"].includes(arch) && ds?.meta.task.startsWith("sequence_") && (
        ds.meta.word_prediction
          ? <WordPredictor runId={r.id} spec={ds.meta.word_prediction} classNames={names} />
          : <SequencePredictor runId={r.id} task={ds.meta.task} inputShape={ds.meta.input_shape} classNames={names} />
      )}
      {r.status === "completed" && arch === "autoencoder" && ds && (
        <AutoencoderTester runId={r.id} inputShape={ds.meta.input_shape} task={ds.meta.task} />
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

function DigitPredictor({ runId, classNames }: { runId: number; classNames?: string[] }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const [result, setResult] = useState<PredictorResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const clear = () => {
    const c = canvas.current;
    if (!c) return;
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, c.width, c.height);
    setResult(null);
    setError("");
  };
  useEffect(clear, [runId]);

  const point = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const c = canvas.current!;
    const box = c.getBoundingClientRect();
    return {x: (e.clientX - box.left) * c.width / box.width, y: (e.clientY - box.top) * c.height / box.height};
  };
  const start = (e: React.PointerEvent<HTMLCanvasElement>) => {
    drawing.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    const p = point(e), ctx = e.currentTarget.getContext("2d")!;
    ctx.beginPath(); ctx.moveTo(p.x, p.y);
  };
  const move = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawing.current) return;
    const p = point(e), ctx = e.currentTarget.getContext("2d")!;
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 22; ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.lineTo(p.x, p.y); ctx.stroke();
  };
  const predict = async () => {
    const c = canvas.current!;
    const small = document.createElement("canvas"); small.width = 28; small.height = 28;
    const sctx = small.getContext("2d")!;
    sctx.drawImage(c, 0, 0, 28, 28);
    const rgba = sctx.getImageData(0, 0, 28, 28).data;
    const image = Array.from({length: 28}, (_, y) => Array.from({length: 28}, (_, x) => rgba[(y * 28 + x) * 4] / 255));
    setBusy(true); setError("");
    try { setResult(await api.predict(runId, image)); }
    catch (e) { setError(e instanceof Error ? e.message : "Não foi possível classificar o desenho."); }
    finally { setBusy(false); }
  };
  const probs = result?.probabilities?.[0] ?? [];
  const top = probs.map((p, i) => ({label: classNames?.[i] ?? String(i), p})).sort((a, b) => b.p - a.p).slice(0, 3);
  return <div className="section digit-predictor">
    <div className="eyebrow">Testar o modelo</div>
    <h4>Desenhe um algarismo</h4>
    <p className="small muted">Desenhe em branco no quadro preto, como nas imagens do MNIST.</p>
    <div className="row" style={{alignItems: "flex-start", flexWrap: "wrap"}}>
      <canvas ref={canvas} width={280} height={280} className="digit-canvas" onPointerDown={start} onPointerMove={move} onPointerUp={() => drawing.current = false} onPointerCancel={() => drawing.current = false} />
      <div style={{minWidth: 180}}>
        <div className="row"><button className="btn primary" onClick={predict} disabled={busy}>{busy ? "Analisando…" : "Reconhecer"}</button><button className="btn ghost" onClick={clear}>Limpar</button></div>
        {result && <div className="prediction-result"><span>Predição</span><b>{result.labels?.[0] ?? result.predictions?.[0] ?? "—"}</b>{top.map(v => <div key={v.label} className="small">{v.label}: {(v.p * 100).toFixed(1)}%</div>)}</div>}
        {error && <div className="err small">{error}</div>}
      </div>
    </div>
  </div>;
}

type PredictorResult = { predictions?: number[]; labels?: string[] | null; probabilities?: number[][]; reconstruction_error?: number[]; latent?: number[][]; reconstruction?: number[][][][] | number[][] };

function numericInput(value: string, expected: number) {
  const values = value.trim().split(/[\s,;]+/).filter(Boolean).map(Number);
  if (values.some((v) => !Number.isFinite(v))) throw new Error("Use apenas números separados por vírgula.");
  if (values.length !== expected) throw new Error(`Esta rede espera ${expected} valor${expected === 1 ? "" : "es"}.`);
  return values;
}

function SequencePredictor({ runId, task, inputShape, classNames }: { runId: number; task: string; inputShape: number[]; classNames?: string[] }) {
  const size = inputShape.reduce((total, value) => total * value, 1);
  const steps = inputShape[0] ?? size;
  const features = inputShape[1] ?? 1;
  const example = Array.from({ length: size }, (_, i) => task === "sequence_classification" ? (i % 3 === 0 ? 1 : 0) : Math.sin((i / Math.max(1, size - 1)) * Math.PI * 2).toFixed(2)).join(", ");
  const [values, setValues] = useState(example);
  const [result, setResult] = useState<PredictorResult | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const predict = async () => {
    try {
      setBusy(true); setError("");
      setResult(await api.predict(runId, numericInput(values, size)));
    } catch (e) { setError(e instanceof Error ? e.message : "Não foi possível testar a sequência."); }
    finally { setBusy(false); }
  };
  const probabilities = result?.probabilities?.[0] ?? [];
  const top = probabilities.map((p, i) => ({ label: classNames?.[i] ?? String(i), p })).sort((a, b) => b.p - a.p).slice(0, 2);
  const regression = task === "sequence_regression";
  return <div className="section model-tester">
    <div className="eyebrow">Testar o modelo</div>
    <h4>{regression ? "Preveja o próximo valor da sequência" : "Classifique uma sequência de bits"}</h4>
    <p className="small muted">
      Informe {steps} passo{steps === 1 ? "" : "s"}{features > 1 ? ` com ${features} atributos cada` : ""}, separados por vírgula.
      {regression ? " O exemplo é uma onda normalizada para experimentar a inferência." : " A classe indica se a quantidade de 1s é par ou ímpar."}
    </p>
    <textarea className="sequence-input" value={values} onChange={(e) => { setValues(e.target.value); setResult(null); }} rows={3} aria-label="Valores da sequência" />
    <div className="row" style={{ marginTop: 8 }}>
      <button className="btn primary" onClick={predict} disabled={busy}>{busy ? "Analisando…" : "Testar sequência"}</button>
      <button className="btn ghost" onClick={() => { setValues(example); setResult(null); setError(""); }}>Usar exemplo</button>
    </div>
    {result && <div className="prediction-result compact-result">
      <span>{regression ? "Próximo valor previsto" : "Classificação"}</span>
      <b>{regression ? result.predictions?.[0] ?? "—" : result.labels?.[0] ?? result.predictions?.[0] ?? "—"}</b>
      {!regression && top.map((item) => <div key={item.label} className="small">{item.label}: {(item.p * 100).toFixed(1)}%</div>)}
    </div>}
    {error && <div className="err small">{error}</div>}
  </div>;
}

function WordPredictor({ runId, spec, classNames }: { runId: number; spec: { vocabulary: string[]; window: number; examples: string[] }; classNames?: string[] }) {
  const [phrase, setPhrase] = useState(spec.examples[0] ?? "");
  const [result, setResult] = useState<PredictorResult | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const encode = (text: string) => {
    const words = text.toLocaleLowerCase("pt-BR").trim().split(/\s+/).filter(Boolean).slice(-spec.window);
    if (!words.length) throw new Error("Escreva ao menos uma palavra.");
    const unknown = words.find((word) => !spec.vocabulary.includes(word));
    if (unknown) throw new Error(`A palavra “${unknown}” não faz parte deste corpus didático.`);
    const padIndex = spec.vocabulary.length;
    const padded = [null, ...words].slice(-spec.window);
    while (padded.length < spec.window) padded.unshift(null);
    return padded.map((word) => {
      const vector = Array(spec.vocabulary.length + 1).fill(0);
      vector[word === null ? padIndex : spec.vocabulary.indexOf(word)] = 1;
      return vector;
    });
  };
  const predict = async () => {
    try { setBusy(true); setError(""); setResult(await api.predict(runId, encode(phrase))); }
    catch (e) { setError(e instanceof Error ? e.message : "Não foi possível prever a próxima palavra."); }
    finally { setBusy(false); }
  };
  const top = (result?.probabilities?.[0] ?? []).map((p, i) => ({ word: classNames?.[i] ?? String(i), p })).sort((a, b) => b.p - a.p).slice(0, 3);
  return <div className="section model-tester">
    <div className="eyebrow">Testar o modelo</div>
    <h4>Preveja a próxima palavra</h4>
    <p className="small muted">Escreva até {spec.window} palavras do corpus. Elas são convertidas em vetores one-hot e enviadas à rede recorrente.</p>
    <input className="word-input" value={phrase} onChange={(e) => { setPhrase(e.target.value); setResult(null); }} aria-label="Contexto de palavras" />
    <div className="row" style={{ marginTop: 8, flexWrap: "wrap" }}>
      <button className="btn primary" onClick={predict} disabled={busy}>{busy ? "Prevendo…" : "Prever próxima palavra"}</button>
      {spec.examples.map((example) => <button key={example} className="btn ghost" onClick={() => { setPhrase(example); setResult(null); setError(""); }}>{example}</button>)}
    </div>
    {result && <div className="prediction-result compact-result"><span>Próxima palavra sugerida</span><b>{result.labels?.[0] ?? result.predictions?.[0] ?? "—"}</b>{top.map((item) => <div key={item.word} className="small">{item.word}: {(item.p * 100).toFixed(1)}%</div>)}</div>}
    {error && <div className="err small">{error}</div>}
  </div>;
}

function AutoencoderTester({ runId, inputShape, task }: { runId: number; inputShape: number[]; task: string }) {
  const image = task === "image_classification" && inputShape.length === 3;
  return image
    ? <AutoencoderImageTester runId={runId} shape={inputShape} />
    : <AutoencoderVectorTester runId={runId} inputShape={inputShape} />;
}

function AutoencoderImageTester({ runId, shape }: { runId: number; shape: number[] }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const height = shape[1], width = shape[2];
  const [input, setInput] = useState<number[][] | null>(null);
  const [result, setResult] = useState<PredictorResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const clear = () => {
    const c = canvas.current; if (!c) return;
    const ctx = c.getContext("2d")!; ctx.fillStyle = "#000"; ctx.fillRect(0, 0, c.width, c.height);
    setInput(null); setResult(null); setError("");
  };
  useEffect(clear, [runId]);
  const point = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const c = canvas.current!, box = c.getBoundingClientRect();
    return { x: (e.clientX - box.left) * c.width / box.width, y: (e.clientY - box.top) * c.height / box.height };
  };
  const start = (e: React.PointerEvent<HTMLCanvasElement>) => {
    drawing.current = true; e.currentTarget.setPointerCapture(e.pointerId);
    const p = point(e), ctx = e.currentTarget.getContext("2d")!; ctx.beginPath(); ctx.moveTo(p.x, p.y);
  };
  const move = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawing.current) return;
    const p = point(e), ctx = e.currentTarget.getContext("2d")!;
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 22; ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.lineTo(p.x, p.y); ctx.stroke();
  };
  const reconstruct = async () => {
    const c = canvas.current!, small = document.createElement("canvas"); small.width = width; small.height = height;
    const ctx = small.getContext("2d")!; ctx.drawImage(c, 0, 0, width, height);
    const rgba = ctx.getImageData(0, 0, width, height).data;
    const sample = Array.from({ length: height }, (_, y) => Array.from({ length: width }, (_, x) => rgba[(y * width + x) * 4] / 255));
    try { setBusy(true); setError(""); setInput(sample); setResult(await api.predict(runId, sample)); }
    catch (e) { setError(e instanceof Error ? e.message : "Não foi possível reconstruir o desenho."); }
    finally { setBusy(false); }
  };
  const reconstruction = result?.reconstruction?.[0];
  const rebuilt = Array.isArray(reconstruction?.[0]) ? reconstruction[0] as number[][] : null;
  return <div className="section model-tester">
    <div className="eyebrow">Testar o modelo</div>
    <h4>Desenhe e observe a reconstrução</h4>
    <p className="small muted">O encoder comprime seu desenho; o decoder tenta reconstruí-lo. Compare a imagem e o erro.</p>
    <div className="row" style={{ alignItems: "flex-start", flexWrap: "wrap" }}>
      <div><canvas ref={canvas} width={280} height={280} className="digit-canvas" onPointerDown={start} onPointerMove={move} onPointerUp={() => drawing.current = false} onPointerCancel={() => drawing.current = false} />
        <div className="row" style={{ marginTop: 8 }}><button className="btn primary" onClick={reconstruct} disabled={busy}>{busy ? "Reconstruindo…" : "Reconstruir"}</button><button className="btn ghost" onClick={clear}>Limpar</button></div></div>
      {input && rebuilt && <div className="ae-comparison"><div><span>entrada</span><GrayImage pixels={input} scale={10} /></div><div><span>reconstrução</span><GrayImage pixels={rebuilt} scale={10} /></div></div>}
      {result && <div className="prediction-result compact-result"><span>Erro de reconstrução (MSE)</span><b>{result.reconstruction_error?.[0]?.toFixed(6) ?? "—"}</b><div className="small muted">Espaço latente: {result.latent?.[0]?.map((v) => v.toFixed(2)).join(", ") ?? "—"}</div></div>}
    </div>
    {error && <div className="err small">{error}</div>}
  </div>;
}

function AutoencoderVectorTester({ runId, inputShape }: { runId: number; inputShape: number[] }) {
  const size = inputShape.reduce((total, value) => total * value, 1);
  const example = Array.from({ length: size }, () => "0").join(", ");
  const [values, setValues] = useState(example);
  const [result, setResult] = useState<PredictorResult | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const reconstruct = async () => {
    try { setBusy(true); setError(""); setResult(await api.predict(runId, numericInput(values, size))); }
    catch (e) { setError(e instanceof Error ? e.message : "Não foi possível reconstruir a amostra."); }
    finally { setBusy(false); }
  };
  const reconstruction = result?.reconstruction?.[0] as number[] | undefined;
  return <div className="section model-tester">
    <div className="eyebrow">Testar o modelo</div>
    <h4>Reconstrua uma amostra</h4>
    <p className="small muted">Informe os {size} valores de uma amostra para observar sua compressão e reconstrução.</p>
    <textarea className="sequence-input" value={values} onChange={(e) => { setValues(e.target.value); setResult(null); }} rows={3} aria-label="Valores da amostra" />
    <div className="row" style={{ marginTop: 8 }}><button className="btn primary" onClick={reconstruct} disabled={busy}>{busy ? "Reconstruindo…" : "Reconstruir"}</button><button className="btn ghost" onClick={() => { setValues(example); setResult(null); setError(""); }}>Limpar</button></div>
    {result && <div className="prediction-result compact-result"><span>Erro de reconstrução (MSE)</span><b>{result.reconstruction_error?.[0]?.toFixed(6) ?? "—"}</b><div className="small muted">Reconstrução: {reconstruction?.map((v) => v.toFixed(3)).join(", ") ?? "—"}</div></div>}
    {error && <div className="err small">{error}</div>}
  </div>;
}
