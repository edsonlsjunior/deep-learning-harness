import { useEffect, useRef } from "react";

// Mesma paleta do material "Redes Neurais Lab" (blue, coral, green, violet, amber, cyan…)
export const PALETTE = ["#4777f4", "#f2735b", "#39b98a", "#8067e8", "#f5b73f", "#22c8dd", "#ff5fa2", "#8bd450", "#10233d", "#8b98ad"];

/* ------------------------------------------------------------ curvas */
export function LineChart({
  series,
  height = 180,
  title,
  yLabel,
  xLabel = "época",
}: {
  series: { name: string; values: number[]; color: string; dashed?: boolean }[];
  height?: number;
  title?: string;
  yLabel?: string;
  xLabel?: string;
}) {
  const W = 520;
  const H = height;
  const pad = { l: 46, r: 12, t: title ? 38 : 26, b: 24 };
  const all = series.flatMap((s) => s.values).filter((v) => Number.isFinite(v));
  const n = Math.max(...series.map((s) => s.values.length), 1);
  if (!all.length) return <div className="muted small">sem dados ainda…</div>;
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  if (hi - lo < 1e-9) {
    hi += 0.5;
    lo -= 0.5;
  }
  const x = (i: number) => pad.l + (n <= 1 ? 0 : (i / (n - 1)) * (W - pad.l - pad.r));
  const y = (v: number) => pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b);
  const ticks = [lo, (lo + hi) / 2, hi];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart">
      {title && (
        <text x={pad.l} y={14} className="chart-title">
          {title}
        </text>
      )}
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} className="grid" />
          <text x={pad.l - 6} y={y(t) + 4} textAnchor="end" className="tick">
            {fmt(t)}
          </text>
        </g>
      ))}
      <text x={W - pad.r} y={H - 6} textAnchor="end" className="tick">
        {xLabel} {n}
      </text>
      {yLabel && (
        <text x={10} y={H / 2} className="tick" transform={`rotate(-90 10 ${H / 2})`} textAnchor="middle">
          {yLabel}
        </text>
      )}
      {series.map((s) => (
        <polyline
          key={s.name}
          fill="none"
          stroke={s.color}
          strokeWidth={2}
          strokeDasharray={s.dashed ? "5 4" : undefined}
          points={s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ")}
        />
      ))}
      {series.map((s, i) => (
        <g key={s.name} transform={`translate(${W - pad.r - 110 * (series.length - i)}, ${title ? 18 : 4})`}>
          <rect width={10} height={3} y={4} fill={s.color} />
          <text x={14} y={9} className="tick">
            {s.name}
          </text>
        </g>
      ))}
    </svg>
  );
}

export function fmt(v: number) {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 0.01 || a === 0) return v.toFixed(3);
  return v.toExponential(1);
}

/* ------------------------------------------------------------ canvas helpers */
function useCanvas(draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void, deps: unknown[]) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d")!;
    ctx.clearRect(0, 0, c.width, c.height);
    draw(ctx, c.width, c.height);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return ref;
}

function hexA(hex: string, a: number) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

/** Fronteira de decisão: grade de classes previstas + pontos reais. */
export function BoundaryPlot({ plot, size = 320 }: { plot: any; size?: number }) {
  const ref = useCanvas(
    (ctx, w, h) => {
      const g: number[][] = plot.grid;
      const rows = g.length;
      const cols = g[0].length;
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
          ctx.fillStyle = hexA(PALETTE[g[r][c] % PALETTE.length], 0.22);
          ctx.fillRect((c / cols) * w, h - ((r + 1) / rows) * h, w / cols + 1, h / rows + 1);
        }
      const [x0, x1] = plot.x_range;
      const [y0, y1] = plot.y_range;
      plot.points.forEach((p: number[], i: number) => {
        ctx.beginPath();
        ctx.arc(((p[0] - x0) / (x1 - x0)) * w, h - ((p[1] - y0) / (y1 - y0)) * h, plot.points.length < 10 ? 8 : 3.5, 0, 7);
        ctx.fillStyle = PALETTE[plot.labels[i] % PALETTE.length];
        ctx.fill();
        ctx.strokeStyle = "#ffffffcc";
        ctx.stroke();
      });
    },
    [plot],
  );
  return <canvas ref={ref} width={size} height={size} className="plot" />;
}

export function ScatterPlot({
  groups,
  size = 320,
}: {
  groups: { points: number[][]; color: string | ((i: number) => string); r?: number }[];
  size?: number;
}) {
  const ref = useCanvas(
    (ctx, w, h) => {
      const all = groups.flatMap((g) => g.points);
      if (!all.length) return;
      const xs = all.map((p) => p[0]);
      const ys = all.map((p) => p[1]);
      const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
      const px = (v: number) => 12 + ((v - x0) / (x1 - x0 || 1)) * (w - 24);
      const py = (v: number) => h - 12 - ((v - y0) / (y1 - y0 || 1)) * (h - 24);
      for (const g of groups)
        g.points.forEach((p, i) => {
          ctx.beginPath();
          ctx.arc(px(p[0]), py(p[1]), g.r ?? 3, 0, 7);
          ctx.fillStyle = typeof g.color === "string" ? g.color : g.color(i);
          ctx.fill();
        });
    },
    [groups],
  );
  return <canvas ref={ref} width={size} height={size} className="plot" />;
}

/** Curva y=f(x) ajustada + pontos (regressão com 1 atributo). */
export function CurvePlot({ plot }: { plot: any }) {
  const W = 520;
  const H = 240;
  const xs: number[] = [...plot.x, ...plot.points];
  const ys: number[] = [...plot.y_pred, ...plot.targets];
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const px = (v: number) => 20 + ((v - x0) / (x1 - x0 || 1)) * (W - 40);
  const py = (v: number) => H - 20 - ((v - y0) / (y1 - y0 || 1)) * (H - 40);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart">
      {plot.points.map((x: number, i: number) => (
        <circle key={i} cx={px(x)} cy={py(plot.targets[i])} r={2.5} fill={PALETTE[0]} opacity={0.6} />
      ))}
      <polyline fill="none" stroke={PALETTE[1]} strokeWidth={2.5} points={plot.x.map((x: number, i: number) => `${px(x)},${py(plot.y_pred[i])}`).join(" ")} />
    </svg>
  );
}

/** Imagem em tons de cinza (matriz HxW com valores 0..1). */
export function GrayImage({ pixels, scale = 4 }: { pixels: number[][]; scale?: number }) {
  const h = pixels.length;
  const w = pixels[0].length;
  const s = Math.max(1, Math.round((scale * 28) / Math.max(h, w)));
  const ref = useCanvas(
    (ctx) => {
      for (let r = 0; r < h; r++)
        for (let c = 0; c < w; c++) {
          const v = Math.round(Math.max(0, Math.min(1, pixels[r][c])) * 255);
          ctx.fillStyle = `rgb(${v},${v},${v})`;
          ctx.fillRect(c * s, r * s, s, s);
        }
    },
    [pixels],
  );
  return <canvas ref={ref} width={w * s} height={h * s} className="img" />;
}

export function ImageRow({ label, images }: { label: string; images: number[][][] }) {
  return (
    <div className="img-row">
      <span className="muted small img-label">{label}</span>
      {images.map((im, i) => (
        <GrayImage key={i} pixels={im} scale={2.4} />
      ))}
    </div>
  );
}

export function ConfusionMatrix({ cm, names }: { cm: number[][]; names?: string[] }) {
  const max = Math.max(...cm.flat(), 1);
  return (
    <table className="cm">
      <thead>
        <tr>
          <th className="muted small">real ↓ / previsto →</th>
          {cm.map((_, j) => (
            <th key={j}>{names?.[j] ?? j}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {cm.map((row, i) => (
          <tr key={i}>
            <th>{names?.[i] ?? i}</th>
            {row.map((v, j) => (
              <td key={j} style={{ background: hexA(i === j ? "#39b98a" : "#f2735b", 0.12 + (0.75 * v) / max) }}>
                {v}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function Sparkline({ values, color = PALETTE[0] }: { values: number[]; color?: string }) {
  if (values.length < 2) return null;
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * 80},${20 - ((v - lo) / (hi - lo || 1)) * 18 - 1}`).join(" ");
  return (
    <svg width={80} height={20} className="spark">
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} />
    </svg>
  );
}
