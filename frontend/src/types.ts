export type Task =
  | "tabular_classification"
  | "tabular_regression"
  | "image_classification"
  | "sequence_regression"
  | "sequence_classification"
  | "unsupervised";

export interface ProjectSummary {
  id: number;
  name: string;
  description: string;
  created_at: string;
  n_datasets: number;
  n_models: number;
  n_runs: number;
}

export interface DatasetMeta {
  task: Task;
  n_samples: number;
  input_shape: number[];
  n_classes?: number;
  class_names?: string[];
  class_counts?: number[];
  feature_names?: string[];
  description?: string;
  target_stats?: Record<string, number>;
  note?: string;
}

export interface Dataset {
  id: number;
  project_id: number;
  name: string;
  source: string;
  params: Record<string, unknown>;
  meta: DatasetMeta;
  created_at: string;
}

export interface Model {
  id: number;
  project_id: number;
  name: string;
  architecture: string;
  config: Record<string, unknown>;
  created_at: string;
}

export type HistoryEntry = Record<string, number>;

export interface Run {
  id: number;
  project_id: number;
  model_id: number;
  dataset_id: number;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  hyperparams: Record<string, unknown>;
  history: HistoryEntry[];
  metrics: Record<string, any>;
  extras?: Record<string, any>;
  error?: string | null;
  created_at: string;
  finished_at?: string | null;
}

export interface Project extends Omit<ProjectSummary, "n_datasets" | "n_models" | "n_runs"> {
  datasets: Dataset[];
  models: Model[];
  runs: Run[];
}

export interface Architecture {
  name: string;
  aula: number;
  description: string;
  tasks: Task[];
  default_config: Record<string, unknown>;
  config_doc: Record<string, string> | string;
  default_hyperparams: Record<string, unknown>;
}

export interface CatalogEntry {
  source: string;
  aula: number;
  description: string;
}

/** Eventos normalizados que o backend repassa do stream-json do Claude Code. */
export type ChatEvent =
  | { type: "harness"; command: string[] }
  | { type: "init"; session_id: string; model: string; tools: string[]; mcp_servers: { name: string; status: string }[] }
  | { type: "text_delta"; text: string }
  | { type: "assistant_text"; text: string }
  | { type: "tool_use"; id: string; name: string; input: unknown }
  | { type: "tool_result"; tool_use_id: string; content: string; is_error: boolean }
  | { type: "result"; session_id: string; is_error: boolean; duration_ms: number; num_turns: number; total_cost_usd: number; usage: any; result: string }
  | { type: "error"; message: string }
  | { type: "done" };

export interface FileInfo {
  path: string;
  filename: string;
  format: string;
  size_kb: number;
  n_rows?: number;
  n_columns?: number;
  columns?: { name: string; kind: "numeric" | "categorical"; n_unique: number; n_missing: number; examples: string[] }[];
  arrays?: Record<string, { shape: number[]; dtype: string }>;
}

export interface KaggleDataset {
  ref: string;
  title: string;
  subtitle: string;
  size_mb: number;
  downloads: number;
  votes: number;
  usability: number;
  license: string;
  updated: string;
  url: string;
}
