import type { Architecture, CatalogEntry, ChatEvent, Dataset, FileInfo, KaggleDataset, Model, Project, ProjectSummary, Run } from "./types";

/** Formatos de base aceitos no upload (espelha backend/app/files.py). */
export const DATA_ACCEPT = ".csv,.tsv,.txt,.json,.jsonl,.xlsx,.xls,.parquet,.npz";

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method,
    headers: body instanceof FormData || body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body instanceof FormData ? body : body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = r.statusText;
    try {
      detail = (await r.json()).detail ?? detail;
    } catch {
      /* corpo não-JSON */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return r.json();
}

export const api = {
  approveAction: (runId: string, id: string, approved: boolean) => req("POST", `/api/harness/runs/${runId}/approvals/${id}`, {approved}),
  detectProviders: () => req<{id: string; name: string; configured: boolean; default_model: string; available: boolean; status: string; models: string[]}[]>("POST", "/api/providers/detect"),
  providers: () => req<{id: string; name: string; configured: boolean; default_model: string}[]>("GET", "/api/providers"),
  architectures: () => req<Architecture[]>("GET", "/api/architectures"),
  catalog: () => req<CatalogEntry[]>("GET", "/api/datasets/catalog"),
  projects: () => req<ProjectSummary[]>("GET", "/api/projects"),
  project: (id: number) => req<Project>("GET", `/api/projects/${id}`),
  createProject: (name: string, description = "") => req<Project>("POST", "/api/projects", { name, description }),
  deleteProject: (id: number) => req("DELETE", `/api/projects/${id}`),
  addDataset: (pid: number, source: string, name: string | null, params: object) =>
    req<Dataset>("POST", `/api/projects/${pid}/datasets`, { source, name, params }),
  uploadFile: (pid: number, form: FormData) => req<Dataset>("POST", `/api/projects/${pid}/datasets/upload`, form),
  stageUpload: (file: File) => {
    const f = new FormData();
    f.append("file", file);
    return req<FileInfo>("POST", "/api/uploads", f);
  },
  kaggleSearch: (q: string, fileType = "") =>
    req<KaggleDataset[]>("GET", `/api/kaggle/search?q=${encodeURIComponent(q)}${fileType ? `&file_type=${fileType}` : ""}`),
  previewDataset: (id: number, n = 8) => req<any>("GET", `/api/datasets/${id}/preview?n=${n}`),
  deleteDataset: (id: number) => req("DELETE", `/api/datasets/${id}`),
  createModel: (pid: number, name: string, architecture: string, config: object) =>
    req<Model>("POST", `/api/projects/${pid}/models`, { name, architecture, config }),
  deleteModel: (id: number) => req("DELETE", `/api/models/${id}`),
  startRun: (model_id: number, dataset_id: number, hyperparams: object) =>
    req<Run>("POST", "/api/runs", { model_id, dataset_id, hyperparams }),
  run: (id: number) => req<Run>("GET", `/api/runs/${id}`),
  cancelRun: (id: number) => req("POST", `/api/runs/${id}/cancel`),
  deleteRun: (id: number) => req("DELETE", `/api/runs/${id}`),
  predict: (id: number, inputs: unknown[]) => req<{predictions: number[]; labels?: string[] | null; probabilities?: number[][]}>("POST", `/api/runs/${id}/predict`, {inputs}),
  generate: (id: number, n = 16) => req<any>("POST", `/api/runs/${id}/generate?n=${n}`),
  harness: () => req<any>("GET", "/api/harness"),
};

/** POST /api/chat devolve Server-Sent Events; lemos o corpo como stream. */
export async function streamChat(
  body: { message: string; session_id?: string | null; project_id?: number | null; model?: string | null; provider?: string; history?: {role: string; content: string}[]; policy?: string },
  onEvent: (e: ChatEvent) => void,
  signal: AbortSignal,
) {
  const r = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.body) throw new Error("Sem corpo na resposta");
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i: number;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i);
      buf = buf.slice(i + 2);
      for (const line of chunk.split("\n")) {
        if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)));
      }
    }
  }
}
