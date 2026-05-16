/**
 * Model management: download, list, remove, info.
 *
 * Models are GGUF files. We support two source forms in `pull`:
 *   1. Direct URL ending in .gguf
 *   2. HuggingFace shorthand "hf:user/repo/file.gguf"
 *
 * Downloads are streamed to disk with progress reporting.
 */
import { createWriteStream } from "node:fs";
import { stat, unlink, rename } from "node:fs/promises";
import { join, basename } from "node:path";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";
import {
  MODELS_DIR,
  ensureDirs,
  loadRegistry,
  saveRegistry,
  type RegisteredModel,
} from "./config.ts";

export interface PullProgress {
  /** Bytes downloaded so far. */
  downloaded: number;
  /** Total bytes (may be undefined if server didn't send Content-Length). */
  total?: number;
  /** Percent 0..100, only when total is known. */
  percent?: number;
}

export type ProgressCallback = (p: PullProgress) => void;

/**
 * Resolve a user-facing source string to a download URL + suggested id.
 */
export function resolveSource(source: string): { url: string; id: string; filename: string } {
  // hf:user/repo/file.gguf -> https://huggingface.co/user/repo/resolve/main/file.gguf
  if (source.startsWith("hf:")) {
    const rest = source.slice(3);
    const parts = rest.split("/");
    if (parts.length < 3) {
      throw new Error(`Invalid hf: source. Expected hf:user/repo/file.gguf, got ${source}`);
    }
    const user = parts[0];
    const repo = parts[1];
    const file = parts.slice(2).join("/");
    const url = `https://huggingface.co/${user}/${repo}/resolve/main/${file}`;
    const id = `${repo}:${stripExt(basename(file))}`.toLowerCase();
    return { url, id, filename: basename(file) };
  }

  // Plain URL
  try {
    const parsed = new URL(source);
    const filename = basename(parsed.pathname) || "model.gguf";
    if (!filename.toLowerCase().endsWith(".gguf")) {
      throw new Error(`URL does not look like a .gguf file: ${source}`);
    }
    const id = stripExt(filename).toLowerCase();
    return { url: parsed.toString(), id, filename };
  } catch (err) {
    throw new Error(`Could not parse model source "${source}": ${(err as Error).message}`);
  }
}

function stripExt(name: string): string {
  return name.replace(/\.gguf$/i, "");
}

/**
 * Download a GGUF model and register it. If `id` already exists, the call
 * fails unless `overwrite` is true.
 */
export async function pullModel(opts: {
  source: string;
  id?: string;
  overwrite?: boolean;
  onProgress?: ProgressCallback;
  signal?: AbortSignal;
}): Promise<RegisteredModel> {
  await ensureDirs();
  const { url, id: defaultId, filename } = resolveSource(opts.source);
  const id = opts.id ?? defaultId;

  const reg = await loadRegistry();
  const existing = reg.models.find((m) => m.id === id);
  if (existing && !opts.overwrite) {
    throw new Error(`Model "${id}" already exists. Use overwrite to replace.`);
  }

  const finalPath = join(MODELS_DIR, `${sanitize(id)}.gguf`);
  const tempPath = `${finalPath}.part`;

  const res = await fetch(url, { signal: opts.signal, redirect: "follow" });
  if (!res.ok || !res.body) {
    throw new Error(`Download failed (${res.status} ${res.statusText}) for ${url}`);
  }

  const totalHeader = res.headers.get("content-length");
  const total = totalHeader ? Number(totalHeader) : undefined;
  let downloaded = 0;

  // Wrap the web stream into a Node stream and tee progress.
  const nodeStream = Readable.fromWeb(res.body as unknown as import("node:stream/web").ReadableStream);
  nodeStream.on("data", (chunk: Buffer) => {
    downloaded += chunk.length;
    opts.onProgress?.({
      downloaded,
      total,
      percent: total ? Math.min(100, (downloaded / total) * 100) : undefined,
    });
  });

  try {
    await pipeline(nodeStream, createWriteStream(tempPath));
  } catch (err) {
    await unlink(tempPath).catch(() => undefined);
    throw err;
  }

  await rename(tempPath, finalPath);
  const stats = await stat(finalPath);

  const model: RegisteredModel = {
    id,
    name: filename,
    path: finalPath,
    source: url,
    sizeBytes: stats.size,
    addedAt: Date.now(),
  };

  const next = {
    models: [...reg.models.filter((m) => m.id !== id), model],
  };
  await saveRegistry(next);
  return model;
}

export async function listModels(): Promise<RegisteredModel[]> {
  const reg = await loadRegistry();
  return reg.models.slice().sort((a, b) => a.id.localeCompare(b.id));
}

export async function getModel(id: string): Promise<RegisteredModel | undefined> {
  const reg = await loadRegistry();
  return reg.models.find((m) => m.id === id);
}

export async function removeModel(id: string, opts: { deleteFile?: boolean } = {}): Promise<boolean> {
  const reg = await loadRegistry();
  const idx = reg.models.findIndex((m) => m.id === id);
  if (idx === -1) return false;
  const [model] = reg.models.splice(idx, 1);
  await saveRegistry(reg);
  if (opts.deleteFile !== false) {
    await unlink(model.path).catch(() => undefined);
  }
  return true;
}

export async function updateModel(id: string, patch: Partial<RegisteredModel>): Promise<RegisteredModel | undefined> {
  const reg = await loadRegistry();
  const idx = reg.models.findIndex((m) => m.id === id);
  if (idx === -1) return undefined;
  const updated = { ...reg.models[idx], ...patch, id }; // keep id immutable
  reg.models[idx] = updated;
  await saveRegistry(reg);
  return updated;
}

function sanitize(id: string): string {
  return id.replace(/[^A-Za-z0-9._-]+/g, "_");
}

export function formatBytes(bytes?: number): string {
  if (!bytes || bytes <= 0) return "?";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}
