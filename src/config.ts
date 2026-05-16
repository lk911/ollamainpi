/**
 * Config and storage paths for ollamainpi.
 *
 * Models and metadata live under ~/.pi/ollamainpi/ to keep them isolated from
 * any host Ollama installation and to make uninstall trivial.
 */
import { homedir } from "node:os";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join } from "node:path";

export const OLLAMAINPI_HOME = join(homedir(), ".pi", "ollamainpi");
export const MODELS_DIR = join(OLLAMAINPI_HOME, "models");
export const REGISTRY_FILE = join(OLLAMAINPI_HOME, "registry.json");
export const SETTINGS_FILE = join(OLLAMAINPI_HOME, "settings.json");

export interface RegisteredModel {
  /** Short tag the user uses, e.g. "llama-3.2-3b" or "qwen2.5-coder:7b". */
  id: string;
  /** Display name. */
  name: string;
  /** Absolute path to the GGUF file on disk. */
  path: string;
  /** Source URL the model was pulled from (huggingface direct URL). */
  source?: string;
  /** Context length the user wants to use; if omitted, model default. */
  contextSize?: number;
  /** GPU layers to offload (-1 = all, 0 = CPU only). */
  gpuLayers?: number | "max" | "auto";
  /** Approximate file size in bytes (for /llm:list). */
  sizeBytes?: number;
  /** When the model was added. */
  addedAt: number;
}

export interface Registry {
  models: RegisteredModel[];
}

export interface Settings {
  /** Port for the local OpenAI-compatible HTTP server. */
  serverPort: number;
  /** Whether to auto-start the HTTP server on session_start. */
  autoStartServer: boolean;
  /** Default GPU layers strategy for newly pulled models. */
  defaultGpuLayers: number | "max" | "auto";
  /** Default context window used when a model doesn't specify one. */
  defaultContextSize: number;
}

const DEFAULT_SETTINGS: Settings = {
  serverPort: 11435, // intentionally different from Ollama's 11434
  autoStartServer: true,
  defaultGpuLayers: "auto",
  defaultContextSize: 4096,
};

export async function ensureDirs(): Promise<void> {
  await mkdir(MODELS_DIR, { recursive: true });
}

export async function loadRegistry(): Promise<Registry> {
  if (!existsSync(REGISTRY_FILE)) return { models: [] };
  try {
    const raw = await readFile(REGISTRY_FILE, "utf8");
    const parsed = JSON.parse(raw) as Registry;
    if (!Array.isArray(parsed.models)) return { models: [] };
    return parsed;
  } catch {
    return { models: [] };
  }
}

export async function saveRegistry(reg: Registry): Promise<void> {
  await ensureDirs();
  await writeFile(REGISTRY_FILE, JSON.stringify(reg, null, 2), "utf8");
}

export async function loadSettings(): Promise<Settings> {
  if (!existsSync(SETTINGS_FILE)) return { ...DEFAULT_SETTINGS };
  try {
    const raw = await readFile(SETTINGS_FILE, "utf8");
    return { ...DEFAULT_SETTINGS, ...(JSON.parse(raw) as Partial<Settings>) };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export async function saveSettings(s: Settings): Promise<void> {
  await ensureDirs();
  await writeFile(SETTINGS_FILE, JSON.stringify(s, null, 2), "utf8");
}
