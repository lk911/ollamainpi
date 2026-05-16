/**
 * Inference engine wrapping node-llama-cpp.
 *
 * - Lazy-loads the Llama runtime on first use (heavy native init).
 * - Caches a single loaded model + context so successive inferences are fast.
 * - Reloads when the requested model id changes.
 * - Supports GPU acceleration via node-llama-cpp's `gpu: "auto"` selector.
 */
import type { RegisteredModel } from "./config.ts";
import { loadSettings } from "./config.ts";
import { getModel } from "./model-manager.ts";

// node-llama-cpp is heavy and binds native code; import types only at the
// top level and load the runtime lazily.
type LlamaRuntime = Awaited<ReturnType<typeof loadLlama>>;
type LoadedModel = {
  id: string;
  model: any; // LlamaModel
  context: any; // LlamaContext
};

let llamaPromise: Promise<any> | undefined;
let loaded: LoadedModel | undefined;

async function loadLlama() {
  if (!llamaPromise) {
    llamaPromise = import("node-llama-cpp").then((m) => m.getLlama({ gpu: "auto" as any }));
  }
  return llamaPromise;
}

/**
 * Make sure the given model is loaded. If a different model is currently
 * loaded, dispose of it first to free GPU memory.
 */
export async function ensureLoaded(id: string): Promise<LoadedModel> {
  if (loaded && loaded.id === id) return loaded;

  const reg = await getModel(id);
  if (!reg) throw new Error(`Unknown model "${id}". Use /llm:pull to download it first.`);

  const settings = await loadSettings();
  const llama: any = await loadLlama();

  // Dispose previous
  if (loaded) {
    try {
      await loaded.context?.dispose?.();
      await loaded.model?.dispose?.();
    } catch {
      /* ignore */
    }
    loaded = undefined;
  }

  const gpuLayers = reg.gpuLayers ?? settings.defaultGpuLayers;
  const contextSize = reg.contextSize ?? settings.defaultContextSize;

  const model = await llama.loadModel({
    modelPath: reg.path,
    gpuLayers: gpuLayers === "auto" || gpuLayers === "max" ? undefined : gpuLayers,
  });

  const context = await model.createContext({ contextSize });
  loaded = { id, model, context };
  return loaded;
}

export async function disposeLoaded(): Promise<void> {
  if (!loaded) return;
  try {
    await loaded.context?.dispose?.();
    await loaded.model?.dispose?.();
  } catch {
    /* ignore */
  }
  loaded = undefined;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface CompletionOptions {
  temperature?: number;
  topP?: number;
  topK?: number;
  maxTokens?: number;
  stop?: string[];
  signal?: AbortSignal;
  onToken?: (token: string) => void;
}

/**
 * Chat completion. Streams tokens via onToken if provided and also returns the
 * full assembled string.
 */
export async function chatComplete(
  id: string,
  messages: ChatMessage[],
  options: CompletionOptions = {},
): Promise<string> {
  const { context, model } = await ensureLoaded(id);
  const { LlamaChatSession } = await import("node-llama-cpp");

  // Pull the first system message out (LlamaChatSession accepts a single one).
  const systemMessage = messages.find((m) => m.role === "system")?.content;
  const dialog = messages.filter((m) => m.role !== "system");

  const session = new LlamaChatSession({
    contextSequence: context.getSequence(),
    systemPrompt: systemMessage,
  });

  // Replay all but the final user turn as conversational history so the model
  // sees the full context. We only call prompt() with the final user message.
  for (let i = 0; i < dialog.length - 1; i += 2) {
    const u = dialog[i];
    const a = dialog[i + 1];
    if (u?.role === "user" && a?.role === "assistant") {
      // LlamaChatSession lacks a public history-injection API; the simplest
      // portable approach is to prepend prior turns into the next prompt.
      // We do that by accumulating them into a single user prompt below.
    }
  }

  // Build a single prompt that includes the conversational history.
  const historyText = dialog
    .slice(0, -1)
    .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`)
    .join("\n");
  const lastUser = [...dialog].reverse().find((m) => m.role === "user");
  if (!lastUser) throw new Error("chatComplete requires at least one user message.");

  const promptText = historyText ? `${historyText}\nUser: ${lastUser.content}` : lastUser.content;

  let full = "";
  const response = await session.prompt(promptText, {
    temperature: options.temperature,
    topP: options.topP,
    topK: options.topK,
    maxTokens: options.maxTokens,
    customStopTriggers: options.stop,
    signal: options.signal,
    onTextChunk: (chunk: string) => {
      full += chunk;
      options.onToken?.(chunk);
    },
  } as any);

  // Some node-llama-cpp versions return the full text, some only stream. Fall
  // back to whichever has content.
  return response || full;
  // Keep `model` referenced so TS doesn't warn it's unused.
  void model;
}

/**
 * Tokenize a piece of text using the currently loaded model. Useful for
 * /llm:tokens to estimate input size.
 */
export async function tokenize(id: string, text: string): Promise<number> {
  const { model } = await ensureLoaded(id);
  const tokens: ArrayLike<unknown> = model.tokenize(text);
  return tokens.length;
}
