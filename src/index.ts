/**
 * ollamainpi - Ollama-like local LLM manager for PI.
 *
 * What this extension does:
 *  - Manages a registry of local GGUF models under ~/.pi/ollamainpi/.
 *  - Downloads models from HuggingFace or a direct .gguf URL (/llm:pull).
 *  - Loads them via node-llama-cpp with GPU acceleration when available.
 *  - Exposes an OpenAI-compatible HTTP server on 127.0.0.1:11435.
 *  - Registers an "ollamainpi" provider with PI so local models show up in
 *    /model just like cloud models.
 *  - Provides slash commands (/llm:list, /llm:pull, /llm:rm, /llm:run, ...) and
 *    a local_llm tool so the agent itself can run a local model from inside
 *    a conversation.
 *
 * Async factory: we register the provider with the list of currently
 * downloaded models BEFORE pi finishes startup so `pi --list-models` works
 * and the model is selectable immediately.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import {
  ensureDirs,
  loadSettings,
  saveSettings,
  type RegisteredModel,
} from "./config.ts";
import {
  formatBytes,
  listModels,
  pullModel,
  removeModel,
  updateModel,
} from "./model-manager.ts";
import { chatComplete, disposeLoaded, ensureLoaded, tokenize } from "./inference.ts";
import { getServerPort, startServer, stopServer } from "./server.ts";

const PROVIDER_NAME = "ollamainpi";

/**
 * Build the provider config that PI consumes. We point it at our own local
 * HTTP server so PI's "openai-completions" code path can talk to us with no
 * special integration on its side.
 */
function buildProviderConfig(models: RegisteredModel[], port: number) {
  return {
    name: "Ollama in PI (local)",
    baseUrl: `http://127.0.0.1:${port}/v1`,
    // PI requires an apiKey field; the value is irrelevant for a local server
    // but cannot be empty. Use a literal sentinel.
    apiKey: "ollamainpi-local",
    api: "openai-completions" as const,
    models: models.map((m) => ({
      id: m.id,
      name: m.name || m.id,
      reasoning: false,
      input: ["text"] as ("text" | "image")[],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: m.contextSize ?? 4096,
      maxTokens: 4096,
    })),
  };
}

/**
 * Re-register the provider with the current registry. Call this any time the
 * set of models changes (pull, rm, config change).
 */
async function refreshProvider(pi: ExtensionAPI): Promise<void> {
  const port = getServerPort();
  if (!port) return; // server not up yet; will register when it starts
  const models = await listModels();
  pi.registerProvider(PROVIDER_NAME, buildProviderConfig(models, port) as any);
}

export default async function (pi: ExtensionAPI): Promise<void> {
  await ensureDirs();
  const settings = await loadSettings();

  // ----- Provider registration (eager so /model + --list-models work) -----
  // Register first with the configured port so the provider exists even
  // before the server has actually finished binding. The server starts a
  // moment later in session_start; by then refreshProvider() will replace
  // the registration with the live port if it differs.
  const initialModels = await listModels();
  pi.registerProvider(
    PROVIDER_NAME,
    buildProviderConfig(initialModels, settings.serverPort) as any,
  );

  // ----- Lifecycle: start/stop server -----
  pi.on("session_start", async (_event, ctx) => {
    const s = await loadSettings();
    if (!s.autoStartServer) return;
    try {
      await startServer({
        port: s.serverPort,
        log: (m) => {
          if (ctx.hasUI) ctx.ui.notify(m, "info");
        },
      });
      await refreshProvider(pi);
      if (ctx.hasUI) {
        const plural = initialModels.length === 1 ? "" : "s";
        ctx.ui.setStatus(
          "ollamainpi",
          `local LLMs: ${initialModels.length} model${plural} @ :${s.serverPort}`,
        );
      }
    } catch (err) {
      if (ctx.hasUI) {
        ctx.ui.notify(`ollamainpi server failed: ${(err as Error).message}`, "error");
      }
    }
  });

  pi.on("session_shutdown", async () => {
    await stopServer().catch(() => undefined);
    await disposeLoaded().catch(() => undefined);
  });

  // ----- Commands -----

  // /llm:list - show installed models
  pi.registerCommand("llm:list", {
    description: "List local LLM models managed by ollamainpi",
    handler: async (_args, ctx) => {
      const models = await listModels();
      if (models.length === 0) {
        ctx.ui.notify("No local models. Use /llm:pull <source> to download one.", "info");
        return;
      }
      const lines = models.map(
        (m) => `${m.id.padEnd(28)}  ${formatBytes(m.sizeBytes).padStart(8)}  ${m.path}`,
      );
      ctx.ui.notify(`Installed models:\n${lines.join("\n")}`, "info");
    },
  });

  // /llm:pull <source> [as <id>]
  pi.registerCommand("llm:pull", {
    description:
      "Download a GGUF model. Usage: /llm:pull hf:user/repo/file.gguf [as my-id]  or  /llm:pull https://.../file.gguf [as my-id]",
    handler: async (args, ctx) => {
      const raw = (args ?? "").trim();
      if (!raw) {
        ctx.ui.notify("Usage: /llm:pull <hf:user/repo/file.gguf | url> [as <id>]", "warning");
        return;
      }
      // Parse optional "as <id>"
      let source = raw;
      let id: string | undefined;
      const asMatch = raw.match(/^(.+?)\s+as\s+(\S+)\s*$/);
      if (asMatch) {
        source = asMatch[1];
        id = asMatch[2];
      }

      ctx.ui.setStatus("ollamainpi", `pulling ${source}...`);
      let lastLog = 0;
      try {
        const model = await pullModel({
          source,
          id,
          onProgress: (p) => {
            const now = Date.now();
            if (now - lastLog < 500) return;
            lastLog = now;
            const pct = p.percent !== undefined ? `${p.percent.toFixed(1)}%` : "?";
            const totalPart = p.total ? ` / ${formatBytes(p.total)}` : "";
            ctx.ui.setStatus(
              "ollamainpi",
              `pulling ${id ?? source}: ${pct} (${formatBytes(p.downloaded)}${totalPart})`,
            );
          },
        });
        ctx.ui.setStatus("ollamainpi", undefined);
        ctx.ui.notify(
          `Pulled ${model.id} (${formatBytes(model.sizeBytes)}) -> ${model.path}`,
          "info",
        );
        await refreshProvider(pi);
      } catch (err) {
        ctx.ui.setStatus("ollamainpi", undefined);
        ctx.ui.notify(`Pull failed: ${(err as Error).message}`, "error");
      }
    },
  });

  // /llm:rm <id>
  pi.registerCommand("llm:rm", {
    description: "Remove a local model. Usage: /llm:rm <id>",
    getArgumentCompletions: async (prefix: string) => {
      const models = await listModels();
      return models
        .filter((m) => m.id.startsWith(prefix))
        .map((m) => ({ value: m.id, label: m.id, description: formatBytes(m.sizeBytes) }));
    },
    handler: async (args, ctx) => {
      const id = (args ?? "").trim();
      if (!id) {
        ctx.ui.notify("Usage: /llm:rm <id>", "warning");
        return;
      }
      const ok = await ctx.ui.confirm("Remove model?", `Delete ${id} and its GGUF file from disk?`);
      if (!ok) return;
      const removed = await removeModel(id);
      if (!removed) {
        ctx.ui.notify(`No such model: ${id}`, "error");
        return;
      }
      ctx.ui.notify(`Removed ${id}`, "info");
      await refreshProvider(pi);
    },
  });

  // /llm:run <id> <prompt...>
  pi.registerCommand("llm:run", {
    description: "Run a one-shot prompt against a local model. Usage: /llm:run <id> <prompt>",
    getArgumentCompletions: async (prefix: string) => {
      const models = await listModels();
      return models
        .filter((m) => m.id.startsWith(prefix))
        .map((m) => ({ value: m.id, label: m.id }));
    },
    handler: async (args, ctx) => {
      const raw = (args ?? "").trim();
      const spaceIdx = raw.indexOf(" ");
      if (!raw || spaceIdx === -1) {
        ctx.ui.notify("Usage: /llm:run <id> <prompt>", "warning");
        return;
      }
      const id = raw.slice(0, spaceIdx);
      const prompt = raw.slice(spaceIdx + 1);
      ctx.ui.setStatus("ollamainpi", `running ${id}...`);
      try {
        const text = await chatComplete(id, [{ role: "user", content: prompt }]);
        ctx.ui.setStatus("ollamainpi", undefined);
        ctx.ui.notify(`[${id}]\n${text}`, "info");
      } catch (err) {
        ctx.ui.setStatus("ollamainpi", undefined);
        ctx.ui.notify(`Inference failed: ${(err as Error).message}`, "error");
      }
    },
  });

  // /llm:config <id> [ctx=<n>] [gpu=<n|auto|max>]
  pi.registerCommand("llm:config", {
    description: "Configure a local model. Usage: /llm:config <id> ctx=<n> gpu=<n|auto|max>",
    getArgumentCompletions: async (prefix: string) => {
      const models = await listModels();
      return models
        .filter((m) => m.id.startsWith(prefix))
        .map((m) => ({ value: m.id, label: m.id }));
    },
    handler: async (args, ctx) => {
      const parts = (args ?? "").trim().split(/\s+/);
      const id = parts.shift();
      if (!id) {
        ctx.ui.notify("Usage: /llm:config <id> ctx=<n> gpu=<n|auto|max>", "warning");
        return;
      }
      const patch: Partial<RegisteredModel> = {};
      for (const tok of parts) {
        const [k, v] = tok.split("=");
        if (k === "ctx") patch.contextSize = Number(v);
        else if (k === "gpu") {
          patch.gpuLayers = v === "auto" || v === "max" ? v : Number(v);
        }
      }
      const updated = await updateModel(id, patch);
      if (!updated) {
        ctx.ui.notify(`No such model: ${id}`, "error");
        return;
      }
      ctx.ui.notify(
        `Updated ${id}: ctx=${updated.contextSize ?? "default"} gpu=${updated.gpuLayers ?? "auto"}`,
        "info",
      );
      // Force a reload on next inference so new settings take effect.
      await disposeLoaded();
      await refreshProvider(pi);
    },
  });

  // /llm:server start|stop|status
  pi.registerCommand("llm:server", {
    description: "Control the local OpenAI-compatible server: start | stop | status",
    getArgumentCompletions: (prefix: string) =>
      ["start", "stop", "status"]
        .filter((v) => v.startsWith(prefix))
        .map((v) => ({ value: v, label: v })),
    handler: async (args, ctx) => {
      const cmd = (args ?? "").trim();
      const s = await loadSettings();
      if (cmd === "start") {
        await startServer({ port: s.serverPort, log: (m) => ctx.ui.notify(m, "info") });
        await refreshProvider(pi);
      } else if (cmd === "stop") {
        await stopServer();
        ctx.ui.notify("ollamainpi server stopped", "info");
      } else {
        const port = getServerPort();
        ctx.ui.notify(
          port ? `running on http://127.0.0.1:${port}` : "not running",
          "info",
        );
      }
    },
  });

  // /llm:port <number>
  pi.registerCommand("llm:port", {
    description: "Set the port for the local server. Usage: /llm:port <number>",
    handler: async (args, ctx) => {
      const port = Number((args ?? "").trim());
      if (!Number.isInteger(port) || port <= 0 || port > 65535) {
        ctx.ui.notify("Usage: /llm:port <1..65535>", "warning");
        return;
      }
      const s = await loadSettings();
      s.serverPort = port;
      await saveSettings(s);
      await stopServer();
      await startServer({ port, log: (m) => ctx.ui.notify(m, "info") });
      await refreshProvider(pi);
    },
  });

  // /llm:tokens <id> <text>
  pi.registerCommand("llm:tokens", {
    description: "Tokenize text with a local model to estimate input size",
    handler: async (args, ctx) => {
      const raw = (args ?? "").trim();
      const idx = raw.indexOf(" ");
      if (idx === -1) {
        ctx.ui.notify("Usage: /llm:tokens <id> <text>", "warning");
        return;
      }
      const id = raw.slice(0, idx);
      const text = raw.slice(idx + 1);
      try {
        const n = await tokenize(id, text);
        ctx.ui.notify(`${n} tokens`, "info");
      } catch (err) {
        ctx.ui.notify(`tokenize failed: ${(err as Error).message}`, "error");
      }
    },
  });

  // ----- A tool the agent itself can call -----
  pi.registerTool({
    name: "local_llm",
    label: "Local LLM",
    description:
      "Run a prompt against a locally hosted GGUF model managed by ollamainpi. Use for cheap drafting, classification, or offline inference. Does not access the internet.",
    promptSnippet: "Run a prompt against a local GGUF model (free, offline).",
    promptGuidelines: [
      "Use local_llm when the user explicitly asks to run a local model, or when the task is cheap drafting or classification that does not need the primary model.",
    ],
    parameters: Type.Object({
      model: Type.String({ description: "Local model id, as returned by /llm:list" }),
      prompt: Type.String({ description: "User prompt to send to the local model" }),
      system: Type.Optional(Type.String({ description: "Optional system prompt" })),
      maxTokens: Type.Optional(Type.Number({ description: "Optional max tokens" })),
      temperature: Type.Optional(
        Type.Number({ description: "Sampling temperature (default 0.7)" }),
      ),
    }),
    async execute(_toolCallId, params, signal, onUpdate, _ctx) {
      const messages = [
        ...(params.system ? [{ role: "system" as const, content: params.system }] : []),
        { role: "user" as const, content: params.prompt },
      ];
      // Ensure model is loaded once and stream tokens for progress.
      await ensureLoaded(params.model);
      let buffer = "";
      const text = await chatComplete(params.model, messages, {
        signal,
        temperature: params.temperature,
        maxTokens: params.maxTokens,
        onToken: (tok) => {
          buffer += tok;
          onUpdate?.({ content: [{ type: "text", text: buffer }] });
        },
      });
      return {
        content: [{ type: "text", text: text || buffer }],
        details: { model: params.model, output: text || buffer },
      };
    },
  });

  // ----- Flag to disable auto-starting the server on launch -----
  pi.registerFlag("no-ollamainpi-server", {
    description: "Do not auto-start the local ollamainpi HTTP server",
    type: "boolean",
    default: false,
  } as any);

  if ((pi as any).getFlag?.("no-ollamainpi-server")) {
    const s = await loadSettings();
    s.autoStartServer = false;
    await saveSettings(s);
  }
}
