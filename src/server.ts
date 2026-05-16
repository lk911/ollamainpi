/**
 * OpenAI-compatible HTTP server.
 *
 * Implements the subset of endpoints PI's "openai-completions" provider needs:
 *   GET  /v1/models                 - list registered local models
 *   POST /v1/chat/completions       - chat completion (streaming + non-streaming)
 *
 * Plus a few convenience endpoints:
 *   GET  /health                    - { ok: true }
 *   GET  /v1/internal/loaded        - which model is currently resident in memory
 *
 * The server uses Node's built-in http module - no external deps.
 */
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { randomUUID } from "node:crypto";
import { chatComplete, type ChatMessage } from "./inference.ts";
import { listModels } from "./model-manager.ts";

let server: Server | undefined;
let activePort: number | undefined;

export interface StartOptions {
  port: number;
  /** Called with a friendly log message on lifecycle events. */
  log?: (msg: string) => void;
}

export async function startServer(opts: StartOptions): Promise<void> {
  if (server) return;
  server = createServer((req, res) => {
    handle(req, res).catch((err) => {
      opts.log?.(`server error: ${(err as Error).message}`);
      if (!res.headersSent) {
        res.statusCode = 500;
        res.setHeader("content-type", "application/json");
      }
      res.end(JSON.stringify({ error: { message: (err as Error).message } }));
    });
  });

  await new Promise<void>((resolve, reject) => {
    server!.once("error", reject);
    server!.listen(opts.port, "127.0.0.1", () => {
      activePort = opts.port;
      opts.log?.(`ollamainpi server listening on http://127.0.0.1:${opts.port}`);
      resolve();
    });
  });
}

export async function stopServer(): Promise<void> {
  if (!server) return;
  await new Promise<void>((resolve) => server!.close(() => resolve()));
  server = undefined;
  activePort = undefined;
}

export function getServerPort(): number | undefined {
  return activePort;
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);

  if (req.method === "GET" && url.pathname === "/health") {
    return json(res, 200, { ok: true });
  }

  if (req.method === "GET" && url.pathname === "/v1/models") {
    const models = await listModels();
    return json(res, 200, {
      object: "list",
      data: models.map((m) => ({
        id: m.id,
        object: "model",
        owned_by: "ollamainpi",
        created: Math.floor(m.addedAt / 1000),
      })),
    });
  }

  if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
    return handleChatCompletion(req, res);
  }

  return json(res, 404, { error: { message: `Not found: ${req.method} ${url.pathname}` } });
}

async function handleChatCompletion(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  const model = String(body.model ?? "");
  const messages = Array.isArray(body.messages) ? (body.messages as ChatMessage[]) : [];
  const stream = Boolean(body.stream);
  const temperature = typeof body.temperature === "number" ? body.temperature : undefined;
  const topP = typeof body.top_p === "number" ? body.top_p : undefined;
  const maxTokens = typeof body.max_tokens === "number" ? body.max_tokens : undefined;
  const stop = Array.isArray(body.stop) ? body.stop.map(String) : undefined;

  if (!model) return json(res, 400, { error: { message: "model is required" } });
  if (messages.length === 0)
    return json(res, 400, { error: { message: "messages is required" } });

  const id = randomUUID();
  const created = Math.floor(Date.now() / 1000);

  if (stream) {
    res.statusCode = 200;
    res.setHeader("content-type", "text/event-stream");
    res.setHeader("cache-control", "no-cache");
    res.setHeader("connection", "keep-alive");

    const send = (obj: unknown) => res.write(`data: ${JSON.stringify(obj)}\n\n`);

    const controller = new AbortController();
    req.on("close", () => controller.abort());

    try {
      // Initial role chunk for spec compliance.
      send({
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
      });

      await chatComplete(model, messages, {
        temperature,
        topP,
        maxTokens,
        stop,
        signal: controller.signal,
        onToken: (chunk) => {
          send({
            id,
            object: "chat.completion.chunk",
            created,
            model,
            choices: [{ index: 0, delta: { content: chunk }, finish_reason: null }],
          });
        },
      });

      send({
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      });
      res.write("data: [DONE]\n\n");
      res.end();
    } catch (err) {
      send({ error: { message: (err as Error).message } });
      res.end();
    }
    return;
  }

  // Non-streaming
  const text = await chatComplete(model, messages, { temperature, topP, maxTokens, stop });
  return json(res, 200, {
    id,
    object: "chat.completion",
    created,
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: text },
        finish_reason: "stop",
      },
    ],
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
  });
}

function json(res: ServerResponse, status: number, body: unknown): void {
  res.statusCode = status;
  res.setHeader("content-type", "application/json");
  res.end(JSON.stringify(body));
}

async function readJson(req: IncomingMessage): Promise<any> {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  const raw = Buffer.concat(chunks).toString("utf8");
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch {
    throw new Error("Request body is not valid JSON");
  }
}
