import Anthropic from "@anthropic-ai/sdk";
import { classifyError, logLlmCall } from "../log";
import type { LlmRunOptions, LlmRunResult } from "../llm";

const DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/anthropic";
const DEFAULT_MODEL = "claude-sonnet-4-6";

let cachedClient: Anthropic | null = null;
function getClient(): Anthropic {
  if (cachedClient) return cachedClient;
  const apiKey = process.env.ZHIPU_API_KEY || process.env.LLM_API_KEY;
  if (!apiKey) {
    throw new Error(
      "ZHIPU_API_KEY (or generic LLM_API_KEY) is required for LLM_BACKEND=zhipu. Set it in .env.local.",
    );
  }
  const baseURL =
    process.env.ZHIPU_BASE_URL?.trim()
    || process.env.LLM_BASE_URL?.trim()
    || DEFAULT_BASE_URL;
  cachedClient = new Anthropic({ apiKey, baseURL });
  return cachedClient;
}

export function zhipuModel(): string {
  return process.env.LLM_MODEL?.trim() || DEFAULT_MODEL;
}

export async function runZhipu({
  systemPrompt,
  userPrompt,
  timeoutMs = 180_000,
}: LlmRunOptions): Promise<LlmRunResult> {
  const client = getClient();
  const model = zhipuModel();
  const started = Date.now();
  const inputChars = systemPrompt.length + userPrompt.length;

  try {
    const resp = await client.messages.create(
      {
        model,
        max_tokens: 8192,
        system: systemPrompt,
        messages: [{ role: "user", content: userPrompt }],
      },
      { timeout: timeoutMs },
    );
    const text = resp.content
      .filter((block): block is Anthropic.TextBlock => block.type === "text")
      .map((block) => block.text)
      .join("")
      .trim();
    const durationMs = Date.now() - started;
    logLlmCall({
      ts: new Date(started).toISOString(),
      backend: "zhipu",
      model,
      durationMs,
      success: true,
      inputChars,
      outputChars: text.length,
      errorCategory: null,
      errorSnippet: null,
    });
    return { text, durationMs };
  } catch (err) {
    const durationMs = Date.now() - started;
    const msg = err instanceof Error ? err.message : String(err);
    logLlmCall({
      ts: new Date(started).toISOString(),
      backend: "zhipu",
      model,
      durationMs,
      success: false,
      inputChars: 0,
      outputChars: 0,
      errorCategory: classifyError(msg),
      errorSnippet: msg.slice(0, 200),
    });
    throw err;
  }
}
