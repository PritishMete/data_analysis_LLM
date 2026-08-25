import Ajv2020 from "ajv/dist/2020";
import type { ErrorObject } from "ajv";
import addFormats from "ajv-formats";
import planSchema from "../../shared/insightflow-plan.schema.json";

type ModelBinding = {
  run(model: string, input: unknown): Promise<unknown>;
};

export interface Env {
  AI: ModelBinding;
  ALLOWED_ORIGINS?: string;
  MODEL_ID?: string;
  MAX_REQUEST_BYTES?: string;
  RATE_LIMIT_PER_MINUTE?: string;
  RATE_LIMIT_WINDOW_MS?: string;
}

type PlanRequest = {
  request_id?: string;
  request_text?: string;
  locale?: string;
  client_version?: string;
};

type PlanResponse =
  | {
      ok: true;
      request_id: string;
      plan: unknown;
      model_id: string;
      repaired: boolean;
    }
  | {
      ok: false;
      request_id: string;
      error: {
        code: string;
        message: string;
      };
    };

type RateWindow = {
  count: number;
  resetAt: number;
};

const MODEL_ID_DEFAULT = "@cf/meta/llama-3.1-8b-instruct-fast";
const MAX_REQUEST_BYTES_DEFAULT = 8192;
const RATE_LIMIT_PER_MINUTE_DEFAULT = 30;
const RATE_LIMIT_WINDOW_MS_DEFAULT = 60_000;
const CONTENT_TYPE_JSON = "application/json; charset=utf-8";

const ajv = new Ajv2020({
  allErrors: true,
  strict: true,
  allowUnionTypes: true
});
addFormats(ajv);

const validatePlan = ajv.compile(planSchema);
const rateState = new Map<string, RateWindow>();

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    try {
      if (request.method === "OPTIONS") {
        return handleOptions(request, env);
      }

      if (request.method !== "POST" || new URL(request.url).pathname !== "/v1/plan") {
        return json(
          {
            ok: false,
            request_id: safeRequestId(),
            error: { code: "NOT_FOUND", message: "Route not found." }
          },
          404,
          request,
          env
        );
      }

      const origin = request.headers.get("Origin");
      if (!isAllowedOrigin(origin, env)) {
        return json(
          {
            ok: false,
            request_id: safeRequestId(),
            error: { code: "ORIGIN_DENIED", message: "Origin not allowed." }
          },
          403,
          request,
          env
        );
      }

      const clientIp = request.headers.get("CF-Connecting-IP") || request.headers.get("X-Forwarded-For") || "unknown";
      const rateLimitResult = enforceRateLimit(clientIp, env);
      if (!rateLimitResult.ok) {
        return json(
          {
            ok: false,
            request_id: safeRequestId(),
            error: { code: "RATE_LIMITED", message: "Too many requests." }
          },
          429,
          request,
          env,
          { "Retry-After": Math.ceil(rateLimitResult.retryAfterMs / 1000).toString() }
        );
      }

      const contentLength = request.headers.get("Content-Length");
      const maxRequestBytes = numberEnv(env.MAX_REQUEST_BYTES, MAX_REQUEST_BYTES_DEFAULT);
      if (contentLength && Number.parseInt(contentLength, 10) > maxRequestBytes) {
        return json(
          {
            ok: false,
            request_id: safeRequestId(),
            error: { code: "PAYLOAD_TOO_LARGE", message: "Request body too large." }
          },
          413,
          request,
          env
        );
      }

      const requestId = safeRequestId();
      const bodyText = await readLimitedBody(request, maxRequestBytes);
      const body = safeParseJson<PlanRequest>(bodyText);
      if (!body) {
        return json(
          {
            ok: false,
            request_id: requestId,
            error: { code: "INVALID_JSON", message: "Request body must be valid JSON." }
          },
          400,
          request,
          env
        );
      }

      const requestText = body.request_text?.trim();
      if (!requestText) {
        return json(
          {
            ok: false,
            request_id: requestId,
            error: { code: "MISSING_REQUEST_TEXT", message: "Request text is required." }
          },
          400,
          request,
          env
        );
      }
      if (requestText.length > 2000) {
        return json(
          {
            ok: false,
            request_id: requestId,
            error: { code: "REQUEST_TOO_LONG", message: "Request text is too long." }
          },
          400,
          request,
          env
        );
      }

      const planPayload = await buildPlan({
        env,
        requestId,
        requestText,
        locale: body.locale,
        clientVersion: body.client_version
      });

      return json(
        {
          ok: true,
          request_id: requestId,
          plan: planPayload.plan,
          model_id: planPayload.modelId,
          repaired: planPayload.repaired
        },
        200,
        request,
        env
      );
    } catch {
      return json(
        {
          ok: false,
          request_id: safeRequestId(),
          error: { code: "INTERNAL_ERROR", message: "Unable to build a plan." }
        },
        500,
        request,
        env
      );
    }
  }
} satisfies ExportedHandler<Env>;

async function buildPlan(args: {
  env: Env;
  requestId: string;
  requestText: string;
  locale?: string;
  clientVersion?: string;
}): Promise<{ plan: unknown; modelId: string; repaired: boolean }> {
  const modelId = args.env.MODEL_ID || MODEL_ID_DEFAULT;
  const responseSchema = planSchema;

  const systemPrompt = [
    "You are InsightFlow's planning engine.",
    "Return a single JSON object that matches the provided JSON Schema.",
    "Do not write code, formulas, SQL, Python, JavaScript, or Dart.",
    "Do not mention workbook contents, columns, filenames, sheets, values, or statistics.",
    "Only produce operation plans.",
    "Preserve all stated filter conditions.",
    "If a required semantic target is ambiguous or unresolved, include it in the plan rather than dropping it."
  ].join(" ");

  const userPrompt = [
    `request_id: ${args.requestId}`,
    `locale: ${args.locale || "en-US"}`,
    `client_version: ${args.clientVersion || "unknown"}`,
    `user_request: ${args.requestText}`
  ].join("\n");

  const first = await callModel(args.env, modelId, systemPrompt, userPrompt, responseSchema);
  const parsedFirst = parseModelPlan(first);
  if (parsedFirst.ok && validatePlan(parsedFirst.value)) {
    return { plan: parsedFirst.value, modelId, repaired: false };
  }

  const repairInput = JSON.stringify(
    {
      invalid_output: first,
      error_summary: parsedFirst.ok ? normalizeAjvErrors(validatePlan.errors) : parsedFirst.error
    },
    null,
    2
  );

  const repairPrompt = [
    "Repair the following plan so it becomes valid JSON matching the schema.",
    "Return JSON only.",
    repairInput
  ].join("\n\n");

  const second = await callModel(args.env, modelId, systemPrompt, repairPrompt, responseSchema);
  const parsedSecond = parseModelPlan(second);
  if (parsedSecond.ok && validatePlan(parsedSecond.value)) {
    return { plan: parsedSecond.value, modelId, repaired: true };
  }

  throw new Error("Plan validation failed.");
}

async function callModel(
  env: Env,
  modelId: string,
  systemPrompt: string,
  userPrompt: string,
  schema: unknown
): Promise<unknown> {
  const result = await env.AI.run(modelId, {
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt }
    ],
    response_format: {
      type: "json_schema",
      json_schema: schema
    },
    temperature: 0,
    top_p: 1,
    max_tokens: 1200
  });
  return result;
}

export function parseModelPlan(raw: unknown): { ok: true; value: unknown } | { ok: false; error: string } {
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    if ("response" in raw && typeof (raw as { response?: unknown }).response === "string") {
      return tryParseJsonString((raw as { response: string }).response);
    }
    return { ok: true, value: raw };
  }
  if (typeof raw === "string") {
    return tryParseJsonString(raw);
  }
  return { ok: false, error: "Unexpected model output shape." };
}

export function tryParseJsonString(input: string): { ok: true; value: unknown } | { ok: false; error: string } {
  const trimmed = stripCodeFences(input).trim();
  const candidate = extractBalancedJson(trimmed);
  if (!candidate) {
    return { ok: false, error: "No JSON object found." };
  }
  try {
    return { ok: true, value: JSON.parse(candidate) };
  } catch {
    return { ok: false, error: "Invalid JSON." };
  }
}

export function stripCodeFences(input: string): string {
  return input.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "");
}

export function extractBalancedJson(input: string): string | null {
  const start = input.indexOf("{");
  if (start < 0) {
    return null;
  }

  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let i = start; i < input.length; i += 1) {
    const ch = input[i];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (ch === "\\") {
        escaped = true;
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }

    if (ch === '"') {
      inString = true;
      continue;
    }
    if (ch === "{") {
      depth += 1;
    } else if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        return input.slice(start, i + 1);
      }
    }
  }
  return null;
}

export function safeParseJson<T>(input: string): T | null {
  try {
    return JSON.parse(input) as T;
  } catch {
    return null;
  }
}

async function readLimitedBody(request: Request, maxBytes: number): Promise<string> {
  const text = await request.text();
  if (new TextEncoder().encode(text).byteLength > maxBytes) {
    throw new Error("body_too_large");
  }
  return text;
}

function isAllowedOrigin(origin: string | null, env: Env): boolean {
  if (!origin) {
    return false;
  }
  const allowed = parseCsv(env.ALLOWED_ORIGINS);
  return allowed.includes(origin);
}

export function parseCsv(value: string | undefined): string[] {
  return (value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function enforceRateLimit(ip: string, env: Env): { ok: true } | { ok: false; retryAfterMs: number } {
  const limit = numberEnv(env.RATE_LIMIT_PER_MINUTE, RATE_LIMIT_PER_MINUTE_DEFAULT);
  const windowMs = numberEnv(env.RATE_LIMIT_WINDOW_MS, RATE_LIMIT_WINDOW_MS_DEFAULT);
  const now = Date.now();
  const slot = rateState.get(ip);
  if (!slot || slot.resetAt <= now) {
    rateState.set(ip, { count: 1, resetAt: now + windowMs });
    return { ok: true };
  }

  if (slot.count >= limit) {
    return { ok: false, retryAfterMs: Math.max(1000, slot.resetAt - now) };
  }

  slot.count += 1;
  rateState.set(ip, slot);
  return { ok: true };
}

export function numberEnv(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

export function normalizeAjvErrors(errors: ErrorObject[] | null | undefined): string {
  if (!errors || errors.length === 0) {
    return "Plan validation failed.";
  }
  return errors
    .slice(0, 5)
    .map((error) => `${error.instancePath || "/"} ${error.message || "invalid"}`)
    .join("; ");
}

function safeRequestId(): string {
  return crypto.randomUUID();
}

function json(
  body: PlanResponse,
  status: number,
  request: Request,
  env: Env,
  extraHeaders: Record<string, string> = {}
): Response {
  const headers = new Headers({
    "content-type": CONTENT_TYPE_JSON,
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
    ...corsHeaders(request, env),
    ...extraHeaders
  });
  return Response.json(body, { status, headers });
}

function handleOptions(request: Request, env: Env): Response {
  const origin = request.headers.get("Origin");
  if (!isAllowedOrigin(origin, env)) {
    return new Response(null, { status: 403 });
  }
  return new Response(null, {
    status: 204,
    headers: corsHeaders(request, env)
  });
}

function corsHeaders(request: Request, env: Env): Record<string, string> {
  const origin = request.headers.get("Origin");
  if (!isAllowedOrigin(origin, env)) {
    return {};
  }
  return {
    "access-control-allow-origin": origin || "",
    "access-control-allow-methods": "POST, OPTIONS",
    "access-control-allow-headers": "content-type",
    "access-control-max-age": "86400",
    vary: "origin"
  };
}
