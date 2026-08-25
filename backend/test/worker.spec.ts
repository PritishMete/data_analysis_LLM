import { describe, expect, it } from "vitest";
import worker, { extractBalancedJson, normalizeAjvErrors } from "../src/index";

describe("JSON extraction", () => {
  it("extracts balanced JSON from code fences", () => {
    const input = "```json\n{\"a\":1,\"b\":{\"c\":2}}\n```";
    expect(extractBalancedJson(input)).toBe("{\"a\":1,\"b\":{\"c\":2}}");
  });
});

describe("error normalization", () => {
  it("formats a short summary", () => {
    const summary = normalizeAjvErrors([
      { instancePath: "/intent", message: "must match required schema" } as never
    ]);
    expect(summary).toContain("/intent");
  });
});

describe("plan route", () => {
  it("builds a plan without workbook metadata in the prompt", async () => {
    const seenInputs: unknown[] = [];
    const env = {
      AI: {
        run: async (_model: string, input: unknown) => {
          seenInputs.push(input);
          return {
            schema_version: "1.0",
            request_id: "req-0001",
            intent: { task_class: "filter", summary: "Filter restaurants" },
            semantic_targets: [],
            operations: [
              {
                type: "filter_rows",
                where: {
                  logic: "AND",
                  conditions: [
                    {
                      target_ref: "rating",
                      operator: "lt",
                      value: 3.9,
                      value_type: "number",
                      case_sensitive: false
                    }
                  ]
                }
              }
            ],
            output: {
              sheet_name_seed: "rating",
              open_automatically: true,
              artifact_kind: "worksheet"
            },
            confidence: 1
          };
        }
      },
      ALLOWED_ORIGINS: "https://app.example",
      MODEL_ID: "@cf/meta/llama-3.1-8b-instruct-fast"
    };

    const response = await worker.fetch(
      new Request("https://worker.example/v1/plan", {
        method: "POST",
        headers: {
          "content-type": "application/json",
          origin: "https://app.example"
        },
        body: JSON.stringify({
          request_id: "req-0001",
          request_text: "Show restaurants having rating below 3.9.",
          locale: "en-US",
          client_version: "web"
        })
      }),
      env as never
    );

    expect(response.status).toBe(200);
    const body = (await response.json()) as { ok: boolean; plan: { request_id: string } };
    expect(body.ok).toBe(true);
    expect(body.plan.request_id).toBe("req-0001");
    expect(seenInputs).toHaveLength(1);
    const modelInput = seenInputs[0] as { messages: Array<{ content: string }> };
    const userPrompt = modelInput.messages[1]!.content;
    expect(userPrompt).toContain("Show restaurants having rating below 3.9.");
    expect(userPrompt).toContain("locale: en-US");
    expect(userPrompt).toContain("client_version: web");
    expect(userPrompt).not.toContain("Sheet1");
  });
});
