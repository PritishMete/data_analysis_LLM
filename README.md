# InsightFlow

InsightFlow is a browser-first AI data analyst that keeps workbook data local and uses a cloud LLM only to produce a structured JSON operation plan.

## Architecture Summary

- Frontend: Flutter Web hosted on GitHub Pages
- Backend: Cloudflare Worker in TypeScript
- Model: Cloudflare Workers AI
- Shared contract: JSON Schema
- Workbook execution: entirely in the browser
- Heavy processing: browser Web Worker where practical

## Privacy Boundaries

- Workbook files never leave the browser.
- Column names, sheet names, filenames, cell values, sample rows, unique values, statistics, generated reports, and workbook errors never go to the backend or LLM.
- User prompts are treated as sensitive and are not logged or stored.
- The LLM only receives the natural-language request plus the plan contract.
- The backend never sees workbook content.
- All plan validation and execution happen locally after allowlist checks.

## Repo Layout

- `shared/` contains the JSON Schema contract.
- `backend/` contains the Cloudflare Worker and tests.
- `frontend/` contains the Flutter Web app, models, client, validator, fallback parser, and tests.

## Setup

### Prerequisites

- Node.js 20+
- pnpm or npm
- Flutter stable
- A Cloudflare account with Workers AI enabled

### Backend

```bash
cd backend
npm install
npm test
```

### Frontend

```bash
cd frontend
flutter pub get
flutter test
```

## Deployment

### Cloudflare Worker

1. Set backend secrets:

```bash
cd backend
npx wrangler secret put CLOUDFLARE_ACCOUNT_ID
npx wrangler secret put CLOUDFLARE_API_TOKEN
```

2. Configure allowlisted origins in `wrangler.toml` or as an environment variable.
3. Deploy:

```bash
cd backend
npx wrangler deploy
```

### GitHub Pages

1. Build the Flutter web app:

```bash
cd frontend
flutter build web --release --base-href /data_analysis_LLM/ --dart-define=INSIGHTFLOW_PLAN_ENDPOINT=https://<your-worker>.workers.dev/v1/plan
```

2. Publish the `frontend/build/web` directory to GitHub Pages.
3. Set `backend/wrangler.toml` `ALLOWED_ORIGINS` to your GitHub Pages origin, then deploy the worker.

## Recommended Free-Tier Hosting

- Frontend: GitHub Pages
- Backend: Cloudflare Workers Free
- LLM: Cloudflare Workers AI Free/Paid usage depending on quota

This combination is the best fit for a lightweight static frontend plus a very small plan-broker backend.

## Verification Checklist

- LLM never receives workbook content.
- Backend request body contains only the user request and minimal client metadata.
- Local validator blocks unsafe or unknown operations.
- Required semantic targets never fail silently.
- Missing column mappings stop execution and prompt the user.
- Currency conversion is blocked unless a rate source and timestamp are available.
- All filters in AND/OR queries are preserved and evaluated locally.
- Sheet creation auto-switches to the new sheet.
