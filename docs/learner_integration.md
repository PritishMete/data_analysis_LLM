# Learner integration and deployment

The learner exposes the InsightFlow-compatible `POST /v1/plan` contract. It
accepts the request shape already produced by InsightFlow and returns the
existing safe plan envelope. Routing uses both the request text and the active
dataset schema:

| Dataset | Adapter directory | Skill |
|---|---|---|
| Myntra | `models/adapters/myntra` | `myntra.query.v1` |
| Finance | `models/adapters/finance` | `finance.query.v1` |
| Restaurants | `models/adapters/restaurant` | `restaurant.query.v1` |

If the schema is ambiguous, confidence is below
`LEARNER_ROUTER_MIN_CONFIDENCE`, or `adapter_config.json` is missing, the
learner returns `routing.fallback=true` and uses the existing planner. This is
the safety path for low-confidence requests and keeps Gemini/teacher behavior
available while checkpoints are being mounted.

Run locally:

```powershell
$env:LEARNER_ADAPTERS_ENABLED="true"
$env:LEARNER_ADAPTER_ROOT="$PWD/models/adapters"
uvicorn insight_learning.api.app:app --reload --port 8020
```

Deploy with `Dockerfile` or the included `render.yaml`. The adapter folders
must contain the actual Kaggle checkpoint files before deployment; manifests
alone intentionally do not activate a model.

Example independent request:

```powershell
Invoke-RestMethod http://127.0.0.1:8020/v1/plan -Method Post -ContentType 'application/json' -Body (@{
  text='show the top 5 brands by rating'
  intent='rank'
  query_features=@{semantic_roles=@('categorical','numeric'); operators=@('rank')}
  dataset_profile=@{fields=@(
    @{id='product_name'; semantic_role='description'; dtype='str'},
    @{id='brand_name'; semantic_role='category'; dtype='str'},
    @{id='rating'; semantic_role='measure'; dtype='float'}
  )}
} | ConvertTo-Json -Depth 8)
```
