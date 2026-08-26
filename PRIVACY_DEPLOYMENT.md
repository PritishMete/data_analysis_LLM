# InsightFlow privacy deployment

## Safe production setting

Set this Render environment variable:

`INSIGHTFLOW_PRIVACY_MODE=local_only`

This is also the default when the variable is absent.

In `local_only` mode the backend rejects multipart dataset uploads before the
route handler reads the request body. Dataset-bearing endpoints such as
`/analyze`, `/smart_query`, `/clean_data`, `/sentiment_analysis`, location and
categorization routes, Excel context/session routes, and `/v2/ingest/dataset`
are therefore unavailable for remote workbook processing.

The Flutter secure build in this package uses local parsing, local quality
scanning, local filtering and local deduplication for uploaded files. It does
not send workbook rows to the hosted API.

## Health check

`GET /health` remains available.

`GET /privacy` reports the current server privacy mode without exposing data.

## If remote processing is intentionally required

Set:

`INSIGHTFLOW_PRIVACY_MODE=remote_allowed`

Only do this when you intentionally want workbook data processed by the
hosted backend and any configured external AI provider. That mode is not the
privacy-preserving local-only deployment.

## Frontend deployment

Build the Flutter web application normally. The secure build keeps
`secureLocalOnly = true` and therefore does not call the dataset-processing
API. Uploaded-file pipelines use the local Dart engine for filtering and
 deduplication. Excel workbook operations continue to use Office.js when the
application is running inside Excel.
