# Secure-local categorization and INR conversion

The backend supports the secure-local architecture used by the Flutter frontend.

- `/agentic_command` accepts only user text + schema metadata and may use Gemini to understand the request.
- `/agentic_categorize` remains a legacy remote-processing endpoint and is blocked by the local-only privacy boundary.
- `/currency/rates` returns only an exchange rate for two ISO currency codes and accepts no dataset fields, rows, files, or workbook data.
- Raw workbook data is never required by the hosted API for local categorization.
