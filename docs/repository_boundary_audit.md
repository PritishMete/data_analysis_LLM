# Repository Boundary Audit

## Allowed Teacher -> Student Payload

- `intent`
- aliased `field_id` values
- `semantic_role`
- `dtype`
- `query_features`
- plan summaries
- validation outcomes
- quality scores
- tool sequences

## Explicitly Disallowed

- raw workbook rows
- cell values
- file bytes
- filenames
- sheet names
- previews
- customer names
- email addresses
- phone numbers
- account identifiers
- free-form result tables

## Notes

The teacher app is responsible for aliasing column names before any learned
planning request leaves the process boundary. The student stores only
generalized experiences and promotes repeated patterns into candidate or
trusted strategies.
