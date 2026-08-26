# Planner Baseline Report

## Corpus Snapshot
- inspected: 41
- eligible: 11
- rejected: 30
- duplicates removed: 1
- family count: 11
- intent count: 4
- tool graph count: 6
- average quality: 0.9700
- train/validation/test: 9/2/0

## Readiness
- ready_for_prototype: False
- readiness_score: 0.3243
- reason: eligible_examples_below_threshold,family_count_below_threshold,intent_count_below_threshold
- balance_warnings: none

## Training Pass
- seed runs: 8
- learned reruns: 8
- operation runs: 4
- bridge accepted during learned reruns: 0
- Gemini fallback calls during seed pass: 8
- Gemini fallback calls during learned pass: 22

## Holdout Metrics
- intent accuracy: 1.000
- tool selection accuracy: 1.000
- tool sequence accuracy: 1.000
- predicate coverage: 1.000
- logical structure preservation: 1.000
- semantic-role coverage: 1.000
- plan validity: 1.000
- invalid-tool rate: 0.000
- privacy-violation rate: 0.000

## Export Files
- train: E:\LLM\runtime\training\dataset-v1\train.jsonl
- validation: E:\LLM\runtime\training\dataset-v1\validation.jsonl
- test: E:\LLM\runtime\training\dataset-v1\test.jsonl
- manifest: E:\LLM\runtime\training\dataset-v1\manifest.json
- report: E:\LLM\runtime\training\dataset-v1\report.json

## Validation Notes
- invalidation response: {"invalidated": true, "invalidation": {"corpus_version": 1, "created_at": "2026-08-26T15:14:53.428871+00:00", "family_fingerprint": null, "reason": "correction", "source_id": "5755b1319db958649dc970d1ee1d9dc1"}}
- dedupe report eligible_examples: 11
- training seed accuracy: 100.0%
- learned rerun accuracy: 100.0%
- holdout baseline accuracy: 100.0%
- holdout learned accuracy: 100.0%
