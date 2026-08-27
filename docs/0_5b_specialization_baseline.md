# 0.5B Planner Specialization Baseline

This document captures the current untouched `Qwen/Qwen2.5-0.5B-Instruct`
planner baseline on the local GTX 1650 machine.

## Baseline metrics

- valid_json_rate: 1.0
- schema_valid_rate: 1.0
- plan_validity_rate: 0.0
- intent_accuracy: 1.0
- tool_selection_f1: 0.0
- tool_sequence_accuracy: 0.0
- predicate_coverage: 0.17647058823529413
- logical_structure_accuracy: 0.7647058823529411
- semantic_role_coverage: 0.11764705882352941
- invalid_tool_rate: 0.0

## Failure modes

The current model is structurally valid but semantically weak. The main failure
families are:

- missing tool semantics
- wrong tool selection
- wrong tool order
- missing predicate coverage
- wrong semantic-role binding
- incomplete multi-step plans
- unsupported or fallback planning

The benchmark report records the current aggregate counts safely, without raw
queries or dataset values.

## Training target

The specialization workflow keeps the model narrow:

- input: intent, semantic schema, predicate graph, logical structure,
  available tools, output contract
- output: a canonical analytics plan

No prose, raw values, workbook data, or file names are used in the target
format.

## Success gates

The post-finetune checkpoint should satisfy:

- valid_json_rate >= 0.99
- schema_valid_rate >= 0.99
- plan_validity_rate >= 0.90
- tool_selection_f1 >= 0.90
- predicate_coverage >= 0.90
- logical_structure_accuracy >= 0.90
- semantic_role_coverage >= 0.90
- invalid_tool_rate <= 0.01

