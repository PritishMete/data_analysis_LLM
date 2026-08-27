# Manual Runtime Learning Proof

## Runtime Roots
- Student runtime root: `C:\Users\jiban\AppData\Local\Temp\insight_learning_manual_student_20260827_run4`
- Teacher runtime root: `C:\Users\jiban\AppData\Local\Temp\teacher_manual_runtime_20260827_run3`

## Baseline Status
- `experience_count`: 0
- `eligible_experience_count`: 0
- `rejected_experience_count`: 0
- `plan_template_count`: 0
- `candidate_strategy_count`: 0
- `trusted_strategy_count`: 0
- `learner_plan_requests`: 0
- `learner_plan_accepts`: 0
- `learner_plan_rejections`: 0
- `learner_experience_requests`: 0
- `learner_experience_accepts`: 0

## First Live Query
- Query family: boolean AND boolean AND numeric comparison
- First request hit `/v1/plan` before `/v1/experience`
- First plan source: `bootstrap_skill`
- `learner_attempted`: true
- `learner_accepted`: true
- `gemini_called`: false
- `gemini_call_count`: 0

## First Eligibility Result
- The first successful abstract-plan experience became eligible.
- `eligible_experience_count` increased to `1`
- `privacy_gate_passed`: true
- Rejection reasons still reflected export gating, not privacy leakage.

## Rejection Reasons Observed
- `strategy_lifecycle_not_trusted`
- Earlier mismatched evidence produced `execution_failed`, which was fixed by aligning the safe abstract plan with the validator.

## Strategy Lifecycle
- `candidate_strategy_count` increased to `2`
- `trusted_strategy_count` remained `0`
- `plan_template_count` increased to `2`

## Learned Plan Acceptance
- The filter family was accepted from `bootstrap_skill`
- `learner_plan_accepts` increased across repeated live requests
- `learner_accepted` remained true on the repeated learned-path requests

## Gemini Bypass Proof
- Learned filter requests reported:
  - `gemini_called = false`
  - `gemini_call_count = 0`
- Before and after the learned request, the explicit counter remained at `0`

## Second Structural Family
- Aggregate family tested: average sales by region
- Result: accepted as a safe learned plan
- `gemini_called`: false
- `plan_source`: `bootstrap_skill`

## Multi-Step Result
- An unsupported multi-step style request fell back to the operation parser path and returned an internal routing error because `google.adk` is unavailable in this environment.
- No unsafe overconfident learned plan was returned.

## Unknown-Query Fallback
- A structurally novel request fell back safely instead of becoming a trusted learned plan.
- The response stayed in the operation/error path and did not bypass validation.

## Restart Persistence
- After stopping and restarting the student on the same runtime root:
  - `experience_count` persisted at `3`
  - `eligible_experience_count` persisted at `1`
  - the same filter family still reused the learned bootstrap skill
  - `gemini_call_count` remained `0` on the learned-path replay

## Online Readiness
- Readiness improved from baseline by gaining one eligible example and live experience history.
- Overall readiness remained `false` because the dataset is still far below the prototype thresholds.
- Reported reason included:
  - `eligible_examples_below_threshold`
  - `family_count_below_threshold`
  - `intent_count_below_threshold`
  - `largest_intent_share=1.000`
  - `largest_tool_graph_share=1.000`

## Test Results
- Student suite: `411 passed`
- Teacher suite: `399 passed, 1 skipped`

## Notes
- No fine-tuning was performed.
- Generated datasets, PDFs, and runtime stores were kept local and uncommitted.
