import 'package:flutter/material.dart';
import 'package:insightflow/models/operation_plan.dart';

class PlanPreviewDialog extends StatelessWidget {
  final InsightFlowPlan plan;
  final VoidCallback onConfirm;
  final VoidCallback onCancel;

  const PlanPreviewDialog({
    super.key,
    required this.plan,
    required this.onConfirm,
    required this.onCancel,
  });

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: Text('Preview ${plan.output.sheetNameSeed}'),
      content: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 560, maxHeight: 520),
        child: SingleChildScrollView(
          child: DefaultTextStyle(
            style: Theme.of(context).textTheme.bodyMedium!,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text('Intent: ${plan.intent.summary}'),
                const SizedBox(height: 12),
                Text('Operations', style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 8),
                ...plan.operations.map(
                  (operation) => Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: Text('• ${operation.type.name}'),
                  ),
                ),
                if (plan.semanticTargets.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  Text('Semantic targets', style: Theme.of(context).textTheme.titleMedium),
                  const SizedBox(height: 8),
                  ...plan.semanticTargets.map(
                    (target) => Padding(
                      padding: const EdgeInsets.only(bottom: 6),
                      child: Text('• ${target.hint} (${target.kind})'),
                    ),
                  ),
                ],
                if (plan.warnings.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  Text('Warnings', style: Theme.of(context).textTheme.titleMedium),
                  const SizedBox(height: 8),
                  ...plan.warnings.map((warning) => Text('• $warning')),
                ],
              ],
            ),
          ),
        ),
      ),
      actions: [
        TextButton(onPressed: onCancel, child: const Text('Cancel')),
        FilledButton(onPressed: onConfirm, child: const Text('Apply locally')),
      ],
    );
  }
}
