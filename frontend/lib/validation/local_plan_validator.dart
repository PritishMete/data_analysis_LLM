import 'package:insightflow/models/operation_plan.dart';
import 'package:insightflow/validation/operation_allowlist.dart';

class LocalPlanValidationResult {
  final bool ok;
  final List<String> errors;
  final List<String> unresolvedTargets;

  const LocalPlanValidationResult(
      {required this.ok,
      required this.errors,
      required this.unresolvedTargets});
}

class LocalPlanValidator {
  const LocalPlanValidator();

  LocalPlanValidationResult validate(InsightFlowPlan plan,
      {required bool Function(String targetId) canResolveTarget}) {
    final errors = <String>[];
    final unresolvedTargets = <String>[];

    if (plan.schemaVersion != '1.0') {
      errors.add('Unsupported schema version.');
    }
    if (!OperationAllowlist.allowedOperationTypes
        .containsAll(plan.operations.map((op) => op.type))) {
      errors.add('Plan contains disallowed operations.');
    }

    for (final target in plan.semanticTargets) {
      if (target.required && !canResolveTarget(target.targetId)) {
        unresolvedTargets.add(target.targetId);
      }
    }

    for (final op in plan.operations) {
      switch (op) {
        case FilterRowsOperation():
          _validateConditionGroup(
              op.where, canResolveTarget, errors, unresolvedTargets);
        case SortRowsOperation():
          for (final key in op.keys) {
            if (!canResolveTarget(key.targetRef)) {
              unresolvedTargets.add(key.targetRef);
            }
          }
        case CategorizeColumnsOperation():
          if (!{'all', 'matched', 'selected'}.contains(op.scope)) {
            errors.add('Invalid categorize scope.');
          }
        case NormalizeColumnsOperation():
          if (!{'all', 'matched', 'selected'}.contains(op.scope)) {
            errors.add('Invalid normalize scope.');
          }
        case ConvertCurrencyOperation():
          if (op.exchangeRateSource.trim().isEmpty) {
            errors.add('Currency conversion requires an exchange-rate source.');
          }
          if (op.rateTimestamp.toUtc().isAfter(
              DateTime.now().toUtc().add(const Duration(minutes: 1)))) {
            errors.add('Currency conversion rate timestamp is invalid.');
          }
        case CreatePivotOperation():
          if (op.values.isEmpty) {
            errors.add('Pivot requires at least one value field.');
          }
          for (final value in op.values) {
            if (!OperationAllowlist.allowedAggregations
                .contains(value.aggregation)) {
              errors.add('Pivot aggregation is not allowed.');
            }
            if (!canResolveTarget(value.targetRef)) {
              unresolvedTargets.add(value.targetRef);
            }
          }
        case BuildChartOperation():
          if (op.x.trim().isEmpty) {
            errors.add('Chart requires an x field.');
          }
        case RenameSheetOperation():
          if (op.nameSeed.trim().isEmpty) {
            errors.add('Sheet name seed is required.');
          }
        case GroupedAggregationOperation():
          if (op.groupBy.isNotEmpty) {
            for (final targetRef in op.groupBy) {
              if (!canResolveTarget(targetRef)) {
                unresolvedTargets.add(targetRef);
              }
            }
          }
          if (op.metrics.isEmpty) {
            errors.add('Grouped aggregation requires at least one metric.');
          }
          for (final metric in op.metrics) {
            if (!OperationAllowlist.allowedAggregations
                .contains(metric.aggregation)) {
              errors.add('Grouped aggregation uses a disallowed aggregation.');
            }
            if (!canResolveTarget(metric.targetRef)) {
              unresolvedTargets.add(metric.targetRef);
            }
          }
        case SummaryStatisticsOperation():
          for (final statistic in op.statistics) {
            if (!OperationAllowlist.allowedSummaryStatistics
                .contains(statistic)) {
              errors.add('Summary statistic is not allowed.');
            }
          }
          for (final targetRef in op.columns) {
            if (!canResolveTarget(targetRef)) {
              unresolvedTargets.add(targetRef);
            }
          }
        case MissingValueAnalysisOperation():
          for (final targetRef in op.columns) {
            if (!canResolveTarget(targetRef)) {
              unresolvedTargets.add(targetRef);
            }
          }
        case DuplicateDetectionOperation():
          for (final targetRef in op.keys) {
            if (!canResolveTarget(targetRef)) {
              unresolvedTargets.add(targetRef);
            }
          }
        case OutlierDetectionOperation():
          if (op.threshold <= 0) {
            errors.add('Outlier threshold must be positive.');
          }
          if (!OperationAllowlist.allowedOutlierMethods.contains(op.method)) {
            errors.add('Outlier method is not allowed.');
          }
          for (final targetRef in op.columns) {
            if (!canResolveTarget(targetRef)) {
              unresolvedTargets.add(targetRef);
            }
          }
      }
    }

    final ok = errors.isEmpty && unresolvedTargets.isEmpty;
    return LocalPlanValidationResult(
        ok: ok,
        errors: errors,
        unresolvedTargets: unresolvedTargets.toSet().toList(growable: false));
  }

  void _validateConditionGroup(
      ConditionGroup group,
      bool Function(String targetId) canResolveTarget,
      List<String> errors,
      List<String> unresolvedTargets) {
    for (final item in group.conditions) {
      if (item is ConditionGroup) {
        _validateConditionGroup(
            item, canResolveTarget, errors, unresolvedTargets);
        continue;
      }
      final condition = item as Condition;
      if (!OperationAllowlist.allowedConditionOperators
          .contains(condition.operator)) {
        errors.add('Condition operator is not allowed.');
      }
      if (!canResolveTarget(condition.targetRef)) {
        unresolvedTargets.add(condition.targetRef);
      }
    }
  }
}
