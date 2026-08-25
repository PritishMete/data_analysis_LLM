import 'package:insightflow/models/operation_plan.dart';

class OperationAllowlist {
  static const allowedOperationTypes = {
    OperationType.filterRows,
    OperationType.sortRows,
    OperationType.categorizeColumns,
    OperationType.normalizeColumns,
    OperationType.convertCurrency,
    OperationType.createPivot,
    OperationType.buildChart,
    OperationType.renameSheet
  };

  static const allowedConditionOperators = {
    ConditionOperator.eq,
    ConditionOperator.neq,
    ConditionOperator.gt,
    ConditionOperator.gte,
    ConditionOperator.lt,
    ConditionOperator.lte,
    ConditionOperator.contains,
    ConditionOperator.notContains,
    ConditionOperator.inList,
    ConditionOperator.notIn,
    ConditionOperator.between,
    ConditionOperator.isNull,
    ConditionOperator.notNull
  };

  static const allowedAggregations = {
    Aggregation.sum,
    Aggregation.count,
    Aggregation.average,
    Aggregation.min,
    Aggregation.max,
    Aggregation.median,
    Aggregation.distinctCount
  };
}

