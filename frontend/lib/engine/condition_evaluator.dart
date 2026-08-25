import 'package:insightflow/models/operation_plan.dart';

class ConditionEvaluator {
  const ConditionEvaluator();

  List<Map<String, Object?>> filterRows(
    List<Map<String, Object?>> rows,
    ConditionGroup group,
    Object? Function(String targetRef, Map<String, Object?> row) resolveValue
  ) {
    return rows.where((row) => evaluateGroup(group, row, resolveValue)).toList(growable: false);
  }

  bool evaluateGroup(
    ConditionGroup group,
    Map<String, Object?> row,
    Object? Function(String targetRef, Map<String, Object?> row) resolveValue
  ) {
    final results = group.conditions.map((item) {
      if (item is ConditionGroup) {
        return evaluateGroup(item, row, resolveValue);
      }
      return evaluateCondition(item as Condition, row, resolveValue);
    }).toList(growable: false);

    return switch (group.logic) {
      ConditionLogic.and => results.every((item) => item),
      ConditionLogic.or => results.any((item) => item)
    };
  }

  bool evaluateCondition(
    Condition condition,
    Map<String, Object?> row,
    Object? Function(String targetRef, Map<String, Object?> row) resolveValue
  ) {
    final value = resolveValue(condition.targetRef, row);
    final rhs = condition.value;

    return switch (condition.operator) {
      ConditionOperator.eq => _equals(value, rhs, condition.caseSensitive),
      ConditionOperator.neq => !_equals(value, rhs, condition.caseSensitive),
      ConditionOperator.gt => _compareNumbers(value, rhs) > 0,
      ConditionOperator.gte => _compareNumbers(value, rhs) >= 0,
      ConditionOperator.lt => _compareNumbers(value, rhs) < 0,
      ConditionOperator.lte => _compareNumbers(value, rhs) <= 0,
      ConditionOperator.contains => _contains(value, rhs, condition.caseSensitive),
      ConditionOperator.notContains => !_contains(value, rhs, condition.caseSensitive),
      ConditionOperator.inList => _inList(value, rhs, condition.caseSensitive),
      ConditionOperator.notIn => !_inList(value, rhs, condition.caseSensitive),
      ConditionOperator.between => _between(value, rhs),
      ConditionOperator.isNull => value == null,
      ConditionOperator.notNull => value != null
    };
  }

  bool _equals(Object? lhs, Object? rhs, bool caseSensitive) {
    if (lhs == null || rhs == null) {
      return lhs == rhs;
    }
    final left = lhs.toString();
    final right = rhs.toString();
    return caseSensitive ? left == right : left.toLowerCase() == right.toLowerCase();
  }

  int _compareNumbers(Object? lhs, Object? rhs) {
    final left = double.tryParse(lhs?.toString() ?? '');
    final right = double.tryParse(rhs?.toString() ?? '');
    if (left == null || right == null) {
      return 0;
    }
    return left.compareTo(right);
  }

  bool _contains(Object? lhs, Object? rhs, bool caseSensitive) {
    if (lhs == null || rhs == null) {
      return false;
    }
    final left = lhs.toString();
    final right = rhs.toString();
    return caseSensitive ? left.contains(right) : left.toLowerCase().contains(right.toLowerCase());
  }

  bool _inList(Object? lhs, Object? rhs, bool caseSensitive) {
    final list = rhs is List ? rhs : <Object?>[rhs];
    return list.any((item) => _equals(lhs, item, caseSensitive));
  }

  bool _between(Object? lhs, Object? rhs) {
    if (lhs == null || rhs is! List || rhs.length != 2) {
      return false;
    }
    final value = double.tryParse(lhs.toString());
    final low = double.tryParse(rhs[0].toString());
    final high = double.tryParse(rhs[1].toString());
    if (value == null || low == null || high == null) {
      return false;
    }
    return value >= low && value <= high;
  }
}
