import 'dart:convert';

enum TaskClass {
  filter,
  transform,
  analyze,
  pivot,
  chart,
  export,
  multiStep,
  unknown
}

enum ArtifactKind { worksheet, pivotTable, chart, report, worksheetAndChart }

enum OperationType {
  filterRows,
  sortRows,
  categorizeColumns,
  normalizeColumns,
  convertCurrency,
  createPivot,
  buildChart,
  renameSheet,
  groupedAggregation,
  summaryStatistics,
  missingValueAnalysis,
  duplicateDetection,
  outlierDetection
}

enum ConditionLogic { and, or }

enum ConditionOperator {
  eq,
  neq,
  gt,
  gte,
  lt,
  lte,
  contains,
  notContains,
  inList,
  notIn,
  between,
  isNull,
  notNull
}

enum ValueType { string, number, boolean, date, currency, list, any }

enum Aggregation { sum, count, average, min, max, median, distinctCount }

enum SortDirection { asc, desc }

enum SummaryStatisticKind {
  count,
  missingCount,
  mean,
  median,
  stdDev,
  min,
  max,
  variance,
  q1,
  q3,
  iqr
}

enum OutlierMethod { iqr, zscore }

String _enumName(Object value) => value.toString().split('.').last;

const _taskClassToJson = {
  TaskClass.filter: 'filter',
  TaskClass.transform: 'transform',
  TaskClass.analyze: 'analyze',
  TaskClass.pivot: 'pivot',
  TaskClass.chart: 'chart',
  TaskClass.export: 'export',
  TaskClass.multiStep: 'multi_step',
  TaskClass.unknown: 'unknown'
};

const _artifactKindToJson = {
  ArtifactKind.worksheet: 'worksheet',
  ArtifactKind.pivotTable: 'pivot_table',
  ArtifactKind.chart: 'chart',
  ArtifactKind.report: 'report',
  ArtifactKind.worksheetAndChart: 'worksheet_and_chart'
};

const _operationTypeToJson = {
  OperationType.filterRows: 'filter_rows',
  OperationType.sortRows: 'sort_rows',
  OperationType.categorizeColumns: 'categorize_columns',
  OperationType.normalizeColumns: 'normalize_columns',
  OperationType.convertCurrency: 'convert_currency',
  OperationType.createPivot: 'create_pivot',
  OperationType.buildChart: 'build_chart',
  OperationType.renameSheet: 'rename_sheet',
  OperationType.groupedAggregation: 'grouped_aggregation',
  OperationType.summaryStatistics: 'summary_statistics',
  OperationType.missingValueAnalysis: 'missing_value_analysis',
  OperationType.duplicateDetection: 'duplicate_detection',
  OperationType.outlierDetection: 'outlier_detection'
};

const _conditionLogicToJson = {
  ConditionLogic.and: 'AND',
  ConditionLogic.or: 'OR'
};

const _conditionOperatorToJson = {
  ConditionOperator.eq: 'eq',
  ConditionOperator.neq: 'neq',
  ConditionOperator.gt: 'gt',
  ConditionOperator.gte: 'gte',
  ConditionOperator.lt: 'lt',
  ConditionOperator.lte: 'lte',
  ConditionOperator.contains: 'contains',
  ConditionOperator.notContains: 'not_contains',
  ConditionOperator.inList: 'in',
  ConditionOperator.notIn: 'not_in',
  ConditionOperator.between: 'between',
  ConditionOperator.isNull: 'is_null',
  ConditionOperator.notNull: 'not_null'
};

const _aggregationToJson = {
  Aggregation.sum: 'sum',
  Aggregation.count: 'count',
  Aggregation.average: 'average',
  Aggregation.min: 'min',
  Aggregation.max: 'max',
  Aggregation.median: 'median',
  Aggregation.distinctCount: 'distinct_count'
};

const _summaryStatisticToJson = {
  SummaryStatisticKind.count: 'count',
  SummaryStatisticKind.missingCount: 'missing_count',
  SummaryStatisticKind.mean: 'mean',
  SummaryStatisticKind.median: 'median',
  SummaryStatisticKind.stdDev: 'std_dev',
  SummaryStatisticKind.min: 'min',
  SummaryStatisticKind.max: 'max',
  SummaryStatisticKind.variance: 'variance',
  SummaryStatisticKind.q1: 'q1',
  SummaryStatisticKind.q3: 'q3',
  SummaryStatisticKind.iqr: 'iqr'
};

const _outlierMethodToJson = {
  OutlierMethod.iqr: 'iqr',
  OutlierMethod.zscore: 'zscore'
};

TaskClass taskClassFromJson(String value) =>
    _taskClassToJson.entries.firstWhere((entry) => entry.value == value).key;
String taskClassToJson(TaskClass value) => _taskClassToJson[value]!;

ArtifactKind artifactKindFromJson(String value) =>
    _artifactKindToJson.entries.firstWhere((entry) => entry.value == value).key;
String artifactKindToJson(ArtifactKind value) => _artifactKindToJson[value]!;

OperationType operationTypeFromJson(String value) =>
    _operationTypeToJson.entries
        .firstWhere((entry) => entry.value == value)
        .key;
String operationTypeToJson(OperationType value) => _operationTypeToJson[value]!;

ConditionLogic conditionLogicFromJson(String value) =>
    _conditionLogicToJson.entries
        .firstWhere((entry) => entry.value == value)
        .key;
String conditionLogicToJson(ConditionLogic value) =>
    _conditionLogicToJson[value]!;

ConditionOperator conditionOperatorFromJson(String value) =>
    _conditionOperatorToJson.entries
        .firstWhere((entry) => entry.value == value)
        .key;
String conditionOperatorToJson(ConditionOperator value) =>
    _conditionOperatorToJson[value]!;

ValueType valueTypeFromJson(String value) =>
    ValueType.values.firstWhere((item) => _enumName(item) == value);
String valueTypeToJson(ValueType value) => _enumName(value);

Aggregation aggregationFromJson(String value) =>
    _aggregationToJson.entries.firstWhere((entry) => entry.value == value).key;
String aggregationToJson(Aggregation value) => _aggregationToJson[value]!;

SortDirection sortDirectionFromJson(String value) =>
    SortDirection.values.firstWhere((item) => _enumName(item) == value);
String sortDirectionToJson(SortDirection value) => _enumName(value);

SummaryStatisticKind summaryStatisticKindFromJson(String value) =>
    _summaryStatisticToJson.entries
        .firstWhere((entry) => entry.value == value)
        .key;
String summaryStatisticKindToJson(SummaryStatisticKind value) =>
    _summaryStatisticToJson[value]!;

OutlierMethod outlierMethodFromJson(String value) =>
    _outlierMethodToJson.entries
        .firstWhere((entry) => entry.value == value)
        .key;
String outlierMethodToJson(OutlierMethod value) => _outlierMethodToJson[value]!;

class InsightFlowPlan {
  final String schemaVersion;
  final String requestId;
  final PlanIntent intent;
  final List<SemanticTarget> semanticTargets;
  final List<PlanOperation> operations;
  final OutputSpec output;
  final bool needsUserConfirmation;
  final List<String> clarifyingQuestions;
  final List<String> warnings;
  final double confidence;

  const InsightFlowPlan(
      {required this.schemaVersion,
      required this.requestId,
      required this.intent,
      required this.semanticTargets,
      required this.operations,
      required this.output,
      required this.needsUserConfirmation,
      required this.clarifyingQuestions,
      required this.warnings,
      required this.confidence});

  factory InsightFlowPlan.fromJson(Map<String, dynamic> json) {
    return InsightFlowPlan(
        schemaVersion: json['schema_version'] as String,
        requestId: json['request_id'] as String,
        intent: PlanIntent.fromJson(json['intent'] as Map<String, dynamic>),
        semanticTargets:
            (json['semantic_targets'] as List<dynamic>? ?? const [])
                .cast<Map<String, dynamic>>()
                .map(SemanticTarget.fromJson)
                .toList(growable: false),
        operations: (json['operations'] as List<dynamic>)
            .cast<Map<String, dynamic>>()
            .map(PlanOperation.fromJson)
            .toList(growable: false),
        output: OutputSpec.fromJson(json['output'] as Map<String, dynamic>),
        needsUserConfirmation:
            json['needs_user_confirmation'] as bool? ?? false,
        clarifyingQuestions:
            (json['clarifying_questions'] as List<dynamic>? ?? const [])
                .cast<String>(),
        warnings:
            (json['warnings'] as List<dynamic>? ?? const []).cast<String>(),
        confidence: (json['confidence'] as num).toDouble());
  }

  Map<String, dynamic> toJson() => {
        'schema_version': schemaVersion,
        'request_id': requestId,
        'intent': intent.toJson(),
        'semantic_targets': semanticTargets
            .map((item) => item.toJson())
            .toList(growable: false),
        'operations':
            operations.map((item) => item.toJson()).toList(growable: false),
        'output': output.toJson(),
        'needs_user_confirmation': needsUserConfirmation,
        'clarifying_questions': clarifyingQuestions,
        'warnings': warnings,
        'confidence': confidence
      };
}

class PlanIntent {
  final TaskClass taskClass;
  final String summary;

  const PlanIntent({required this.taskClass, required this.summary});

  factory PlanIntent.fromJson(Map<String, dynamic> json) {
    return PlanIntent(
        taskClass: taskClassFromJson(json['task_class'] as String),
        summary: json['summary'] as String);
  }

  Map<String, dynamic> toJson() =>
      {'task_class': taskClassToJson(taskClass), 'summary': summary};
}

class SemanticTarget {
  final String targetId;
  final String kind;
  final String hint;
  final String? expectedType;
  final String? cardinality;
  final bool required;
  final bool nullable;

  const SemanticTarget(
      {required this.targetId,
      required this.kind,
      required this.hint,
      this.expectedType,
      this.cardinality,
      required this.required,
      required this.nullable});

  factory SemanticTarget.fromJson(Map<String, dynamic> json) {
    return SemanticTarget(
        targetId: json['target_id'] as String,
        kind: json['kind'] as String,
        hint: json['hint'] as String,
        expectedType: json['expected_type'] as String?,
        cardinality: json['cardinality'] as String?,
        required: json['required'] as bool? ?? true,
        nullable: json['nullable'] as bool? ?? true);
  }

  Map<String, dynamic> toJson() => {
        'target_id': targetId,
        'kind': kind,
        'hint': hint,
        if (expectedType != null) 'expected_type': expectedType,
        if (cardinality != null) 'cardinality': cardinality,
        'required': required,
        'nullable': nullable
      };
}

class OutputSpec {
  final String sheetNameSeed;
  final bool openAutomatically;
  final ArtifactKind artifactKind;

  const OutputSpec(
      {required this.sheetNameSeed,
      required this.openAutomatically,
      required this.artifactKind});

  factory OutputSpec.fromJson(Map<String, dynamic> json) {
    return OutputSpec(
        sheetNameSeed: json['sheet_name_seed'] as String,
        openAutomatically: json['open_automatically'] as bool,
        artifactKind: artifactKindFromJson(json['artifact_kind'] as String));
  }

  Map<String, dynamic> toJson() => {
        'sheet_name_seed': sheetNameSeed,
        'open_automatically': openAutomatically,
        'artifact_kind': artifactKindToJson(artifactKind)
      };
}

abstract class PlanOperation {
  final OperationType type;

  const PlanOperation(this.type);

  Map<String, dynamic> toJson();

  static PlanOperation fromJson(Map<String, dynamic> json) {
    final type = operationTypeFromJson(json['type'] as String);
    return switch (type) {
      OperationType.filterRows => FilterRowsOperation.fromJson(json),
      OperationType.sortRows => SortRowsOperation.fromJson(json),
      OperationType.categorizeColumns =>
        CategorizeColumnsOperation.fromJson(json),
      OperationType.normalizeColumns =>
        NormalizeColumnsOperation.fromJson(json),
      OperationType.convertCurrency => ConvertCurrencyOperation.fromJson(json),
      OperationType.createPivot => CreatePivotOperation.fromJson(json),
      OperationType.buildChart => BuildChartOperation.fromJson(json),
      OperationType.renameSheet => RenameSheetOperation.fromJson(json),
      OperationType.groupedAggregation =>
        GroupedAggregationOperation.fromJson(json),
      OperationType.summaryStatistics =>
        SummaryStatisticsOperation.fromJson(json),
      OperationType.missingValueAnalysis =>
        MissingValueAnalysisOperation.fromJson(json),
      OperationType.duplicateDetection =>
        DuplicateDetectionOperation.fromJson(json),
      OperationType.outlierDetection => OutlierDetectionOperation.fromJson(json)
    };
  }
}

class FilterRowsOperation extends PlanOperation {
  final ConditionGroup where;

  const FilterRowsOperation({required this.where})
      : super(OperationType.filterRows);

  factory FilterRowsOperation.fromJson(Map<String, dynamic> json) =>
      FilterRowsOperation(
          where:
              ConditionGroup.fromJson(json['where'] as Map<String, dynamic>));

  @override
  Map<String, dynamic> toJson() =>
      {'type': 'filter_rows', 'where': where.toJson()};
}

class SortRowsOperation extends PlanOperation {
  final List<SortKey> keys;

  const SortRowsOperation({required this.keys}) : super(OperationType.sortRows);

  factory SortRowsOperation.fromJson(Map<String, dynamic> json) =>
      SortRowsOperation(
          keys: (json['keys'] as List<dynamic>)
              .cast<Map<String, dynamic>>()
              .map(SortKey.fromJson)
              .toList(growable: false));

  @override
  Map<String, dynamic> toJson() => {
        'type': 'sort_rows',
        'keys': keys.map((item) => item.toJson()).toList(growable: false)
      };
}

class CategorizeColumnsOperation extends PlanOperation {
  final String scope;
  final String strategy;

  const CategorizeColumnsOperation(
      {required this.scope, required this.strategy})
      : super(OperationType.categorizeColumns);

  factory CategorizeColumnsOperation.fromJson(Map<String, dynamic> json) =>
      CategorizeColumnsOperation(
          scope: json['scope'] as String,
          strategy: json['strategy'] as String? ?? 'hybrid');

  @override
  Map<String, dynamic> toJson() =>
      {'type': 'categorize_columns', 'scope': scope, 'strategy': strategy};
}

class NormalizeColumnsOperation extends PlanOperation {
  final String scope;
  final List<String> actions;

  const NormalizeColumnsOperation({required this.scope, required this.actions})
      : super(OperationType.normalizeColumns);

  factory NormalizeColumnsOperation.fromJson(Map<String, dynamic> json) =>
      NormalizeColumnsOperation(
          scope: json['scope'] as String,
          actions:
              (json['actions'] as List<dynamic>? ?? const []).cast<String>());

  @override
  Map<String, dynamic> toJson() =>
      {'type': 'normalize_columns', 'scope': scope, 'actions': actions};
}

class GroupedAggregationOperation extends PlanOperation {
  final List<String> groupBy;
  final List<GroupAggregationMetric> metrics;
  final bool includeTotals;

  const GroupedAggregationOperation({
    required this.groupBy,
    required this.metrics,
    required this.includeTotals,
  }) : super(OperationType.groupedAggregation);

  factory GroupedAggregationOperation.fromJson(Map<String, dynamic> json) =>
      GroupedAggregationOperation(
        groupBy:
            (json['group_by'] as List<dynamic>? ?? const []).cast<String>(),
        metrics: (json['metrics'] as List<dynamic>)
            .cast<Map<String, dynamic>>()
            .map(GroupAggregationMetric.fromJson)
            .toList(growable: false),
        includeTotals: json['include_totals'] as bool? ?? false,
      );

  @override
  Map<String, dynamic> toJson() => {
        'type': 'grouped_aggregation',
        'group_by': groupBy,
        'metrics': metrics.map((item) => item.toJson()).toList(growable: false),
        'include_totals': includeTotals
      };
}

class GroupAggregationMetric {
  final String targetRef;
  final Aggregation aggregation;
  final String? alias;

  const GroupAggregationMetric({
    required this.targetRef,
    required this.aggregation,
    this.alias,
  });

  factory GroupAggregationMetric.fromJson(Map<String, dynamic> json) =>
      GroupAggregationMetric(
        targetRef: json['target_ref'] as String,
        aggregation: aggregationFromJson(json['aggregation'] as String),
        alias: json['alias'] as String?,
      );

  Map<String, dynamic> toJson() => {
        'target_ref': targetRef,
        'aggregation': aggregationToJson(aggregation),
        if (alias != null) 'alias': alias
      };
}

class SummaryStatisticsOperation extends PlanOperation {
  final List<String> columns;
  final List<SummaryStatisticKind> statistics;

  const SummaryStatisticsOperation({
    required this.columns,
    required this.statistics,
  }) : super(OperationType.summaryStatistics);

  factory SummaryStatisticsOperation.fromJson(Map<String, dynamic> json) =>
      SummaryStatisticsOperation(
        columns: (json['columns'] as List<dynamic>? ?? const []).cast<String>(),
        statistics: (json['statistics'] as List<dynamic>? ??
                const [
                  'count',
                  'missing_count',
                  'mean',
                  'median',
                  'std_dev',
                  'min',
                  'max'
                ])
            .cast<String>()
            .map(summaryStatisticKindFromJson)
            .toList(growable: false),
      );

  @override
  Map<String, dynamic> toJson() => {
        'type': 'summary_statistics',
        'columns': columns,
        'statistics':
            statistics.map(summaryStatisticKindToJson).toList(growable: false)
      };
}

class MissingValueAnalysisOperation extends PlanOperation {
  final List<String> columns;
  final bool includePercentages;

  const MissingValueAnalysisOperation({
    required this.columns,
    required this.includePercentages,
  }) : super(OperationType.missingValueAnalysis);

  factory MissingValueAnalysisOperation.fromJson(Map<String, dynamic> json) =>
      MissingValueAnalysisOperation(
        columns: (json['columns'] as List<dynamic>? ?? const []).cast<String>(),
        includePercentages: json['include_percentages'] as bool? ?? true,
      );

  @override
  Map<String, dynamic> toJson() => {
        'type': 'missing_value_analysis',
        'columns': columns,
        'include_percentages': includePercentages
      };
}

class DuplicateDetectionOperation extends PlanOperation {
  final List<String> keys;
  final bool includeCounts;
  final bool caseSensitive;

  const DuplicateDetectionOperation({
    required this.keys,
    required this.includeCounts,
    required this.caseSensitive,
  }) : super(OperationType.duplicateDetection);

  factory DuplicateDetectionOperation.fromJson(Map<String, dynamic> json) =>
      DuplicateDetectionOperation(
        keys: (json['keys'] as List<dynamic>? ?? const []).cast<String>(),
        includeCounts: json['include_counts'] as bool? ?? true,
        caseSensitive: json['case_sensitive'] as bool? ?? false,
      );

  @override
  Map<String, dynamic> toJson() => {
        'type': 'duplicate_detection',
        'keys': keys,
        'include_counts': includeCounts,
        'case_sensitive': caseSensitive
      };
}

class OutlierDetectionOperation extends PlanOperation {
  final List<String> columns;
  final OutlierMethod method;
  final double threshold;

  const OutlierDetectionOperation({
    required this.columns,
    required this.method,
    required this.threshold,
  }) : super(OperationType.outlierDetection);

  factory OutlierDetectionOperation.fromJson(Map<String, dynamic> json) =>
      OutlierDetectionOperation(
        columns: (json['columns'] as List<dynamic>? ?? const []).cast<String>(),
        method: outlierMethodFromJson(json['method'] as String? ?? 'iqr'),
        threshold: (json['threshold'] as num? ?? 1.5).toDouble(),
      );

  @override
  Map<String, dynamic> toJson() => {
        'type': 'outlier_detection',
        'columns': columns,
        'method': outlierMethodToJson(method),
        'threshold': threshold
      };
}

class ConvertCurrencyOperation extends PlanOperation {
  final String targetCurrency;
  final String scope;
  final String exchangeRateSource;
  final DateTime rateTimestamp;
  final String roundingMode;

  const ConvertCurrencyOperation(
      {required this.targetCurrency,
      required this.scope,
      required this.exchangeRateSource,
      required this.rateTimestamp,
      required this.roundingMode})
      : super(OperationType.convertCurrency);

  factory ConvertCurrencyOperation.fromJson(Map<String, dynamic> json) =>
      ConvertCurrencyOperation(
          targetCurrency: json['target_currency'] as String,
          scope: json['scope'] as String? ?? 'matched',
          exchangeRateSource: json['exchange_rate_source'] as String,
          rateTimestamp: DateTime.parse(json['rate_timestamp'] as String),
          roundingMode: json['rounding_mode'] as String? ?? 'round');

  @override
  Map<String, dynamic> toJson() => {
        'type': 'convert_currency',
        'target_currency': targetCurrency,
        'scope': scope,
        'exchange_rate_source': exchangeRateSource,
        'rate_timestamp': rateTimestamp.toUtc().toIso8601String(),
        'rounding_mode': roundingMode
      };
}

class CreatePivotOperation extends PlanOperation {
  final List<String> rows;
  final List<String> columns;
  final List<PivotValue> values;

  const CreatePivotOperation(
      {required this.rows, required this.columns, required this.values})
      : super(OperationType.createPivot);

  factory CreatePivotOperation.fromJson(Map<String, dynamic> json) =>
      CreatePivotOperation(
          rows: (json['rows'] as List<dynamic>? ?? const []).cast<String>(),
          columns:
              (json['columns'] as List<dynamic>? ?? const []).cast<String>(),
          values: (json['values'] as List<dynamic>)
              .cast<Map<String, dynamic>>()
              .map(PivotValue.fromJson)
              .toList(growable: false));

  @override
  Map<String, dynamic> toJson() => {
        'type': 'create_pivot',
        'rows': rows,
        'columns': columns,
        'values': values.map((item) => item.toJson()).toList(growable: false)
      };
}

class BuildChartOperation extends PlanOperation {
  final String chartType;
  final String x;
  final List<String> y;
  final String? series;

  const BuildChartOperation(
      {required this.chartType,
      required this.x,
      required this.y,
      required this.series})
      : super(OperationType.buildChart);

  factory BuildChartOperation.fromJson(Map<String, dynamic> json) =>
      BuildChartOperation(
          chartType: json['chart_type'] as String,
          x: json['x'] as String,
          y: (json['y'] as List<dynamic>? ?? const []).cast<String>(),
          series: json['series'] as String?);

  @override
  Map<String, dynamic> toJson() => {
        'type': 'build_chart',
        'chart_type': chartType,
        'x': x,
        'y': y,
        if (series != null) 'series': series
      };
}

class RenameSheetOperation extends PlanOperation {
  final String nameSeed;

  const RenameSheetOperation({required this.nameSeed})
      : super(OperationType.renameSheet);

  factory RenameSheetOperation.fromJson(Map<String, dynamic> json) =>
      RenameSheetOperation(nameSeed: json['name_seed'] as String);

  @override
  Map<String, dynamic> toJson() =>
      {'type': 'rename_sheet', 'name_seed': nameSeed};
}

class ConditionGroup {
  final ConditionLogic logic;
  final List<Object> conditions;

  const ConditionGroup({required this.logic, required this.conditions});

  factory ConditionGroup.fromJson(Map<String, dynamic> json) {
    final items = (json['conditions'] as List<dynamic>).map((item) {
      if (item is Map<String, dynamic> && item.containsKey('logic')) {
        return ConditionGroup.fromJson(item);
      }
      return Condition.fromJson(item as Map<String, dynamic>);
    }).toList(growable: false);
    return ConditionGroup(
        logic: conditionLogicFromJson(json['logic'] as String),
        conditions: items);
  }

  Map<String, dynamic> toJson() => {
        'logic': conditionLogicToJson(logic),
        'conditions': conditions
            .map((item) => item is ConditionGroup
                ? item.toJson()
                : (item as Condition).toJson())
            .toList(growable: false)
      };
}

class Condition {
  final String targetRef;
  final ConditionOperator operator;
  final Object? value;
  final ValueType valueType;
  final bool caseSensitive;

  const Condition(
      {required this.targetRef,
      required this.operator,
      required this.value,
      required this.valueType,
      required this.caseSensitive});

  factory Condition.fromJson(Map<String, dynamic> json) => Condition(
      targetRef: json['target_ref'] as String,
      operator: conditionOperatorFromJson(json['operator'] as String),
      value: json['value'],
      valueType: valueTypeFromJson(json['value_type'] as String? ?? 'any'),
      caseSensitive: json['case_sensitive'] as bool? ?? false);

  Map<String, dynamic> toJson() => {
        'target_ref': targetRef,
        'operator': conditionOperatorToJson(operator),
        'value': value,
        'value_type': valueTypeToJson(valueType),
        'case_sensitive': caseSensitive
      };
}

class SortKey {
  final String targetRef;
  final SortDirection direction;

  const SortKey({required this.targetRef, required this.direction});

  factory SortKey.fromJson(Map<String, dynamic> json) => SortKey(
      targetRef: json['target_ref'] as String,
      direction: sortDirectionFromJson(json['direction'] as String));

  Map<String, dynamic> toJson() =>
      {'target_ref': targetRef, 'direction': sortDirectionToJson(direction)};
}

class PivotValue {
  final String targetRef;
  final Aggregation aggregation;

  const PivotValue({required this.targetRef, required this.aggregation});

  factory PivotValue.fromJson(Map<String, dynamic> json) => PivotValue(
      targetRef: json['target_ref'] as String,
      aggregation: aggregationFromJson(json['aggregation'] as String));

  Map<String, dynamic> toJson() =>
      {'target_ref': targetRef, 'aggregation': aggregationToJson(aggregation)};
}

InsightFlowPlan insightFlowPlanFromJson(String source) {
  return InsightFlowPlan.fromJson(jsonDecode(source) as Map<String, dynamic>);
}
