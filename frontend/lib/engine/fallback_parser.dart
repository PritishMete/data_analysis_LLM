import 'package:insightflow/models/operation_plan.dart';

class FallbackParseResult {
  final InsightFlowPlan? plan;
  final String? message;

  const FallbackParseResult({required this.plan, required this.message});
}

class DeterministicFallbackParser {
  const DeterministicFallbackParser();

  FallbackParseResult parse(String requestText, {required String requestId}) {
    final normalized = requestText.toLowerCase();

    final entityQuery = _parseRestaurantQuery(normalized, requestId);
    if (entityQuery != null) {
      return entityQuery;
    }

    if (normalized.contains('categorize') && normalized.contains('fix all columns')) {
      return FallbackParseResult(
        plan: _buildPlan(
          requestId,
          'Categorize and fix all columns.',
          const [],
          [
            CategorizeColumnsOperation(scope: 'all', strategy: 'hybrid'),
            NormalizeColumnsOperation(scope: 'all', actions: const [
              'trim_whitespace',
              'standardize_case',
              'parse_dates',
              'parse_numbers',
              'fill_blanks',
              'remove_currency_symbols'
            ])
          ],
          'categorize_fix_all'
        ),
        message: null
      );
    }

    if (normalized.contains('normalize country') && normalized.contains('gender') && normalized.contains('bool')) {
      return FallbackParseResult(
        plan: _buildPlan(
          requestId,
          'Normalize country, gender and bool.',
          [
            const SemanticTarget(targetId: 'country', kind: 'category', hint: 'country', expectedType: 'string', cardinality: 'single', required: true, nullable: true),
            const SemanticTarget(targetId: 'gender', kind: 'category', hint: 'gender', expectedType: 'string', cardinality: 'single', required: true, nullable: true),
            const SemanticTarget(targetId: 'bool', kind: 'flag', hint: 'bool', expectedType: 'boolean', cardinality: 'many', required: true, nullable: true)
          ],
          [
            const NormalizeColumnsOperation(scope: 'matched', actions: ['trim_whitespace', 'standardize_case', 'fill_blanks'])
          ],
          'normalize_country_gender_bool'
        ),
        message: null
      );
    }

    if (normalized.contains('create a pivot table')) {
      return FallbackParseResult(
        plan: _buildPlan(
          requestId,
          'Create a pivot table showing average rating by city.',
          [
            const SemanticTarget(targetId: 'city', kind: 'category', hint: 'city', expectedType: 'string', cardinality: 'single', required: true, nullable: false),
            const SemanticTarget(targetId: 'rating', kind: 'metric', hint: 'rating', expectedType: 'number', cardinality: 'single', required: true, nullable: false)
          ],
          [
            CreatePivotOperation(rows: ['city'], columns: const [], values: const [
              PivotValue(targetRef: 'rating', aggregation: Aggregation.average)
            ])
          ],
          'average_rating_by_city',
          artifactKind: ArtifactKind.pivotTable
        ),
        message: 'Pivot layout needs local field selection before execution.'
      );
    }

    if (normalized.contains('convert') && normalized.contains('currency')) {
      return const FallbackParseResult(
        plan: null,
        message: 'Currency conversion requires an exchange-rate source and timestamp.'
      );
    }

    return const FallbackParseResult(plan: null, message: 'No deterministic fallback matched this request.');
  }

  FallbackParseResult? _parseRestaurantQuery(String normalized, String requestId) {
    if (!normalized.contains('restaurant') && !normalized.contains('restaurants')) {
      return null;
    }

    final entity = _extractEntity(normalized);
    final city = _extractCity(normalized);
    final rating = _extractRating(normalized);
    final wantsDelivery = normalized.contains('delivery');
    final wantsBooking = normalized.contains('booking') || normalized.contains('reservation');

    if (entity == null && city == null && rating == null && !wantsDelivery && !wantsBooking) {
      return null;
    }

    final targets = <SemanticTarget>[];
    final conditions = <Object>[];

    if (entity != null) {
      targets.add(SemanticTarget(
        targetId: 'entity',
        kind: 'entity',
        hint: entity,
        expectedType: 'string',
        cardinality: 'single',
        required: true,
        nullable: false,
      ));
      conditions.add(Condition(
        targetRef: 'entity',
        operator: ConditionOperator.contains,
        value: entity,
        valueType: ValueType.string,
        caseSensitive: false,
      ));
    }

    if (wantsDelivery) {
      targets.add(const SemanticTarget(
        targetId: 'delivery',
        kind: 'flag',
        hint: 'delivery',
        expectedType: 'boolean',
        cardinality: 'single',
        required: true,
        nullable: false,
      ));
      conditions.add(Condition(
        targetRef: 'delivery',
        operator: ConditionOperator.eq,
        value: true,
        valueType: ValueType.boolean,
        caseSensitive: false,
      ));
    }

    if (wantsBooking) {
      targets.add(const SemanticTarget(
        targetId: 'booking',
        kind: 'flag',
        hint: 'booking',
        expectedType: 'boolean',
        cardinality: 'single',
        required: true,
        nullable: false,
      ));
      conditions.add(Condition(
        targetRef: 'booking',
        operator: ConditionOperator.eq,
        value: true,
        valueType: ValueType.boolean,
        caseSensitive: false,
      ));
    }

    if (city != null) {
      targets.add(const SemanticTarget(
        targetId: 'city',
        kind: 'category',
        hint: 'city',
        expectedType: 'string',
        cardinality: 'single',
        required: true,
        nullable: false,
      ));
      conditions.add(Condition(
        targetRef: 'city',
        operator: ConditionOperator.eq,
        value: city,
        valueType: ValueType.string,
        caseSensitive: false,
      ));
    }

    if (rating != null) {
      targets.add(const SemanticTarget(
        targetId: 'rating',
        kind: 'metric',
        hint: 'rating',
        expectedType: 'number',
        cardinality: 'single',
        required: true,
        nullable: false,
      ));
      conditions.add(Condition(
        targetRef: 'rating',
        operator: rating.operator,
        value: rating.value,
        valueType: ValueType.number,
        caseSensitive: false,
      ));
    }

    if (conditions.isEmpty) {
      return null;
    }

    return FallbackParseResult(
      plan: _buildPlan(
        requestId,
        'Filter restaurants locally.',
        targets,
        [
          FilterRowsOperation(
            where: ConditionGroup(
              logic: ConditionLogic.and,
              conditions: conditions,
            ),
          ),
        ],
        'filtered_restaurants',
      ),
      message: null,
    );
  }

  String? _extractEntity(String normalized) {
    if (normalized.contains('pizza hut')) {
      return 'Pizza Hut';
    }
    if (normalized.contains('domino\'s pizza') || normalized.contains('dominos pizza') || normalized.contains('domino\'s')) {
      return 'Domino\'s Pizza';
    }
    return null;
  }

  String? _extractCity(String normalized) {
    final cleaned = normalized.replaceAll(RegExp(r'[^a-z0-9\s]+'), ' ');
    final match = RegExp(r'\bin\s+([a-z][a-z\s]+)').firstMatch(cleaned);
    if (match == null) {
      return null;
    }
    return _titleCase(match.group(1)!.trim());
  }

  ({ConditionOperator operator, double value})? _extractRating(String normalized) {
    final match = RegExp(r'rating\s+(above|over|greater than|below|under|less than|at least|at most|>=|<=|>|<)\s*([0-9]+(?:\.[0-9]+)?)').firstMatch(normalized);
    if (match == null) {
      return null;
    }
    final comparator = match.group(1)!;
    final value = double.parse(match.group(2)!);
    if (comparator == 'above' || comparator == 'over' || comparator == 'greater than' || comparator == '>') {
      return (operator: ConditionOperator.gt, value: value);
    }
    if (comparator == 'at least' || comparator == '>=') {
      return (operator: ConditionOperator.gte, value: value);
    }
    if (comparator == 'below' || comparator == 'under' || comparator == 'less than' || comparator == '<') {
      return (operator: ConditionOperator.lt, value: value);
    }
    if (comparator == 'at most' || comparator == '<=') {
      return (operator: ConditionOperator.lte, value: value);
    }
    return (operator: ConditionOperator.gt, value: value);
  }

  String _titleCase(String input) {
    return input
        .split(RegExp(r'\s+'))
        .where((part) => part.isNotEmpty)
        .map((part) => part[0].toUpperCase() + part.substring(1).toLowerCase())
        .join(' ');
  }

  InsightFlowPlan _buildPlan(
    String requestId,
    String summary,
    List<SemanticTarget> semanticTargets,
    List<PlanOperation> operations,
    String sheetSeed, {
    ArtifactKind artifactKind = ArtifactKind.worksheet
  }
  ) {
    return InsightFlowPlan(
      schemaVersion: '1.0',
      requestId: requestId,
      intent: PlanIntent(taskClass: TaskClass.multiStep, summary: summary),
      semanticTargets: semanticTargets,
      operations: operations,
      output: OutputSpec(sheetNameSeed: sheetSeed, openAutomatically: true, artifactKind: artifactKind),
      needsUserConfirmation: false,
      clarifyingQuestions: const [],
      warnings: const [],
      confidence: 0.75
    );
  }
}
