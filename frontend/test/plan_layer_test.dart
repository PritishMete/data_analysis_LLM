import 'package:flutter_test/flutter_test.dart';
import 'package:insightflow/engine/condition_evaluator.dart';
import 'package:insightflow/engine/fallback_parser.dart';
import 'package:insightflow/engine/sheet_namer.dart';
import 'package:insightflow/models/operation_plan.dart';
import 'package:insightflow/semantic/entity_discovery.dart';
import 'package:insightflow/semantic/semantic_resolution.dart';
import 'package:insightflow/validation/local_plan_validator.dart';

void main() {
  group('Condition evaluator', () {
    test('preserves all AND conditions', () {
      final evaluator = const ConditionEvaluator();
      final rows = [
        {'name': 'Pizza Hut', 'delivery': true, 'booking': true, 'rating': 4.1},
        {'name': 'Pizza Hut', 'delivery': true, 'booking': false, 'rating': 4.3},
        {'name': 'Dominos', 'delivery': true, 'booking': true, 'rating': 3.4}
      ];
      final group = ConditionGroup(
        logic: ConditionLogic.and,
        conditions: [
          const Condition(targetRef: 'name', operator: ConditionOperator.contains, value: 'Pizza Hut', valueType: ValueType.string, caseSensitive: false),
          const Condition(targetRef: 'delivery', operator: ConditionOperator.eq, value: true, valueType: ValueType.boolean, caseSensitive: false),
          const Condition(targetRef: 'booking', operator: ConditionOperator.eq, value: true, valueType: ValueType.boolean, caseSensitive: false),
          const Condition(targetRef: 'rating', operator: ConditionOperator.gt, value: 3.5, valueType: ValueType.number, caseSensitive: false)
        ]
      );

      final filtered = evaluator.filterRows(rows, group, (targetRef, row) => row[targetRef]);
      expect(filtered, hasLength(1));
      expect(filtered.first['name'], 'Pizza Hut');
    });

    test('supports nested OR conditions', () {
      final evaluator = const ConditionEvaluator();
      final rows = [
        {'city': 'Kolkata', 'rating': 3.8},
        {'city': 'Delhi', 'rating': 3.8},
        {'city': 'Mumbai', 'rating': 4.2}
      ];
      final group = ConditionGroup(
        logic: ConditionLogic.and,
        conditions: [
          ConditionGroup(
            logic: ConditionLogic.or,
            conditions: [
              const Condition(targetRef: 'city', operator: ConditionOperator.eq, value: 'Kolkata', valueType: ValueType.string, caseSensitive: false),
              const Condition(targetRef: 'city', operator: ConditionOperator.eq, value: 'Delhi', valueType: ValueType.string, caseSensitive: false)
            ]
          ),
          const Condition(targetRef: 'rating', operator: ConditionOperator.lt, value: 3.9, valueType: ValueType.number, caseSensitive: false)
        ]
      );

      final filtered = evaluator.filterRows(rows, group, (targetRef, row) => row[targetRef]);
      expect(filtered, hasLength(2));
    });
  });

  group('Fallback parser', () {
    test('parses categorize and fix all columns', () {
      final parser = const DeterministicFallbackParser();
      final result = parser.parse('Categorize and fix all columns.', requestId: 'req-1');
      expect(result.plan, isNotNull);
      expect(result.plan!.operations, hasLength(2));
    });

    test('refuses currency conversion without rate metadata', () {
      final parser = const DeterministicFallbackParser();
      final result = parser.parse('Convert the currency column to INR.', requestId: 'req-2');
      expect(result.plan, isNull);
      expect(result.message, contains('exchange-rate source'));
    });
  });

  group('Sheet names', () {
    test('creates short understandable names', () {
      final generator = const SheetNameGenerator();
      final name = generator.fromRequest('Show Pizza Hut restaurants having online delivery and table booking with rating above 3.5.');
      expect(name, isNotEmpty);
      expect(name.length, lessThanOrEqualTo(24));
    });
  });

  group('Semantic resolution', () {
    test('discovers entity columns locally', () {
      const resolver = LocalSemanticColumnResolver();
      const snapshot = TableSnapshot(
        columns: [
          ColumnProfile(columnId: 'c1', header: 'Restaurant Name', inferredType: 'string', textness: 0.9, numericness: 0.0, booleanness: 0.0, dateness: 0.0, headerTokens: ['restaurant', 'name']),
          ColumnProfile(columnId: 'c2', header: 'Rating', inferredType: 'number', textness: 0.0, numericness: 1.0, booleanness: 0.0, dateness: 0.0, headerTokens: ['rating'])
        ],
        rows: [
          {'c1': 'Pizza Hut', 'c2': 4.1},
          {'c1': 'Dominos', 'c2': 3.6}
        ]
      );

      final discovered = resolver.discoverEntityColumn('Pizza Hut', snapshot);
      expect(discovered, 'c1');
    });

    test('returns null when entity not found', () {
      const discovery = LocalEntityDiscovery();
      const snapshot = TableSnapshot(
        columns: [
          ColumnProfile(columnId: 'c1', header: 'Restaurant Name', inferredType: 'string', textness: 0.9, numericness: 0.0, booleanness: 0.0, dateness: 0.0, headerTokens: ['restaurant', 'name'])
        ],
        rows: [
          {'c1': 'Burger King'}
        ]
      );

      final result = discovery.discover('Pizza Hut', snapshot);
      expect(result.found, isFalse);
    });
  });

  group('Validator', () {
    test('rejects unresolved required targets', () {
      const validator = LocalPlanValidator();
      final plan = InsightFlowPlan(
        schemaVersion: '1.0',
        requestId: 'req-3',
        intent: const PlanIntent(taskClass: TaskClass.filter, summary: 'filter'),
        semanticTargets: const [
          SemanticTarget(targetId: 'rating', kind: 'metric', hint: 'rating', expectedType: 'number', cardinality: 'single', required: true, nullable: false)
        ],
        operations: const [
          FilterRowsOperation(
            where: ConditionGroup(
              logic: ConditionLogic.and,
              conditions: [
                Condition(targetRef: 'rating', operator: ConditionOperator.gt, value: 3.5, valueType: ValueType.number, caseSensitive: false)
              ]
            )
          )
        ],
        output: const OutputSpec(sheetNameSeed: 'rating', openAutomatically: true, artifactKind: ArtifactKind.worksheet),
        needsUserConfirmation: false,
        clarifyingQuestions: const [],
        warnings: const [],
        confidence: 1
      );

      final result = validator.validate(plan, canResolveTarget: (targetId) => false);
      expect(result.ok, isFalse);
      expect(result.unresolvedTargets, contains('rating'));
    });
  });
}
