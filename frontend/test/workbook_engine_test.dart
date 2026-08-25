import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:insightflow/api/plan_client.dart';
import 'package:insightflow/app/insightflow_controller.dart';
import 'package:insightflow/engine/fallback_parser.dart';
import 'package:insightflow/models/operation_plan.dart';
import 'package:insightflow/workbook/exchange_rate_service.dart';
import 'package:insightflow/workbook/workbook_engine.dart';
import 'package:insightflow/workbook/workbook_exporter.dart';
import 'package:insightflow/workbook/workbook_importer.dart';
import 'package:insightflow/workbook/workbook_models.dart';
import 'package:insightflow/semantic/semantic_resolution.dart';

WorkbookSession _buildSession() {
  const csv = '''
Restaurant Name,Online Delivery,Table Booking,Rating,City,Country,Gender,Bool,Currency
Pizza Hut,true,true,4.2,Kolkata,USA,M,yes,100
Domino's Pizza,true,true,3.8,Kolkata,India,F,no,200
Burger King,false,false,3.6,Delhi,UK,m,1,300
Other Bistro,true,false,4.5,Mumbai,India,man,available,400
''';
  final bytes = Uint8List.fromList(utf8.encode(csv));
  return const WorkbookImporter().importBytes(bytes, fileName: 'restaurants.csv').session;
}

Future<WorkbookExecutionResult> _executePlan(String requestText, {ExchangeRateService? exchangeRateService}) async {
  final session = _buildSession();
  final fallback = const DeterministicFallbackParser();
  final requestId = 'req-test';
  final parsed = fallback.parse(requestText, requestId: requestId);
  final plan = parsed.plan;
  expect(plan, isNotNull);
  return const WorkbookExecutionEngine().execute(
    session: session,
    plan: plan!,
    requestText: requestText,
    exchangeRateService: exchangeRateService ?? const ManualExchangeRateService(snapshot: null),
  );
}

void main() {
  group('Workbook execution', () {
    test('preserves all filter conditions for Pizza Hut query', () async {
      final result = await _executePlan(
        'Show Pizza Hut restaurants having online delivery and table booking with rating above 3.5.',
      );

      expect(result.success, isTrue, reason: result.errors.join(' | '));
      expect(result.session.activeSheet.rows, hasLength(1));
      final row = result.session.activeSheet.rows.single;
      expect(row[0], 'Pizza Hut');
      expect(row[1], isTrue);
      expect(row[2], isTrue);
      expect((row[3] as num).toDouble(), greaterThan(3.5));
    });

    test('preserves entity, delivery, booking and city for Domino\'s query', () async {
      final result = await _executePlan(
        'Show Domino\'s Pizza restaurants with delivery and booking in Kolkata.',
      );

      expect(result.success, isTrue, reason: result.errors.join(' | '));
      expect(result.session.activeSheet.rows, hasLength(1));
      final row = result.session.activeSheet.rows.single;
      expect(row[0], contains('Domino'));
      expect(row[1], isTrue);
      expect(row[2], isTrue);
      expect(row[4], 'Kolkata');
    });

    test('filters by rating below threshold', () async {
      final result = await _executePlan('Show restaurants having rating below 3.9.');

      expect(result.success, isTrue, reason: result.errors.join(' | '));
      expect(result.session.activeSheet.rows, hasLength(2));
      for (final row in result.session.activeSheet.rows) {
        expect((row[3] as num).toDouble(), lessThan(3.9));
      }
    });

    test('categorizes and fixes all columns locally', () async {
      final result = await _executePlan('Categorize and fix all columns.');

      expect(result.success, isTrue, reason: result.errors.join(' | '));
      final row = result.session.activeSheet.rows.first;
      expect(row[5], 'United States');
      expect(row[6], 'Male');
      expect(row[7], isTrue);
    });

    test('normalizes country, gender and bool locally', () async {
      final result = await _executePlan('Normalize country, gender and bool.');

      expect(result.success, isTrue, reason: result.errors.join(' | '));
      final row = result.session.activeSheet.rows.first;
      expect(row[5], 'United States');
      expect(row[6], 'Male');
      expect(row[7], isTrue);
    });

    test('converts currency to INR with explicit rate metadata', () async {
      final session = _buildSession();
      final timestamp = DateTime.utc(2026, 8, 25);
      final plan = InsightFlowPlan(
        schemaVersion: '1.0',
        requestId: 'req-currency',
        intent: const PlanIntent(taskClass: TaskClass.transform, summary: 'Convert currency'),
        semanticTargets: const [
          SemanticTarget(
            targetId: 'currency',
            kind: 'currency',
            hint: 'currency',
            expectedType: 'currency',
            cardinality: 'single',
            required: true,
            nullable: false,
          ),
        ],
        operations: [
          ConvertCurrencyOperation(
            targetCurrency: 'INR',
            scope: 'matched',
            exchangeRateSource: 'manual',
            rateTimestamp: timestamp,
            roundingMode: 'round',
          ),
        ],
        output: const OutputSpec(sheetNameSeed: 'currency_inr', openAutomatically: true, artifactKind: ArtifactKind.worksheet),
        needsUserConfirmation: false,
        clarifyingQuestions: const [],
        warnings: const [],
        confidence: 1,
      );
      final rateService = ManualExchangeRateService(
        snapshot: ExchangeRateSnapshot(
          sourceCurrency: 'USD',
          targetCurrency: 'INR',
          rate: 83.0,
          rateSource: 'manual',
          timestamp: timestamp,
        ),
      );

      final result = await const WorkbookExecutionEngine().execute(
        session: session,
        plan: plan,
        requestText: 'Convert the currency column to INR.',
        exchangeRateService: rateService,
      );

      expect(result.success, isTrue, reason: result.errors.join(' | '));
      expect(result.session.activeSheet.originalHeaders[8], 'Currency (₹)');
      expect((result.session.activeSheet.rows.first[8] as num).toDouble(), closeTo(8300, 0.001));
    });

    test('converts symbol-prefixed currency values locally', () async {
      final timestamp = DateTime.utc(2026, 8, 25);
      final session = WorkbookSession(
        workbookName: 'symbols.xlsx',
        sheets: [
          WorkbookSheet(
            id: 'sheet1',
            name: 'sheet1',
            columnIds: const ['c1'],
            originalHeaders: const ['Currency'],
            rows: const [
              [r'$100.00'],
              ['USD 200.50'],
              ['₹300'],
            ],
            profiles: const [
              ColumnProfile(
                columnId: 'c1',
                header: 'Currency',
                inferredType: 'string',
                textness: 1.0,
                numericness: 0.0,
                booleanness: 0.0,
                dateness: 0.0,
                headerTokens: ['currency'],
              ),
            ],
            isOriginal: true,
          ),
        ],
        activeSheetIndex: 0,
        sourceFileName: 'symbols.xlsx',
        charts: const [],
      );
      final plan = InsightFlowPlan(
        schemaVersion: '1.0',
        requestId: 'req-symbols',
        intent: const PlanIntent(taskClass: TaskClass.transform, summary: 'Convert currency'),
        semanticTargets: const [
          SemanticTarget(
            targetId: 'currency',
            kind: 'currency',
            hint: 'currency',
            expectedType: 'currency',
            cardinality: 'single',
            required: true,
            nullable: false,
          ),
        ],
        operations: [
          ConvertCurrencyOperation(
            targetCurrency: 'INR',
            scope: 'matched',
            exchangeRateSource: 'manual',
            rateTimestamp: timestamp,
            roundingMode: 'round',
          ),
        ],
        output: const OutputSpec(sheetNameSeed: 'currency_inr', openAutomatically: true, artifactKind: ArtifactKind.worksheet),
        needsUserConfirmation: false,
        clarifyingQuestions: const [],
        warnings: const [],
        confidence: 1,
      );
      final result = await const WorkbookExecutionEngine().execute(
        session: session,
        plan: plan,
        requestText: 'Convert currency to INR.',
        exchangeRateService: ManualExchangeRateService(
          snapshot: ExchangeRateSnapshot(
            sourceCurrency: 'USD',
            targetCurrency: 'INR',
            rate: 83.0,
            rateSource: 'manual',
            timestamp: timestamp,
          ),
        ),
      );

      expect(result.success, isTrue, reason: result.errors.join(' | '));
      expect(result.session.activeSheet.originalHeaders.single, 'Currency (₹)');
      expect((result.session.activeSheet.rows[0][0] as num).toDouble(), closeTo(8300, 0.001));
      expect((result.session.activeSheet.rows[1][0] as num).toDouble(), closeTo(16641.5, 0.001));
      expect((result.session.activeSheet.rows[2][0] as num).toDouble(), closeTo(24900, 0.001));
    });

    test('creates a pivot table showing average rating by city', () async {
      final result = await _executePlan('Create a pivot table showing average rating by city.');

      expect(result.success, isTrue);
      expect(result.session.activeSheet.originalHeaders.first, 'City');
      expect(result.session.activeSheet.originalHeaders.last, 'Average Rating');
      expect(result.session.activeSheet.rows, isNotEmpty);
    });
  });

  group('Plan client privacy', () {
    test('sends only request text and planning metadata', () async {
      Map<String, dynamic>? capturedBody;
      final client = MockClient((request) async {
        capturedBody = jsonDecode(request.body) as Map<String, dynamic>;
        return http.Response(
          jsonEncode({
          'ok': true,
          'request_id': 'req-123',
          'plan': {
            'schema_version': '1.0',
            'request_id': 'req-123',
            'intent': {'task_class': 'filter', 'summary': 'filter'},
            'semantic_targets': [],
            'operations': [
              {
                'type': 'filter_rows',
                'where': {
                  'logic': 'AND',
                  'conditions': [
                    {
                      'target_ref': 'rating',
                      'operator': 'gt',
                      'value': 3.9,
                      'value_type': 'number',
                      'case_sensitive': false,
                    }
                  ],
                },
              }
            ],
            'output': {
              'sheet_name_seed': 'rating',
              'open_automatically': true,
              'artifact_kind': 'worksheet',
            },
            'confidence': 1,
          },
          'model_id': 'test-model',
          'repaired': false,
          }),
          200,
          headers: {'content-type': 'application/json'},
        );
      });

      final api = PlanClient(endpoint: Uri.parse('https://example.com/v1/plan'), client: client);
      final response = await api.requestPlan(
        const PlanRequestPayload(
          requestId: 'req-privacy',
          requestText: 'Show restaurants having rating below 3.9.',
          locale: 'en-US',
          clientVersion: 'web',
        ),
      );

      expect(response.ok, isTrue);
      expect(capturedBody, isNotNull);
      expect(capturedBody!['request_text'], 'Show restaurants having rating below 3.9.');
      expect(capturedBody!['locale'], 'en-US');
      expect(capturedBody!['client_version'], 'web');
      expect(capturedBody!['request_text'], isNot(contains('Sheet1')));
      expect(jsonEncode(capturedBody!), isNot(contains('Pizza Hut')));
    });
  });

  group('Currency history', () {
    test('records exchange-rate details locally', () {
      final controller = InsightFlowController(
        planClient: PlanClient(
          endpoint: Uri.parse('https://example.com/v1/plan'),
          client: MockClient((request) async => http.Response('{}', 200)),
        ),
        importer: const WorkbookImporter(),
        exporter: const WorkbookExporter(),
        engine: const WorkbookExecutionEngine(),
        fallbackParser: const DeterministicFallbackParser(),
      );

      controller.configureCurrencyRate(
        sourceCurrency: 'usd',
        targetCurrency: 'inr',
        exchangeRate: 83.0,
        rateSource: 'manual',
        rateTimestamp: DateTime.utc(2026, 8, 25, 12),
      );

      expect(controller.hasCurrencyRateConfig, isTrue);
      expect(controller.operationHistory, hasLength(1));
      expect(controller.operationHistory.single.kind, 'currency_rate');
      expect(controller.operationHistory.single.metadata['source_currency'], 'USD');
      expect(controller.operationHistory.single.metadata['target_currency'], 'INR');
      expect(controller.operationHistory.single.metadata['rate_source'], 'manual');
    });
  });
}
