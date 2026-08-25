import 'dart:math';

import 'package:insightflow/engine/condition_evaluator.dart';
import 'package:insightflow/engine/sheet_namer.dart';
import 'package:insightflow/models/operation_plan.dart';
import 'package:insightflow/semantic/entity_discovery.dart';
import 'package:insightflow/semantic/semantic_resolution.dart';
import 'package:insightflow/validation/local_plan_validator.dart';
import 'package:insightflow/workbook/exchange_rate_service.dart';
import 'package:insightflow/workbook/currency_utils.dart';
import 'package:insightflow/workbook/workbook_models.dart';
import 'package:insightflow/workbook/workbook_utils.dart';

class WorkbookExecutionResult {
  final WorkbookSession session;
  final List<String> messages;
  final List<String> warnings;
  final List<String> errors;
  final String? activeSheetId;

  const WorkbookExecutionResult({
    required this.session,
    required this.messages,
    required this.warnings,
    required this.errors,
    required this.activeSheetId,
  });

  bool get success => errors.isEmpty;
}

class WorkbookPlanException implements Exception {
  final String code;
  final String message;

  const WorkbookPlanException(this.code, this.message);

  @override
  String toString() => '$code: $message';
}

class NeedsUserSelectionException extends WorkbookPlanException {
  final String targetId;
  final List<String> candidates;

  const NeedsUserSelectionException({
    required this.targetId,
    required this.candidates,
    required String message,
  }) : super('NEEDS_COLUMN_SELECTION', message);
}

class ValueNotFoundException extends WorkbookPlanException {
  const ValueNotFoundException(String message) : super('VALUE_NOT_FOUND', message);
}

class MissingPivotSpecificationException extends WorkbookPlanException {
  const MissingPivotSpecificationException(String message) : super('NEEDS_PIVOT_SPECIFICATION', message);
}

class MissingCurrencyRateException extends WorkbookPlanException {
  const MissingCurrencyRateException(String message) : super('MISSING_EXCHANGE_RATE', message);
}

class WorkbookExecutionEngine {
  final ConditionEvaluator _conditionEvaluator = const ConditionEvaluator();
  final SheetNameGenerator _sheetNameGenerator = const SheetNameGenerator();
  final LocalSemanticColumnResolver _semanticResolver = const LocalSemanticColumnResolver();
  final LocalEntityDiscovery _entityDiscovery = const LocalEntityDiscovery();

  const WorkbookExecutionEngine();

  ResolvedTargets prepareTargets({
    required WorkbookSession session,
    required InsightFlowPlan plan,
    Map<String, String> forcedTargetBindings = const {},
  }) {
    final snapshot = _snapshot(session.activeSheet);
    return _resolveTargets(plan, snapshot, forcedTargetBindings);
  }

  Future<WorkbookExecutionResult> execute({
    required WorkbookSession session,
    required InsightFlowPlan plan,
    required String requestText,
    required ExchangeRateService exchangeRateService,
    Map<String, String> forcedTargetBindings = const {},
  }) async {
    final resolution = prepareTargets(
      session: session,
      plan: plan,
      forcedTargetBindings: forcedTargetBindings,
    );
    final validator = const LocalPlanValidator();
    final validation = validator.validate(
      plan,
      canResolveTarget: (targetId) => resolution.resolutions[targetId]?.resolvedColumnId != null,
    );

    if (!validation.ok) {
      return WorkbookExecutionResult(
        session: session,
        messages: const [],
        warnings: const [],
        errors: validation.errors.isNotEmpty
            ? validation.errors
            : ['Unresolved targets: ${validation.unresolvedTargets.join(', ')}'],
        activeSheetId: session.activeSheet.id,
      );
    }

    final workingSheets = session.sheets.map(_MutableSheet.fromSheet).toList(growable: true);
    var activeSheet = workingSheets[session.activeSheetIndex];
    final originalSheetId = activeSheet.id;
    final messages = <String>[];
    final warnings = <String>[];
    final charts = session.charts.map((chart) => chart.clone()).toList(growable: true);
    var resultSheetName = _sheetNameGenerator.fromPlan(plan, requestText: requestText);
    var createdSheet = false;
    try {
      for (final operation in plan.operations) {
        switch (operation) {
          case FilterRowsOperation():
            final filteredRows = _applyFilter(activeSheet, operation.where, resolution);
            activeSheet.rows
              ..clear()
              ..addAll(filteredRows);
            activeSheet.refreshProfiles();
            messages.add('Filtered rows locally.');
          case SortRowsOperation():
            _sortRows(activeSheet, operation.keys, resolution);
            messages.add('Sorted rows locally.');
          case CategorizeColumnsOperation():
            final indices = _scopeColumns(activeSheet, operation.scope, resolution, mode: 'categorize');
            final outcome = _categorizeColumns(activeSheet, indices, operation.strategy);
            messages.addAll(outcome.messages);
            warnings.addAll(outcome.warnings);
          case NormalizeColumnsOperation():
            final indices = _scopeColumns(activeSheet, operation.scope, resolution, mode: 'normalize');
            final outcome = _normalizeColumns(activeSheet, indices, operation.actions);
            messages.addAll(outcome.messages);
            warnings.addAll(outcome.warnings);
          case ConvertCurrencyOperation():
            final outcome = await _convertCurrency(activeSheet, operation, exchangeRateService, resolution);
            messages.addAll(outcome.messages);
            warnings.addAll(outcome.warnings);
          case CreatePivotOperation():
            final pivot = _createPivot(activeSheet, operation, resolution);
            final pivotName = sanitizeSheetName(
              resultSheetName.isEmpty ? _sheetNameGenerator.fromPlan(plan, requestText: requestText) : resultSheetName,
              existingNames: workingSheets.map((sheet) => sheet.name).toSet(),
            );
            pivot.id = pivotName;
            pivot.name = pivotName;
            activeSheet = pivot;
            workingSheets.add(activeSheet);
            createdSheet = true;
            messages.add('Created pivot table locally.');
          case BuildChartOperation():
            final chartName = sanitizeSheetName(
              '${_sheetNameGenerator.fromPlan(plan, requestText: requestText)}_Chart',
              existingNames: workingSheets.map((sheet) => sheet.name).toSet(),
            );
            final chart = _buildChartSpec(activeSheet, operation, resolution, requestText);
            charts.add(chart);
            activeSheet = _MutableSheet.fromSheet(_createResultSheet(source: activeSheet, name: chartName, original: false));
            workingSheets.add(activeSheet);
            createdSheet = true;
            messages.add('Generated chart data locally.');
          case RenameSheetOperation():
            resultSheetName = sanitizeSheetName(
              operation.nameSeed,
              existingNames: workingSheets.map((sheet) => sheet.name).toSet(),
            );
            messages.add('Renamed result sheet.');
        }
      }

      if (!createdSheet) {
        final finalName = sanitizeSheetName(
          resultSheetName,
          existingNames: workingSheets.map((sheet) => sheet.name).toSet(),
        );
        activeSheet = _MutableSheet.fromSheet(_createResultSheet(source: activeSheet, name: finalName, original: false));
        workingSheets.add(activeSheet);
      } else {
        final finalName = sanitizeSheetName(
          resultSheetName,
          existingNames: workingSheets.map((sheet) => sheet.name).toSet(),
        );
        activeSheet.name = finalName;
        activeSheet.id = finalName;
        workingSheets[workingSheets.length - 1] = activeSheet;
      }

      final activeSheetIndex = workingSheets.length - 1;
      final sessionResult = WorkbookSession(
        workbookName: session.workbookName,
        sheets: workingSheets.map((sheet) => sheet.toSheet()).toList(growable: false),
        activeSheetIndex: activeSheetIndex,
        sourceFileName: session.sourceFileName,
        charts: charts,
      );

      return WorkbookExecutionResult(
        session: sessionResult,
        messages: messages,
        warnings: warnings,
        errors: const [],
        activeSheetId: sessionResult.activeSheet.id,
      );
    } on WorkbookPlanException catch (error) {
      return WorkbookExecutionResult(
        session: session,
        messages: messages,
        warnings: warnings,
        errors: [error.message],
        activeSheetId: originalSheetId,
      );
    } catch (error) {
      return WorkbookExecutionResult(
        session: session,
        messages: messages,
        warnings: warnings,
        errors: ['Unexpected workbook execution failure.'],
        activeSheetId: originalSheetId,
      );
    }
  }

  ResolvedTargets _resolveTargets(
    InsightFlowPlan plan,
    WorkbookSnapshot snapshot,
    Map<String, String> forcedBindings,
  ) {
    final resolutions = <String, ColumnResolution>{};
    final columnByTarget = <String, String>{};
    final targetsByName = <String, SemanticTarget>{};

    for (final target in plan.semanticTargets) {
      targetsByName[target.targetId.toLowerCase()] = target;
      targetsByName[target.hint.toLowerCase()] = target;

      if (forcedBindings.containsKey(target.targetId)) {
        final selectedColumn = forcedBindings[target.targetId]!;
        final resolution = ColumnResolution(
          targetId: target.targetId,
          resolvedColumnId: selectedColumn,
          confidence: 1,
          needsUserSelection: false,
          candidateColumnIds: [selectedColumn],
        );
        resolutions[target.targetId] = resolution;
        columnByTarget[target.targetId] = selectedColumn;
        continue;
      }

      final resolved = _resolveSemanticTarget(target, snapshot);
      resolutions[target.targetId] = resolved;
      if (resolved.resolvedColumnId != null) {
        columnByTarget[target.targetId] = resolved.resolvedColumnId!;
      }
    }

    return ResolvedTargets(
      resolutions: resolutions,
      columnByTarget: columnByTarget,
      targetsByName: targetsByName,
    );
  }

  ColumnResolution _resolveSemanticTarget(SemanticTarget target, WorkbookSnapshot snapshot) {
    if (target.kind == 'entity' && target.hint.trim().isNotEmpty) {
      final discovered = _entityDiscovery.discover(target.hint, snapshot.toTableSnapshot());
      if (!discovered.found || discovered.columnId == null) {
        throw ValueNotFoundException('Value not found: ${target.hint}');
      }
      return ColumnResolution(
        targetId: target.targetId,
        resolvedColumnId: discovered.columnId,
        confidence: 1,
        needsUserSelection: false,
        candidateColumnIds: discovered.candidateColumnIds,
      );
    }

    final resolverResult = _semanticResolver.resolveTarget(target, snapshot.toTableSnapshot());
    if (resolverResult.needsUserSelection || resolverResult.resolvedColumnId == null) {
      if (target.required) {
        throw NeedsUserSelectionException(
          targetId: target.targetId,
          candidates: resolverResult.candidateColumnIds,
          message: 'Select a column for ${target.hint}.',
        );
      }
    }
    return resolverResult;
  }

  WorkbookSnapshot _snapshot(WorkbookSheet sheet) {
    return WorkbookSnapshot(
      sheet: sheet,
      rows: sheet.rows.map((row) => List<Object?>.from(row, growable: true)).toList(growable: true),
      columnIds: List<String>.from(sheet.columnIds, growable: false),
      headers: List<String>.from(sheet.originalHeaders, growable: false),
      profiles: sheet.profiles
          .map(
            (profile) => ColumnProfile(
              columnId: profile.columnId,
              header: profile.header,
              inferredType: profile.inferredType,
              textness: profile.textness,
              numericness: profile.numericness,
              booleanness: profile.booleanness,
              dateness: profile.dateness,
              headerTokens: List<String>.from(profile.headerTokens, growable: false),
            ),
          )
          .toList(growable: false),
    );
  }

  WorkbookSheet _createResultSheet({
    required _MutableSheet source,
    required String name,
    required bool original,
  }) {
    return source.toSheet(id: name, name: name, isOriginal: original);
  }

  List<List<Object?>> _applyFilter(
    _MutableSheet sheet,
    ConditionGroup where,
    ResolvedTargets resolution,
  ) {
    final rows = sheet.rows
        .map((row) => _rowMap(sheet, row))
        .where((row) => _conditionEvaluator.evaluateGroup(
              where,
              row,
              (targetRef, rowMap) => _resolveValue(sheet, targetRef, rowMap, resolution),
            ))
        .map((row) => sheet.rowFromMap(row))
        .toList(growable: false);
    return rows;
  }

  void _sortRows(
    _MutableSheet sheet,
    List<SortKey> keys,
    ResolvedTargets resolution,
  ) {
    final indexedRows = sheet.rows.asMap().entries.map((entry) {
      return _IndexedRow(index: entry.key, row: entry.value);
    }).toList(growable: false);

    indexedRows.sort((left, right) {
      final leftMap = _rowMap(sheet, left.row);
      final rightMap = _rowMap(sheet, right.row);
      for (final key in keys) {
        final columnId = _resolveColumnRef(sheet, key.targetRef, resolution);
        final comparison = _compareValues(leftMap[columnId], rightMap[columnId]);
        if (comparison != 0) {
          return key.direction == SortDirection.asc ? comparison : -comparison;
        }
      }
      return left.index.compareTo(right.index);
    });

    sheet.rows
      ..clear()
      ..addAll(indexedRows.map((entry) => entry.row));
  }

  _ColumnMutationResult _categorizeColumns(
    _MutableSheet sheet,
    List<int> indices,
    String strategy,
  ) {
    final messages = <String>[];
    final warnings = <String>[];

    for (final index in indices) {
      final header = sheet.originalHeaders[index];
      final profile = sheet.profiles[index];
      final category = _pickStrategy(header, profile, strategy);
      for (var rowIndex = 0; rowIndex < sheet.rows.length; rowIndex++) {
        sheet.rows[rowIndex][index] = _applyNormalizationStrategy(sheet.rows[rowIndex][index], category);
      }
      messages.add('Categorized ${header.isEmpty ? 'Column ${index + 1}' : header}.');
    }

    sheet.refreshProfiles();
    return _ColumnMutationResult(messages: messages, warnings: warnings);
  }

  _ColumnMutationResult _normalizeColumns(
    _MutableSheet sheet,
    List<int> indices,
    List<String> actions,
  ) {
    final messages = <String>[];
    final warnings = <String>[];

    for (final index in indices) {
      final header = sheet.originalHeaders[index];
      for (var rowIndex = 0; rowIndex < sheet.rows.length; rowIndex++) {
        final current = index < sheet.rows[rowIndex].length ? sheet.rows[rowIndex][index] : null;
        sheet.rows[rowIndex][index] = _applyActions(current, header, actions);
      }
      messages.add('Normalized ${header.isEmpty ? 'Column ${index + 1}' : header}.');
    }

    sheet.refreshProfiles();
    return _ColumnMutationResult(messages: messages, warnings: warnings);
  }

  Future<_ColumnMutationResult> _convertCurrency(
    _MutableSheet sheet,
    ConvertCurrencyOperation op,
    ExchangeRateService exchangeRateService,
    ResolvedTargets resolution,
  ) async {
    final snapshot = await exchangeRateService.resolve(
      targetCurrency: op.targetCurrency.toUpperCase(),
    );
    if (snapshot == null) {
      throw const MissingCurrencyRateException(
        'Currency conversion requires a configured exchange-rate source and timestamp.',
      );
    }

    final indices = _scopeColumns(sheet, op.scope, resolution, mode: 'currency');
    if (indices.isEmpty) {
      throw const MissingCurrencyRateException('No currency column was found to convert.');
    }

    final messages = <String>[];
    var convertedCount = 0;
    var skippedCount = 0;
    for (final index in indices) {
      for (var rowIndex = 0; rowIndex < sheet.rows.length; rowIndex++) {
        final current = index < sheet.rows[rowIndex].length ? sheet.rows[rowIndex][index] : null;
        final numeric = parseCurrencyAmount(current, sourceCurrency: snapshot.sourceCurrency);
        if (numeric == null) {
          skippedCount += 1;
          continue;
        }
        final converted = _applyRounding(numeric * snapshot.rate, op.roundingMode);
        sheet.rows[rowIndex][index] = converted;
        convertedCount += 1;
      }

      final header = sheet.originalHeaders[index];
      if (normalizeHeader(header).contains('currency') && normalizeCurrencyCode(snapshot.targetCurrency) == 'INR') {
        sheet.originalHeaders[index] = formatCurrencyHeader(header, snapshot.targetCurrency);
      }
      messages.add('Converted ${sheet.originalHeaders[index]} to ${snapshot.targetCurrency}.');
    }

    sheet.refreshProfiles();
    if (skippedCount > 0) {
      messages.add('Converted $convertedCount values and skipped $skippedCount non-currency cells.');
    }
    return _ColumnMutationResult(messages: messages, warnings: const []);
  }

  AnalysisChartSpec _buildChartSpec(
    _MutableSheet sheet,
    BuildChartOperation op,
    ResolvedTargets resolution,
    String requestText,
  ) {
    final xColumnId = _resolveColumnRef(sheet, op.x, resolution);
    final yColumns = op.y.isEmpty
        ? <String>[xColumnId]
        : op.y.map((ref) => _resolveColumnRef(sheet, ref, resolution)).toList(growable: false);
    final xIndex = sheet.columnIds.indexOf(xColumnId);
    final yIndex = sheet.columnIds.indexOf(yColumns.first);
    if (xIndex < 0 || yIndex < 0) {
      throw NeedsUserSelectionException(
        targetId: op.x,
        candidates: const [],
        message: 'Select a chart column.',
      );
    }

    final points = <ChartPoint>[];
    for (var rowIndex = 0; rowIndex < sheet.rows.length; rowIndex++) {
      final row = sheet.rows[rowIndex];
      final label = xIndex < row.length ? row[xIndex]?.toString() ?? '' : '';
      final numeric = _tryNumeric(yIndex < row.length ? row[yIndex] : null) ?? 0;
      points.add(ChartPoint(x: rowIndex.toDouble(), y: numeric, label: label));
    }

    return AnalysisChartSpec(
      type: op.chartType,
      title: requestText,
      xLabel: sheet.originalHeaders[xIndex].isEmpty ? sheet.columnIds[xIndex] : sheet.originalHeaders[xIndex],
      yLabel: sheet.originalHeaders[yIndex].isEmpty ? sheet.columnIds[yIndex] : sheet.originalHeaders[yIndex],
      points: points,
      categories: points.map((point) => point.label).toList(growable: false),
    );
  }

  _MutableSheet _createPivot(
    _MutableSheet sheet,
    CreatePivotOperation op,
    ResolvedTargets resolution,
  ) {
    if (op.rows.isEmpty || op.values.isEmpty) {
      throw const MissingPivotSpecificationException('Choose pivot rows and value fields locally.');
    }

    final rowIds = op.rows.map((ref) => _resolveColumnRef(sheet, ref, resolution)).toList(growable: false);
    final columnIds = op.columns.map((ref) => _resolveColumnRef(sheet, ref, resolution)).toList(growable: false);
    final valueId = _resolveColumnRef(sheet, op.values.first.targetRef, resolution);
    final valueIndex = sheet.columnIds.indexOf(valueId);

    if (valueIndex < 0) {
      throw NeedsUserSelectionException(
        targetId: op.values.first.targetRef,
        candidates: const [],
        message: 'Select a pivot value field.',
      );
    }

    final groups = <String, List<_PivotEntry>>{};
    final keyOrder = <String>[];
    for (final row in sheet.rows) {
      final rowKey = rowIds.map((columnId) => _stringify(_rowValue(sheet, row, columnId))).join(' | ');
      final columnKey = columnIds.isEmpty ? 'all' : columnIds.map((columnId) => _stringify(_rowValue(sheet, row, columnId))).join(' | ');
      final entry = _PivotEntry(
        rowLabelValues: rowIds.map((columnId) => _rowValue(sheet, row, columnId)).toList(growable: false),
        columnLabel: columnKey,
        metricValue: _tryNumeric(_rowValue(sheet, row, valueId)),
      );
      groups.putIfAbsent(rowKey, () {
        keyOrder.add(rowKey);
        return <_PivotEntry>[];
      }).add(entry);
    }

    final distinctColumns = columnIds.isEmpty
        ? ['all']
        : groups.values
            .expand((entries) => entries.map((entry) => entry.columnLabel))
            .toSet()
            .toList(growable: false);

    final headers = [
      ...rowIds.map((columnId) => _headerFor(sheet, columnId)),
      if (columnIds.isEmpty) _aggregateHeader(sheet, valueId, op.values.first.aggregation) else ...distinctColumns.map((column) => column),
    ];

    final rows = <List<Object?>>[];
    for (final rowKey in keyOrder) {
      final entries = groups[rowKey] ?? const <_PivotEntry>[];
      final first = entries.isEmpty ? null : entries.first;
      final rowValues = <Object?>[];
      if (first != null) {
        rowValues.addAll(first.rowLabelValues);
      }

      if (columnIds.isEmpty) {
        final values = entries.map((entry) => entry.metricValue).whereType<double>().toList(growable: false);
        rowValues.add(_aggregate(values, op.values.first.aggregation));
      } else {
        for (final columnKey in distinctColumns) {
          final values = entries.where((entry) => entry.columnLabel == columnKey).map((entry) => entry.metricValue).whereType<double>().toList(growable: false);
          rowValues.add(_aggregate(values, op.values.first.aggregation));
        }
      }
      rows.add(rowValues);
    }

    final outputColumnIds = List<String>.generate(headers.length, columnId, growable: false);
    final profiles = _buildProfiles(outputColumnIds, headers, rows);
    return _MutableSheet(
      id: _sheetNameGenerator.fromRequest('pivot'),
      name: _sheetNameGenerator.fromRequest('pivot'),
      columnIds: outputColumnIds,
      originalHeaders: headers,
      rows: rows,
      profiles: profiles,
      isOriginal: false,
    );
  }

  List<ColumnProfile> _buildProfiles(List<String> columnIds, List<String> headers, List<List<Object?>> rows) {
    return List<ColumnProfile>.generate(columnIds.length, (index) {
      final values = rows.map((row) => index < row.length ? row[index] : null).toList(growable: false);
      final nonNull = values.where((value) => !isBlankValue(value)).toList(growable: false);
      final numericCount = nonNull.whereType<num>().length;
      final textCount = nonNull.whereType<String>().length;
      final boolCount = nonNull.whereType<bool>().length;
      final dateCount = nonNull.whereType<DateTime>().length;
      final total = max(1, nonNull.length);
      return ColumnProfile(
        columnId: columnIds[index],
        header: headers[index],
        inferredType: numericCount >= total * 0.6
            ? 'number'
            : boolCount >= total * 0.6
                ? 'boolean'
                : dateCount >= total * 0.6
                    ? 'date'
                    : textCount >= total * 0.6
                        ? 'string'
                        : 'any',
        textness: textCount / total,
        numericness: numericCount / total,
        booleanness: boolCount / total,
        dateness: dateCount / total,
        headerTokens: tokenizeHeader(headers[index]),
      );
    }, growable: false);
  }

  int _pickStrategy(String header, ColumnProfile profile, String explicitStrategy) {
    final normalizedHeader = normalizeHeader(header);
    if (normalizedHeader.contains('country')) return 1;
    if (normalizedHeader.contains('gender')) return 2;
    if (normalizedHeader.contains('bool') || normalizedHeader.contains('flag') || normalizedHeader.contains('delivery') || normalizedHeader.contains('booking')) return 3;
    if (normalizedHeader.contains('currency') || normalizedHeader.contains('price') || normalizedHeader.contains('amount') || normalizedHeader.contains('cost')) return 4;
    if (normalizedHeader.contains('date') || normalizedHeader.contains('time')) return 5;
    if (normalizedHeader.contains('city')) return 6;
    if (normalizedHeader.contains('rating') || normalizedHeader.contains('score')) return 7;
    if (explicitStrategy == 'header_based') {
      return profile.textness > 0.5 ? 8 : 0;
    }
    if (explicitStrategy == 'type_based') {
      if (profile.booleanness > 0.5) return 3;
      if (profile.numericness > 0.5) return 4;
      if (profile.dateness > 0.5) return 5;
      return 8;
    }
    if (profile.booleanness > 0.5) return 3;
    if (profile.numericness > 0.5) return 4;
    if (profile.dateness > 0.5) return 5;
    return 8;
  }

  Object? _applyNormalizationStrategy(Object? value, int strategy) {
    return switch (strategy) {
      1 => _normalizeCountry(value),
      2 => _normalizeGender(value),
      3 => _normalizeBoolean(value),
      4 => _normalizeNumericText(value),
      5 => _normalizeDateText(value),
      6 => _normalizeCity(value),
      7 => _normalizeRating(value),
      _ => _trimWhitespace(value),
    };
  }

  Object? _applyActions(Object? value, String header, List<String> actions) {
    var result = value;
    for (final action in actions) {
      result = switch (action) {
        'trim_whitespace' => _trimWhitespace(result),
        'standardize_case' => _standardizeCase(result),
        'parse_dates' => _normalizeDateText(result),
        'parse_numbers' => _normalizeNumericText(result),
        'split_text' => result,
        'deduplicate_labels' => result,
        'fill_blanks' => result ?? '',
        'remove_currency_symbols' => _removeCurrencySymbols(result),
        _ => result,
      };
    }

    final normalizedHeader = normalizeHeader(header);
    if (normalizedHeader.contains('country')) return _normalizeCountry(result);
    if (normalizedHeader.contains('region')) return _normalizeRegion(result);
    if (normalizedHeader.contains('city')) return _normalizeCity(result);
    if (normalizedHeader.contains('gender')) return _normalizeGender(result);
    if (normalizedHeader.contains('bool') || normalizedHeader.contains('flag') || normalizedHeader.contains('delivery') || normalizedHeader.contains('booking')) return _normalizeBoolean(result);
    if (normalizedHeader.contains('rating') || normalizedHeader.contains('score')) return _normalizeRating(result);
    return result;
  }

  Object? _trimWhitespace(Object? value) => value is String ? value.trim() : value;

  Object? _standardizeCase(Object? value) => value is String ? value.trim().toLowerCase() : value;

  Object? _removeCurrencySymbols(Object? value) => value is String ? value.replaceAll(RegExp(r'[₹$€£,]'), '').trim() : value;

  Object? _normalizeCountry(Object? value) {
    if (value == null) return null;
    final text = value.toString().trim();
    final normalized = text.toLowerCase();
    const map = {
      'usa': 'United States',
      'u.s.a.': 'United States',
      'us': 'United States',
      'uk': 'United Kingdom',
      'india': 'India',
      'in': 'India',
    };
    return map[normalized] ?? _titleCase(text);
  }

  Object? _normalizeRegion(Object? value) {
    if (value == null) return null;
    return _titleCase(value.toString());
  }

  Object? _normalizeCity(Object? value) {
    if (value == null) return null;
    return _titleCase(value.toString());
  }

  Object? _normalizeGender(Object? value) {
    if (value == null) return null;
    final normalized = value.toString().trim().toLowerCase();
    if (['m', 'male', 'man'].contains(normalized)) return 'Male';
    if (['f', 'female', 'woman'].contains(normalized)) return 'Female';
    return _titleCase(normalized);
  }

  Object? _normalizeBoolean(Object? value) {
    if (value == null) return null;
    final normalized = value.toString().trim().toLowerCase();
    if (['yes', 'y', 'true', '1', 't', 'available', 'on'].contains(normalized)) return true;
    if (['no', 'n', 'false', '0', 'f', 'not', 'unavailable', 'off'].contains(normalized)) return false;
    return value;
  }

  Object? _normalizeNumericText(Object? value) {
    if (value == null) return null;
    if (value is num) return value;
    final cleaned = value.toString().replaceAll(RegExp(r'[^0-9.\-]'), '');
    return double.tryParse(cleaned) ?? value;
  }

  Object? _normalizeDateText(Object? value) {
    if (value == null) return null;
    if (value is DateTime) return value;
    return DateTime.tryParse(value.toString()) ?? value;
  }

  Object? _normalizeRating(Object? value) {
    final numeric = _normalizeNumericText(value);
    if (numeric is double) {
      return double.parse(numeric.toStringAsFixed(2));
    }
    if (numeric is num) {
      return numeric.toDouble();
    }
    return numeric;
  }

  String _titleCase(String input) {
    return input
        .split(RegExp(r'\s+'))
        .where((part) => part.isNotEmpty)
        .map((part) => part[0].toUpperCase() + part.substring(1).toLowerCase())
        .join(' ');
  }

  Object? _resolveValue(_MutableSheet sheet, String targetRef, Map<String, Object?> row, ResolvedTargets resolution) {
    final columnId = _resolveColumnRef(sheet, targetRef, resolution);
    return row[columnId];
  }

  String _resolveColumnRef(_MutableSheet sheet, String targetRef, ResolvedTargets resolution) {
    if (resolution.columnByTarget.containsKey(targetRef)) {
      return resolution.columnByTarget[targetRef]!;
    }

    if (sheet.columnIds.contains(targetRef)) {
      return targetRef;
    }

    final normalized = normalizeHeader(targetRef);
    final exact = <String>[];
    for (var index = 0; index < sheet.columnIds.length; index++) {
      final header = normalizeHeader(sheet.originalHeaders[index]);
      if (header == normalized) {
        exact.add(sheet.columnIds[index]);
      }
    }

    if (exact.length == 1) {
      return exact.first;
    }
    if (exact.length > 1) {
      throw NeedsUserSelectionException(
        targetId: targetRef,
        candidates: exact,
        message: 'Select a column for $targetRef.',
      );
    }

    final fuzzy = <String>[];
    for (var index = 0; index < sheet.columnIds.length; index++) {
      final header = normalizeHeader(sheet.originalHeaders[index]);
      if (header.contains(normalized) || normalized.contains(header)) {
        fuzzy.add(sheet.columnIds[index]);
      }
    }

    if (fuzzy.length == 1) {
      return fuzzy.first;
    }
    if (fuzzy.length > 1) {
      throw NeedsUserSelectionException(
        targetId: targetRef,
        candidates: fuzzy,
        message: 'Select a column for $targetRef.',
      );
    }

    throw NeedsUserSelectionException(
      targetId: targetRef,
      candidates: sheet.columnIds,
      message: 'Select a column for $targetRef.',
    );
  }

  List<int> _scopeColumns(
    _MutableSheet sheet,
    String scope,
    ResolvedTargets resolution, {
    required String mode,
  }) {
    switch (scope) {
      case 'all':
        return List<int>.generate(sheet.columnIds.length, (index) => index, growable: false);
      case 'selected':
        return resolution.columnByTarget.values
            .map((columnId) => sheet.columnIds.indexOf(columnId))
            .where((index) => index >= 0)
            .toSet()
            .toList(growable: false);
      case 'matched':
        return List<int>.generate(sheet.columnIds.length, (index) => index, growable: false)
            .where((index) => _isMatchedColumn(sheet, index, mode))
            .toList(growable: false);
      default:
        return List<int>.generate(sheet.columnIds.length, (index) => index, growable: false);
    }
  }

  bool _isMatchedColumn(_MutableSheet sheet, int index, String mode) {
    final header = normalizeHeader(sheet.originalHeaders[index]);
    final profile = sheet.profiles[index];
    final candidates = <bool>[
      header.contains('country'),
      header.contains('region'),
      header.contains('city'),
      header.contains('gender'),
      header.contains('rating'),
      header.contains('score'),
      header.contains('currency'),
      header.contains('price'),
      header.contains('amount'),
      header.contains('cost'),
      header.contains('bool'),
      header.contains('flag'),
      header.contains('delivery'),
      header.contains('booking'),
      header.contains('date'),
      header.contains('time'),
    ];
    if (mode == 'currency') {
      return header.contains('currency') || header.contains('price') || header.contains('amount') || header.contains('cost');
    }
    if (mode == 'normalize') {
      return candidates.any((item) => item) || profile.booleanness > 0.5 || profile.numericness > 0.5 || profile.dateness > 0.5;
    }
    return candidates.any((item) => item) || profile.textness > 0.5;
  }

  String _headerFor(_MutableSheet sheet, String columnId) {
    final index = sheet.columnIds.indexOf(columnId);
    if (index < 0) return columnId;
    return sheet.originalHeaders[index].isEmpty ? columnId : sheet.originalHeaders[index];
  }

  String _aggregateHeader(_MutableSheet sheet, String valueId, Aggregation aggregation) {
    final header = _headerFor(sheet, valueId);
    final label = switch (aggregation) {
      Aggregation.sum => 'Sum',
      Aggregation.count => 'Count',
      Aggregation.average => 'Average',
      Aggregation.min => 'Min',
      Aggregation.max => 'Max',
      Aggregation.median => 'Median',
      Aggregation.distinctCount => 'Distinct Count',
    };
    return '$label $header';
  }

  Object? _rowValue(_MutableSheet sheet, List<Object?> row, String columnId) {
    final index = sheet.columnIds.indexOf(columnId);
    if (index < 0 || index >= row.length) return null;
    return row[index];
  }

  Map<String, Object?> _rowMap(_MutableSheet sheet, List<Object?> row) {
    final mapped = <String, Object?>{};
    for (var index = 0; index < sheet.columnIds.length; index++) {
      mapped[sheet.columnIds[index]] = index < row.length ? row[index] : null;
    }
    return mapped;
  }

  String _stringify(Object? value) => value?.toString() ?? '';

  int _compareValues(Object? a, Object? b) {
    if (a == null && b == null) return 0;
    if (a == null) return -1;
    if (b == null) return 1;

    final numA = _tryNumeric(a);
    final numB = _tryNumeric(b);
    if (numA != null && numB != null) {
      return numA.compareTo(numB);
    }

    final dateA = _tryDate(a);
    final dateB = _tryDate(b);
    if (dateA != null && dateB != null) {
      return dateA.compareTo(dateB);
    }

    return a.toString().toLowerCase().compareTo(b.toString().toLowerCase());
  }

  double? _tryNumeric(Object? value) {
    if (value is num) return value.toDouble();
    if (value is String) {
      final cleaned = value.replaceAll(RegExp(r'[^0-9.\-]'), '');
      return double.tryParse(cleaned);
    }
    return null;
  }

  DateTime? _tryDate(Object? value) {
    if (value is DateTime) return value;
    if (value is String) return DateTime.tryParse(value);
    return null;
  }

  double _applyRounding(double value, String mode) {
    return switch (mode) {
      'floor' => value.floorToDouble(),
      'ceil' => value.ceilToDouble(),
      'bankers' => _bankersRound(value),
      _ => double.parse(value.toStringAsFixed(2)),
    };
  }

  double _bankersRound(double value) {
    final floorValue = value.floorToDouble();
    final fraction = value - floorValue;
    if (fraction > 0.5) {
      return value.ceilToDouble();
    }
    if (fraction < 0.5) {
      return floorValue;
    }
    return floorValue.toInt().isEven ? floorValue : value.ceilToDouble();
  }

  double _aggregate(List<double> values, Aggregation aggregation) {
    if (values.isEmpty) {
      return 0;
    }
    switch (aggregation) {
      case Aggregation.sum:
        return values.fold<double>(0, (sum, value) => sum + value);
      case Aggregation.count:
        return values.length.toDouble();
      case Aggregation.average:
        return values.fold<double>(0, (sum, value) => sum + value) / values.length;
      case Aggregation.min:
        return values.reduce(min);
      case Aggregation.max:
        return values.reduce(max);
      case Aggregation.median:
        final sorted = List<double>.from(values)..sort();
        return sorted[sorted.length ~/ 2];
      case Aggregation.distinctCount:
        return values.toSet().length.toDouble();
    }
  }
}

class WorkbookSnapshot {
  final WorkbookSheet sheet;
  final List<List<Object?>> rows;
  final List<String> columnIds;
  final List<String> headers;
  final List<ColumnProfile> profiles;

  const WorkbookSnapshot({
    required this.sheet,
    required this.rows,
    required this.columnIds,
    required this.headers,
    required this.profiles,
  });

  TableSnapshot toTableSnapshot() {
    return TableSnapshot(
      columns: profiles,
      rows: rows.map((row) {
        final mapped = <String, Object?>{};
        for (var index = 0; index < columnIds.length; index++) {
          mapped[columnIds[index]] = index < row.length ? row[index] : null;
        }
        return mapped;
      }).toList(growable: false),
    );
  }
}

class ResolvedTargets {
  final Map<String, ColumnResolution> resolutions;
  final Map<String, String> columnByTarget;
  final Map<String, SemanticTarget> targetsByName;

  const ResolvedTargets({
    required this.resolutions,
    required this.columnByTarget,
    required this.targetsByName,
  });
}

class _MutableSheet {
  String id;
  String name;
  List<String> columnIds;
  List<String> originalHeaders;
  List<List<Object?>> rows;
  List<ColumnProfile> profiles;
  bool isOriginal;

  _MutableSheet({
    required this.id,
    required this.name,
    required this.columnIds,
    required this.originalHeaders,
    required this.rows,
    required this.profiles,
    required this.isOriginal,
  });

  factory _MutableSheet.fromSheet(WorkbookSheet sheet) {
    return _MutableSheet(
      id: sheet.id,
      name: sheet.name,
      columnIds: List<String>.from(sheet.columnIds, growable: false),
      originalHeaders: List<String>.from(sheet.originalHeaders, growable: false),
      rows: sheet.rows.map((row) => List<Object?>.from(row, growable: true)).toList(growable: true),
      profiles: sheet.profiles
          .map(
            (profile) => ColumnProfile(
              columnId: profile.columnId,
              header: profile.header,
              inferredType: profile.inferredType,
              textness: profile.textness,
              numericness: profile.numericness,
              booleanness: profile.booleanness,
              dateness: profile.dateness,
              headerTokens: List<String>.from(profile.headerTokens, growable: false),
            ),
          )
          .toList(growable: false),
      isOriginal: sheet.isOriginal,
    );
  }

  WorkbookSheet toSheet({String? id, String? name, bool? isOriginal}) {
    return WorkbookSheet(
      id: id ?? this.id,
      name: name ?? this.name,
      columnIds: List<String>.from(columnIds, growable: false),
      originalHeaders: List<String>.from(originalHeaders, growable: false),
      rows: rows.map((row) => List<Object?>.from(row, growable: false)).toList(growable: false),
      profiles: profiles
          .map(
            (profile) => ColumnProfile(
              columnId: profile.columnId,
              header: profile.header,
              inferredType: profile.inferredType,
              textness: profile.textness,
              numericness: profile.numericness,
              booleanness: profile.booleanness,
              dateness: profile.dateness,
              headerTokens: List<String>.from(profile.headerTokens, growable: false),
            ),
          )
          .toList(growable: false),
      isOriginal: isOriginal ?? this.isOriginal,
    );
  }

  Map<String, Object?> rowMap(List<Object?> row) {
    final mapped = <String, Object?>{};
    for (var index = 0; index < columnIds.length; index++) {
      mapped[columnIds[index]] = index < row.length ? row[index] : null;
    }
    return mapped;
  }

  List<Object?> rowFromMap(Map<String, Object?> rowMap) {
    return columnIds.map((columnId) => rowMap[columnId]).toList(growable: false);
  }

  void refreshProfiles() {
    profiles = List<ColumnProfile>.generate(columnIds.length, (index) {
      final values = rows.map((row) => index < row.length ? row[index] : null).toList(growable: false);
      final nonNull = values.where((value) => !isBlankValue(value)).toList(growable: false);
      final numericCount = nonNull.whereType<num>().length;
      final textCount = nonNull.whereType<String>().length;
      final boolCount = nonNull.whereType<bool>().length;
      final dateCount = nonNull.whereType<DateTime>().length;
      final total = max(1, nonNull.length);
      return ColumnProfile(
        columnId: columnIds[index],
        header: originalHeaders[index],
        inferredType: numericCount >= total * 0.6
            ? 'number'
            : boolCount >= total * 0.6
                ? 'boolean'
                : dateCount >= total * 0.6
                    ? 'date'
                    : textCount >= total * 0.6
                        ? 'string'
                        : 'any',
        textness: textCount / total,
        numericness: numericCount / total,
        booleanness: boolCount / total,
        dateness: dateCount / total,
        headerTokens: tokenizeHeader(originalHeaders[index]),
      );
    }, growable: false);
  }
}

class _IndexedRow {
  final int index;
  final List<Object?> row;

  const _IndexedRow({required this.index, required this.row});
}

class _PivotEntry {
  final List<Object?> rowLabelValues;
  final String columnLabel;
  final double? metricValue;

  const _PivotEntry({
    required this.rowLabelValues,
    required this.columnLabel,
    required this.metricValue,
  });
}

class _ColumnMutationResult {
  final List<String> messages;
  final List<String> warnings;

  const _ColumnMutationResult({
    required this.messages,
    required this.warnings,
  });
}
