import 'dart:typed_data';

import 'package:csv/csv.dart';
import 'package:excel/excel.dart';
import 'package:insightflow/semantic/semantic_resolution.dart';
import 'package:insightflow/workbook/workbook_models.dart';
import 'package:insightflow/workbook/workbook_utils.dart';

class WorkbookImportResult {
  final WorkbookSession session;
  final List<String> warnings;

  const WorkbookImportResult({
    required this.session,
    required this.warnings
  });
}

class WorkbookImporter {
  const WorkbookImporter();

  WorkbookImportResult importBytes(Uint8List bytes, {required String fileName}) {
    final lower = fileName.toLowerCase();
    if (lower.endsWith('.csv')) {
      return _importCsv(bytes, fileName: fileName);
    }
    return _importExcel(bytes, fileName: fileName);
  }

  WorkbookImportResult _importCsv(Uint8List bytes, {required String fileName}) {
    final text = String.fromCharCodes(bytes);
    final rows = Csv().decode(text);
    final normalized = rows
        .map((row) => row.map((cell) => _normalizeImportedCell(cell)).toList(growable: false))
        .toList(growable: false);
    final sheet = _buildSheet(
      sheetName: _sheetTitleFromFile(fileName),
      rows: normalized,
      original: true
    );
    return WorkbookImportResult(
      session: WorkbookSession(
        workbookName: fileName,
        sheets: [sheet],
        activeSheetIndex: 0,
        sourceFileName: fileName,
        charts: const []
      ),
      warnings: _collectWarnings(normalized)
    );
  }

  WorkbookImportResult _importExcel(Uint8List bytes, {required String fileName}) {
    final workbook = Excel.decodeBytes(bytes);
    final sheets = <WorkbookSheet>[];
    final warnings = <String>[];
    for (final entry in workbook.tables.entries) {
      final rawRows = entry.value.rows;
      final normalized = rawRows
          .map((row) => row.map((cell) => _normalizeImportedCell(cell?.value)).toList(growable: false))
          .toList(growable: false);
      sheets.add(_buildSheet(sheetName: entry.key, rows: normalized, original: sheets.isEmpty));
      warnings.addAll(_collectWarnings(normalized));
    }

    if (sheets.isEmpty) {
      sheets.add(_buildSheet(sheetName: _sheetTitleFromFile(fileName), rows: const [], original: true));
    }

    return WorkbookImportResult(
      session: WorkbookSession(
        workbookName: fileName,
        sheets: sheets,
        activeSheetIndex: 0,
        sourceFileName: fileName,
        charts: const []
      ),
      warnings: warnings.toSet().toList(growable: false)
    );
  }

  WorkbookSheet _buildSheet({
    required String sheetName,
    required List<List<Object?>> rows,
    required bool original
  }) {
    final paddedRows = _padRows(rows);
    final headers = paddedRows.isEmpty ? <Object?>[] : paddedRows.first;
    final columnCount = paddedRows.isEmpty ? 0 : paddedRows.first.length;
    final columnIds = List<String>.generate(columnCount, columnId, growable: false);
    final originalHeaders = <String>[];
    final displayHeaders = <String>[];
    final seen = <String, int>{};
    for (var i = 0; i < columnCount; i++) {
      final header = headers.isNotEmpty && i < headers.length ? headers[i]?.toString().trim() ?? '' : '';
      originalHeaders.add(header);
      final base = header.isEmpty ? 'Column ${i + 1}' : header;
      final count = seen[base] ?? 0;
      seen[base] = count + 1;
      displayHeaders.add(count == 0 ? base : '$base (${count + 1})');
    }

    final dataRows = paddedRows.length <= 1 ? <List<Object?>>[] : paddedRows.skip(1).map((row) => _padRow(row, columnCount)).toList(growable: false);
    final profiles = _buildProfiles(columnIds, displayHeaders, dataRows);

    return WorkbookSheet(
      id: sheetName,
      name: sheetName,
      columnIds: columnIds,
      originalHeaders: originalHeaders,
      rows: dataRows,
      profiles: profiles,
      isOriginal: original
    );
  }

  List<List<Object?>> _padRows(List<List<Object?>> rows) {
    final width = rows.fold<int>(0, (maxWidth, row) => row.length > maxWidth ? row.length : maxWidth);
    return rows.map((row) => _padRow(row, width)).toList(growable: false);
  }

  List<Object?> _padRow(List<Object?> row, int width) {
    if (row.length >= width) {
      return List<Object?>.from(row, growable: false);
    }
    final next = List<Object?>.from(row, growable: true);
    while (next.length < width) {
      next.add(null);
    }
    return next;
  }

  Object? _normalizeImportedCell(dynamic cell) {
    if (cell == null) {
      return null;
    }
    if (cell is String) {
      final trimmed = cell.trim();
      final lowered = trimmed.toLowerCase();
      if (['true', 'yes', 'y', '1', 't', 'on'].contains(lowered)) {
        return true;
      }
      if (['false', 'no', 'n', '0', 'f', 'off'].contains(lowered)) {
        return false;
      }
      final numeric = double.tryParse(trimmed.replaceAll(RegExp(r'[^0-9.\-]'), ''));
      if (numeric != null && trimmed.isNotEmpty) {
        return trimmed.contains('.') ? numeric : numeric.toInt();
      }
      final parsedDate = DateTime.tryParse(trimmed);
      if (parsedDate != null) {
        return parsedDate;
      }
      return trimmed;
    }
    if (cell is num || cell is bool || cell is DateTime) {
      return cell;
    }
    return cell.toString();
  }

  List<ColumnProfile> _buildProfiles(List<String> columnIds, List<String> headers, List<List<Object?>> rows) {
    final profiles = <ColumnProfile>[];
    for (var index = 0; index < columnIds.length; index++) {
      final columnValues = rows.map((row) => index < row.length ? row[index] : null).toList(growable: false);
      profiles.add(_profileColumn(columnIds[index], headers[index], columnValues));
    }
    return profiles;
  }

  ColumnProfile _profileColumn(String columnId, String header, List<Object?> values) {
    final nonNull = values.where((value) => !isBlankValue(value)).toList(growable: false);
    final numericCount = nonNull.where((value) => _looksNumeric(value)).length;
    final boolCount = nonNull.where((value) => _looksBoolean(value)).length;
    final dateCount = nonNull.where((value) => _looksDate(value)).length;
    final textCount = nonNull.whereType<String>().length;
    final total = nonNull.isEmpty ? 1 : nonNull.length;
    final headerTokens = tokenizeHeader(header);

    final ratingLike = headerTokens.any((token) => {'rating', 'score', 'stars'}.contains(token));
    final inferredType = ratingLike
        ? 'number'
        : boolCount >= total * 0.6
            ? 'boolean'
            : dateCount >= total * 0.6
                ? 'date'
                : numericCount >= total * 0.6
                    ? 'number'
                    : textCount >= total * 0.6
                        ? 'string'
                        : 'any';

    return ColumnProfile(
      columnId: columnId,
      header: header,
      inferredType: inferredType,
      textness: total == 0 ? 0 : textCount / total,
      numericness: total == 0 ? 0 : numericCount / total,
      booleanness: total == 0 ? 0 : boolCount / total,
      dateness: total == 0 ? 0 : dateCount / total,
      headerTokens: headerTokens
    );
  }

  bool _looksNumeric(Object? value) {
    if (value is num) {
      return true;
    }
    if (value is String) {
      return double.tryParse(value.replaceAll(RegExp(r'[^0-9.\-]'), '')) != null;
    }
    return false;
  }

  bool _looksBoolean(Object? value) {
    if (value is bool) {
      return true;
    }
    if (value is String) {
      final normalized = value.trim().toLowerCase();
      return {'yes', 'no', 'y', 'n', 'true', 'false', '1', '0', 't', 'f'}.contains(normalized);
    }
    if (value is num) {
      return value == 0 || value == 1;
    }
    return false;
  }

  bool _looksDate(Object? value) {
    if (value is DateTime) {
      return true;
    }
    if (value is String) {
      return DateTime.tryParse(value) != null;
    }
    return false;
  }

  String _sheetTitleFromFile(String fileName) {
    final name = fileName.replaceAll(RegExp(r'\.[^.]+$'), '');
    return name.isEmpty ? 'Sheet1' : name;
  }

  List<String> _collectWarnings(List<List<Object?>> rows) {
    final warnings = <String>[];
    if (rows.isEmpty) {
      warnings.add('Imported file is empty.');
    } else if (rows.first.isEmpty) {
      warnings.add('Imported file has no usable columns.');
    }
    return warnings;
  }
}
