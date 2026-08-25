import 'dart:typed_data';

import 'package:excel/excel.dart';
import 'package:insightflow/workbook/currency_utils.dart';
import 'package:insightflow/workbook/workbook_models.dart';
import 'package:insightflow/workbook/workbook_utils.dart';

class WorkbookExporter {
  const WorkbookExporter();

  Uint8List exportToXlsx(WorkbookSession session) {
    final excel = Excel.createExcel();
    final defaultSheetName = excel.getDefaultSheet() ?? excel.sheets.keys.first;
    final usedNames = <String>{};

    if (session.sheets.isEmpty) {
      final blank = excel[defaultSheetName];
      blank.updateCell(
        CellIndex.indexByColumnRow(columnIndex: 0, rowIndex: 0),
        TextCellValue('Empty workbook'),
      );
      return Uint8List.fromList(excel.encode() ?? <int>[]);
    }

    for (var sheetIndex = 0; sheetIndex < session.sheets.length; sheetIndex++) {
      final sheet = session.sheets[sheetIndex];
      final safeName = sanitizeSheetName(sheet.name, existingNames: usedNames);
      usedNames.add(safeName);

      if (sheetIndex == 0) {
        if (defaultSheetName != safeName) {
          excel.rename(defaultSheetName, safeName);
        }
      } else {
        excel[safeName];
      }

      final target = excel[safeName];
      _writeSheet(target, sheet);
    }

    final activeSheet = sanitizeSheetName(session.activeSheet.name, existingNames: usedNames);
    excel.setDefaultSheet(activeSheet);
    return Uint8List.fromList(excel.encode() ?? <int>[]);
  }

  void _writeSheet(Sheet target, WorkbookSheet sheet) {
    final headerValues = sheet.originalHeaders.map(_toCellValue).toList(growable: false);
    for (var columnIndex = 0; columnIndex < headerValues.length; columnIndex++) {
      target.updateCell(
        CellIndex.indexByColumnRow(columnIndex: columnIndex, rowIndex: 0),
        headerValues[columnIndex] ?? TextCellValue(''),
      );
    }

    for (var rowIndex = 0; rowIndex < sheet.rows.length; rowIndex++) {
      final row = sheet.rows[rowIndex];
      for (var columnIndex = 0; columnIndex < sheet.columnCount; columnIndex++) {
        final value = columnIndex < row.length ? row[columnIndex] : null;
        final cellIndex = CellIndex.indexByColumnRow(columnIndex: columnIndex, rowIndex: rowIndex + 1);
        final cellValue = _toCellValue(value) ?? TextCellValue('');
        final header = columnIndex < sheet.originalHeaders.length ? sheet.originalHeaders[columnIndex] : '';
        final style = _currencyStyleIfNeeded(header, value);
        if (style == null) {
          target.updateCell(cellIndex, cellValue);
        } else {
          target.updateCell(cellIndex, cellValue, cellStyle: style);
        }
      }
    }

    for (var columnIndex = 0; columnIndex < sheet.columnCount; columnIndex++) {
      target.setColumnAutoFit(columnIndex);
    }
  }

  CellStyle? _currencyStyleIfNeeded(String header, Object? value) {
    final normalized = normalizeHeader(header);
    if (!looksLikeCurrencyHeader(normalized) || !_looksNumeric(value)) {
      return null;
    }
    return CellStyle(
      numberFormat: CustomNumericNumFormat(formatCode: r'[$₹-en-IN]#,##0.00'),
    );
  }

  bool _looksNumeric(Object? value) {
    return value is num || (value is String && double.tryParse(value.replaceAll(RegExp(r'[^0-9.\-]'), '')) != null);
  }

  CellValue? _toCellValue(Object? value) {
    if (value == null) {
      return null;
    }
    if (value is CellValue) {
      return value;
    }
    if (value is bool) {
      return BoolCellValue(value);
    }
    if (value is int) {
      return IntCellValue(value);
    }
    if (value is double) {
      return DoubleCellValue(value);
    }
    if (value is DateTime) {
      return DateTimeCellValue.fromDateTime(value);
    }
    return TextCellValue(value.toString());
  }
}
