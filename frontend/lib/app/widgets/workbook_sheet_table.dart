import 'dart:math';

import 'package:flutter/material.dart';
import 'package:insightflow/workbook/currency_utils.dart';
import 'package:insightflow/workbook/workbook_models.dart';
import 'package:insightflow/workbook/workbook_utils.dart';

class WorkbookSheetTable extends StatelessWidget {
  final WorkbookSheet sheet;

  const WorkbookSheetTable({
    super.key,
    required this.sheet,
  });

  @override
  Widget build(BuildContext context) {
    final labels = List<String>.generate(sheet.columnCount, (index) {
      final header = sheet.originalHeaders[index];
      return header.isEmpty ? 'Column ${index + 1}' : header;
    }, growable: false);

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    sheet.name,
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                ),
                Text('${sheet.rowCount} rows'),
              ],
            ),
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child: PaginatedDataTable(
                columns: labels.map((label) => DataColumn(label: Text(label))).toList(growable: false),
                source: _WorkbookDataSource(sheet),
                rowsPerPage: max(1, min(10, sheet.rowCount)),
                showFirstLastButtons: true,
                availableRowsPerPage: const [5, 10, 25, 50],
                columnSpacing: 20,
                horizontalMargin: 12,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _WorkbookDataSource extends DataTableSource {
  final WorkbookSheet sheet;

  _WorkbookDataSource(this.sheet);

  @override
  DataRow? getRow(int index) {
    if (index < 0 || index >= sheet.rows.length) {
      return null;
    }
    final row = sheet.rows[index];
    return DataRow(
      cells: List<DataCell>.generate(sheet.columnCount, (columnIndex) {
        final value = columnIndex < row.length ? row[columnIndex] : null;
        return DataCell(Text(_formatCell(sheet, columnIndex, value)));
      }, growable: false),
    );
  }

  @override
  bool get isRowCountApproximate => false;

  @override
  int get rowCount => sheet.rows.length;

  @override
  int get selectedRowCount => 0;

  String _formatCell(WorkbookSheet sheet, int columnIndex, Object? value) {
    if (value == null) {
      return '';
    }

    final header = columnIndex < sheet.originalHeaders.length ? sheet.originalHeaders[columnIndex] : '';
    final normalized = normalizeHeader(header);
    if (looksLikeCurrencyHeader(normalized) && value is num) {
      return formatCurrencyAmount(value, header.contains('₹') ? 'INR' : 'USD');
    }
    if (value is DateTime) {
      return value.toIso8601String();
    }
    return value.toString();
  }
}
