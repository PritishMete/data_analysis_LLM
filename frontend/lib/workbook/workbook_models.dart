import 'package:insightflow/semantic/semantic_resolution.dart';

class WorkbookSession {
  final String workbookName;
  final List<WorkbookSheet> sheets;
  final int activeSheetIndex;
  final String? sourceFileName;
  final List<AnalysisChartSpec> charts;

  const WorkbookSession({
    required this.workbookName,
    required this.sheets,
    required this.activeSheetIndex,
    required this.sourceFileName,
    required this.charts
  });

  WorkbookSheet get activeSheet => sheets[activeSheetIndex];

  WorkbookSession copyWith({
    String? workbookName,
    List<WorkbookSheet>? sheets,
    int? activeSheetIndex,
    String? sourceFileName,
    List<AnalysisChartSpec>? charts
  }) {
    return WorkbookSession(
      workbookName: workbookName ?? this.workbookName,
      sheets: sheets ?? this.sheets.map((sheet) => sheet.clone()).toList(growable: false),
      activeSheetIndex: activeSheetIndex ?? this.activeSheetIndex,
      sourceFileName: sourceFileName ?? this.sourceFileName,
      charts: charts ?? this.charts.map((chart) => chart.clone()).toList(growable: false)
    );
  }

  WorkbookSession clone() => copyWith();

  WorkbookSession withActiveSheet(String sheetId) {
    final index = sheets.indexWhere((sheet) => sheet.id == sheetId);
    if (index < 0) {
      return this;
    }
    return copyWith(activeSheetIndex: index);
  }

  WorkbookSession withCharts(List<AnalysisChartSpec> nextCharts) {
    return copyWith(charts: nextCharts);
  }
}

class WorkbookSheet {
  final String id;
  final String name;
  final List<String> columnIds;
  final List<String> originalHeaders;
  final List<List<Object?>> rows;
  final List<ColumnProfile> profiles;
  final bool isOriginal;

  const WorkbookSheet({
    required this.id,
    required this.name,
    required this.columnIds,
    required this.originalHeaders,
    required this.rows,
    required this.profiles,
    required this.isOriginal
  });

  int get rowCount => rows.length;
  int get columnCount => columnIds.length;

  WorkbookSheet clone() {
    return WorkbookSheet(
      id: id,
      name: name,
      columnIds: List<String>.from(columnIds, growable: false),
      originalHeaders: List<String>.from(originalHeaders, growable: false),
      rows: rows.map((row) => List<Object?>.from(row, growable: false)).toList(growable: false),
      profiles: profiles.map((profile) => ColumnProfile(
        columnId: profile.columnId,
        header: profile.header,
        inferredType: profile.inferredType,
        textness: profile.textness,
        numericness: profile.numericness,
        booleanness: profile.booleanness,
        dateness: profile.dateness,
        headerTokens: List<String>.from(profile.headerTokens, growable: false)
      )).toList(growable: false),
      isOriginal: isOriginal
    );
  }

  WorkbookSheet copyWith({
    String? id,
    String? name,
    List<String>? columnIds,
    List<String>? originalHeaders,
    List<List<Object?>>? rows,
    List<ColumnProfile>? profiles,
    bool? isOriginal
  }) {
    return WorkbookSheet(
      id: id ?? this.id,
      name: name ?? this.name,
      columnIds: columnIds ?? List<String>.from(this.columnIds, growable: false),
      originalHeaders: originalHeaders ?? List<String>.from(this.originalHeaders, growable: false),
      rows: rows ?? this.rows.map((row) => List<Object?>.from(row, growable: false)).toList(growable: false),
      profiles: profiles ?? this.profiles.map((profile) => ColumnProfile(
        columnId: profile.columnId,
        header: profile.header,
        inferredType: profile.inferredType,
        textness: profile.textness,
        numericness: profile.numericness,
        booleanness: profile.booleanness,
        dateness: profile.dateness,
        headerTokens: List<String>.from(profile.headerTokens, growable: false)
      )).toList(growable: false),
      isOriginal: isOriginal ?? this.isOriginal
    );
  }
}

class AnalysisChartSpec {
  final String type;
  final String title;
  final String xLabel;
  final String yLabel;
  final List<ChartPoint> points;
  final List<String> categories;

  const AnalysisChartSpec({
    required this.type,
    required this.title,
    required this.xLabel,
    required this.yLabel,
    required this.points,
    required this.categories
  });

  AnalysisChartSpec clone() {
    return AnalysisChartSpec(
      type: type,
      title: title,
      xLabel: xLabel,
      yLabel: yLabel,
      points: points.map((point) => ChartPoint(x: point.x, y: point.y, label: point.label)).toList(growable: false),
      categories: List<String>.from(categories, growable: false)
    );
  }
}

class ChartPoint {
  final double x;
  final double y;
  final String label;

  const ChartPoint({
    required this.x,
    required this.y,
    required this.label
  });
}

