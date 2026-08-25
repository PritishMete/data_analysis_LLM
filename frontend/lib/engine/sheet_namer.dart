import 'package:insightflow/models/operation_plan.dart';

class SheetNameGenerator {
  const SheetNameGenerator();

  String fromRequest(String requestText) {
    final words = requestText
        .replaceAll(RegExp(r'[^A-Za-z0-9 ]+'), ' ')
        .split(RegExp(r'\s+'))
        .where((word) => word.isNotEmpty)
        .map((word) => _normalizeWord(word))
        .where((word) => word.length > 2)
        .take(3)
        .toList(growable: false);

    final seed = words.isEmpty ? 'InsightFlow' : words.join('_');
    return _shorten(seed);
  }

  String fromPlan(InsightFlowPlan plan, {String? requestText}) {
    final artifact = plan.output.artifactKind;
    final base = switch (artifact) {
      ArtifactKind.pivotTable => 'Pivot',
      ArtifactKind.chart => 'Chart',
      ArtifactKind.report => 'Report',
      ArtifactKind.worksheetAndChart => 'Result',
      ArtifactKind.worksheet => 'Sheet'
    };
    final requestSeed = requestText == null ? plan.output.sheetNameSeed : fromRequest(requestText);
    return _shorten('${base}_$requestSeed');
  }

  String _normalizeWord(String word) {
    final trimmed = word.replaceAll(RegExp(r'[^A-Za-z0-9]'), '');
    if (trimmed.isEmpty) {
      return '';
    }
    return trimmed[0].toUpperCase() + trimmed.substring(1).toLowerCase();
  }

  String _shorten(String value) {
    final compact = value.replaceAll(RegExp(r'[^A-Za-z0-9_]+'), '_').replaceAll(RegExp(r'_{2,}'), '_');
    return compact.length <= 24 ? compact : compact.substring(0, 24).replaceAll(RegExp(r'_+$'), '');
  }
}
