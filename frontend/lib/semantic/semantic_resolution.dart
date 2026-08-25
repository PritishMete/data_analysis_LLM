import 'package:insightflow/models/operation_plan.dart';

class ColumnProfile {
  final String columnId;
  final String? header;
  final String inferredType;
  final double textness;
  final double numericness;
  final double booleanness;
  final double dateness;
  final List<String> headerTokens;

  const ColumnProfile({
    required this.columnId,
    required this.header,
    required this.inferredType,
    required this.textness,
    required this.numericness,
    required this.booleanness,
    required this.dateness,
    required this.headerTokens
  });
}

class TableSnapshot {
  final List<ColumnProfile> columns;
  final List<Map<String, Object?>> rows;

  const TableSnapshot({required this.columns, required this.rows});
}

class ColumnResolution {
  final String targetId;
  final String? resolvedColumnId;
  final double confidence;
  final bool needsUserSelection;
  final List<String> candidateColumnIds;

  const ColumnResolution({
    required this.targetId,
    required this.resolvedColumnId,
    required this.confidence,
    required this.needsUserSelection,
    required this.candidateColumnIds
  });
}

abstract class SemanticColumnResolver {
  ColumnResolution resolveTarget(SemanticTarget target, TableSnapshot snapshot);
  String? discoverEntityColumn(String entityValue, TableSnapshot snapshot);
}

class LocalSemanticColumnResolver implements SemanticColumnResolver {
  const LocalSemanticColumnResolver();

  @override
  ColumnResolution resolveTarget(SemanticTarget target, TableSnapshot snapshot) {
    final hint = target.hint.toLowerCase();
    final scored = <({String columnId, double score})>[];

    for (final column in snapshot.columns) {
      final score = _scoreColumn(hint, target.kind, column);
      if (score > 0) {
        scored.add((columnId: column.columnId, score: score));
      }
    }

    scored.sort((a, b) => b.score.compareTo(a.score));
    if (scored.isEmpty) {
      return ColumnResolution(
        targetId: target.targetId,
        resolvedColumnId: null,
        confidence: 0,
        needsUserSelection: true,
        candidateColumnIds: const []
      );
    }

    final best = scored.first;
    final second = scored.length > 1 ? scored[1] : null;
    final gap = second == null ? best.score : best.score - second.score;
    final needsUserSelection = best.score < 0.65 || gap < 0.15;

    return ColumnResolution(
      targetId: target.targetId,
      resolvedColumnId: needsUserSelection ? null : best.columnId,
      confidence: best.score,
      needsUserSelection: needsUserSelection,
      candidateColumnIds: scored.take(3).map((item) => item.columnId).toList(growable: false)
    );
  }

  @override
  String? discoverEntityColumn(String entityValue, TableSnapshot snapshot) {
    final needle = _normalize(entityValue);
    final best = <String, double>{};

    for (final column in snapshot.columns) {
      if (column.inferredType != 'string' && column.textness < 0.45) {
        continue;
      }
      var score = 0.0;
      final header = (column.header ?? '').toLowerCase();
      if (header.contains(needle)) {
        score += 0.5;
      }
      for (final row in snapshot.rows) {
        final value = row[column.columnId];
        if (value is String && _normalize(value).contains(needle)) {
          score += 0.2;
          break;
        }
      }
      if (score > 0) {
        best[column.columnId] = score;
      }
    }

    if (best.isEmpty) {
      return null;
    }
    final sorted = best.entries.toList()..sort((a, b) => b.value.compareTo(a.value));
    return sorted.first.key;
  }

  double _scoreColumn(String hint, String kind, ColumnProfile column) {
    var score = 0.0;
    final header = (column.header ?? '').toLowerCase();
    final headerJoined = column.headerTokens.join(' ');

    if (header == hint) score += 0.8;
    if (header.contains(hint)) score += 0.5;
    if (headerJoined.contains(hint)) score += 0.45;

    final synonymBoost = _synonymBoost(hint, header, kind, column);
    score += synonymBoost;

    score += _typeCompatibility(kind, column);
    return score.clamp(0, 1.0);
  }

  double _typeCompatibility(String kind, ColumnProfile column) {
    return switch (kind) {
      'flag' => column.booleanness > 0.4 ? 0.3 : 0,
      'metric' => column.numericness > 0.4 ? 0.3 : 0,
      'date' => column.dateness > 0.4 ? 0.3 : 0,
      'currency' => column.numericness > 0.4 ? 0.25 : 0,
      'entity' => column.textness > 0.3 ? 0.2 : 0,
      'text' => column.textness > 0.3 ? 0.2 : 0,
      'category' => column.textness > 0.3 ? 0.2 : 0,
      _ => 0.1
    };
  }

  double _synonymBoost(String hint, String header, String kind, ColumnProfile column) {
    const synonymMap = <String, List<String>>{
      'rating': ['score', 'stars', 'review'],
      'city': ['town', 'location', 'municipality'],
      'country': ['nation', 'region'],
      'gender': ['sex'],
      'bool': ['flag', 'is_', 'has_', 'enabled', 'available', 'yesno'],
      'delivery': ['online delivery', 'delivery'],
      'booking': ['table booking', 'reservation'],
      'restaurant': ['restaurant', 'business', 'entity', 'name']
    };

    final tokens = synonymMap[hint] ?? const [];
    for (final token in tokens) {
      if (header.contains(token)) {
        return 0.35;
      }
    }
    if (kind == 'flag' && (header.contains('is_') || header.contains('has_'))) {
      return 0.3;
    }
    if (column.headerTokens.contains(hint)) {
      return 0.4;
    }
    return 0;
  }
}

String _normalize(String value) {
  return value.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]+'), '');
}
