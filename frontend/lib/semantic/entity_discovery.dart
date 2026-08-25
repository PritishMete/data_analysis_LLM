import 'package:insightflow/semantic/semantic_resolution.dart';

class EntityDiscoveryResult {
  final String entityValue;
  final String? columnId;
  final bool found;
  final List<String> candidateColumnIds;

  const EntityDiscoveryResult({
    required this.entityValue,
    required this.columnId,
    required this.found,
    required this.candidateColumnIds
  });
}

class LocalEntityDiscovery {
  const LocalEntityDiscovery();

  EntityDiscoveryResult discover(String entityValue, TableSnapshot snapshot) {
    final resolver = const LocalSemanticColumnResolver();
    final columnId = resolver.discoverEntityColumn(entityValue, snapshot);
    if (columnId == null) {
      return EntityDiscoveryResult(entityValue: entityValue, columnId: null, found: false, candidateColumnIds: const []);
    }

    return EntityDiscoveryResult(
      entityValue: entityValue,
      columnId: columnId,
      found: true,
      candidateColumnIds: [columnId]
    );
  }

  bool valueExistsInColumn(String entityValue, TableSnapshot snapshot, String columnId) {
    final needle = entityValue.toLowerCase();
    for (final row in snapshot.rows) {
      final value = row[columnId];
      if (value is String && value.toLowerCase().contains(needle)) {
        return true;
      }
    }
    return false;
  }
}

