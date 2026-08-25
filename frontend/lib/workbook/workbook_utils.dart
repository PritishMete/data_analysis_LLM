import 'dart:math';

String normalizeHeader(String input) {
  return input.trim().toLowerCase().replaceAll(RegExp(r'[^a-z0-9]+'), ' ').replaceAll(RegExp(r'\s+'), ' ').trim();
}

String sanitizeSheetName(String input, {Set<String> existingNames = const {}}) {
  final invalid = RegExp(r'[:\\/?*\[\]]');
  var value = input.replaceAll(invalid, ' ').replaceAll(RegExp(r'\s+'), ' ').trim();
  if (value.isEmpty) {
    value = 'Result';
  }
  if (value.length > 31) {
    value = value.substring(0, 31).trim();
  }
  if (value.isEmpty) {
    value = 'Result';
  }

  var candidate = value;
  var suffix = 2;
  while (existingNames.contains(candidate)) {
    final room = max(1, 31 - 3);
    final base = value.length > room ? value.substring(0, room).trim() : value;
    candidate = '$base ($suffix)';
    if (candidate.length > 31) {
      candidate = candidate.substring(0, 31).trim();
    }
    suffix += 1;
  }
  return candidate;
}

String columnId(int index) => 'c${index + 1}';

List<String> tokenizeHeader(String header) {
  return normalizeHeader(header)
      .split(' ')
      .where((part) => part.isNotEmpty)
      .toList(growable: false);
}

bool isBlankValue(Object? value) => value == null || (value is String && value.trim().isEmpty);

