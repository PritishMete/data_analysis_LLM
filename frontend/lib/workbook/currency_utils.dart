const Map<String, String> kCurrencySymbols = {
  'USD': r'$',
  'INR': '₹',
  'GBP': '£',
  'AED': 'د.إ',
  'BDT': '৳',
  'RUB': '₽',
  'SGD': r'S$',
  'CAD': r'C$',
};

String normalizeCurrencyCode(String value) {
  final trimmed = value.trim().toUpperCase();
  const symbolToCode = {
    r'$': 'USD',
    '₹': 'INR',
    '£': 'GBP',
    '₽': 'RUB',
    '৳': 'BDT',
  };
  return symbolToCode[trimmed] ?? trimmed;
}

String currencySymbolForCode(String code) {
  final normalized = normalizeCurrencyCode(code);
  return kCurrencySymbols[normalized] ?? normalized;
}

bool looksLikeCurrencyHeader(String header) {
  final normalized = header.trim().toLowerCase();
  return normalized.contains('currency') ||
      normalized.contains('price') ||
      normalized.contains('amount') ||
      normalized.contains('cost') ||
      normalized.contains('revenue') ||
      normalized.contains('salary') ||
      normalized.contains('income') ||
      normalized.contains('value');
}

double? parseCurrencyAmount(Object? value, {String? sourceCurrency}) {
  if (value == null) {
    return null;
  }
  if (value is num) {
    return value.toDouble();
  }
  if (value is String) {
    final trimmed = value.trim();
    if (trimmed.isEmpty) {
      return null;
    }

    final cleaned = trimmed
        .replaceAll(RegExp(r'\b(?:USD|INR|GBP|AED|BDT|RUB|SGD|CAD)\b', caseSensitive: false), ' ')
        .replaceAll(RegExp(r'[₹$£₽৳,]'), '')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
    final numeric = double.tryParse(cleaned.replaceAll(RegExp(r'[^0-9.\-]'), ''));
    if (numeric != null) {
      return numeric;
    }

    if (sourceCurrency != null) {
      final sourceSymbol = currencySymbolForCode(sourceCurrency);
      final sourcePrefixRemoved = trimmed
          .replaceFirst(RegExp('^${RegExp.escape(sourceCurrency)}\\s*', caseSensitive: false), '')
          .replaceFirst(RegExp('^${RegExp.escape(sourceSymbol)}\\s*'), '')
          .trim();
      final fallback = double.tryParse(sourcePrefixRemoved.replaceAll(RegExp(r'[^0-9.\-]'), ''));
      if (fallback != null) {
        return fallback;
      }
    }
  }
  return null;
}

String formatCurrencyAmount(num amount, String currencyCode) {
  final normalized = normalizeCurrencyCode(currencyCode);
  final symbol = currencySymbolForCode(normalized);
  final text = _formatNumber(amount.toDouble());
  if (normalized == 'INR') {
    return '₹$text';
  }
  if (symbol == normalized) {
    return '$normalized $text';
  }
  return '$symbol$text';
}

String formatCurrencyHeader(String header, String currencyCode) {
  final normalized = normalizeCurrencyCode(currencyCode);
  if (normalizeCurrencyCode(header) == normalized) {
    return header;
  }
  if (normalizeCurrencyCode(header) == 'CURRENCY' || header.trim().toLowerCase() == 'currency') {
    if (normalized == 'INR') {
      return 'Currency (₹)';
    }
    return 'Currency ($normalized)';
  }
  return header;
}

String _formatNumber(double value) {
  final fixed = value.toStringAsFixed(2);
  final parts = fixed.split('.');
  final integer = parts.first;
  final fraction = parts.length > 1 ? parts[1] : '00';
  final buffer = StringBuffer();
  var digits = 0;
  for (var index = integer.length - 1; index >= 0; index -= 1) {
    buffer.write(integer[index]);
    digits += 1;
    if (digits % 3 == 0 && index > 0) {
      buffer.write(',');
    }
  }
  final formattedInteger = buffer.toString().split('').reversed.join();
  return '$formattedInteger.$fraction';
}
