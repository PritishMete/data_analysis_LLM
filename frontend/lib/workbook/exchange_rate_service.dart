class ExchangeRateSnapshot {
  final String sourceCurrency;
  final String targetCurrency;
  final double rate;
  final String rateSource;
  final DateTime timestamp;

  const ExchangeRateSnapshot({
    required this.sourceCurrency,
    required this.targetCurrency,
    required this.rate,
    required this.rateSource,
    required this.timestamp
  });
}

abstract class ExchangeRateService {
  Future<ExchangeRateSnapshot?> resolve({
    required String targetCurrency
  });
}

class ManualExchangeRateService implements ExchangeRateService {
  final ExchangeRateSnapshot? snapshot;

  const ManualExchangeRateService({required this.snapshot});

  @override
  Future<ExchangeRateSnapshot?> resolve({
    required String targetCurrency
  }) async {
    if (snapshot == null) {
      return null;
    }
    if (snapshot!.targetCurrency.toUpperCase() != targetCurrency.toUpperCase()) {
      return null;
    }
    return snapshot;
  }
}

