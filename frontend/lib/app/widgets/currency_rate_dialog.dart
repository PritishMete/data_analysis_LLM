import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:insightflow/workbook/exchange_rate_service.dart';

class CurrencyRateDialogResult {
  final String sourceCurrency;
  final String targetCurrency;
  final double exchangeRate;
  final String rateSource;
  final DateTime rateTimestamp;

  const CurrencyRateDialogResult({
    required this.sourceCurrency,
    required this.targetCurrency,
    required this.exchangeRate,
    required this.rateSource,
    required this.rateTimestamp,
  });
}

Future<CurrencyRateDialogResult?> showCurrencyRateDialog(
  BuildContext context, {
  ExchangeRateSnapshot? initialSnapshot,
}) {
  return showDialog<CurrencyRateDialogResult>(
    context: context,
    barrierDismissible: false,
    builder: (dialogContext) {
      return _CurrencyRateDialog(initialSnapshot: initialSnapshot);
    },
  );
}

class _CurrencyRateDialog extends StatefulWidget {
  final ExchangeRateSnapshot? initialSnapshot;

  const _CurrencyRateDialog({required this.initialSnapshot});

  @override
  State<_CurrencyRateDialog> createState() => _CurrencyRateDialogState();
}

class _CurrencyRateDialogState extends State<_CurrencyRateDialog> {
  late final TextEditingController _sourceController;
  late final TextEditingController _targetController;
  late final TextEditingController _rateController;
  late final TextEditingController _providerController;
  late final TextEditingController _timestampController;
  final GlobalKey<FormState> _formKey = GlobalKey<FormState>();

  @override
  void initState() {
    super.initState();
    final snapshot = widget.initialSnapshot;
    _sourceController = TextEditingController(text: snapshot?.sourceCurrency ?? 'USD');
    _targetController = TextEditingController(text: snapshot?.targetCurrency ?? 'INR');
    _rateController = TextEditingController(text: snapshot?.rate.toString() ?? '');
    _providerController = TextEditingController(text: snapshot?.rateSource ?? '');
    _timestampController = TextEditingController(
      text: snapshot?.timestamp.toUtc().toIso8601String() ?? DateTime.now().toUtc().toIso8601String(),
    );
  }

  @override
  void dispose() {
    _sourceController.dispose();
    _targetController.dispose();
    _rateController.dispose();
    _providerController.dispose();
    _timestampController.dispose();
    super.dispose();
  }

  void _save() {
    if (!(_formKey.currentState?.validate() ?? false)) {
      return;
    }
    final result = CurrencyRateDialogResult(
      sourceCurrency: _sourceController.text.trim().toUpperCase(),
      targetCurrency: _targetController.text.trim().toUpperCase(),
      exchangeRate: double.parse(_rateController.text.trim()),
      rateSource: _providerController.text.trim(),
      rateTimestamp: DateTime.parse(_timestampController.text.trim()).toUtc(),
    );
    Navigator.of(context).pop(result);
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Configure exchange rate'),
      content: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 520),
        child: Form(
          key: _formKey,
          child: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextFormField(
                  controller: _sourceController,
                  decoration: const InputDecoration(
                    labelText: 'Source currency',
                    hintText: r'USD, GBP, INR, $, ₹, £',
                  ),
                  textCapitalization: TextCapitalization.characters,
                  validator: (value) {
                    final text = value?.trim() ?? '';
                    if (text.isEmpty) {
                      return 'Source currency is required.';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 12),
                TextFormField(
                  controller: _targetController,
                  decoration: const InputDecoration(
                    labelText: 'Target currency',
                    hintText: 'INR',
                  ),
                  textCapitalization: TextCapitalization.characters,
                  validator: (value) {
                    final text = value?.trim() ?? '';
                    if (text.isEmpty) {
                      return 'Target currency is required.';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 12),
                TextFormField(
                  controller: _rateController,
                  decoration: const InputDecoration(
                    labelText: 'Exchange rate',
                    hintText: '83.15',
                  ),
                  keyboardType: const TextInputType.numberWithOptions(decimal: true),
                  inputFormatters: [FilteringTextInputFormatter.allow(RegExp(r'[0-9.\-]'))],
                  validator: (value) {
                    final parsed = double.tryParse(value?.trim() ?? '');
                    if (parsed == null) {
                      return 'Enter a valid exchange rate.';
                    }
                    if (parsed <= 0) {
                      return 'Exchange rate must be positive.';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 12),
                TextFormField(
                  controller: _providerController,
                  decoration: const InputDecoration(
                    labelText: 'Rate provider/source',
                    hintText: 'Manual, RBI, ECB, etc.',
                  ),
                  validator: (value) {
                    final text = value?.trim() ?? '';
                    if (text.isEmpty) {
                      return 'Rate provider/source is required.';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 12),
                TextFormField(
                  controller: _timestampController,
                  decoration: const InputDecoration(
                    labelText: 'Rate timestamp',
                    hintText: '2026-08-25T00:00:00Z',
                  ),
                  validator: (value) {
                    final text = value?.trim() ?? '';
                    if (text.isEmpty) {
                      return 'Rate timestamp is required.';
                    }
                    try {
                      DateTime.parse(text);
                    } catch (_) {
                      return 'Enter a valid ISO-8601 timestamp.';
                    }
                    return null;
                  },
                ),
              ],
            ),
          ),
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
        FilledButton(
          onPressed: _save,
          child: const Text('Save rate'),
        ),
      ],
    );
  }
}
