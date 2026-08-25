import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:insightflow/api/plan_client.dart';
import 'package:insightflow/engine/fallback_parser.dart';
import 'package:insightflow/models/operation_plan.dart';
import 'package:insightflow/semantic/semantic_resolution.dart';
import 'package:insightflow/workbook/exchange_rate_service.dart';
import 'package:insightflow/workbook/workbook_engine.dart';
import 'package:insightflow/workbook/workbook_exporter.dart';
import 'package:insightflow/workbook/workbook_importer.dart';
import 'package:insightflow/workbook/workbook_models.dart';

class OperationHistoryEntry {
  final String kind;
  final String summary;
  final DateTime timestamp;
  final Map<String, String> metadata;

  const OperationHistoryEntry({
    required this.kind,
    required this.summary,
    required this.timestamp,
    required this.metadata,
  });
}

enum InsightFlowStatus {
  idle,
  loadingFile,
  planning,
  awaitingPlanConfirmation,
  awaitingColumnSelection,
  executing,
  ready,
  error,
}

enum ChatRole { user, assistant, system }

class ChatMessage {
  final ChatRole role;
  final String text;
  final DateTime timestamp;

  const ChatMessage({
    required this.role,
    required this.text,
    required this.timestamp,
  });
}

class PendingSelection {
  final String targetId;
  final List<String> candidateColumnIds;

  const PendingSelection({
    required this.targetId,
    required this.candidateColumnIds,
  });
}

class InsightFlowController extends ChangeNotifier {
  final PlanClient planClient;
  final WorkbookImporter importer;
  final WorkbookExporter exporter;
  final WorkbookExecutionEngine engine;
  final DeterministicFallbackParser fallbackParser;

  final List<ChatMessage> _history = <ChatMessage>[];
  final List<OperationHistoryEntry> _operationHistory = <OperationHistoryEntry>[];
  final List<WorkbookSession> _undoStack = <WorkbookSession>[];
  final Random _random = Random.secure();

  WorkbookSession? _session;
  InsightFlowPlan? _pendingPlan;
  String? _pendingRequestText;
  PendingSelection? _pendingSelection;
  Map<String, String> _forcedBindings = const {};
  ExchangeRateSnapshot? _currencyRateSnapshot;
  InsightFlowStatus _status = InsightFlowStatus.idle;
  String? _errorMessage;
  String? _successMessage;
  String? _loadedFileName;

  InsightFlowController({
    required this.planClient,
    required this.importer,
    required this.exporter,
    required this.engine,
    required this.fallbackParser,
  });

  InsightFlowStatus get status => _status;
  WorkbookSession? get session => _session;
  InsightFlowPlan? get pendingPlan => _pendingPlan;
  PendingSelection? get pendingSelection => _pendingSelection;
  String? get errorMessage => _errorMessage;
  String? get successMessage => _successMessage;
  String? get loadedFileName => _loadedFileName;
  List<ChatMessage> get history => List<ChatMessage>.unmodifiable(_history);
  List<OperationHistoryEntry> get operationHistory => List<OperationHistoryEntry>.unmodifiable(_operationHistory);
  bool get hasWorkbook => _session != null;
  bool get hasPendingPlan => _pendingPlan != null;
  bool get canUndo => _undoStack.isNotEmpty;
  bool get hasCurrencyRateConfig => _currencyRateSnapshot != null;
  ExchangeRateSnapshot? get currencyRateSnapshot => _currencyRateSnapshot;

  Future<void> importWorkbook(Uint8List bytes, String fileName) async {
    _status = InsightFlowStatus.loadingFile;
    _errorMessage = null;
    _successMessage = null;
    notifyListeners();

    try {
      final result = importer.importBytes(bytes, fileName: fileName);
      _session = result.session;
      _loadedFileName = fileName;
      _history
        ..clear()
        ..add(
          ChatMessage(
            role: ChatRole.system,
            text: 'Loaded $fileName locally.',
            timestamp: DateTime.now(),
          ),
        );
      _operationHistory
        ..clear()
        ..add(
          OperationHistoryEntry(
            kind: 'import',
            summary: 'Loaded $fileName locally.',
            timestamp: DateTime.now(),
            metadata: {
              'file_name': fileName,
            },
          ),
        );
      _pendingPlan = null;
      _pendingSelection = null;
      _forcedBindings = const {};
      _currencyRateSnapshot = null;
      _undoStack.clear();
      _status = InsightFlowStatus.ready;
      _successMessage = result.warnings.isEmpty ? 'Workbook loaded locally.' : result.warnings.join(' ');
      notifyListeners();
    } catch (_) {
      _status = InsightFlowStatus.error;
      _errorMessage = 'Unable to read the workbook locally.';
      notifyListeners();
    }
  }

  Future<void> submitRequest(String requestText, {String locale = 'en-US', String clientVersion = 'web'}) async {
    if (_session == null) {
      _errorMessage = 'Load a workbook first.';
      _status = InsightFlowStatus.error;
      notifyListeners();
      return;
    }

    final trimmed = requestText.trim();
    if (trimmed.isEmpty) {
      _errorMessage = 'Enter a request.';
      _status = InsightFlowStatus.error;
      notifyListeners();
      return;
    }

    _history.add(ChatMessage(role: ChatRole.user, text: trimmed, timestamp: DateTime.now()));
    _pendingRequestText = trimmed;
    _errorMessage = null;
    _successMessage = null;
    _status = InsightFlowStatus.planning;
    notifyListeners();

    final requestId = _newRequestId();
    InsightFlowPlan? plan;
    try {
      final response = await planClient.requestPlan(
        PlanRequestPayload(
          requestId: requestId,
          requestText: trimmed,
          locale: locale,
          clientVersion: clientVersion,
        ),
      );
      if (response.ok) {
        plan = response.plan;
      } else {
        final fallback = fallbackParser.parse(trimmed, requestId: requestId);
        plan = fallback.plan;
        if (plan == null) {
          _status = InsightFlowStatus.error;
          _errorMessage = response.errorMessage ?? fallback.message ?? 'Unable to plan the request.';
          _history.add(ChatMessage(role: ChatRole.assistant, text: _errorMessage!, timestamp: DateTime.now()));
          notifyListeners();
          return;
        }
      }
    } catch (_) {
      final fallback = fallbackParser.parse(trimmed, requestId: requestId);
      plan = fallback.plan;
      if (plan == null) {
        _status = InsightFlowStatus.error;
        _errorMessage = fallback.message ?? 'Unable to plan the request.';
        _history.add(ChatMessage(role: ChatRole.assistant, text: _errorMessage!, timestamp: DateTime.now()));
        notifyListeners();
        return;
      }
    }

    _pendingPlan = plan;
    _status = InsightFlowStatus.awaitingPlanConfirmation;
    _history.add(ChatMessage(role: ChatRole.assistant, text: 'Plan ready. Review it before execution.', timestamp: DateTime.now()));
    notifyListeners();
  }

  Future<void> confirmPendingPlan() async {
    final plan = _pendingPlan;
    final session = _session;
    final requestText = _pendingRequestText;
    if (plan == null || session == null || requestText == null) {
      return;
    }

    _status = InsightFlowStatus.executing;
    _errorMessage = null;
    _successMessage = null;
    notifyListeners();

    try {
      if (_requiresCurrencyRate(plan) && _currencyRateSnapshot == null) {
        _status = InsightFlowStatus.error;
        _errorMessage = 'Configure the currency rate locally before converting values.';
        _history.add(ChatMessage(role: ChatRole.assistant, text: _errorMessage!, timestamp: DateTime.now()));
        notifyListeners();
        return;
      }

      final resolution = engine.prepareTargets(
        session: session,
        plan: plan,
        forcedTargetBindings: _forcedBindings,
      );

      final unresolved = resolution.resolutions.values.firstWhere(
        (item) => item.needsUserSelection || item.resolvedColumnId == null,
        orElse: () => const ColumnResolution(
          targetId: '',
          resolvedColumnId: null,
          confidence: 0,
          needsUserSelection: false,
          candidateColumnIds: [],
        ),
      );

      if (unresolved.targetId.isNotEmpty) {
        _pendingSelection = PendingSelection(
          targetId: unresolved.targetId,
          candidateColumnIds: unresolved.candidateColumnIds.isEmpty ? session.activeSheet.columnIds : unresolved.candidateColumnIds,
        );
        _status = InsightFlowStatus.awaitingColumnSelection;
        notifyListeners();
        return;
      }

      final result = await engine.execute(
        session: session,
        plan: plan,
        requestText: requestText,
        exchangeRateService: ManualExchangeRateService(snapshot: _currencyRateSnapshot),
        forcedTargetBindings: _forcedBindings,
      );

      if (result.success) {
        _undoStack.add(session.clone());
        _session = result.session;
        _pendingPlan = null;
        _pendingSelection = null;
        _forcedBindings = const {};
        _status = InsightFlowStatus.ready;
        _successMessage = result.messages.isEmpty ? 'Operation completed.' : result.messages.join(' ');
        _history.add(ChatMessage(role: ChatRole.assistant, text: _successMessage!, timestamp: DateTime.now()));
        _recordOperation(
          kind: 'plan_execution',
          summary: _successMessage!,
          metadata: {
            'request_id': plan.requestId,
            if (_currencyRateSnapshot != null) 'currency_rate': _describeCurrencyRate(_currencyRateSnapshot!),
          },
        );
      } else {
        _status = InsightFlowStatus.error;
        _errorMessage = result.errors.join(' ');
        _history.add(ChatMessage(role: ChatRole.assistant, text: _errorMessage!, timestamp: DateTime.now()));
      }
      notifyListeners();
    } on NeedsUserSelectionException catch (error) {
      _pendingSelection = PendingSelection(
        targetId: error.targetId,
        candidateColumnIds: error.candidates.isEmpty ? session.activeSheet.columnIds : error.candidates,
      );
      _status = InsightFlowStatus.awaitingColumnSelection;
      notifyListeners();
      return;
    } on ValueNotFoundException catch (error) {
      _status = InsightFlowStatus.error;
      _errorMessage = error.message;
      _history.add(ChatMessage(role: ChatRole.assistant, text: _errorMessage!, timestamp: DateTime.now()));
      notifyListeners();
      return;
    } on MissingCurrencyRateException catch (error) {
      _status = InsightFlowStatus.error;
      _errorMessage = error.message;
      _history.add(ChatMessage(role: ChatRole.assistant, text: _errorMessage!, timestamp: DateTime.now()));
      notifyListeners();
      return;
    } on MissingPivotSpecificationException catch (error) {
      _status = InsightFlowStatus.error;
      _errorMessage = error.message;
      _history.add(ChatMessage(role: ChatRole.assistant, text: _errorMessage!, timestamp: DateTime.now()));
      notifyListeners();
      return;
    } catch (_) {
      _status = InsightFlowStatus.error;
      _errorMessage = 'Unable to execute the plan locally.';
      _history.add(ChatMessage(role: ChatRole.assistant, text: _errorMessage!, timestamp: DateTime.now()));
      notifyListeners();
    }
  }

  Future<void> chooseColumn(String columnId) async {
    final pendingSelection = _pendingSelection;
    if (pendingSelection == null) {
      return;
    }
    _forcedBindings = <String, String>{..._forcedBindings, pendingSelection.targetId: columnId};
    _pendingSelection = null;
    _status = InsightFlowStatus.executing;
    notifyListeners();
    await confirmPendingPlan();
  }

  void cancelPendingPlan() {
    _pendingPlan = null;
    _pendingSelection = null;
    _forcedBindings = const {};
    _status = _session == null ? InsightFlowStatus.idle : InsightFlowStatus.ready;
    notifyListeners();
  }

  void setActiveSheet(String sheetId) {
    final session = _session;
    if (session == null) {
      return;
    }
    _session = session.withActiveSheet(sheetId);
    notifyListeners();
  }

  void undo() {
    if (_undoStack.isEmpty) {
      return;
    }
    _session = _undoStack.removeLast();
    _status = InsightFlowStatus.ready;
    _successMessage = 'Restored the previous workbook state.';
    notifyListeners();
  }

  void configureCurrencyRate({
    required String sourceCurrency,
    required String targetCurrency,
    required double exchangeRate,
    required String rateSource,
    required DateTime rateTimestamp,
  }) {
    final normalizedSource = sourceCurrency.trim();
    final normalizedTarget = targetCurrency.trim();
    final normalizedRateSource = rateSource.trim();
    if (normalizedSource.isEmpty) {
      _errorMessage = 'Source currency is required.';
      _status = InsightFlowStatus.error;
      notifyListeners();
      return;
    }
    if (normalizedTarget.isEmpty) {
      _errorMessage = 'Target currency is required.';
      _status = InsightFlowStatus.error;
      notifyListeners();
      return;
    }
    if (exchangeRate <= 0) {
      _errorMessage = 'Exchange rate must be positive.';
      _status = InsightFlowStatus.error;
      notifyListeners();
      return;
    }

    final snapshot = ExchangeRateSnapshot(
      sourceCurrency: normalizedSource.toUpperCase(),
      targetCurrency: normalizedTarget.toUpperCase(),
      rate: exchangeRate,
      rateSource: normalizedRateSource,
      timestamp: rateTimestamp.toUtc(),
    );
    _currencyRateSnapshot = snapshot;
    final summary = 'Configured ${snapshot.sourceCurrency} → ${snapshot.targetCurrency} at ${snapshot.rate} from ${snapshot.rateSource}.';
    _operationHistory.add(
      OperationHistoryEntry(
        kind: 'currency_rate',
        summary: summary,
        timestamp: DateTime.now(),
        metadata: {
          'source_currency': snapshot.sourceCurrency,
          'target_currency': snapshot.targetCurrency,
          'exchange_rate': snapshot.rate.toString(),
          'rate_source': snapshot.rateSource,
          'rate_timestamp': snapshot.timestamp.toIso8601String(),
        },
      ),
    );
    _history.add(ChatMessage(role: ChatRole.system, text: summary, timestamp: DateTime.now()));
    _status = _pendingPlan == null
        ? (_session == null ? InsightFlowStatus.idle : InsightFlowStatus.ready)
        : InsightFlowStatus.awaitingPlanConfirmation;
    _errorMessage = null;
    _successMessage = 'Currency rate saved locally.';
    notifyListeners();
  }

  Future<Uint8List?> exportWorkbook() async {
    final session = _session;
    if (session == null) {
      return null;
    }
    return exporter.exportToXlsx(session);
  }

  void clearMessages() {
    _errorMessage = null;
    _successMessage = null;
    notifyListeners();
  }

  List<String> sheetColumnLabels(WorkbookSheet sheet) {
    return List<String>.generate(sheet.columnCount, (index) {
      final header = sheet.originalHeaders[index];
      return header.isEmpty ? 'Column ${index + 1}' : header;
    }, growable: false);
  }

  String? get activeSheetId => _session?.activeSheet.id;

  String _newRequestId() {
    return 'req-${DateTime.now().microsecondsSinceEpoch}-${_random.nextInt(1 << 32)}';
  }

  bool _requiresCurrencyRate(InsightFlowPlan plan) {
    return plan.operations.any((operation) => operation is ConvertCurrencyOperation);
  }

  void _recordOperation({
    required String kind,
    required String summary,
    required Map<String, String> metadata,
  }) {
    _operationHistory.add(
      OperationHistoryEntry(
        kind: kind,
        summary: summary,
        timestamp: DateTime.now(),
        metadata: metadata,
      ),
    );
  }

  String _describeCurrencyRate(ExchangeRateSnapshot snapshot) {
    return '${snapshot.sourceCurrency} → ${snapshot.targetCurrency} @ ${snapshot.rate} (${snapshot.rateSource})';
  }
}
