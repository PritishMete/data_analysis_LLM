import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:insightflow/app/insightflow_controller.dart';
import 'package:insightflow/app/widgets/currency_rate_dialog.dart';
import 'package:insightflow/app/widgets/chat_panel.dart';
import 'package:insightflow/app/widgets/chart_panel.dart';
import 'package:insightflow/app/widgets/column_selection_dialog.dart';
import 'package:insightflow/app/widgets/plan_preview_dialog.dart';
import 'package:insightflow/app/widgets/workbook_sheet_table.dart';
import 'package:insightflow/models/operation_plan.dart';
import 'package:insightflow/workbook/workbook_models.dart';
import 'package:insightflow/workbook/workbook_download.dart';

class InsightFlowScreen extends StatefulWidget {
  final InsightFlowController controller;

  const InsightFlowScreen({
    super.key,
    required this.controller,
  });

  @override
  State<InsightFlowScreen> createState() => _InsightFlowScreenState();
}

class _InsightFlowScreenState extends State<InsightFlowScreen> {
  final Set<String> _openedPlanDialogs = <String>{};
  String? _openedSelectionTarget;
  String? _openedRateDialogPlanId;

  @override
  void initState() {
    super.initState();
    widget.controller.addListener(_handleControllerChanged);
  }

  @override
  void didUpdateWidget(covariant InsightFlowScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.controller != widget.controller) {
      oldWidget.controller.removeListener(_handleControllerChanged);
      widget.controller.addListener(_handleControllerChanged);
    }
  }

  @override
  void dispose() {
    widget.controller.removeListener(_handleControllerChanged);
    super.dispose();
  }

  void _handleControllerChanged() {
    if (!mounted) {
      return;
    }
    final controller = widget.controller;
    final plan = controller.pendingPlan;
    if (plan == null) {
      _openedRateDialogPlanId = null;
    }
    if (plan != null && _requiresCurrencyConfig(plan) && !controller.hasCurrencyRateConfig && _openedRateDialogPlanId != plan.requestId) {
      _openedRateDialogPlanId = plan.requestId;
      WidgetsBinding.instance.addPostFrameCallback((_) => _showCurrencyRateDialog(controller, cancelPendingPlanOnDismiss: true));
      return;
    }
    if (plan != null && !_openedPlanDialogs.contains(plan.requestId) && controller.status == InsightFlowStatus.awaitingPlanConfirmation) {
      _openedPlanDialogs.add(plan.requestId);
      WidgetsBinding.instance.addPostFrameCallback((_) => _showPlanPreview(controller, plan));
      return;
    }

    final selection = controller.pendingSelection;
    if (selection == null && controller.status != InsightFlowStatus.awaitingColumnSelection) {
      _openedSelectionTarget = null;
    }
    if (selection != null && _openedSelectionTarget != selection.targetId && controller.status == InsightFlowStatus.awaitingColumnSelection) {
      _openedSelectionTarget = selection.targetId;
      WidgetsBinding.instance.addPostFrameCallback((_) => _showColumnSelection(controller, selection));
    }
  }

  Future<void> _showPlanPreview(InsightFlowController controller, InsightFlowPlan plan) async {
    if (!mounted) {
      return;
    }
    await showDialog<void>(
      context: context,
      barrierDismissible: false,
      builder: (dialogContext) {
        return PlanPreviewDialog(
          plan: plan,
          onCancel: () {
            Navigator.of(dialogContext).pop();
            controller.cancelPendingPlan();
          },
          onConfirm: () {
            Navigator.of(dialogContext).pop();
            controller.confirmPendingPlan();
          },
        );
      },
    );
  }

  Future<void> _showColumnSelection(InsightFlowController controller, PendingSelection selection) async {
    final session = controller.session;
    if (!mounted || session == null) {
      return;
    }
    final labels = controller.sheetColumnLabels(session.activeSheet);
    final choices = selection.candidateColumnIds.map((columnId) {
      final index = session.activeSheet.columnIds.indexOf(columnId);
      final label = index >= 0 ? labels[index] : columnId;
      return ColumnChoice(columnId: columnId, label: label);
    }).toList(growable: false);
    await showDialog<void>(
      context: context,
      barrierDismissible: false,
      builder: (dialogContext) {
        return ColumnSelectionDialog(
          targetLabel: selection.targetId,
          choices: choices,
          onSelected: (columnId) {
            Navigator.of(dialogContext).pop();
            controller.chooseColumn(columnId);
          },
          onCancel: () {
            Navigator.of(dialogContext).pop();
            controller.cancelPendingPlan();
          },
        );
      },
    );
  }

  Future<void> _showCurrencyRateDialog(
    InsightFlowController controller, {
    required bool cancelPendingPlanOnDismiss,
  }) async {
    if (!mounted) {
      return;
    }
    final result = await showCurrencyRateDialog(context, initialSnapshot: controller.currencyRateSnapshot);
    _openedRateDialogPlanId = null;
    if (result == null) {
      if (cancelPendingPlanOnDismiss) {
        controller.cancelPendingPlan();
      }
      return;
    }
    controller.configureCurrencyRate(
      sourceCurrency: result.sourceCurrency,
      targetCurrency: result.targetCurrency,
      exchangeRate: result.exchangeRate,
      rateSource: result.rateSource,
      rateTimestamp: result.rateTimestamp,
    );
  }

  Future<void> _pickWorkbook() async {
    final result = await FilePicker.pickFiles(
      withData: true,
      type: FileType.custom,
      allowedExtensions: const ['csv', 'xlsx'],
    );
    if (result == null || result.files.isEmpty) {
      return;
    }
    final file = result.files.single;
    final bytes = file.bytes;
    if (bytes == null) {
      return;
    }
    await widget.controller.importWorkbook(bytes, file.name);
  }

  Future<void> _exportWorkbook() async {
    final bytes = await widget.controller.exportWorkbook();
    if (bytes == null) {
      return;
    }
    final fileName = _exportFileName(widget.controller.loadedFileName);
    downloadWorkbookBytes(bytes, fileName);
  }

  String _exportFileName(String? sourceName) {
    final base = sourceName == null ? 'insightflow' : sourceName.replaceAll(RegExp(r'\.[^.]+$'), '');
    return '$base-insightflow.xlsx';
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: widget.controller,
      builder: (context, _) {
        final controller = widget.controller;
        final session = controller.session;
        final isBusy = controller.status == InsightFlowStatus.loadingFile ||
            controller.status == InsightFlowStatus.planning ||
            controller.status == InsightFlowStatus.executing;

        return Scaffold(
          appBar: AppBar(
            title: const Text('InsightFlow'),
            actions: [
              TextButton.icon(
                onPressed: isBusy ? null : _pickWorkbook,
                icon: const Icon(Icons.upload_file),
                label: const Text('Open workbook'),
              ),
              TextButton.icon(
                onPressed: isBusy ? null : () => _showCurrencyRateDialog(controller, cancelPendingPlanOnDismiss: false),
                icon: const Icon(Icons.currency_exchange),
                label: const Text('Exchange rate'),
              ),
              TextButton.icon(
                onPressed: controller.canUndo ? controller.undo : null,
                icon: const Icon(Icons.undo),
                label: const Text('Undo'),
              ),
              TextButton.icon(
                onPressed: session == null ? null : _exportWorkbook,
                icon: const Icon(Icons.download),
                label: const Text('Export XLSX'),
              ),
              const SizedBox(width: 12),
            ],
          ),
          body: Stack(
            children: [
              LayoutBuilder(
                builder: (context, constraints) {
                  final wide = constraints.maxWidth >= 1100;
                  final content = _buildContent(context, controller, isBusy);
                  if (wide) {
                    return Row(
                      children: [
                        Expanded(flex: 7, child: content),
                        const SizedBox(width: 12),
                        SizedBox(width: 420, child: _buildChat(context, controller, isBusy)),
                      ],
                    );
                  }
                  return Column(
                    children: [
                      Expanded(flex: 7, child: content),
                      const SizedBox(height: 12),
                      SizedBox(height: 360, child: _buildChat(context, controller, isBusy)),
                    ],
                  );
                },
              ),
              if (isBusy)
                const Positioned.fill(
                  child: IgnorePointer(
                    child: ColoredBox(
                      color: Color(0x11000000),
                      child: Center(child: CircularProgressIndicator()),
                    ),
                  ),
                ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildContent(BuildContext context, InsightFlowController controller, bool isBusy) {
    final session = controller.session;
    return Padding(
      padding: const EdgeInsets.all(12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _StatusStrip(
            status: controller.status,
            errorMessage: controller.errorMessage,
            successMessage: controller.successMessage,
          ),
          const SizedBox(height: 12),
          if (session == null)
            const Expanded(
              child: _EmptyState(),
            )
          else
            Expanded(
              child: ListView(
                children: [
                  _SheetStrip(
                    session: session,
                    onSelected: controller.setActiveSheet,
                  ),
                  if (controller.operationHistory.isNotEmpty) ...[
                    const SizedBox(height: 12),
                    _OperationHistoryCard(entries: controller.operationHistory),
                  ],
                  const SizedBox(height: 12),
                  WorkbookSheetTable(sheet: session.activeSheet),
                  const SizedBox(height: 12),
                  if (session.charts.isNotEmpty) ChartPanel(charts: session.charts),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildChat(BuildContext context, InsightFlowController controller, bool isBusy) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(0, 12, 12, 12),
      child: ChatPanel(
        history: controller.history,
        busy: isBusy,
        onSubmit: (text) => controller.submitRequest(text),
        onClear: controller.clearMessages,
      ),
    );
  }

  bool _requiresCurrencyConfig(InsightFlowPlan plan) {
    return plan.operations.any((operation) => operation is ConvertCurrencyOperation);
  }
}

class _StatusStrip extends StatelessWidget {
  final InsightFlowStatus status;
  final String? errorMessage;
  final String? successMessage;

  const _StatusStrip({
    required this.status,
    required this.errorMessage,
    required this.successMessage,
  });

  @override
  Widget build(BuildContext context) {
    final message = errorMessage ?? successMessage;
    if (message == null && status == InsightFlowStatus.idle) {
      return const SizedBox.shrink();
    }

    final color = errorMessage != null ? Colors.red.shade50 : Colors.green.shade50;
    final border = errorMessage != null ? Colors.red.shade200 : Colors.green.shade200;
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color,
        border: Border.all(color: border),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        message ?? 'Ready.',
        style: Theme.of(context).textTheme.bodyMedium,
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState();

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.table_chart_outlined, size: 56, color: Theme.of(context).colorScheme.primary),
              const SizedBox(height: 12),
              Text('Open a CSV or XLSX file to begin.', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              const Text('All workbook processing stays inside the browser.'),
            ],
          ),
        ),
      ),
    );
  }
}

class _SheetStrip extends StatelessWidget {
  final WorkbookSession session;
  final ValueChanged<String> onSelected;

  const _SheetStrip({
    required this.session,
    required this.onSelected,
  });

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: session.sheets.map((sheet) {
        final selected = sheet.id == session.activeSheet.id;
        return ChoiceChip(
          selected: selected,
          label: Text(sheet.name),
          onSelected: (_) => onSelected(sheet.id),
        );
      }).toList(growable: false),
    );
  }
}

class _OperationHistoryCard extends StatelessWidget {
  final List<OperationHistoryEntry> entries;

  const _OperationHistoryCard({required this.entries});

  @override
  Widget build(BuildContext context) {
    final recent = entries.reversed.take(5).toList(growable: false);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Local history', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            ...recent.map((entry) => Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Text('${entry.timestamp.toLocal().toIso8601String()}  ${entry.summary}'),
                )),
          ],
        ),
      ),
    );
  }
}
