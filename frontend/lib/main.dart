import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:insightflow/api/plan_client.dart';
import 'package:insightflow/app/insightflow_controller.dart';
import 'package:insightflow/app/insightflow_screen.dart';
import 'package:insightflow/engine/fallback_parser.dart';
import 'package:insightflow/workbook/workbook_engine.dart';
import 'package:insightflow/workbook/workbook_exporter.dart';
import 'package:insightflow/workbook/workbook_importer.dart';

void main() {
  const endpoint = String.fromEnvironment('INSIGHTFLOW_PLAN_ENDPOINT', defaultValue: '');
  if (kReleaseMode && endpoint.isEmpty) {
    runApp(const _ConfigErrorApp());
    return;
  }
  runApp(InsightFlowApp(planEndpoint: endpoint.isEmpty ? Uri.parse('http://127.0.0.1:8787/v1/plan') : Uri.parse(endpoint)));
}

class InsightFlowApp extends StatelessWidget {
  final Uri planEndpoint;

  const InsightFlowApp({super.key, required this.planEndpoint});

  @override
  Widget build(BuildContext context) {
    final controller = InsightFlowController(
      planClient: PlanClient(
        endpoint: planEndpoint,
      ),
      importer: const WorkbookImporter(),
      exporter: const WorkbookExporter(),
      engine: const WorkbookExecutionEngine(),
      fallbackParser: const DeterministicFallbackParser(),
    );

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'InsightFlow',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF13548C)),
      ),
      home: InsightFlowScreen(controller: controller),
    );
  }
}

class _ConfigErrorApp extends StatelessWidget {
  const _ConfigErrorApp();

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 640),
              child: const Text(
                'InsightFlow is running in release mode without INSIGHTFLOW_PLAN_ENDPOINT.\n\n'
                'Build the GitHub Pages frontend with a Cloudflare Worker endpoint, for example:\n'
                'flutter build web --release --base-href /data_analysis_LLM/ --dart-define=INSIGHTFLOW_PLAN_ENDPOINT=https://<your-worker>.workers.dev/v1/plan',
                textAlign: TextAlign.center,
              ),
            ),
          ),
        ),
      ),
    );
  }
}
