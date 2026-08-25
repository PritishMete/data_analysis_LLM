import 'package:flutter/material.dart';
import 'package:insightflow/api/plan_client.dart';
import 'package:insightflow/app/insightflow_controller.dart';
import 'package:insightflow/app/insightflow_screen.dart';
import 'package:insightflow/engine/fallback_parser.dart';
import 'package:insightflow/workbook/workbook_engine.dart';
import 'package:insightflow/workbook/workbook_exporter.dart';
import 'package:insightflow/workbook/workbook_importer.dart';

void main() {
  runApp(const InsightFlowApp());
}

class InsightFlowApp extends StatelessWidget {
  const InsightFlowApp({super.key});

  @override
  Widget build(BuildContext context) {
    final controller = InsightFlowController(
      planClient: PlanClient(
        endpoint: Uri.parse(const String.fromEnvironment(
          'INSIGHTFLOW_PLAN_ENDPOINT',
          defaultValue: 'http://127.0.0.1:8787/v1/plan',
        )),
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
