import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:insightflow/workbook/workbook_models.dart';

class ChartPanel extends StatelessWidget {
  final List<AnalysisChartSpec> charts;

  const ChartPanel({
    super.key,
    required this.charts,
  });

  @override
  Widget build(BuildContext context) {
    if (charts.isEmpty) {
      return const SizedBox.shrink();
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: charts.map((chart) => Padding(
        padding: const EdgeInsets.only(bottom: 12),
        child: _ChartCard(chart: chart),
      )).toList(growable: false),
    );
  }
}

class _ChartCard extends StatelessWidget {
  final AnalysisChartSpec chart;

  const _ChartCard({required this.chart});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(chart.title, style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            SizedBox(height: 240, child: _buildChart()),
          ],
        ),
      ),
    );
  }

  Widget _buildChart() {
    final points = chart.points;
    if (points.isEmpty) {
      return const Center(child: Text('No chart data.'));
    }

    final type = chart.type.toLowerCase();
    if (type == 'pie') {
      return PieChart(
        PieChartData(
          sectionsSpace: 2,
          centerSpaceRadius: 30,
          sections: points.take(8).map((point) {
            return PieChartSectionData(
              value: point.y <= 0 ? 1 : point.y,
              title: point.label.isEmpty ? point.y.toStringAsFixed(1) : point.label,
              radius: 56,
            );
          }).toList(growable: false),
        ),
      );
    }

    if (type == 'scatter') {
      return ScatterChart(
        ScatterChartData(
          scatterSpots: points.map((point) {
            return ScatterSpot(point.x, point.y);
          }).toList(growable: false),
          scatterTouchData: ScatterTouchData(enabled: true),
          titlesData: FlTitlesData(show: true),
        ),
      );
    }

    final spots = points.map((point) => FlSpot(point.x, point.y)).toList(growable: false);
    return LineChart(
      LineChartData(
        gridData: const FlGridData(show: true),
        titlesData: const FlTitlesData(show: true),
        borderData: FlBorderData(show: true),
        lineBarsData: [
          LineChartBarData(
            spots: spots,
            isCurved: type == 'area',
            barWidth: 3,
            dotData: const FlDotData(show: false),
          ),
        ],
      ),
    );
  }
}
