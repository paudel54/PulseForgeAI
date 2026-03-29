import 'dart:math' as math;

import 'package:flutter/material.dart';

import 'models.dart';

class SignalChartCard extends StatelessWidget {
  const SignalChartCard({
    super.key,
    required this.title,
    required this.samples,
    required this.color,
    required this.subtitle,
  });

  final String title;
  final List<double> samples;
  final Color color;
  final String subtitle;

  @override
  Widget build(BuildContext context) {
    return Card(
      clipBehavior: Clip.antiAlias,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(title, style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 4),
            Text(
              subtitle,
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 150,
              child: CustomPaint(
                painter: _SignalPainter(
                  samples: samples,
                  color: color,
                  gridColor: Theme.of(context).colorScheme.outlineVariant,
                ),
                child: const SizedBox.expand(),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class MetricTile extends StatelessWidget {
  const MetricTile({
    super.key,
    required this.label,
    required this.value,
    this.emphasis = false,
  });

  final String label;
  final String value;
  final bool emphasis;

  @override
  Widget build(BuildContext context) {
    final color = emphasis
        ? Theme.of(context).colorScheme.primary
        : Theme.of(context).colorScheme.onSurface;
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(14),
      ),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(
              label,
              style: Theme.of(context).textTheme.labelMedium,
            ),
            const SizedBox(height: 8),
            Text(
              value,
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    color: color,
                    fontWeight: FontWeight.w700,
                  ),
            ),
          ],
        ),
      ),
    );
  }
}

class DeviceBadge extends StatelessWidget {
  const DeviceBadge({
    super.key,
    required this.device,
    required this.selected,
    required this.onTap,
  });

  final PolarScanDevice device;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return ChoiceChip(
      label: Text('${device.name} (${device.identifier})'),
      selected: selected,
      onSelected: (_) => onTap(),
      avatar: device.rssi == null ? null : Text('${device.rssi}'),
    );
  }
}

class LogPanel extends StatelessWidget {
  const LogPanel({
    super.key,
    required this.logs,
  });

  final List<LogEntry> logs;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text('Log', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            SizedBox(
              height: 180,
              child: ListView.separated(
                itemCount: logs.length,
                itemBuilder: (context, index) {
                  final entry = logs[index];
                  return Text(
                    '[${entry.formattedTimestamp}] ${entry.message}',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          fontFamily: 'monospace',
                        ),
                  );
                },
                separatorBuilder: (_, __) => const SizedBox(height: 8),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _SignalPainter extends CustomPainter {
  const _SignalPainter({
    required this.samples,
    required this.color,
    required this.gridColor,
  });

  final List<double> samples;
  final Color color;
  final Color gridColor;

  @override
  void paint(Canvas canvas, Size size) {
    final gridPaint = Paint()
      ..color = gridColor.withOpacity(0.25)
      ..strokeWidth = 1;
    for (var i = 1; i < 4; i += 1) {
      final dy = size.height * (i / 4);
      canvas.drawLine(Offset(0, dy), Offset(size.width, dy), gridPaint);
    }

    if (samples.length < 2) {
      return;
    }

    final minValue = samples.reduce(math.min);
    final maxValue = samples.reduce(math.max);
    final span = (maxValue - minValue).abs() < 1e-6 ? 1.0 : (maxValue - minValue);
    final path = Path();

    for (var i = 0; i < samples.length; i += 1) {
      final x = size.width * (i / (samples.length - 1));
      final normalized = (samples[i] - minValue) / span;
      final y = size.height - (normalized * size.height);
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }

    final paint = Paint()
      ..color = color
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant _SignalPainter oldDelegate) {
    return oldDelegate.samples != samples ||
        oldDelegate.color != color ||
        oldDelegate.gridColor != gridColor;
  }
}
