import 'package:flutter/material.dart';

import 'models.dart';
import 'services.dart';
import 'widgets.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({
    super.key,
    required this.controller,
    required this.onEditIntake,
  });

  final PulseForgeController controller;
  final VoidCallback onEditIntake;

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  late final TextEditingController _mqttHostController;
  late final TextEditingController _mqttPortController;
  String? _selectedDeviceId;

  @override
  void initState() {
    super.initState();
    _mqttHostController = TextEditingController(text: widget.controller.mqttHost);
    _mqttPortController =
        TextEditingController(text: widget.controller.mqttPort.toString());
  }

  @override
  void dispose() {
    _mqttHostController.dispose();
    _mqttPortController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = widget.controller;
    final latestWindow = controller.latestWindow;
    final latestHrv = controller.latestHrv;

    return Scaffold(
      appBar: AppBar(
        title: Text('PulseForge Mobile - ${controller.profile.subjectId}'),
        actions: <Widget>[
          IconButton(
            tooltip: 'Edit intake',
            onPressed: widget.onEditIntake,
            icon: const Icon(Icons.assignment_outlined),
          ),
        ],
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          final compact = constraints.maxWidth < 1180;
          final left = _buildMainColumn(context, controller, latestWindow, latestHrv);
          final right = _buildSideColumn(context, controller, latestWindow, latestHrv);

          if (compact) {
            return ListView(
              padding: const EdgeInsets.all(20),
              children: <Widget>[
                left,
                const SizedBox(height: 20),
                right,
              ],
            );
          }

          return Padding(
            padding: const EdgeInsets.all(20),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Expanded(flex: 3, child: left),
                const SizedBox(width: 20),
                Expanded(flex: 2, child: right),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildMainColumn(
    BuildContext context,
    PulseForgeController controller,
    WindowAnalysis latestWindow,
    HrvSnapshot latestHrv,
  ) {
    return Column(
      children: <Widget>[
        _buildConnectionCard(context, controller),
        const SizedBox(height: 16),
        SignalChartCard(
          title: 'ECG',
          samples: controller.ecgSamples,
          color: Colors.lightBlueAccent,
          subtitle: '130 Hz live ECG stream',
        ),
        const SizedBox(height: 16),
        SignalChartCard(
          title: 'Accelerometer magnitude',
          samples: controller.accSamples.map((sample) => sample.magnitude).toList(),
          color: Colors.greenAccent,
          subtitle: '100 Hz acceleration magnitude',
        ),
        const SizedBox(height: 16),
        SignalChartCard(
          title: 'Heart rate',
          samples: controller.hrSamples,
          color: Colors.pinkAccent,
          subtitle: 'Most recent heart-rate samples',
        ),
      ],
    );
  }

  Widget _buildSideColumn(
    BuildContext context,
    PulseForgeController controller,
    WindowAnalysis latestWindow,
    HrvSnapshot latestHrv,
  ) {
    return Column(
      children: <Widget>[
        _buildRecordingCard(context, controller),
        const SizedBox(height: 16),
        _buildMetricsCard(context, controller, latestWindow, latestHrv),
        const SizedBox(height: 16),
        LogPanel(logs: controller.logs),
      ],
    );
  }

  Widget _buildConnectionCard(BuildContext context, PulseForgeController controller) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text('Device connection', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: controller.devices
                  .map(
                    (device) => DeviceBadge(
                      device: device,
                      selected: _selectedDeviceId == device.identifier,
                      onTap: () {
                        setState(() {
                          _selectedDeviceId = device.identifier;
                        });
                      },
                    ),
                  )
                  .toList(),
            ),
            if (controller.devices.isEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text(
                  'No devices discovered yet. Scan or start mock mode.',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ),
            const SizedBox(height: 16),
            Row(
              children: <Widget>[
                FilledButton.icon(
                  onPressed: controller.scanDevices,
                  icon: const Icon(Icons.bluetooth_searching),
                  label: const Text('Scan'),
                ),
                const SizedBox(width: 12),
                FilledButton.tonalIcon(
                  onPressed: _selectedDeviceId == null
                      ? null
                      : () {
                          final selected = controller.devices.firstWhere(
                            (device) => device.identifier == _selectedDeviceId,
                          );
                          controller.connectSelectedDevice(selected);
                        },
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Connect'),
                ),
                const SizedBox(width: 12),
                OutlinedButton.icon(
                  onPressed: controller.connectMock,
                  icon: const Icon(Icons.science_outlined),
                  label: const Text('Mock'),
                ),
                const SizedBox(width: 12),
                OutlinedButton.icon(
                  onPressed: controller.disconnect,
                  icon: const Icon(Icons.stop_circle_outlined),
                  label: const Text('Disconnect'),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Wrap(
              spacing: 12,
              runSpacing: 12,
              children: <Widget>[
                _statusChip(
                  context,
                  'Connection',
                  controller.connectionState.name,
                  color: controller.connectionState == LiveConnectionState.connected
                      ? Colors.green
                      : Colors.orange,
                ),
                _statusChip(
                  context,
                  'Source',
                  controller.usingMock ? 'Mock sensor' : (controller.connectedDeviceName ?? 'Idle'),
                  color: controller.usingMock ? Colors.teal : Colors.blueGrey,
                ),
                _statusChip(
                  context,
                  'MQTT',
                  controller.isMqttConnected ? 'Connected' : 'Offline',
                  color: controller.isMqttConnected ? Colors.green : Colors.blueGrey,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRecordingCard(BuildContext context, PulseForgeController controller) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text('Session & MQTT', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 12),
            TextField(
              controller: _mqttHostController,
              decoration: const InputDecoration(
                labelText: 'Broker host',
                prefixIcon: Icon(Icons.cloud_outlined),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _mqttPortController,
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(
                labelText: 'Broker port',
                prefixIcon: Icon(Icons.settings_ethernet),
              ),
            ),
            const SizedBox(height: 12),
            Align(
              alignment: Alignment.centerLeft,
              child: FilledButton.tonal(
                onPressed: () {
                  final port = int.tryParse(_mqttPortController.text.trim()) ?? 1883;
                  controller.updateBroker(_mqttHostController.text, port);
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Broker settings updated.')),
                  );
                },
                child: const Text('Apply broker settings'),
              ),
            ),
            const Divider(height: 24),
            Text(
              'Subject: ${controller.profile.subjectId.isEmpty ? 'Not set' : controller.profile.subjectId}',
            ),
            const SizedBox(height: 8),
            Text(
              controller.sessionFile?.path ?? 'No active recording file',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 16),
            Row(
              children: <Widget>[
                FilledButton.icon(
                  onPressed: controller.startRecording,
                  icon: const Icon(Icons.fiber_manual_record),
                  label: const Text('Start recording'),
                ),
                const SizedBox(width: 12),
                OutlinedButton.icon(
                  onPressed: controller.stopRecording,
                  icon: const Icon(Icons.stop),
                  label: const Text('Stop'),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              controller.isRecording
                  ? 'Recording live. Saved windows: ${controller.savedWindows}'
                  : 'Not recording',
              style: Theme.of(context).textTheme.bodyMedium,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMetricsCard(
    BuildContext context,
    PulseForgeController controller,
    WindowAnalysis latestWindow,
    HrvSnapshot latestHrv,
  ) {
    final metrics = <Widget>[
      MetricTile(label: 'SQI', value: _fmt(latestWindow.sqi, digits: 3), emphasis: true),
      MetricTile(label: 'QRS Energy', value: _fmt(latestWindow.qrsEnergy, digits: 3)),
      MetricTile(label: 'Kurtosis', value: _fmt(latestWindow.vitalKurtosis, digits: 2)),
      MetricTile(label: 'Instant HR', value: _fmt(latestWindow.instantHr, suffix: ' bpm')),
      MetricTile(label: 'RMSSD', value: _fmt(latestHrv.rmssdMs, suffix: ' ms')),
      MetricTile(label: 'SDNN', value: _fmt(latestHrv.sdnnMs, suffix: ' ms')),
      MetricTile(label: 'LF/HF', value: _fmt(latestHrv.lfHf, digits: 3)),
      MetricTile(label: 'Mean HR', value: _fmt(latestHrv.meanHr, suffix: ' bpm')),
      MetricTile(
        label: 'HAR activity',
        value: latestWindow.harActivity.label.replaceAll('_', ' '),
      ),
      MetricTile(
        label: 'ACC entropy',
        value: _fmt(latestWindow.accFeatures.spectralEntropy, digits: 3),
      ),
      MetricTile(
        label: 'ACC median freq',
        value: _fmt(latestWindow.accFeatures.medianFreqHz, suffix: ' Hz'),
      ),
      MetricTile(
        label: 'ACC variance',
        value: _fmt(latestWindow.accFeatures.varMagMg2, digits: 1),
      ),
    ];

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text('Analysis', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 12),
            Text(
              '5 s window: ${latestWindow.status} | 30 s HRV: ${latestHrv.status}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 16),
            GridView.count(
              crossAxisCount: 2,
              childAspectRatio: 1.65,
              mainAxisSpacing: 10,
              crossAxisSpacing: 10,
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              children: metrics,
            ),
            const SizedBox(height: 16),
            Text(
              'Mobile simplifications: ECG morphology widths and Google Fit sync are intentionally left lighter than the desktop Python app.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }

  Widget _statusChip(
    BuildContext context,
    String label,
    String value, {
    required Color color,
  }) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: color.withOpacity(0.16),
        borderRadius: BorderRadius.circular(999),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        child: Text('$label: $value'),
      ),
    );
  }

  String _fmt(
    double? value, {
    int digits = 1,
    String suffix = '',
  }) {
    if (value == null || !value.isFinite) {
      return '--';
    }
    return '${value.toStringAsFixed(digits)}$suffix';
  }
}
