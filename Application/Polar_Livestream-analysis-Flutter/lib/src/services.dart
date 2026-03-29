import 'dart:async';
import 'dart:collection';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:carp_polar_package/carp_polar_package.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_hrv/flutter_hrv.dart';
import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:polar/polar.dart';
import 'package:scientisst/scientisst.dart' as scientisst;
import 'package:shared_preferences/shared_preferences.dart';

import 'models.dart';

class SignalToolkitBridge {
  SignalToolkitBridge() : _calculator = scientisst.Calculator();

  final scientisst.Calculator _calculator;

  int get warmupToken => _calculator.addOne(41);
}

class CarpPolarCompatibility {
  const CarpPolarCompatibility();

  List<String> get supportedMeasures => const <String>[
        PolarSamplingPackage.HR,
        PolarSamplingPackage.ECG,
        PolarSamplingPackage.ACCELEROMETER,
      ];

  PolarDevice buildDescriptor(String identifier) {
    return PolarDevice(
      roleName: 'hr-sensor',
      identifier: identifier,
      name: 'H10',
      deviceType: PolarDeviceType.H10,
    );
  }
}

class LocalStorageService {
  static const _profileKey = 'pulseforge_patient_profile';

  Future<PatientProfile> loadProfile() async {
    final prefs = await SharedPreferences.getInstance();
    final json = prefs.getString(_profileKey);
    if (json == null || json.trim().isEmpty) {
      return PatientProfile.initial();
    }
    return PatientProfile.deserialize(json);
  }

  Future<void> saveProfile(PatientProfile profile) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_profileKey, profile.serialize());
  }

  Future<File> startSession(String subjectId) async {
    final root = await _exportsRoot();
    final subjectDir = Directory('${root.path}${Platform.pathSeparator}$subjectId');
    await subjectDir.create(recursive: true);
    final timestamp = _timestampLabel(DateTime.now());
    final file = File(
      '${subjectDir.path}${Platform.pathSeparator}${subjectId}_$timestamp.jsonl',
    );
    if (!await file.exists()) {
      await file.create(recursive: true);
    }
    return file;
  }

  Future<void> writeProfileCopy(
    File sessionFile,
    PatientProfile profile,
  ) async {
    final profileFile = File(
      '${sessionFile.parent.path}${Platform.pathSeparator}intake_state.json',
    );
    await profileFile.writeAsString(
      const JsonEncoder.withIndent('  ').convert(profile.toJson()),
    );
  }

  Future<void> appendPayload(File sessionFile, Map<String, dynamic> payload) async {
    await sessionFile.writeAsString(
      '${jsonEncode(payload)}\n',
      mode: FileMode.append,
      flush: true,
    );
  }

  Future<Directory> _exportsRoot() async {
    final dir = await getApplicationDocumentsDirectory();
    final exports = Directory(
      '${dir.path}${Platform.pathSeparator}pulseforge_exports',
    );
    await exports.create(recursive: true);
    return exports;
  }

  String _timestampLabel(DateTime dt) {
    final y = dt.year.toString().padLeft(4, '0');
    final m = dt.month.toString().padLeft(2, '0');
    final d = dt.day.toString().padLeft(2, '0');
    final hh = dt.hour.toString().padLeft(2, '0');
    final mm = dt.minute.toString().padLeft(2, '0');
    final ss = dt.second.toString().padLeft(2, '0');
    return '$y$m$d_$hh$mm$ss';
  }
}

class MqttTelemetryService {
  MqttServerClient? _client;

  bool get isConnected =>
      _client != null && _client!.connectionStatus?.state == MqttConnectionState.connected;

  Future<void> connect({
    required String host,
    required int port,
    required void Function(String message) onLog,
  }) async {
    if (isConnected) {
      return;
    }

    final clientId = 'pulseforge-mobile-${DateTime.now().millisecondsSinceEpoch}';
    final client = MqttServerClient.withPort(host, clientId, port);
    client.logging(on: false);
    client.keepAlivePeriod = 30;
    client.autoReconnect = true;
    client.onConnected = () => onLog('MQTT connected to $host:$port');
    client.onDisconnected = () => onLog('MQTT disconnected');
    client.connectionMessage = MqttConnectMessage()
        .withClientIdentifier(clientId)
        .startClean()
        .withWillQos(MqttQos.atMostOnce);

    try {
      await client.connect();
      _client = client;
    } catch (error) {
      onLog('MQTT connection failed: $error');
      client.disconnect();
      rethrow;
    }
  }

  void publishJson(String topic, Map<String, dynamic> payload) {
    final client = _client;
    if (client == null || !isConnected) {
      return;
    }
    final builder = MqttClientPayloadBuilder()..addString(jsonEncode(payload));
    client.publishMessage(topic, MqttQos.atMostOnce, builder.payload!);
  }

  Future<void> disconnect() async {
    _client?.disconnect();
    _client = null;
  }
}

class PolarLiveService {
  PolarLiveService() : _polar = Polar();

  final Polar _polar;
  final StreamController<PolarScanDevice> _scanController =
      StreamController<PolarScanDevice>.broadcast();
  final StreamController<List<PolarEcgSample>> _ecgController =
      StreamController<List<PolarEcgSample>>.broadcast();
  final StreamController<List<PolarAccSample>> _accController =
      StreamController<List<PolarAccSample>>.broadcast();
  final StreamController<HrSamplePoint> _hrController =
      StreamController<HrSamplePoint>.broadcast();
  final StreamController<String> _logController =
      StreamController<String>.broadcast();

  Stream<PolarScanDevice> get scanResults => _scanController.stream;
  Stream<List<PolarEcgSample>> get ecgFrames => _ecgController.stream;
  Stream<List<PolarAccSample>> get accFrames => _accController.stream;
  Stream<HrSamplePoint> get hrSamples => _hrController.stream;
  Stream<String> get logs => _logController.stream;

  StreamSubscription<PolarDeviceInfo>? _scanSub;
  StreamSubscription<PolarEcgData>? _ecgSub;
  StreamSubscription<PolarAccData>? _accSub;
  StreamSubscription<PolarHrData>? _hrSub;
  StreamSubscription<PolarDeviceInfo>? _connectingSub;
  StreamSubscription<PolarDeviceInfo>? _connectedSub;
  StreamSubscription<PolarDeviceDisconnectedEvent>? _disconnectedSub;
  StreamSubscription<PolarBatteryLevelEvent>? _batterySub;
  String? _activeIdentifier;

  bool get isConnected => _activeIdentifier != null;

  Future<void> initialize() async {
    _connectingSub ??= _polar.deviceConnecting.listen(
      (event) => _logController.add('Connecting to ${event.name} (${event.deviceId})'),
    );
    _connectedSub ??= _polar.deviceConnected.listen(
      (event) => _logController.add('Connected to ${event.name} (${event.deviceId})'),
    );
    _disconnectedSub ??= _polar.deviceDisconnected.listen(
      (_) => _logController.add('Polar device disconnected'),
    );
    _batterySub ??= _polar.batteryLevel.listen(
      (event) => _logController.add('Battery level: ${event.level}%'),
    );
  }

  Future<void> requestPermissions() async {
    await initialize();
    await _polar.requestPermissions();
    final permissions = <Permission>[
      Permission.bluetoothScan,
      Permission.bluetoothConnect,
      Permission.locationWhenInUse,
    ];
    await permissions.request();
  }

  Future<void> scan({Duration duration = const Duration(seconds: 8)}) async {
    await requestPermissions();
    await _scanSub?.cancel();
    _scanSub = _polar.searchForDevice().listen(
      (event) {
        _scanController.add(
          PolarScanDevice(
            identifier: event.deviceId,
            name: event.name,
            rssi: event.rssi,
          ),
        );
      },
      onError: (Object error, StackTrace stackTrace) {
        _logController.add('Scan failed: $error');
      },
    );
    Future<void>.delayed(duration, () async {
      await _scanSub?.cancel();
      _scanSub = null;
      _logController.add('Scan complete');
    });
  }

  Future<void> connect(String identifier) async {
    await requestPermissions();
    _activeIdentifier = identifier;
    await _polar.connectToDevice(identifier, requestPermissions: false);
    await _startStreams(identifier);
  }

  Future<void> disconnect() async {
    final identifier = _activeIdentifier;
    await _ecgSub?.cancel();
    await _accSub?.cancel();
    await _hrSub?.cancel();
    _ecgSub = null;
    _accSub = null;
    _hrSub = null;
    _activeIdentifier = null;
    if (identifier != null) {
      await _polar.disconnectFromDevice(identifier);
    }
  }

  Future<void> dispose() async {
    await disconnect();
    await _scanSub?.cancel();
    await _connectingSub?.cancel();
    await _connectedSub?.cancel();
    await _disconnectedSub?.cancel();
    await _batterySub?.cancel();
    await _scanController.close();
    await _ecgController.close();
    await _accController.close();
    await _hrController.close();
    await _logController.close();
  }

  Future<void> _startStreams(String identifier) async {
    final hrReady = await _waitForFeature(identifier, PolarSdkFeature.hr);
    if (hrReady) {
      final hrTypes = await _polar.getAvailableHrServiceDataTypes(identifier);
      if (hrTypes.contains(PolarDataType.hr)) {
        _hrSub = _polar.startHrStreaming(identifier).listen(
          (data) {
            for (final sample in data.samples) {
              _hrController.add(
                HrSamplePoint(
                  timestamp: DateTime.now(),
                  hr: sample.hr,
                  rrsMs: List<int>.from(sample.rrsMs),
                  contactStatus: sample.contactStatus,
                ),
              );
            }
          },
          onError: (Object error, StackTrace stackTrace) {
            _logController.add('HR stream error: $error');
          },
        );
      } else {
        _logController.add('HR streaming not available on this device');
      }
    }

    final onlineReady = await _waitForFeature(
      identifier,
      PolarSdkFeature.onlineStreaming,
    );
    if (!onlineReady) {
      _logController.add('Online streaming feature did not become ready');
      return;
    }

    final streamTypes = await _polar.getAvailableOnlineStreamDataTypes(identifier);
    if (streamTypes.contains(PolarDataType.ecg)) {
      _ecgSub = _polar.startEcgStreaming(identifier).listen(
        (data) => _ecgController.add(data.samples),
        onError: (Object error, StackTrace stackTrace) {
          _logController.add('ECG stream error: $error');
        },
      );
    } else {
      _logController.add('ECG streaming not available on this device');
    }

    if (streamTypes.contains(PolarDataType.acc)) {
      _accSub = _polar.startAccStreaming(identifier).listen(
        (data) => _accController.add(data.samples),
        onError: (Object error, StackTrace stackTrace) {
          _logController.add('ACC stream error: $error');
        },
      );
    } else {
      _logController.add('ACC streaming not available on this device');
    }
  }

  Future<bool> _waitForFeature(String identifier, PolarSdkFeature feature) async {
    try {
      await _polar.sdkFeatureReady
          .firstWhere((event) => event.identifier == identifier && event.feature == feature)
          .timeout(const Duration(seconds: 15));
      return true;
    } catch (_) {
      return false;
    }
  }
}

class AnalysisEngine {
  WindowAnalysis computeWindow({
    required List<double> ecg,
    required List<AccVector> acc,
    required List<TimedRrSample> rrSamples,
    required bool contactStatus,
    required int? latestHr,
  }) {
    final qrsEnergy = _computeQrsEnergy(ecg);
    final kurtosis = _computeKurtosis(ecg);
    final rrWindow = rrSamples.where((sample) {
      return sample.timestamp.isAfter(DateTime.now().subtract(const Duration(seconds: 5)));
    }).toList();
    final instantHr = _computeInstantHr(rrWindow, latestHr);
    final accFeatures = _computeAccFeatures(acc);
    final harActivity = _classifyActivity(accFeatures, acc);
    final sqi = _computeSqi(
      ecg: ecg,
      qrsEnergy: qrsEnergy,
      rrCount: rrWindow.length,
      contactStatus: contactStatus,
    );

    return WindowAnalysis(
      timestamp: DateTime.now(),
      sqi: _finiteOrNull(sqi),
      qrsEnergy: _finiteOrNull(qrsEnergy),
      vitalKurtosis: _finiteOrNull(kurtosis),
      instantHr: _finiteOrNull(instantHr),
      nRPeaks: rrWindow.isEmpty ? 0 : rrWindow.length + 1,
      accFeatures: accFeatures,
      harActivity: harActivity,
      rawEcg: ecg,
      status: ecg.isEmpty ? 'Awaiting ECG stream' : 'OK',
    );
  }

  HrvSnapshot computeHrv(List<TimedRrSample> rrSamples) {
    if (rrSamples.length < 6) {
      return const HrvSnapshot(status: 'Need at least 6 RR intervals');
    }

    final values = rrSamples.map((sample) => sample.rrMs.toDouble()).toList();
    final filtered = values.length >= 8
        ? CalculateHrv.filterPeaksRrs(List<double>.from(values))
        : values;

    try {
      final timeDomain = CalculateHrv.calcTimeDomainRrs(filtered);
      double? lfHf;
      if (filtered.length >= 10) {
        try {
          lfHf = _finiteOrNull(
            CalculateHrv.calcFrequencyDomainRrs(filtered).hrvFrequencyDomain['LF/HF'],
          );
        } catch (_) {
          lfHf = null;
        }
      }

      final meanNn = _finiteOrNull(timeDomain['MeanNN']);
      final meanHr =
          meanNn != null && meanNn > 0 ? _finiteOrNull(60000.0 / meanNn) : null;

      return HrvSnapshot(
        rmssdMs: _finiteOrNull(timeDomain['RMSSD']),
        sdnnMs: _finiteOrNull(timeDomain['SDNN']),
        lfHf: lfHf,
        meanHr: meanHr,
        meanNn: meanNn,
        status: 'OK',
      );
    } catch (error) {
      return HrvSnapshot(status: 'HRV error: $error');
    }
  }

  double? _computeInstantHr(List<TimedRrSample> rrWindow, int? latestHr) {
    if (rrWindow.isNotEmpty) {
      final meanRr = rrWindow
              .map((sample) => sample.rrMs.toDouble())
              .reduce((a, b) => a + b) /
          rrWindow.length;
      if (meanRr > 0) {
        return 60000.0 / meanRr;
      }
    }
    return latestHr?.toDouble();
  }

  double? _computeSqi({
    required List<double> ecg,
    required double? qrsEnergy,
    required int rrCount,
    required bool contactStatus,
  }) {
    if (ecg.length < 100) {
      return null;
    }
    final amplitude = _standardDeviation(ecg);
    final clippingRatio = ecg.where((value) => value.abs() > 2500).length / ecg.length;
    final amplitudeScore = ((amplitude - 40) / 500).clamp(0.0, 1.0);
    final rrScore = (rrCount / 4).clamp(0.0, 1.0);
    final contactScore = contactStatus ? 1.0 : 0.25;
    final qrsScore = (qrsEnergy ?? 0.2).clamp(0.0, 1.0);
    final clippingPenalty = (1 - (clippingRatio * 4)).clamp(0.0, 1.0);

    return ((0.25 * amplitudeScore) +
            (0.30 * rrScore) +
            (0.25 * qrsScore) +
            (0.20 * contactScore)) *
        clippingPenalty;
  }

  double? _computeQrsEnergy(List<double> ecg) {
    if (ecg.length < 128) {
      return null;
    }
    final centered = _center(ecg);
    final spectrum = _powerSpectrum(centered, 130);
    if (spectrum.isEmpty) {
      return null;
    }

    double total = 0;
    double qrs = 0;
    for (final entry in spectrum.entries) {
      final frequency = entry.key;
      final power = entry.value;
      if (frequency >= 1 && frequency <= 40) {
        total += power;
      }
      if (frequency >= 5 && frequency <= 15) {
        qrs += power;
      }
    }
    return total > 0 ? qrs / total : null;
  }

  double? _computeKurtosis(List<double> signal) {
    if (signal.length < 4) {
      return null;
    }
    final mean = signal.reduce((a, b) => a + b) / signal.length;
    final centered = signal.map((value) => value - mean).toList();
    final variance = centered
            .map((value) => value * value)
            .reduce((a, b) => a + b) /
        centered.length;
    if (variance <= 0) {
      return null;
    }
    final fourthMoment = centered
            .map((value) => math.pow(value, 4).toDouble())
            .reduce((a, b) => a + b) /
        centered.length;
    return fourthMoment / math.pow(variance, 2).toDouble();
  }

  AccFeatureSnapshot _computeAccFeatures(List<AccVector> acc) {
    if (acc.length < 40) {
      return const AccFeatureSnapshot();
    }
    final magnitudes = acc.map((sample) => sample.magnitude).toList();
    final mean = magnitudes.reduce((a, b) => a + b) / magnitudes.length;
    final variance = _variance(magnitudes);

    final centered = magnitudes.map((value) => value - mean).toList();
    final spectrum = _powerSpectrum(centered, 100);
    final usable = spectrum.entries.where((entry) => entry.key <= 25).toList();
    if (usable.isEmpty) {
      return AccFeatureSnapshot(
        meanMagMg: _finiteOrNull(mean),
        varMagMg2: _finiteOrNull(variance),
      );
    }

    final totalPower = usable.fold<double>(0, (sum, entry) => sum + entry.value);
    double? entropy;
    double? medianFreq;
    if (totalPower > 0) {
      double running = 0;
      var entropyAccumulator = 0.0;
      for (final entry in usable) {
        final probability = entry.value / totalPower;
        if (probability > 0) {
          entropyAccumulator -= probability * (math.log(probability) / math.ln2);
        }
        running += probability;
        medianFreq ??= running >= 0.5 ? entry.key : null;
      }
      entropy = entropyAccumulator / (math.log(usable.length) / math.ln2);
    }

    return AccFeatureSnapshot(
      meanMagMg: _finiteOrNull(mean),
      varMagMg2: _finiteOrNull(variance),
      spectralEntropy: _finiteOrNull(entropy),
      medianFreqHz: _finiteOrNull(medianFreq),
    );
  }

  HarActivity _classifyActivity(
    AccFeatureSnapshot features,
    List<AccVector> acc,
  ) {
    final variance = features.varMagMg2 ?? 0;
    final medianFreq = features.medianFreqHz ?? 0;
    final meanMag = features.meanMagMg ?? 0;

    if (acc.length < 50) {
      return HarActivity.unknown();
    }

    if (variance < 1200) {
      final yMean = acc.map((sample) => sample.y).reduce((a, b) => a + b) / acc.length;
      final zMean = acc.map((sample) => sample.z).reduce((a, b) => a + b) / acc.length;
      final pitch = math.atan2(zMean.abs(), yMean.abs()) * (180 / math.pi);
      final label = pitch > 15 ? 'sitting' : 'standing';
      return HarActivity(
        label: label,
        confidence: <String, double>{label: 1.0},
        heuristic: true,
      );
    }

    if (medianFreq >= 2.4 || meanMag >= 1200) {
      return const HarActivity(
        label: 'running',
        confidence: <String, double>{'running': 0.84, 'walking': 0.16},
        heuristic: true,
      );
    }

    if (medianFreq >= 1.1 && medianFreq < 2.4) {
      final stairBoost = variance > 30000 ? 0.55 : 0.25;
      return HarActivity(
        label: stairBoost > 0.5 ? 'stair_climbing' : 'walking',
        confidence: <String, double>{
          'walking': 1 - stairBoost,
          'stair_climbing': stairBoost,
        },
        heuristic: true,
      );
    }

    return const HarActivity(
      label: 'cycling',
      confidence: <String, double>{'cycling': 0.68, 'walking': 0.32},
      heuristic: true,
    );
  }

  Map<double, double> _powerSpectrum(List<double> signal, double sampleRate) {
    final n = signal.length;
    final spectrum = <double, double>{};
    if (n < 16) {
      return spectrum;
    }

    for (var k = 0; k <= n ~/ 2; k += 1) {
      var real = 0.0;
      var imaginary = 0.0;
      for (var i = 0; i < n; i += 1) {
        final angle = (2 * math.pi * k * i) / n;
        real += signal[i] * math.cos(angle);
        imaginary -= signal[i] * math.sin(angle);
      }
      final power = (real * real) + (imaginary * imaginary);
      spectrum[(k * sampleRate) / n] = power;
    }
    return spectrum;
  }

  List<double> _center(List<double> values) {
    final mean = values.reduce((a, b) => a + b) / values.length;
    return values.map((value) => value - mean).toList();
  }

  double _standardDeviation(List<double> values) {
    return math.sqrt(_variance(values));
  }

  double _variance(List<double> values) {
    if (values.length < 2) {
      return 0;
    }
    final mean = values.reduce((a, b) => a + b) / values.length;
    final sum = values
        .map((value) => (value - mean) * (value - mean))
        .reduce((a, b) => a + b);
    return sum / (values.length - 1);
  }

  double? _finiteOrNull(num? value) {
    if (value == null) {
      return null;
    }
    final asDouble = value.toDouble();
    return asDouble.isFinite ? asDouble : null;
  }
}

class PulseForgeController extends ChangeNotifier {
  PulseForgeController({
    LocalStorageService? storage,
    MqttTelemetryService? mqtt,
    PolarLiveService? polar,
    AnalysisEngine? analysis,
  })  : _storage = storage ?? LocalStorageService(),
        _mqtt = mqtt ?? MqttTelemetryService(),
        _polar = polar ?? PolarLiveService(),
        _analysis = analysis ?? AnalysisEngine();

  final LocalStorageService _storage;
  final MqttTelemetryService _mqtt;
  final PolarLiveService _polar;
  final AnalysisEngine _analysis;
  final CarpPolarCompatibility carpCompatibility = const CarpPolarCompatibility();
  final SignalToolkitBridge signalToolkitBridge = SignalToolkitBridge();

  final ListQueue<double> _ecgBuffer = ListQueue<double>();
  final ListQueue<AccVector> _accBuffer = ListQueue<AccVector>();
  final ListQueue<double> _hrChartBuffer = ListQueue<double>();
  final ListQueue<TimedRrSample> _rrBuffer = ListQueue<TimedRrSample>();
  final List<PolarScanDevice> _devices = <PolarScanDevice>[];
  final List<LogEntry> _logs = <LogEntry>[];

  StreamSubscription<PolarScanDevice>? _scanResultsSub;
  StreamSubscription<List<PolarEcgSample>>? _ecgFramesSub;
  StreamSubscription<List<PolarAccSample>>? _accFramesSub;
  StreamSubscription<HrSamplePoint>? _hrSampleSub;
  StreamSubscription<String>? _polarLogSub;
  Timer? _windowTimer;
  Timer? _hrvTimer;
  Timer? _uiTimer;
  Timer? _mockTimer;

  PatientProfile profile = PatientProfile.initial();
  LiveConnectionState connectionState = LiveConnectionState.idle;
  WindowAnalysis latestWindow = WindowAnalysis.empty();
  HrvSnapshot latestHrv = HrvSnapshot.empty();
  String mqttHost = 'broker.emqx.io';
  int mqttPort = 1883;
  String? connectedDeviceId;
  String? connectedDeviceName;
  int? latestHeartRate;
  bool lastContactStatus = false;
  bool isRecording = false;
  bool usingMock = false;
  File? sessionFile;
  int savedWindows = 0;

  List<PolarScanDevice> get devices => List<PolarScanDevice>.unmodifiable(_devices);
  List<LogEntry> get logs => List<LogEntry>.unmodifiable(_logs.reversed);
  List<double> get ecgSamples => _ecgBuffer.toList();
  List<AccVector> get accSamples => _accBuffer.toList();
  List<double> get hrSamples => _hrChartBuffer.toList();
  bool get isMqttConnected => _mqtt.isConnected;

  Future<void> initialize() async {
    profile = await _storage.loadProfile();
    _polarLogSub = _polar.logs.listen(_log);
    _scanResultsSub = _polar.scanResults.listen(_upsertDevice);
    _ecgFramesSub = _polar.ecgFrames.listen(_handleEcgFrame);
    _accFramesSub = _polar.accFrames.listen(_handleAccFrame);
    _hrSampleSub = _polar.hrSamples.listen(_handleHrSample);
    _windowTimer = Timer.periodic(const Duration(seconds: 5), (_) => _runWindowAnalysis());
    _hrvTimer = Timer.periodic(const Duration(seconds: 2), (_) => _runHrvAnalysis());
    _uiTimer = Timer.periodic(const Duration(milliseconds: 100), (_) => notifyListeners());
    _log('scientisst bridge ready (${signalToolkitBridge.warmupToken})');
    _log('CARP-compatible measures: ${carpCompatibility.supportedMeasures.join(', ')}');
    notifyListeners();
  }

  Future<void> saveProfile(PatientProfile nextProfile) async {
    profile = nextProfile;
    await _storage.saveProfile(nextProfile);
    _log('Saved intake form for ${profile.subjectId}');
    notifyListeners();
  }

  void updateBroker(String host, int port) {
    mqttHost = host.trim().isEmpty ? 'broker.emqx.io' : host.trim();
    mqttPort = port;
    notifyListeners();
  }

  Future<void> scanDevices() async {
    connectionState = LiveConnectionState.scanning;
    _devices.clear();
    _log('Scanning for Polar devices...');
    notifyListeners();
    await _polar.scan();
    Future<void>.delayed(const Duration(seconds: 8), () {
      if (connectionState == LiveConnectionState.scanning) {
        connectionState = _devices.isEmpty
            ? LiveConnectionState.disconnected
            : LiveConnectionState.idle;
        notifyListeners();
      }
    });
  }

  Future<void> connectSelectedDevice(PolarScanDevice device) async {
    await disconnect();
    usingMock = false;
    connectionState = LiveConnectionState.connecting;
    connectedDeviceId = device.identifier;
    connectedDeviceName = device.name;
    _log('Connecting to ${device.name} (${device.identifier})');
    notifyListeners();

    try {
      await _polar.connect(device.identifier);
      await _ensureMqttConnected();
      await _publishInfoTopic();
      connectionState = LiveConnectionState.connected;
      _log('Live Polar stream started');
      notifyListeners();
    } catch (error) {
      connectionState = LiveConnectionState.error;
      _log('Connect failed: $error');
      notifyListeners();
    }
  }

  Future<void> connectMock() async {
    await disconnect();
    usingMock = true;
    connectionState = LiveConnectionState.connected;
    connectedDeviceId = 'MOCK-H10';
    connectedDeviceName = 'Mock Polar H10';
    await _ensureMqttConnected();
    await _publishInfoTopic();
    _startMockData();
    _log('Mock sensor started');
    notifyListeners();
  }

  Future<void> disconnect() async {
    _stopMockData();
    if (_polar.isConnected) {
      await _polar.disconnect();
    }
    await _mqtt.disconnect();
    connectionState = LiveConnectionState.disconnected;
    connectedDeviceId = null;
    connectedDeviceName = null;
    latestHeartRate = null;
    lastContactStatus = false;
    _log('Disconnected');
    notifyListeners();
  }

  Future<void> startRecording() async {
    if (!profile.isValid) {
      _log('Subject ID is required before recording');
      return;
    }
    await _ensureMqttConnected();
    await _publishInfoTopic();
    sessionFile = await _storage.startSession(profile.subjectId);
    await _storage.writeProfileCopy(sessionFile!, profile);
    savedWindows = 0;
    isRecording = true;
    _log('Recording started: ${sessionFile!.path}');
    notifyListeners();
  }

  Future<void> stopRecording() async {
    if (!isRecording) {
      return;
    }
    isRecording = false;
    _log('Recording stopped. Saved $savedWindows window(s).');
    notifyListeners();
  }

  void disposeController() {
    _windowTimer?.cancel();
    _hrvTimer?.cancel();
    _uiTimer?.cancel();
    _stopMockData();
    unawaited(_scanResultsSub?.cancel() ?? Future<void>.value());
    unawaited(_ecgFramesSub?.cancel() ?? Future<void>.value());
    unawaited(_accFramesSub?.cancel() ?? Future<void>.value());
    unawaited(_hrSampleSub?.cancel() ?? Future<void>.value());
    unawaited(_polarLogSub?.cancel() ?? Future<void>.value());
    unawaited(_polar.dispose());
    unawaited(_mqtt.disconnect());
    super.dispose();
  }

  void _upsertDevice(PolarScanDevice device) {
    final index = _devices.indexWhere((item) => item.identifier == device.identifier);
    if (index >= 0) {
      _devices[index] = device;
    } else {
      _devices.add(device);
    }
    notifyListeners();
  }

  void _handleEcgFrame(List<PolarEcgSample> samples) {
    for (final sample in samples) {
      _ecgBuffer.add(sample.voltage.toDouble());
    }
    while (_ecgBuffer.length > 130 * 120) {
      _ecgBuffer.removeFirst();
    }
  }

  void _handleAccFrame(List<PolarAccSample> samples) {
    for (final sample in samples) {
      _accBuffer.add(
        AccVector(
          timestamp: sample.timeStamp,
          x: sample.x.toDouble(),
          y: sample.y.toDouble(),
          z: sample.z.toDouble(),
        ),
      );
    }
    while (_accBuffer.length > 100 * 120) {
      _accBuffer.removeFirst();
    }
  }

  void _handleHrSample(HrSamplePoint sample) {
    latestHeartRate = sample.hr;
    lastContactStatus = sample.contactStatus;
    _hrChartBuffer.add(sample.hr.toDouble());
    while (_hrChartBuffer.length > 120) {
      _hrChartBuffer.removeFirst();
    }
    for (final rr in sample.rrsMs) {
      _rrBuffer.add(TimedRrSample(timestamp: sample.timestamp, rrMs: rr));
    }
    final cutoff = DateTime.now().subtract(const Duration(seconds: 90));
    while (_rrBuffer.isNotEmpty && _rrBuffer.first.timestamp.isBefore(cutoff)) {
      _rrBuffer.removeFirst();
    }
  }

  Future<void> _ensureMqttConnected() async {
    if (!profile.isValid || _mqtt.isConnected) {
      return;
    }
    try {
      await _mqtt.connect(host: mqttHost, port: mqttPort, onLog: _log);
    } catch (_) {
      // Log already emitted in service.
    }
  }

  Future<void> _publishInfoTopic() async {
    if (!profile.isValid || !_mqtt.isConnected) {
      return;
    }
    _mqtt.publishJson('pulseforgeai/${profile.subjectId}/info', profile.toJson());
  }

  Future<void> _runWindowAnalysis() async {
    if (connectionState != LiveConnectionState.connected) {
      return;
    }
    final ecgWindow = _takeLast(_ecgBuffer, 130 * 5);
    final accWindow = _takeLast(_accBuffer, 100 * 5);
    final rrWindow = _rrBuffer.toList();
    latestWindow = _analysis.computeWindow(
      ecg: ecgWindow,
      acc: accWindow,
      rrSamples: rrWindow,
      contactStatus: lastContactStatus,
      latestHr: latestHeartRate,
    );

    final payload = _buildPayload(latestWindow, latestHrv);
    if (_mqtt.isConnected && profile.subjectId.trim().isNotEmpty) {
      _mqtt.publishJson('pulseforgeai/${profile.subjectId}/raw', payload);
    }
    if (isRecording && sessionFile != null) {
      await _storage.appendPayload(sessionFile!, payload);
      savedWindows += 1;
    }
    notifyListeners();
  }

  void _runHrvAnalysis() {
    if (connectionState != LiveConnectionState.connected) {
      return;
    }
    final cutoff = DateTime.now().subtract(const Duration(seconds: 30));
    final rrWindow = _rrBuffer.where((sample) => sample.timestamp.isAfter(cutoff)).toList();
    latestHrv = _analysis.computeHrv(rrWindow);
    notifyListeners();
  }

  Map<String, dynamic> _buildPayload(
    WindowAnalysis window,
    HrvSnapshot hrv,
  ) {
    return <String, dynamic>{
      'unix_timestamp': window.timestamp.millisecondsSinceEpoch / 1000,
      'subject_id': profile.subjectId.trim().isEmpty ? 'unknown' : profile.subjectId,
      'window_s': 5,
      'ecg_quality': <String, dynamic>{
        'sqi': window.sqi,
        'sqi_metrics': <String, dynamic>{
          'qrs_energy': window.qrsEnergy,
          'vital_kurtosis': window.vitalKurtosis,
        },
      },
      'heart_rate': <String, dynamic>{
        'avg_bpm_ble': latestHeartRate?.toDouble(),
        'n_ble_samples': math.min(_hrChartBuffer.length, 5),
        'avg_bpm_ecg': window.instantHr,
        'n_r_peaks': window.nRPeaks,
      },
      'hrv': <String, dynamic>{
        'rmssd_ms': hrv.rmssdMs,
        'sdnn_ms': hrv.sdnnMs,
        'lf_hf_ratio': hrv.lfHf,
        'analysis_window_s': 30,
      },
      'ecg_morphology': <String, dynamic>{
        'qrs_ms': hrv.qrsWidth,
        'qt_ms': hrv.qtWidth,
        'qtc_ms': hrv.qtcWidth,
        'st_ms': hrv.stWidth,
        'p_ms': hrv.pWidth,
      },
      'accelerometer': <String, dynamic>{
        'features': window.accFeatures.toJson(),
        'activity': window.harActivity.toJson(),
      },
      'raw_ecg': window.rawEcg,
    };
  }

  List<T> _takeLast<T>(Iterable<T> values, int count) {
    final list = values.toList();
    if (list.length <= count) {
      return list;
    }
    return list.sublist(list.length - count);
  }

  void _startMockData() {
    _mockTimer = Timer.periodic(const Duration(milliseconds: 20), (timer) {
      final t = timer.tick / 50.0;
      final ecgValue = 350 * math.sin(2 * math.pi * 1.15 * t) +
          80 * math.sin(2 * math.pi * 15 * t) +
          15 * math.sin(2 * math.pi * 40 * t);
      _ecgBuffer.add(ecgValue);
      while (_ecgBuffer.length > 130 * 120) {
        _ecgBuffer.removeFirst();
      }

      if (timer.tick % 5 == 0) {
        final x = 100 * math.sin(2 * math.pi * 1.5 * t);
        final y = 850 + (180 * math.cos(2 * math.pi * 1.5 * t));
        final z = 200 * math.sin(2 * math.pi * 0.75 * t);
        _accBuffer.add(
          AccVector(
            timestamp: DateTime.now(),
            x: x,
            y: y,
            z: z,
          ),
        );
        while (_accBuffer.length > 100 * 120) {
          _accBuffer.removeFirst();
        }
      }

      if (timer.tick % 50 == 0) {
        final hr = 72 + (5 * math.sin(t / 4)).round();
        final rr = (60000 / hr).round();
        _handleHrSample(
          HrSamplePoint(
            timestamp: DateTime.now(),
            hr: hr,
            rrsMs: <int>[rr],
            contactStatus: true,
          ),
        );
      }
    });
  }

  void _stopMockData() {
    _mockTimer?.cancel();
    _mockTimer = null;
  }

  void _log(String message) {
    _logs.add(LogEntry(DateTime.now(), message));
    if (_logs.length > 300) {
      _logs.removeAt(0);
    }
  }
}
