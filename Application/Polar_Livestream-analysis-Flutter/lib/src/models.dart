import 'dart:convert';
import 'dart:math' as math;

enum LiveConnectionState {
  idle,
  scanning,
  connecting,
  connected,
  disconnected,
  error,
}

class PatientProfile {
  const PatientProfile({
    required this.subjectId,
    required this.age,
    required this.sex,
    required this.heightCm,
    required this.weightKg,
    required this.hrTargetLow,
    required this.hrTargetHigh,
    required this.event,
    required this.eventDateIso,
    required this.lvef,
    required this.comorbDia,
    required this.comorbCopd,
    required this.comorbHyp,
    required this.comorbPad,
    required this.comorbRen,
    required this.betaBlocker,
    required this.tobacco,
    required this.activityLevel,
    required this.chestPain,
    required this.dyspnea,
    required this.phq2,
    required this.baselineNotes,
    required this.includeHistoricalBaseline,
  });

  factory PatientProfile.initial() {
    final now = DateTime.now();
    return PatientProfile(
      subjectId: '',
      age: 50,
      sex: 'Male',
      heightCm: 170,
      weightKg: 70,
      hrTargetLow: 80,
      hrTargetHigh: 130,
      event: 'Post-MI',
      eventDateIso: DateTime(now.year, now.month, now.day).toIso8601String(),
      lvef: 55,
      comorbDia: false,
      comorbCopd: false,
      comorbHyp: false,
      comorbPad: false,
      comorbRen: false,
      betaBlocker: 'No',
      tobacco: 'Never',
      activityLevel: 3,
      chestPain: 'None',
      dyspnea: 'None',
      phq2: 0,
      baselineNotes: '',
      includeHistoricalBaseline: false,
    );
  }

  final String subjectId;
  final int age;
  final String sex;
  final int heightCm;
  final double weightKg;
  final int hrTargetLow;
  final int hrTargetHigh;
  final String event;
  final String eventDateIso;
  final int lvef;
  final bool comorbDia;
  final bool comorbCopd;
  final bool comorbHyp;
  final bool comorbPad;
  final bool comorbRen;
  final String betaBlocker;
  final String tobacco;
  final int activityLevel;
  final String chestPain;
  final String dyspnea;
  final int phq2;
  final String baselineNotes;
  final bool includeHistoricalBaseline;

  PatientProfile copyWith({
    String? subjectId,
    int? age,
    String? sex,
    int? heightCm,
    double? weightKg,
    int? hrTargetLow,
    int? hrTargetHigh,
    String? event,
    String? eventDateIso,
    int? lvef,
    bool? comorbDia,
    bool? comorbCopd,
    bool? comorbHyp,
    bool? comorbPad,
    bool? comorbRen,
    String? betaBlocker,
    String? tobacco,
    int? activityLevel,
    String? chestPain,
    String? dyspnea,
    int? phq2,
    String? baselineNotes,
    bool? includeHistoricalBaseline,
  }) {
    return PatientProfile(
      subjectId: subjectId ?? this.subjectId,
      age: age ?? this.age,
      sex: sex ?? this.sex,
      heightCm: heightCm ?? this.heightCm,
      weightKg: weightKg ?? this.weightKg,
      hrTargetLow: hrTargetLow ?? this.hrTargetLow,
      hrTargetHigh: hrTargetHigh ?? this.hrTargetHigh,
      event: event ?? this.event,
      eventDateIso: eventDateIso ?? this.eventDateIso,
      lvef: lvef ?? this.lvef,
      comorbDia: comorbDia ?? this.comorbDia,
      comorbCopd: comorbCopd ?? this.comorbCopd,
      comorbHyp: comorbHyp ?? this.comorbHyp,
      comorbPad: comorbPad ?? this.comorbPad,
      comorbRen: comorbRen ?? this.comorbRen,
      betaBlocker: betaBlocker ?? this.betaBlocker,
      tobacco: tobacco ?? this.tobacco,
      activityLevel: activityLevel ?? this.activityLevel,
      chestPain: chestPain ?? this.chestPain,
      dyspnea: dyspnea ?? this.dyspnea,
      phq2: phq2 ?? this.phq2,
      baselineNotes: baselineNotes ?? this.baselineNotes,
      includeHistoricalBaseline:
          includeHistoricalBaseline ?? this.includeHistoricalBaseline,
    );
  }

  bool get isValid => subjectId.trim().isNotEmpty;

  Map<String, dynamic> toJson() {
    return {
      'subject_id': subjectId,
      'age': age,
      'sex': sex,
      'height_cm': heightCm,
      'weight_kg': weightKg,
      'hr_target_low': hrTargetLow,
      'hr_target_high': hrTargetHigh,
      'event': event,
      'event_date': eventDateIso,
      'lvef': lvef,
      'comorb_dia': comorbDia,
      'comorb_copd': comorbCopd,
      'comorb_hyp': comorbHyp,
      'comorb_pad': comorbPad,
      'comorb_ren': comorbRen,
      'beta_blocker': betaBlocker,
      'tobacco': tobacco,
      'activity_level': activityLevel,
      'chest_pain': chestPain,
      'dyspnea': dyspnea,
      'phq2': phq2,
      'historical_baseline_enabled': includeHistoricalBaseline,
      'historical_baseline_notes': baselineNotes,
    };
  }

  factory PatientProfile.fromJson(Map<String, dynamic> json) {
    return PatientProfile(
      subjectId: (json['subject_id'] as String? ?? '').trim(),
      age: (json['age'] as num? ?? 50).toInt(),
      sex: json['sex'] as String? ?? 'Male',
      heightCm: (json['height_cm'] as num? ?? 170).toInt(),
      weightKg: (json['weight_kg'] as num? ?? 70).toDouble(),
      hrTargetLow: (json['hr_target_low'] as num? ?? 80).toInt(),
      hrTargetHigh: (json['hr_target_high'] as num? ?? 130).toInt(),
      event: json['event'] as String? ?? 'Post-MI',
      eventDateIso: json['event_date'] as String? ?? DateTime.now().toIso8601String(),
      lvef: (json['lvef'] as num? ?? 55).toInt(),
      comorbDia: json['comorb_dia'] as bool? ?? false,
      comorbCopd: json['comorb_copd'] as bool? ?? false,
      comorbHyp: json['comorb_hyp'] as bool? ?? false,
      comorbPad: json['comorb_pad'] as bool? ?? false,
      comorbRen: json['comorb_ren'] as bool? ?? false,
      betaBlocker: json['beta_blocker'] as String? ?? 'No',
      tobacco: json['tobacco'] as String? ?? 'Never',
      activityLevel: (json['activity_level'] as num? ?? 3).toInt(),
      chestPain: json['chest_pain'] as String? ?? 'None',
      dyspnea: json['dyspnea'] as String? ?? 'None',
      phq2: (json['phq2'] as num? ?? 0).toInt(),
      baselineNotes: json['historical_baseline_notes'] as String? ?? '',
      includeHistoricalBaseline:
          json['historical_baseline_enabled'] as bool? ?? false,
    );
  }

  String serialize() => jsonEncode(toJson());

  factory PatientProfile.deserialize(String source) =>
      PatientProfile.fromJson(jsonDecode(source) as Map<String, dynamic>);
}

class PolarScanDevice {
  const PolarScanDevice({
    required this.identifier,
    required this.name,
    required this.rssi,
  });

  final String identifier;
  final String name;
  final int? rssi;
}

class LogEntry {
  const LogEntry(this.timestamp, this.message);

  final DateTime timestamp;
  final String message;

  String get formattedTimestamp {
    final h = timestamp.hour.toString().padLeft(2, '0');
    final m = timestamp.minute.toString().padLeft(2, '0');
    final s = timestamp.second.toString().padLeft(2, '0');
    return '$h:$m:$s';
  }
}

class AccVector {
  const AccVector({
    required this.timestamp,
    required this.x,
    required this.y,
    required this.z,
  });

  final DateTime timestamp;
  final double x;
  final double y;
  final double z;

  double get magnitude => math.sqrt((x * x) + (y * y) + (z * z));
}

class HrSamplePoint {
  const HrSamplePoint({
    required this.timestamp,
    required this.hr,
    required this.rrsMs,
    required this.contactStatus,
  });

  final DateTime timestamp;
  final int hr;
  final List<int> rrsMs;
  final bool contactStatus;
}

class TimedRrSample {
  const TimedRrSample({
    required this.timestamp,
    required this.rrMs,
  });

  final DateTime timestamp;
  final int rrMs;
}

class AccFeatureSnapshot {
  const AccFeatureSnapshot({
    this.meanMagMg,
    this.varMagMg2,
    this.spectralEntropy,
    this.medianFreqHz,
  });

  final double? meanMagMg;
  final double? varMagMg2;
  final double? spectralEntropy;
  final double? medianFreqHz;

  Map<String, dynamic> toJson() => {
        'mean_mag_mg': meanMagMg,
        'var_mag_mg2': varMagMg2,
        'spectral_entropy': spectralEntropy,
        'median_freq_hz': medianFreqHz,
      };
}

class HarActivity {
  const HarActivity({
    required this.label,
    required this.confidence,
    this.heuristic = false,
  });

  factory HarActivity.unknown() =>
      const HarActivity(label: 'unknown', confidence: <String, double>{});

  final String label;
  final Map<String, double> confidence;
  final bool heuristic;

  Map<String, dynamic> toJson() => {
        'label': label,
        'confidence': confidence,
        'heuristic': heuristic,
      };
}

class HrvSnapshot {
  const HrvSnapshot({
    this.rmssdMs,
    this.sdnnMs,
    this.lfHf,
    this.meanHr,
    this.meanNn,
    this.qrsWidth,
    this.qtWidth,
    this.qtcWidth,
    this.stWidth,
    this.pWidth,
    this.status = 'Awaiting data',
  });

  factory HrvSnapshot.empty() => const HrvSnapshot();

  final double? rmssdMs;
  final double? sdnnMs;
  final double? lfHf;
  final double? meanHr;
  final double? meanNn;
  final double? qrsWidth;
  final double? qtWidth;
  final double? qtcWidth;
  final double? stWidth;
  final double? pWidth;
  final String status;
}

class WindowAnalysis {
  const WindowAnalysis({
    required this.timestamp,
    this.sqi,
    this.qrsEnergy,
    this.vitalKurtosis,
    this.instantHr,
    this.nRPeaks = 0,
    this.accFeatures = const AccFeatureSnapshot(),
    this.harActivity =
        const HarActivity(label: 'unknown', confidence: <String, double>{}),
    this.rawEcg = const <double>[],
    this.status = 'Awaiting data',
  });

  factory WindowAnalysis.empty() => WindowAnalysis(
        timestamp: DateTime.now(),
        status: 'Awaiting data',
      );

  final DateTime timestamp;
  final double? sqi;
  final double? qrsEnergy;
  final double? vitalKurtosis;
  final double? instantHr;
  final int nRPeaks;
  final AccFeatureSnapshot accFeatures;
  final HarActivity harActivity;
  final List<double> rawEcg;
  final String status;
}
