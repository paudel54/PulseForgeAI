import 'package:flutter/material.dart';

import 'models.dart';
import 'services.dart';

class IntakeScreen extends StatefulWidget {
  const IntakeScreen({
    super.key,
    required this.controller,
    required this.onContinue,
  });

  final PulseForgeController controller;
  final VoidCallback onContinue;

  @override
  State<IntakeScreen> createState() => _IntakeScreenState();
}

class _IntakeScreenState extends State<IntakeScreen> {
  late PatientProfile _profile;
  late TextEditingController _subjectIdController;
  late TextEditingController _baselineNotesController;
  late DateTime _eventDate;
  var _controllersReady = false;

  @override
  void initState() {
    super.initState();
    _loadProfile(widget.controller.profile);
  }

  @override
  void didUpdateWidget(covariant IntakeScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.controller.profile != widget.controller.profile) {
      _loadProfile(widget.controller.profile);
    }
  }

  @override
  void dispose() {
    _subjectIdController.dispose();
    _baselineNotesController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 4,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Patient Intake'),
          bottom: const TabBar(
            isScrollable: true,
            tabs: <Tab>[
              Tab(text: 'Demographics'),
              Tab(text: 'Clinical'),
              Tab(text: 'Symptoms'),
              Tab(text: 'Baseline'),
            ],
          ),
        ),
        body: Column(
          children: <Widget>[
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 20, 20, 8),
              child: Row(
                children: <Widget>[
                  Expanded(
                    child: TextField(
                      controller: _subjectIdController,
                      decoration: const InputDecoration(
                        labelText: 'Subject ID',
                        hintText: 'Required, e.g. S001',
                        prefixIcon: Icon(Icons.badge_outlined),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            Expanded(
              child: TabBarView(
                children: <Widget>[
                  _buildDemographicsTab(),
                  _buildClinicalTab(),
                  _buildSymptomsTab(),
                  _buildBaselineTab(),
                ],
              ),
            ),
            SafeArea(
              top: false,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(20, 8, 20, 20),
                child: Row(
                  children: <Widget>[
                    OutlinedButton.icon(
                      onPressed: () => setState(() {
                        _loadProfile(widget.controller.profile);
                      }),
                      icon: const Icon(Icons.refresh),
                      label: const Text('Load Saved'),
                    ),
                    const SizedBox(width: 12),
                    OutlinedButton.icon(
                      onPressed: () => setState(() {
                        _loadProfile(PatientProfile.initial());
                      }),
                      icon: const Icon(Icons.clear),
                      label: const Text('Clear'),
                    ),
                    const Spacer(),
                    FilledButton.icon(
                      onPressed: _saveAndContinue,
                      icon: const Icon(Icons.arrow_forward),
                      label: const Text('Save & Continue'),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDemographicsTab() {
    return ListView(
      padding: const EdgeInsets.all(20),
      children: <Widget>[
        _sectionCard(
          title: 'Demographics & physicals',
          child: Column(
            children: <Widget>[
              _numberRow(
                label: 'Age',
                value: _profile.age,
                min: 18,
                max: 120,
                onChanged: (value) => setState(() {
                  _profile = _profile.copyWith(age: value);
                }),
              ),
              const SizedBox(height: 16),
              DropdownButtonFormField<String>(
                value: _profile.sex,
                decoration: const InputDecoration(labelText: 'Biological sex'),
                items: const <String>['Male', 'Female', 'Other']
                    .map((value) => DropdownMenuItem<String>(
                          value: value,
                          child: Text(value),
                        ))
                    .toList(),
                onChanged: (value) => setState(() {
                  _profile = _profile.copyWith(sex: value);
                }),
              ),
              const SizedBox(height: 16),
              _numberRow(
                label: 'Height (cm)',
                value: _profile.heightCm,
                min: 100,
                max: 240,
                onChanged: (value) => setState(() {
                  _profile = _profile.copyWith(heightCm: value);
                }),
              ),
              const SizedBox(height: 16),
              _sliderRow(
                label: 'Weight (kg)',
                value: _profile.weightKg,
                min: 30,
                max: 180,
                divisions: 150,
                onChanged: (value) => setState(() {
                  _profile = _profile.copyWith(weightKg: double.parse(value.toStringAsFixed(1)));
                }),
              ),
              const SizedBox(height: 16),
              Row(
                children: <Widget>[
                  Expanded(
                    child: _numberRow(
                      label: 'Target HR low',
                      value: _profile.hrTargetLow,
                      min: 50,
                      max: 160,
                      onChanged: (value) => setState(() {
                        _profile = _profile.copyWith(hrTargetLow: value);
                      }),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _numberRow(
                      label: 'Target HR high',
                      value: _profile.hrTargetHigh,
                      min: 70,
                      max: 200,
                      onChanged: (value) => setState(() {
                        _profile = _profile.copyWith(hrTargetHigh: value);
                      }),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildClinicalTab() {
    return ListView(
      padding: const EdgeInsets.all(20),
      children: <Widget>[
        _sectionCard(
          title: 'Clinical history',
          child: Column(
            children: <Widget>[
              DropdownButtonFormField<String>(
                value: _profile.event,
                decoration: const InputDecoration(labelText: 'Qualifying event'),
                items: const <String>[
                  'Post-MI',
                  'Coronary Bypass',
                  'Stent',
                  'Stable Angina',
                  'Heart Failure',
                  'None',
                ]
                    .map((value) => DropdownMenuItem<String>(
                          value: value,
                          child: Text(value),
                        ))
                    .toList(),
                onChanged: (value) => setState(() {
                  _profile = _profile.copyWith(event: value);
                }),
              ),
              const SizedBox(height: 16),
              ListTile(
                contentPadding: EdgeInsets.zero,
                leading: const Icon(Icons.calendar_today_outlined),
                title: const Text('Event date'),
                subtitle: Text(
                  MaterialLocalizations.of(context).formatMediumDate(_eventDate),
                ),
                trailing: FilledButton.tonal(
                  onPressed: _pickDate,
                  child: const Text('Change'),
                ),
              ),
              const SizedBox(height: 12),
              _numberRow(
                label: 'LVEF (%)',
                value: _profile.lvef,
                min: 10,
                max: 90,
                onChanged: (value) => setState(() {
                  _profile = _profile.copyWith(lvef: value);
                }),
              ),
              const SizedBox(height: 16),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: <Widget>[
                  FilterChip(
                    label: const Text('Diabetes'),
                    selected: _profile.comorbDia,
                    onSelected: (selected) => setState(() {
                      _profile = _profile.copyWith(comorbDia: selected);
                    }),
                  ),
                  FilterChip(
                    label: const Text('COPD'),
                    selected: _profile.comorbCopd,
                    onSelected: (selected) => setState(() {
                      _profile = _profile.copyWith(comorbCopd: selected);
                    }),
                  ),
                  FilterChip(
                    label: const Text('Hypertension'),
                    selected: _profile.comorbHyp,
                    onSelected: (selected) => setState(() {
                      _profile = _profile.copyWith(comorbHyp: selected);
                    }),
                  ),
                  FilterChip(
                    label: const Text('PAD'),
                    selected: _profile.comorbPad,
                    onSelected: (selected) => setState(() {
                      _profile = _profile.copyWith(comorbPad: selected);
                    }),
                  ),
                  FilterChip(
                    label: const Text('Renal disease'),
                    selected: _profile.comorbRen,
                    onSelected: (selected) => setState(() {
                      _profile = _profile.copyWith(comorbRen: selected);
                    }),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              DropdownButtonFormField<String>(
                value: _profile.betaBlocker,
                decoration: const InputDecoration(labelText: 'Beta blockers'),
                items: const <String>['No', 'Yes']
                    .map((value) => DropdownMenuItem<String>(
                          value: value,
                          child: Text(value),
                        ))
                    .toList(),
                onChanged: (value) => setState(() {
                  _profile = _profile.copyWith(betaBlocker: value);
                }),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildSymptomsTab() {
    return ListView(
      padding: const EdgeInsets.all(20),
      children: <Widget>[
        _sectionCard(
          title: 'Risk & symptoms',
          child: Column(
            children: <Widget>[
              DropdownButtonFormField<String>(
                value: _profile.tobacco,
                decoration: const InputDecoration(labelText: 'Tobacco history'),
                items: const <String>['Never', 'Former', 'Current']
                    .map((value) => DropdownMenuItem<String>(
                          value: value,
                          child: Text(value),
                        ))
                    .toList(),
                onChanged: (value) => setState(() {
                  _profile = _profile.copyWith(tobacco: value);
                }),
              ),
              const SizedBox(height: 16),
              _numberRow(
                label: 'Baseline activity level (1-5)',
                value: _profile.activityLevel,
                min: 1,
                max: 5,
                onChanged: (value) => setState(() {
                  _profile = _profile.copyWith(activityLevel: value);
                }),
              ),
              const SizedBox(height: 16),
              DropdownButtonFormField<String>(
                value: _profile.chestPain,
                decoration: const InputDecoration(labelText: 'Chest pain / angina'),
                items: const <String>['None', 'With Exertion', 'At Rest']
                    .map((value) => DropdownMenuItem<String>(
                          value: value,
                          child: Text(value),
                        ))
                    .toList(),
                onChanged: (value) => setState(() {
                  _profile = _profile.copyWith(chestPain: value);
                }),
              ),
              const SizedBox(height: 16),
              DropdownButtonFormField<String>(
                value: _profile.dyspnea,
                decoration: const InputDecoration(labelText: 'Dyspnea'),
                items: const <String>['None', 'On Exertion', 'Orthopnea']
                    .map((value) => DropdownMenuItem<String>(
                          value: value,
                          child: Text(value),
                        ))
                    .toList(),
                onChanged: (value) => setState(() {
                  _profile = _profile.copyWith(dyspnea: value);
                }),
              ),
              const SizedBox(height: 16),
              _numberRow(
                label: 'PHQ-2 score',
                value: _profile.phq2,
                min: 0,
                max: 6,
                onChanged: (value) => setState(() {
                  _profile = _profile.copyWith(phq2: value);
                }),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildBaselineTab() {
    return ListView(
      padding: const EdgeInsets.all(20),
      children: <Widget>[
        _sectionCard(
          title: 'Historical baseline',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text('Enable historical baseline notes'),
                subtitle: const Text(
                  'Google Fit OAuth is intentionally left out of the mobile port for now. '
                  'Use this section to capture any imported or manually reviewed baseline context.',
                ),
                value: _profile.includeHistoricalBaseline,
                onChanged: (enabled) => setState(() {
                  _profile = _profile.copyWith(includeHistoricalBaseline: enabled);
                }),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: _baselineNotesController,
                minLines: 4,
                maxLines: 8,
                decoration: const InputDecoration(
                  labelText: 'Baseline notes',
                  alignLabelWithHint: true,
                  hintText: 'Example: 1 month resting HR trend, sleep quality, supervised exercise tolerance...',
                ),
              ),
              const SizedBox(height: 12),
              DecoratedBox(
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Padding(
                  padding: EdgeInsets.all(16),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: <Widget>[
                      Icon(Icons.info_outline),
                      SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          'This Flutter port focuses on live telemetry, mobile recording, and MQTT export. '
                          'If you still need the desktop Google Fit intake workflow, keep the Python tool in the loop.',
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _sectionCard({
    required String title,
    required Widget child,
  }) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(title, style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 16),
            child,
          ],
        ),
      ),
    );
  }

  Widget _numberRow({
    required String label,
    required int value,
    required int min,
    required int max,
    required ValueChanged<int> onChanged,
  }) {
    return Row(
      children: <Widget>[
        Expanded(child: Text(label)),
        IconButton(
          onPressed: value > min ? () => onChanged(value - 1) : null,
          icon: const Icon(Icons.remove_circle_outline),
        ),
        SizedBox(
          width: 56,
          child: Center(child: Text('$value')),
        ),
        IconButton(
          onPressed: value < max ? () => onChanged(value + 1) : null,
          icon: const Icon(Icons.add_circle_outline),
        ),
      ],
    );
  }

  Widget _sliderRow({
    required String label,
    required double value,
    required double min,
    required double max,
    required int divisions,
    required ValueChanged<double> onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        Text('$label: ${value.toStringAsFixed(1)}'),
        Slider(
          value: value.clamp(min, max),
          min: min,
          max: max,
          divisions: divisions,
          label: value.toStringAsFixed(1),
          onChanged: onChanged,
        ),
      ],
    );
  }

  Future<void> _pickDate() async {
    final picked = await showDatePicker(
      context: context,
      firstDate: DateTime(2000),
      lastDate: DateTime.now(),
      initialDate: _eventDate,
    );
    if (picked != null) {
      setState(() {
        _eventDate = picked;
      });
    }
  }

  Future<void> _saveAndContinue() async {
    final subjectId = _subjectIdController.text.trim();
    if (subjectId.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Subject ID is required.')),
      );
      return;
    }

    final nextProfile = _profile.copyWith(
      subjectId: subjectId,
      eventDateIso: _eventDate.toIso8601String(),
      baselineNotes: _baselineNotesController.text.trim(),
    );
    await widget.controller.saveProfile(nextProfile);
    if (!mounted) {
      return;
    }
    widget.onContinue();
  }

  void _loadProfile(PatientProfile profile) {
    _profile = profile;
    if (!_controllersReady) {
      _subjectIdController = TextEditingController(text: profile.subjectId);
      _baselineNotesController = TextEditingController(text: profile.baselineNotes);
      _controllersReady = true;
    } else {
      _subjectIdController.text = profile.subjectId;
      _baselineNotesController.text = profile.baselineNotes;
    }
    _eventDate = DateTime.tryParse(profile.eventDateIso) ?? DateTime.now();
  }
}
