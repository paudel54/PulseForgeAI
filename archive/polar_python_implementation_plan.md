# Goal Description
The objective is twofold:
1. Replace the currently utilized `neurokit2` ECG Signal Quality Index (SQI) with the `vital_sqi` package to provide more accurate quality assessment of the 5-second ECG windows.
2. Overhaul the Patient Profile GUI by introducing a comprehensive, 15-question Patient Intake Form categorized into three tabs (Windows). This form will automatically load previous entries from a JSON config, provide a "Clear" function, and must be filled out to configure the subject ID.

## User Review Required
> [!IMPORTANT]
> The `vital_sqi` repository computes over 70 different Signal Quality Indices (e.g., standard statistics, morphological DTW, ectopic RR counts). Since the current dashboard outputs a single `sqi` float value between 0 and 1 (from NeuroKit2), I need to know: **Which specific `vital_sqi` metric do you want to display on the dashboard?** (e.g., Kurtosis, pSQI, kSQI, or a custom combination of them?). If you aren't sure, I can use a standard statistical SQI like Shannon Entropy or Kurtosis as an initial placeholder.

> [!NOTE]
> For the new GUI, I plan to run the Intake Form as a distinct modal dialog (`QDialog`) that pops up **before** the main dashboard is shown. It will block the app until the user either completes the form, loads past JSON, or clicks proceed. The right-hand panel on the dashboard will then just display a summary of the loaded patient ID and a button to "Edit Intake Form" instead of the current small profile inputs. Let me know if you prefer a different layout.

## Proposed Changes

### Dependencies
#### [MODIFY] `requirements.txt`
- Add `vital_sqi>=0.1.0` to the signal processing requirements section.

### Data Acquisition & Processing Layer
#### [MODIFY] `polar_ecg/workers/processing_worker.py`
- Import `vital_sqi` inside the `_compute_5s_window` function.
- Replace `nk.ecg_quality()` implementation. Feed the 5-second `ecg_cleaned` payload into the chosen `vital_sqi.sqi` extraction pipeline.
- Return the extracted float value as the `sqi` result in the emitted `window_result`.

### UI Layer
#### [NEW] `polar_ecg/ui/intake_form.py`
- Create `IntakeFormDialog` inheriting from `QDialog`.
- Implement `QTabWidget` with three tabs: 
  1. Demographics & Physicals (Age, Sex, Height, Weight, target HR). 
  2. Clinical History (Event, Date, LVEF, Co-morbidities, Beta-blockers). 
  3. Risk & Symptoms (Tobacco, Activity lvl, Chest pain, Dyspnea, PHQ-2).
- Apply the Industrial Dark Mode styling (`#0B1120`, `#1F2937`, `#3B82F6`, `#76B900`).
- Add bottom actions: "Load Past JSON", "Save & Continue", and a bold red "Clear" button.
- Implement file operations to save/load user choices to a local `intake_state.json` file.

#### [MODIFY] `polar_ecg/ui/dashboard.py`
- Remove the old `_build_user_profile_panel`.
- Adjust the `_build_recording_panel` to reflect the Subject ID loaded from the Intake form.
- Add an "Edit Intake Form" button to allow re-opening the QDialog while the dashboard is running.

#### [MODIFY] `main.py`
- Instantiate and `exec_()` the `IntakeFormDialog` before calling `MainDashboard.show()`.
- Pass the filled context payload (especially Subject ID) down into the dashboard.

## Open Questions
1. **SQI Metric Formulation**: As mentioned, which specific vital_sqi metric(s) should replace the single NK SQI value?
2. **Schema for `vital_sqi`**: `vital_sqi` operates heavily on standard `scipy` signals but requires formatting. Are we checking the 5s window directly without RR-based SQI since RR intervals in a 5s window are extremely sparse?

## Verification Plan
### Automated Tests
- N/A

### Manual Verification
1. Run `python main.py --mock`. The Intake Form should appear instantly.
2. Verify saving, loading, and clearing the Intake Form populates the GUI.
3. The dashboard should process mock data and the SQI readouts should actively reflect the values calculated by the newly integrated `vital_sqi` library instead of `neurokit2`.
