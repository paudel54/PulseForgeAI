# Polar Livestream Analysis Flutter

Android-first Flutter port of `Application/Polar_Livestream-analysis-Python`.

## What this port keeps

- Patient intake flow with persisted local state
- Polar device scan, connect, disconnect, and live streaming
- Real-time ECG, accelerometer, and heart-rate dashboard
- Rolling 5-second analytics
- Rolling 30-second HRV analytics
- Local JSONL recording/export per subject/session
- MQTT publishing with the same `pulseforgeai/{subject_id}/info` and `pulseforgeai/{subject_id}/raw` topic shape
- Mock mode for UI testing without hardware

## What is intentionally simplified

- Google Fit OAuth and historical sync are not ported yet
- PyTorch HAR fusion inference is replaced with deterministic activity heuristics
- Full NeuroKit-style ECG morphology delineation is not attempted on-device
- `carp_polar_package` is included as a compatibility layer and future CARP integration hook, but the live runtime stream uses the underlying `polar` plugin directly for a lean standalone app
- `scientisst` is included, but the package is still under development, so the signal pipeline uses custom Dart DSP utilities today

## Packages used

- `flutter_hrv` for RR-interval HRV metrics
- `carp_polar_package` for Polar/CARP compatibility descriptors
- `scientisst` as a reserved signal-processing integration point
- `mqtt_client` for telemetry publishing
- `polar` for direct device streaming
- `path_provider` and `shared_preferences` for local mobile persistence
- `permission_handler` for BLE-related runtime permissions

## Android notes

The Android project includes the Polar BLE permissions and MQTT networking permissions required by the chosen package stack. Because this repository did not have Flutter configured locally when this port was added, you should set up Flutter on the machine, run `flutter pub get`, and then `flutter run` from this folder.

## Folder layout

- `lib/main.dart` — application entry and app shell
- `lib/src/models.dart` — typed models and JSON payloads
- `lib/src/services.dart` — Polar, MQTT, storage, analytics, and controller logic
- `lib/src/intake_screen.dart` — intake flow
- `lib/src/dashboard_screen.dart` — live dashboard UI
- `lib/src/widgets.dart` — charts and reusable metric widgets
