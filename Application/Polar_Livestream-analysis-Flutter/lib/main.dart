import 'package:flutter/material.dart';

import 'src/dashboard_screen.dart';
import 'src/intake_screen.dart';
import 'src/services.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const PulseForgeMobileApp());
}

class PulseForgeMobileApp extends StatelessWidget {
  const PulseForgeMobileApp({super.key});

  @override
  Widget build(BuildContext context) {
    final scheme = ColorScheme.fromSeed(
      seedColor: const Color(0xFF3B82F6),
      brightness: Brightness.dark,
    );

    return MaterialApp(
      title: 'PulseForge Mobile',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: scheme,
        brightness: Brightness.dark,
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFF0B1120),
        cardTheme: CardTheme(
          color: const Color(0xFF111827),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: const Color(0xFF1F2937),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(14),
            borderSide: BorderSide.none,
          ),
        ),
      ),
      home: const PulseForgeHome(),
    );
  }
}

class PulseForgeHome extends StatefulWidget {
  const PulseForgeHome({super.key});

  @override
  State<PulseForgeHome> createState() => _PulseForgeHomeState();
}

class _PulseForgeHomeState extends State<PulseForgeHome> {
  late final PulseForgeController _controller;
  var _ready = false;
  var _showDashboard = false;

  @override
  void initState() {
    super.initState();
    _controller = PulseForgeController();
    _bootstrap();
  }

  @override
  void dispose() {
    _controller.disposeController();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_ready) {
      return const Scaffold(
        body: Center(
          child: CircularProgressIndicator(),
        ),
      );
    }

    return AnimatedBuilder(
      animation: _controller,
      builder: (context, _) {
        if (_showDashboard && _controller.profile.isValid) {
          return DashboardScreen(
            controller: _controller,
            onEditIntake: () {
              setState(() {
                _showDashboard = false;
              });
            },
          );
        }

        return IntakeScreen(
          controller: _controller,
          onContinue: () {
            setState(() {
              _showDashboard = true;
            });
          },
        );
      },
    );
  }

  Future<void> _bootstrap() async {
    await _controller.initialize();
    if (!mounted) {
      return;
    }
    setState(() {
      _ready = true;
      _showDashboard = _controller.profile.isValid;
    });
  }
}
